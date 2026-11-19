"""Aprovisionamiento del broker por experimento (04 ADR-03, §4.5) mediante la API de gestión de RabbitMQ.

Por experimento crea:
  * un vhost `fl-<experiment_id>`;
  * exchanges `fl.unicast` (direct) y `fl.broadcast` (fanout);
  * una cola `q.<node_id>` por nodo (durable; longitud máxima con `reject-publish`; TTL), ligada a ambos
    exchanges;
  * un usuario por nodo con permisos mínimos: configurar nada, escribir solo en los dos exchanges y leer
    solo su propia cola.
Las contraseñas se generan aleatoriamente y se devuelven; la CLI las guarda en `broker_credentials/`,
que no se versiona (RNF-10).

Solo usa la biblioteca estándar (urllib); se ejecuta en el PC de control o en la 6.ª Pi, nunca en los
nodos que entrenan.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import secrets
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

NODE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,32}$")
EXPERIMENT_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class BrokerAdminError(RuntimeError):
    pass


@dataclass(frozen=True)
class NodeCredentials:
    node_id: str
    username: str
    password: str
    vhost: str

    def amqp_url(self, host: str = "127.0.0.1", port: int = 5673) -> str:
        q = urllib.parse.quote
        return f"amqp://{q(self.username, safe='')}:{q(self.password, safe='')}@{host}:{port}/{q(self.vhost, safe='')}"


class BrokerAdmin:
    def __init__(self, api_url: str, user: str, password: str, timeout: float = 15.0) -> None:
        self.api_url = api_url.rstrip("/")
        self._auth = "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()
        self.timeout = timeout

    # -------------------------------------------------------------- HTTP
    def _req(self, method: str, path: str, body: Any = None, ok_missing: bool = False) -> Any:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.api_url + path, data=data, method=method,
                                     headers={"Authorization": self._auth, "content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            if ok_missing and exc.code == 404:
                return None
            raise BrokerAdminError(f"{method} {path} → HTTP {exc.code}: {exc.read()[:300]!r}") from None

    @staticmethod
    def _q(s: str) -> str:
        return urllib.parse.quote(s, safe="")

    # -------------------------------------------------------------- experimento
    @staticmethod
    def vhost_name(experiment_id: str, prefix: str = "fl-") -> str:
        if not EXPERIMENT_ID.match(experiment_id):
            raise ValueError(f"experiment_id inválido: {experiment_id!r}")
        return f"{prefix}{experiment_id}"

    def provision_experiment(self, experiment_id: str, node_ids: Iterable[str],
                             broker_cfg: Mapping[str, Any]) -> dict[str, NodeCredentials]:
        node_ids = list(node_ids)
        for n in node_ids:
            if not NODE_ID.match(n):
                raise ValueError(f"node_id inválido: {n!r}")
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("node_ids duplicados")
        vhost = self.vhost_name(experiment_id, broker_cfg.get("vhost_prefix", "fl-"))
        v = self._q(vhost)
        uni, bro = broker_cfg["unicast_exchange"], broker_cfg["broadcast_exchange"]
        qcfg = broker_cfg["queue"]

        self._req("PUT", f"/vhosts/{v}", {"description": f"FL experiment {experiment_id}"})
        self._req("PUT", f"/exchanges/{v}/{self._q(uni)}", {"type": "direct", "durable": True, "auto_delete": False})
        self._req("PUT", f"/exchanges/{v}/{self._q(bro)}", {"type": "fanout", "durable": True, "auto_delete": False})

        args = {"x-queue-type": "classic", "x-max-length": int(qcfg["max_length"]),
                "x-overflow": qcfg["overflow"], "x-message-ttl": int(qcfg["message_ttl_ms"])}
        write_re = "^(" + "|".join(re.escape(x) for x in (uni, bro)) + ")$"
        creds: dict[str, NodeCredentials] = {}
        for n in node_ids:
            queue = f"q.{n}"
            self._req("PUT", f"/queues/{v}/{self._q(queue)}",
                      {"durable": bool(qcfg.get("durable", True)), "auto_delete": False, "arguments": args})
            self._req("POST", f"/bindings/{v}/e/{self._q(uni)}/q/{self._q(queue)}", {"routing_key": n})
            self._req("POST", f"/bindings/{v}/e/{self._q(bro)}/q/{self._q(queue)}", {"routing_key": ""})
            user, password = f"{experiment_id}.{n}", secrets.token_urlsafe(24)
            self._req("PUT", f"/users/{self._q(user)}", {"password": password, "tags": ""})
            self._req("PUT", f"/permissions/{v}/{self._q(user)}",
                      {"configure": "^$", "write": write_re, "read": f"^{re.escape(queue)}$"})
            creds[n] = NodeCredentials(n, user, password, vhost)
        return creds

    def delete_experiment(self, experiment_id: str, prefix: str = "fl-") -> None:
        vhost = self.vhost_name(experiment_id, prefix)
        users = self._req("GET", "/users") or []
        for u in users:
            if u["name"].startswith(f"{experiment_id}."):
                self._req("DELETE", f"/users/{self._q(u['name'])}", ok_missing=True)
        self._req("DELETE", f"/vhosts/{self._q(vhost)}", ok_missing=True)

    # -------------------------------------------------------------- métricas del broker (R-03)
    def node_memory(self) -> dict[str, Any]:
        node = (self._req("GET", "/nodes") or [{}])[0]
        return {"mem_used": node.get("mem_used"), "mem_limit": node.get("mem_limit"),
                "mem_alarm": node.get("mem_alarm")}

    def overview(self) -> dict[str, Any]:
        o = self._req("GET", "/overview") or {}
        return {"rabbitmq_version": o.get("rabbitmq_version"), "erlang_version": o.get("erlang_version")}

    def queue_stats(self, experiment_id: str, prefix: str = "fl-") -> dict[str, dict[str, Any]]:
        v = self._q(self.vhost_name(experiment_id, prefix))
        return {q["name"]: {"messages": q.get("messages", 0), "message_bytes": q.get("message_bytes", 0)}
                for q in (self._req("GET", f"/queues/{v}") or [])}


DEV_AMQP_PORT = 5673    # docker/compose.yaml publica el broker en 5673/15673
DEV_API_URL = "http://127.0.0.1:15673/api"


def admin_from_env(env_file: str | Path | None = None, api_url: str = DEV_API_URL) -> BrokerAdmin:
    """Credenciales del administrador desde variables de entorno o desde docker/.env."""
    user, password = os.environ.get("RABBITMQ_DEFAULT_USER"), os.environ.get("RABBITMQ_DEFAULT_PASS")
    env_file = Path(env_file) if env_file else Path(__file__).resolve().parents[1] / "docker" / ".env"
    if (not user or not password) and env_file.exists():
        kv = dict(line.split("=", 1) for line in env_file.read_text(encoding="utf-8").splitlines() if "=" in line)
        user, password = kv.get("RABBITMQ_DEFAULT_USER"), kv.get("RABBITMQ_DEFAULT_PASS")
    if not user or not password:
        raise BrokerAdminError("Faltan credenciales de administrador del broker")
    return BrokerAdmin(os.environ.get("FL_BROKER_API", api_url), user, password)


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.config import load_config

    ap = argparse.ArgumentParser(description="Aprovisiona o elimina el vhost de un experimento")
    ap.add_argument("action", choices=["create", "delete"])
    ap.add_argument("experiment_id")
    ap.add_argument("--nodes", default="CN1,CN2,CN3,CN4,CN5,infra")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "broker_credentials"))
    args = ap.parse_args(argv)
    broker_cfg = load_config()["communication"]["broker"]
    admin = admin_from_env()
    if args.action == "delete":
        admin.delete_experiment(args.experiment_id, broker_cfg["vhost_prefix"])
        print(f"Eliminado el experimento {args.experiment_id}")
        return 0
    creds = admin.provision_experiment(args.experiment_id, args.nodes.split(","), broker_cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.experiment_id}.json"
    path.write_text(json.dumps({n: c.__dict__ for n, c in creds.items()}, indent=2), encoding="utf-8")
    print(f"Aprovisionado {args.experiment_id}: {len(creds)} nodos; credenciales en {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
