"""Inyección de fallos en contenedores (E9; 04 ADR-12, RF-27).

Un `Fault` = acción sobre un nodo (o sobre el broker) + disparador + duración opcional (con reversión
automática). Acciones (todas con la CLI de Docker, sin ganchos en el código de los nodos):

  kill        docker kill                     (caída abrupta del proceso; no vuelve)
  stop        docker stop -t 0                (= kill para el broker: pierde los mensajes no persistentes)
  pause       docker pause  [+ unpause]       (proceso congelado: sin CPU ni heartbeats AMQP)
  restart     docker restart                  (broker: caída y vuelta; vhosts/usuarios/colas durables persisten)
  disconnect  docker network disconnect [+ connect]   (partición de red: el nodo pierde el broker)
  netem       tc qdisc ... netem <args> [+ borrar]    (latencia, jitter, pérdida; requiere NET_ADMIN)
  start       docker start

Disparadores: `after_s` (segundos desde el lanzamiento) o `when_log` = (nodo, regex) sobre el log del nodo.

Cada acción se registra en `<run>/faults.jsonl` con la hora del **demonio de Docker** (`docker events`,
`timeNano`): en Docker Desktop es el reloj de la VM, el mismo que leen los contenedores, así que el tiempo
de detección = `events.jsonl` del nodo − hora del fallo se calcula con un único reloj (RNF-07).
"""
from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ACTIONS = ("kill", "stop", "pause", "restart", "disconnect", "netem", "start")
REVERT = {"pause": "unpause", "disconnect": "connect", "netem": "netem_clear", "stop": "start"}
EVENT_ACTION = {"kill": ("kill",), "stop": ("kill", "die", "stop"), "pause": ("pause",), "unpause": ("unpause",),
                "restart": ("kill", "die", "restart"), "start": ("start",), "disconnect": ("disconnect",),
                "connect": ("connect",), "netem": ("exec_start",), "netem_clear": ("exec_start",)}


@dataclass
class Fault:
    action: str
    target: str                                  # node_id o "broker"
    after_s: float | None = None
    when_log: tuple[str, str] | None = None      # (node_id, regex)
    duration_s: float | None = None              # revierte la acción al vencer (pause, disconnect, netem, stop)
    netem: str = ""                              # p. ej. "delay 200ms 50ms loss 5%"
    label: str = ""

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"Acción desconocida: {self.action}")
        if (self.after_s is None) == (self.when_log is None):
            raise ValueError("Indica exactamente uno de after_s o when_log")
        if self.duration_s is not None and self.action not in REVERT:
            raise ValueError(f"{self.action} no admite duración")
        if self.action == "netem" and not self.netem:
            raise ValueError("netem requiere argumentos")


def _run(args: list[str], timeout: float = 60) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def _events(container: str, actions: tuple[str, ...], since_host_s: float) -> set[int]:
    """`timeNano` de los eventos de `container` con una de `actions` desde `since_host_s` (margen de 10 s por
    el desfase entre el reloj del PC y el de la VM de Docker)."""
    res = _run(["docker", "events", "--since", f"{since_host_s - 10:.0f}", "--until", f"{time.time() + 2:.0f}",
                "--format", "{{json .}}"], timeout=30)
    ident = _run(["docker", "inspect", "-f", "{{.Id}}", container]).stdout.strip()   # los eventos de red usan el ID
    out = set()
    for line in res.stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        attrs = ev.get("Actor", {}).get("Attributes", {})
        if not ({container, ident} & {attrs.get("name"), attrs.get("container"), ev.get("id")} - {None, ""}):
            continue
        act = str(ev.get("Action", ""))
        if any(act == a or act.startswith(a) for a in actions) and ev.get("timeNano"):
            out.add(int(ev["timeNano"]))
    return out


class FaultInjector:
    """Ejecuta un plan de fallos en un hilo mientras corren los contenedores del experimento."""

    def __init__(self, faults: list[Fault], containers: dict[str, str], network: str, log_path: Path,
                 read_log: Callable[[str], str]) -> None:
        self.faults = faults
        self.containers = containers                 # target → nombre del contenedor (incluye "broker")
        self.network = network
        self.log_path = log_path
        self.read_log = read_log
        self.records: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="fault-injector", daemon=True)
        self._t0 = time.monotonic()

    def start(self) -> None:
        self._t0 = time.monotonic()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=120)

    # -------------------------------------------------------------- acciones
    def _command(self, action: str, f: Fault) -> list[str]:
        c = self.containers[f.target]
        return {
            "kill": ["docker", "kill", c], "stop": ["docker", "stop", "-t", "0", c],
            "pause": ["docker", "pause", c], "unpause": ["docker", "unpause", c],
            "restart": ["docker", "restart", "-t", "0", c], "start": ["docker", "start", c],
            "disconnect": ["docker", "network", "disconnect", self.network, c],
            "connect": ["docker", "network", "connect", self.network, c],
            "netem": ["docker", "exec", c, "tc", "qdisc", "replace", "dev", "eth0", "root", "netem", *f.netem.split()],
            "netem_clear": ["docker", "exec", c, "tc", "qdisc", "del", "dev", "eth0", "root"],
        }[action]

    def _do(self, action: str, f: Fault) -> None:
        # La hora del fallo es el PRIMER evento nuevo que provoca la acción (p. ej. `kill` antes que `die` y
        # `stop`): se comparan los eventos antes y después para no confundirlo con eventos anteriores.
        name, kinds = self.containers[f.target], EVENT_ACTION[action]
        since = time.time()
        before = _events(name, kinds, since)
        res = _run(self._command(action, f))
        new = _events(name, kinds, since) - before
        t_daemon = min(new) if new else None
        rec = {"action": action, "target": f.target, "label": f.label, "netem": f.netem or None,
               "ok": res.returncode == 0, "stderr": res.stderr.strip()[:300] or None,
               "t_daemon_ns": t_daemon, "t_host_ns": int(since * 1e9),
               "t_since_launch_s": round(time.monotonic() - self._t0, 3)}
        self.records.append(rec)
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")

    def _triggered(self, f: Fault) -> bool:
        if f.after_s is not None:
            return time.monotonic() - self._t0 >= f.after_s
        node, pattern = f.when_log
        return re.search(pattern, self.read_log(node)) is not None

    def _loop(self) -> None:
        pending = list(self.faults)
        reverts: list[tuple[float, str, Fault]] = []
        while not self._stop.is_set() and (pending or reverts):
            now = time.monotonic()
            for f in list(pending):
                if self._triggered(f):
                    pending.remove(f)
                    self._do(f.action, f)
                    if f.duration_s is not None:
                        reverts.append((now + f.duration_s, REVERT[f.action], f))
            for item in list(reverts):
                if time.monotonic() >= item[0]:
                    reverts.remove(item)
                    self._do(item[1], item[2])
            self._stop.wait(0.5)
        for _, action, f in reverts:                 # deja el entorno como estaba (p. ej. broker parado)
            self._do(action, f)


def read_faults(run_dir: str | Path) -> list[dict[str, Any]]:
    p = Path(run_dir) / "faults.jsonl"
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

