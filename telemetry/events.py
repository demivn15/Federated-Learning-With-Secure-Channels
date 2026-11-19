"""Registro de eventos de un nodo (E9): `nodes/<node>/events.jsonl`, una línea JSON por evento.

Cada evento lleva dos marcas de tiempo:
  * `t_wall_ns` — reloj de pared del nodo. Solo se compara con marcas tomadas con el **mismo** reloj: en
    Docker, todos los contenedores (y el demonio, que sella los eventos de inyección de fallos) comparten el
    reloj del kernel de la VM; en las Pi no (RNF-07), y ahí el tiempo de detección se mide en el nodo.
  * `t_mono_ns` — reloj monótono local (intervalos dentro del nodo).
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Mapping


def events_path(run_dir: str | Path, node_id: str) -> Path:
    return Path(run_dir) / "nodes" / node_id / "events.jsonl"


class EventLog:
    def __init__(self, path: str | Path, node_id: str) -> None:
        self.path = Path(path)
        self.node_id = node_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def __call__(self, event: str, details: Mapping[str, Any] | None = None) -> None:
        row = {"node_id": self.node_id, "event": event, "t_wall_ns": time.time_ns(),
               "t_mono_ns": time.perf_counter_ns(), **(dict(details) if details else {})}
        line = json.dumps(row, sort_keys=True, default=str)
        with self._lock, open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def read_events(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
