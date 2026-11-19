"""Informe de una corrida con fallos (E9): tiempo hasta detectar el fallo y hasta terminar, por nodo.

    python -m experiments.fault_report results/runs/<exp>

Cruza `faults.jsonl` (hora del fallo según el demonio de Docker) con `nodes/<nodo>/events.jsonl` (hora de
pared del contenedor). En Docker Desktop ambos relojes son el de la VM, así que las diferencias son válidas;
en las Pi no se deben restar relojes de máquinas distintas (RNF-07).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from experiments.faults import read_faults
from telemetry.events import read_events

DETECTION_EVENTS = ("broker_connection_lost", "broker_unreachable")


def node_events(run_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    return {p.parent.name: read_events(p) for p in sorted(Path(run_dir).glob("nodes/*/events.jsonl"))}


def detection_report(run_dir: str | Path, fault_index: int = 0) -> list[dict[str, Any]]:
    """Por nodo: segundos desde el fallo `fault_index` hasta la primera detección, la reconexión, la pérdida
    definitiva del broker y la salida del proceso (None si no ocurrió)."""
    faults = [f for f in read_faults(run_dir) if f.get("t_daemon_ns")]
    if len(faults) <= fault_index:
        raise ValueError("La corrida no tiene ese fallo con hora del demonio")
    t_f = faults[fault_index]["t_daemon_ns"]
    rows = []
    for node, evs in node_events(run_dir).items():
        def first(names: tuple[str, ...]) -> float | None:
            ts = [e["t_wall_ns"] for e in evs if e["event"] in names and e["t_wall_ns"] >= t_f - 1_000_000_000]
            return round((min(ts) - t_f) / 1e9, 3) if ts else None
        exit_ev = [e for e in evs if e["event"] == "node_exit"]
        rows.append({"node": node, "fault": faults[fault_index]["action"], "target": faults[fault_index]["target"],
                     "detect_s": first(DETECTION_EVENTS), "reconnect_s": first(("broker_reconnected",)),
                     "lost_s": first(("broker_lost",)), "exit_s": first(("node_exit",)),
                     "exit_code": exit_ev[-1].get("code") if exit_ev else None})
    return rows


def main(argv: list[str] | None = None) -> int:
    run_dir = Path((argv or sys.argv[1:])[0])
    print(json.dumps({"faults": read_faults(run_dir), "detection": detection_report(run_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
