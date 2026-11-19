"""Ejecutor local de experimentos (RF-25, RF-26): prepara la carpeta de la corrida, aprovisiona el broker,
lanza un proceso por nodo, espera con timeout y limpia.

Estructura de una corrida (`results/runs/<experiment_id>/`):
  manifest.json            manifiesto (experiments/manifest.py)
  credentials.json         credenciales AMQP por nodo (solo si hay broker; `results/` no se versiona)
  logs/<node>.log          salida de cada proceso
  nodes/<node>/kpi.csv     KPIs por ronda (telemetry/kpi_log.py)
  checkpoints/<node>/      checkpoints por ronda (node/checkpoint.py)
El lanzamiento por SSH en las Raspberry Pi se añade en E11; aquí los nodos son procesos locales.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from core.config import PROJECT_ROOT
from experiments.manifest import save_manifest

RUNS_ROOT = PROJECT_ROOT / "results" / "runs"
CommandFactory = Callable[[Path, str], Sequence[str]]


def default_command(run_dir: Path, node_id: str) -> list[str]:
    return [sys.executable, "-m", "node.main", "--run-dir", str(run_dir), "--node", node_id]


@dataclass
class RunResult:
    run_dir: Path
    exit_codes: dict[str, int | None]
    durations_s: dict[str, float]
    timed_out: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.timed_out and all(c == 0 for c in self.exit_codes.values())


def prepare_run(manifest: Mapping[str, Any], node_ids: Sequence[str], runs_root: str | Path = RUNS_ROOT,
                broker_admin: Any | None = None) -> Path:
    run_dir = Path(runs_root) / manifest["experiment_id"]
    if run_dir.exists():
        raise FileExistsError(f"La corrida ya existe: {run_dir}")
    for sub in ("logs", "nodes", "checkpoints"):
        (run_dir / sub).mkdir(parents=True)
    save_manifest(manifest, run_dir)
    if broker_admin is not None:
        bcfg = manifest["config"]["communication"]["broker"]
        creds = broker_admin.provision_experiment(manifest["experiment_id"], node_ids, bcfg)
        (run_dir / "credentials.json").write_text(
            json.dumps({n: c.__dict__ for n, c in creds.items()}, indent=2), encoding="utf-8")
    return run_dir


def launch_local(run_dir: str | Path, node_ids: Sequence[str], timeout_s: float,
                 command: CommandFactory = default_command) -> RunResult:
    run_dir = Path(run_dir)
    procs, logs, t0 = {}, {}, {}
    try:
        for n in node_ids:
            logs[n] = open(run_dir / "logs" / f"{n}.log", "w", encoding="utf-8")
            t0[n] = time.monotonic()
            procs[n] = subprocess.Popen(list(command(run_dir, n)), cwd=PROJECT_ROOT, stdout=logs[n],
                                        stderr=subprocess.STDOUT)
        deadline = time.monotonic() + timeout_s
        durations: dict[str, float] = {}
        while time.monotonic() < deadline and len(durations) < len(procs):
            for n, p in procs.items():
                if n not in durations and p.poll() is not None:
                    durations[n] = time.monotonic() - t0[n]
            time.sleep(0.1)
        timed_out = [n for n in procs if n not in durations]
        for n in timed_out:
            procs[n].kill()
            procs[n].wait()
            durations[n] = time.monotonic() - t0[n]
        return RunResult(run_dir, {n: p.returncode for n, p in procs.items()}, durations, timed_out)
    finally:
        for fh in logs.values():
            fh.close()


def teardown(manifest: Mapping[str, Any], broker_admin: Any | None) -> None:
    if broker_admin is not None:
        broker_admin.delete_experiment(manifest["experiment_id"],
                                       manifest["config"]["communication"]["broker"]["vhost_prefix"])
