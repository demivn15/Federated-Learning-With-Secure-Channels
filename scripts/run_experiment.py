"""Ejecuta un experimento completo en local: manifiesto → broker → nodos → recogida → evaluación → limpieza.

    python scripts/run_experiment.py --topology centralized --method fedavg [--seed 42]
           [--rounds N] [--epochs E] [--threads 2] [--variant model_120_84 ...] [--id nombre]

`--rounds` y `--epochs` sobrescriben la referencia (pruebas rápidas); sin ellos se usa la referencia de D0
(centralizada: 12 rondas, 3 épocas). La corrida queda en `results/runs/<id>/`.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from broker.provision_broker import admin_from_env  # noqa: E402
from core.config import PROJECT_ROOT, load_config  # noqa: E402
from experiments.collect import collect_kpis  # noqa: E402
from experiments.evaluate_run import evaluate_run  # noqa: E402
from experiments.manifest import build_manifest  # noqa: E402
from experiments.runner import RUNS_ROOT, launch_local, prepare_run, teardown  # noqa: E402


def make_config(variants, rounds=None, epochs=None, topology="centralized", overrides=None) -> dict:
    cfg = load_config(legacy_variants=variants)
    cfg = copy.deepcopy(cfg)
    if rounds is not None:
        cfg["topologies"][topology]["rounds"] = {"policy": "fixed", "value": int(rounds)}
    if epochs is not None:
        cfg["training"]["local_epochs"] = int(epochs)
    for path, value in (overrides or {}).items():
        node = cfg
        *parents, last = path.split(".")
        for p in parents:
            node = node[p]
        node[last] = value
    return cfg


def participants(cfg: dict, topology: str) -> list[str]:
    """Nodos que se aprovisionan y lanzan. En P2P la 6.ª Pi no tiene rol: solo el broker (04 §4.4)."""
    if topology == "p2p":
        return sorted(cfg["topologies"]["p2p"]["peers"])
    return list(cfg["platform"]["node_assignment"])


def run_experiment(topology: str, method: str, *, seed: int = 42, rounds: int | None = None,
                   epochs: int | None = None, threads: int | None = None, variants=(), experiment_id=None,
                   processed_dir: Path = PROJECT_ROOT / "data" / "processed", runs_root: Path = RUNS_ROOT,
                   timeout_s: float = 3600, command_overrides: dict | None = None,
                   config_overrides: dict | None = None, study: str | None = None) -> dict:
    cfg = make_config(list(variants), rounds, epochs, topology, config_overrides)
    cfg["data"]["seed"] = seed
    exp_id = experiment_id or f"{topology[:4]}-{method}-s{seed}-{datetime.now():%Y%m%d%H%M%S}-{uuid.uuid4().hex[:4]}"
    manifest = build_manifest(cfg, experiment_id=exp_id, topology=topology, method=method, seed=seed,
                              processed_dir=processed_dir, study=study)
    nodes = participants(cfg, topology)
    admin = admin_from_env()
    run_dir = prepare_run(manifest, nodes, runs_root=runs_root, broker_admin=admin)
    command_overrides = command_overrides or {}

    def command(rd: Path, node: str) -> list[str]:
        if node in command_overrides:
            return command_overrides[node](rd, node)
        cmd = [sys.executable, "-m", "node.main", "--run-dir", str(rd), "--node", node,
               "--processed", str(processed_dir)]
        return cmd + (["--threads", str(threads)] if threads else [])

    try:
        # El servidor arranca primero en la lista para que su cola esté consumiéndose cuanto antes
        order = (["infra"] if "infra" in nodes else []) + [n for n in nodes if n != "infra"]
        result = launch_local(run_dir, order, timeout_s=timeout_s, command=command)
    finally:
        teardown(manifest, admin)
    out = {"run_dir": run_dir, "result": result}
    lead_ok = result.exit_codes.get("infra") == 0 if "infra" in nodes else any(
        c == 0 for c in result.exit_codes.values())
    if lead_ok:
        out["kpis"] = collect_kpis(run_dir)
        out["evaluation"] = evaluate_run(run_dir, processed_dir)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topology", default="centralized")
    ap.add_argument("--method", required=True, choices=["fedavg", "fedprox", "fednova"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--variant", action="append", default=[])
    ap.add_argument("--id")
    ap.add_argument("--study")
    ap.add_argument("--timeout", type=float, default=3600)
    args = ap.parse_args(argv)
    out = run_experiment(args.topology, args.method, seed=args.seed, rounds=args.rounds, epochs=args.epochs,
                         threads=args.threads, variants=args.variant, experiment_id=args.id,
                         timeout_s=args.timeout, study=args.study)
    res = out["result"]
    print(f"Corrida: {out['run_dir']}")
    print(f"Códigos de salida: {res.exit_codes} · duración del servidor {res.durations_s.get('infra', 0):.1f} s")
    if "evaluation" in out:
        print(json.dumps(out["evaluation"]["summary"], indent=2))
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
