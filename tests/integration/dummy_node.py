"""Nodo de prueba para el ejecutor de E4 (los orquestadores reales llegan en E5).

Escribe 6 rondas de KPIs con el esquema real y, si es `infra`, un checkpoint `global` por ronda
(MLP de referencia con semilla = ronda). `--sleep` simula un nodo colgado (prueba de timeout).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.model import build_model  # noqa: E402
from experiments.manifest import load_manifest  # noqa: E402
from node.checkpoint import save_checkpoint  # noqa: E402
from telemetry.kpi_log import KpiLogger, kpi_path  # noqa: E402

ROUNDS = 6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--node", required=True)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--n-features", type=int, default=278)
    a = ap.parse_args()
    time.sleep(a.sleep)
    m = load_manifest(a.run_dir)
    log = KpiLogger(kpi_path(a.run_dir, a.node), topology=m["topology"], experiment_id=m["experiment_id"],
                    manifest_sha256=m["manifest_sha256"], node_id=a.node)
    for r in range(1, ROUNDS + 1):
        if a.node == "infra":
            cfg = dict(m["config"]); cfg["data"] = {**cfg["data"], "seed": r}
            state = {k: v.detach().clone() for k, v in build_model(cfg, a.n_features).state_dict().items()}
            save_checkpoint(a.run_dir, "infra", r, 0, "global", state, {"aggregator_id": None})
            log.log({"role": "server", "method": m["method"], "round": r, "sub_round": 0, "straggler_s": 0.01 * r})
        else:
            log.log({"role": "client", "method": m["method"], "round": r, "sub_round": 0, "n_samples": 100,
                     "tau": 7, "legacy_local_recall": 0.5 + 0.01 * r, "legacy_local_accuracy": 0.8,
                     "legacy_local_precision": 0.6, "legacy_local_f1": 0.55})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
