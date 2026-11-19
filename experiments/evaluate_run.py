"""Evaluador fuera de línea (04 ADR-07): aplica el protocolo de D0 §3.7 a los checkpoints de una corrida.

Se ejecuta en el PC de control, no en las Pi, para no alterar sus KPIs de CPU, energía y tiempo.

Para cada checkpoint de modelo (`global`, `cluster`, `mixed`):
  * umbral = el que maximiza el F1 macro en la VALIDACIÓN común (unión de los 20 % de validación);
  * métricas de red en el TEST común (unión de los 20 % de test) con ese umbral       → alcance "network";
  * métricas por hospital en su test, con el mismo umbral                         → D0 Tablas 5 y 6;
  * varianza del recall de validación entre hospitales                             → D0 Fig. 4;
  * si el checkpoint indica `aggregator_id` (SD): métricas del agregador en sus propios datos → D0 Tabla 7 †.
Resumen = media de las últimas 5 rondas (D0 §3.7). En P2P (`mixed`) primero se promedian los pares de cada
ronda. Si hay KPIs, se resume también la métrica heredada (media local previa a la agregación, D1 H1).

Uso:  python -m experiments.evaluate_run <run_dir> [--processed data/processed]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from core.config import PROJECT_ROOT
from core.data import load_hospital
from core.model import build_model
from evaluation.metrics import best_f1_threshold, binary_metrics, mean_last_k, predict_proba
from experiments.manifest import load_manifest, verify_inputs
from node.checkpoint import list_checkpoints, load_checkpoint

METRICS = ("accuracy", "precision", "recall", "f1")
MODEL_KINDS = ("global", "cluster", "mixed")


def _per_round_table(rows: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Una fila por ronda: el último sub_round de la ronda; en `mixed`, la media de los pares."""
    sub = rows[rows["kind"] == kind]
    if sub.empty:
        return sub
    last = sub.groupby("round")["sub_round"].transform("max")
    sub = sub[sub["sub_round"] == last]
    numeric = [c for c in sub.columns if c not in ("node_id", "kind") and pd.api.types.is_numeric_dtype(sub[c])]
    return sub.groupby("round", as_index=False)[numeric].mean()


def evaluate_run(run_dir: str | Path, processed_dir: str | Path, write: bool = True) -> dict[str, Any]:
    run_dir = Path(run_dir)
    manifest = load_manifest(run_dir)
    verify_inputs(manifest, processed_dir, check_code=False)
    cfg = manifest["config"]
    node_to_h = {n: a["hospital"] for n, a in cfg["platform"]["node_assignment"].items() if "hospital" in a}
    hospitals = sorted(set(node_to_h.values()))
    data = {h: load_hospital(processed_dir, h, cfg["data"]) for h in hospitals}
    X_val = torch.cat([data[h].X_val for h in hospitals]); y_val = torch.cat([data[h].y_val for h in hospitals]).numpy()
    X_te = torch.cat([data[h].X_test for h in hospitals]); y_te = torch.cat([data[h].y_test for h in hospitals]).numpy()
    model = build_model(cfg, data[hospitals[0]].n_features)

    rows = []
    for ref in list_checkpoints(run_dir):
        if ref.kind not in MODEL_KINDS:
            continue
        model.load_state_dict(load_checkpoint(ref))
        t = best_f1_threshold(y_val, predict_proba(model, X_val))
        row: dict[str, Any] = {"node_id": ref.node_id, "round": ref.round, "sub_round": ref.sub_round,
                               "kind": ref.kind, "threshold": t}
        row.update({f"net_{k}": v for k, v in binary_metrics(y_te, predict_proba(model, X_te), t).items()
                    if k in METRICS})
        val_recalls = []
        for h in hospitals:
            mh = binary_metrics(data[h].y_test.numpy(), predict_proba(model, data[h].X_test), t)
            row.update({f"{h}_{k}": mh[k] for k in ("recall", "f1")})
            val_recalls.append(binary_metrics(data[h].y_val.numpy(), predict_proba(model, data[h].X_val), t)["recall"])
        row["recall_variance_val"] = float(np.var(val_recalls))
        agg = ref.meta.get("aggregator_id")
        if agg in node_to_h:                                   # SD: alcance del nodo agregador (D0 Tabla 7 †)
            d = data[node_to_h[agg]]
            ta = best_f1_threshold(d.y_val.numpy(), predict_proba(model, d.X_val))
            row.update({f"agg_{k}": v for k, v in binary_metrics(d.y_test.numpy(), predict_proba(model, d.X_test), ta).items()
                        if k in METRICS})
        rows.append(row)
    if not rows:
        raise ValueError("La corrida no tiene checkpoints de modelo")
    table = pd.DataFrame(rows)

    summary: dict[str, Any] = {"experiment_id": manifest["experiment_id"], "topology": manifest["topology"],
                               "method": manifest["method"], "manifest_sha256": manifest["manifest_sha256"]}
    for kind in MODEL_KINDS:
        per_round = _per_round_table(table, kind)
        if per_round.empty:
            continue
        cols = [c for c in per_round.columns if c.startswith(("net_", "agg_")) or c == "recall_variance_val"]
        values = per_round[cols].dropna(axis=1, how="all")          # p. ej. agg_* solo existe en SD
        summary[kind] = {"rounds": int(per_round["round"].max()),
                         "mean_last_5": mean_last_k(values.to_dict("records"), 5)}
    kpis = run_dir / "kpis_all.csv"
    if kpis.exists():
        k = pd.read_csv(kpis)
        legacy = k.dropna(subset=["legacy_local_recall"]).groupby("round")[
            ["legacy_local_accuracy", "legacy_local_precision", "legacy_local_recall", "legacy_local_f1"]].mean()
        if not legacy.empty:
            summary["legacy_local"] = {"mean_last_5": mean_last_k(legacy.to_dict("records"), 5)}
    if write:
        table.to_csv(run_dir / "evaluation_rounds.csv", index=False)
        (run_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"table": table, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--processed", default=str(PROJECT_ROOT / "data" / "processed"))
    args = ap.parse_args(argv)
    result = evaluate_run(args.run_dir, args.processed)
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
