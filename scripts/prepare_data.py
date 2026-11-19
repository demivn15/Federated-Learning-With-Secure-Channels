"""Reconstruye la partición de D0 (Tabla 1), aplica el split 60/20/20 y ajusta el preprocesador de D0 §3.1.

Uso:
    python scripts/prepare_data.py [--config configs/variants/d0_reference.yaml]
                                   [--silo-dir ../FL_Hierarchical_Hospital/data/silos]
                                   [--out data/processed]

Salidas (en --out):
    splits/<H>_{train,val,test}.csv   filas crudas de cada split
    preprocessor.joblib              ColumnTransformer ajustado solo con los train
    manifest.json                    hashes de entradas y salidas, conteos, nº de features, supuestos

Solo lee los silos originales (P2 se mantiene congelado, RNF-11).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import DEFAULT_REFERENCE, PROJECT_ROOT, load_config  # noqa: E402
from core.data import (MANIFEST_NAME, PREPROCESSOR_NAME, SPLITS, build_preprocessor,  # noqa: E402
                       fit_preprocessor, load_partition, sha256_file, stratified_split)

DEFAULT_SILOS = PROJECT_ROOT.parent / "FL_Hierarchical_Hospital" / "data" / "silos"
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_REFERENCE))
    ap.add_argument("--silo-dir", default=str(DEFAULT_SILOS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    label, seed = data_cfg["label_column"], data_cfg["seed"]
    silo_dir, out = Path(args.silo_dir), Path(args.out)
    (out / "splits").mkdir(parents=True, exist_ok=True)

    partition = load_partition(silo_dir, data_cfg)

    report, outputs, splits_by_h = {}, {}, {}
    for h, df in partition.items():
        spec = data_cfg["hospitals"][h]
        pos = int(df[label].sum())
        splits = stratified_split(df, label, data_cfg["split"], seed)
        splits_by_h[h] = splits
        report[h] = {
            "rows": len(df), "positives": pos, "positives_d0": spec["positives_d0"],
            "positives_diff_vs_d0": pos - spec["positives_d0"],
            "pct_positive": round(100 * pos / len(df), 2),
            "split_rows": {s: len(splits[s]) for s in SPLITS},
            "split_positives": {s: int(splits[s][label].sum()) for s in SPLITS},
        }
        for s in SPLITS:
            rel = f"splits/{h}_{s}.csv"
            # row_id = posición de la fila en el hospital reconstruido (permite verificar que los splits son disjuntos)
            splits[s].to_csv(out / rel, index=True, index_label="row_id", encoding="utf-8", lineterminator="\n")
            outputs[rel] = {"sha256": sha256_file(out / rel), "rows": len(splits[s])}

    pre = fit_preprocessor(build_preprocessor(data_cfg),
                           [splits_by_h[h]["train"] for h in partition], data_cfg)
    n_features = int(len(pre.get_feature_names_out()))
    joblib.dump(pre, out / PREPROCESSOR_NAME)
    outputs[PREPROCESSOR_NAME] = {"sha256": sha256_file(out / PREPROCESSOR_NAME)}

    manifest = {
        "reference_id": cfg["reference_id"],
        "seed": seed,
        "split": data_cfg["split"],
        "preprocessing": data_cfg["preprocessing"],
        "n_features": n_features,
        "expected_n_features": data_cfg["preprocessing"].get("expected_n_features"),
        "inputs": {f: sha256_file(silo_dir / f)
                   for spec in data_cfg["hospitals"].values() for f in spec["source_files"]},
        "outputs": outputs,
        "partition_report": report,
        "totals": {"rows": sum(r["rows"] for r in report.values()),
                   "positives": sum(r["positives"] for r in report.values()),
                   "positives_d0": sum(r["positives_d0"] for r in report.values())},
        "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                        "numpy": np.__version__, "scikit-learn": sklearn.__version__},
        "assumptions": [
            "Preprocesador ajustado con la unión de los train por hospital: la cohorte nacional de D0 §3.1 no está disponible [SUP]",
            "Holdout común = unión de los test (20 %) estratificados por hospital [SUP Q-15]",
            "Hospital 4 = silo_4.csv + silo_6.csv de FL_Hierarchical_Hospital [VER]",
        ],
    }
    with open(out / MANIFEST_NAME, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    print(f"{'Hospital':<9}{'filas':>8}{'positivos':>11}{'D0':>7}{'dif':>6}{'% pos':>8}   train/val/test")
    for h, r in report.items():
        sr = r["split_rows"]
        print(f"{h:<9}{r['rows']:>8}{r['positives']:>11}{r['positives_d0']:>7}"
              f"{r['positives_diff_vs_d0']:>+6}{r['pct_positive']:>8}   {sr['train']}/{sr['val']}/{sr['test']}")
    t = manifest["totals"]
    print(f"{'Total':<9}{t['rows']:>8}{t['positives']:>11}{t['positives_d0']:>7}"
          f"{t['positives'] - t['positives_d0']:>+6}")
    exp = manifest["expected_n_features"]
    print(f"\nFeatures tras el preprocesamiento: {n_features} (referencia del código local: {exp})")
    print(f"Salida: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
