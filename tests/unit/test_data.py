"""Datos de D0: partición (Tabla 1), split 60/20/20 estratificado, preprocesador y verificación de hashes."""
from __future__ import annotations

import shutil

import pandas as pd
import pytest
import torch

from core.data import (SPLITS, load_hospital, load_manifest, load_partition, read_csv, sha256_file)
from tests.conftest import SILO_DIR

HOSPITALS = ("H1", "H2", "H3", "H4", "H5")


def test_partition_matches_d0_table1(cfg):
    part = load_partition(SILO_DIR, cfg["data"])
    assert tuple(part) == HOSPITALS
    for h, spec in cfg["data"]["hospitals"].items():
        assert len(part[h]) == spec["rows"]
        assert int(part[h]["is_premature_ncd"].sum()) == spec["positives_local"]
    h4 = pd.concat([read_csv(SILO_DIR / "silo_4.csv"), read_csv(SILO_DIR / "silo_6.csv")])
    assert len(h4) == 1191 and int(h4["is_premature_ncd"].sum()) == 1185


def test_manifest_and_split_proportions(processed_dir, cfg):
    man = load_manifest(processed_dir)
    assert man["totals"] == {"rows": 39083, "positives": 7184, "positives_d0": 7191}
    assert man["n_features"] == cfg["data"]["preprocessing"]["expected_n_features"] == 278
    for h in HOSPITALS:
        r = man["partition_report"][h]
        rows = r["split_rows"]
        assert sum(rows.values()) == r["rows"]
        assert rows["train"] == pytest.approx(0.6 * r["rows"], abs=1)
        assert rows["val"] == pytest.approx(0.2 * r["rows"], abs=1)
        # Estratificación: la prevalencia de cada split difiere ≤ 1 positivo de la proporcional
        for s in SPLITS:
            expected = r["positives"] * rows[s] / r["rows"]
            assert abs(r["split_positives"][s] - expected) <= 1.0, (h, s)


def test_splits_are_disjoint_and_cover_hospital(processed_dir, cfg):
    for h in HOSPITALS:
        ids = [set(read_csv(processed_dir / f"splits/{h}_{s}.csv")["row_id"]) for s in SPLITS]
        assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), h
        assert set().union(*ids) == set(range(cfg["data"]["hospitals"][h]["rows"])), h


def test_load_hospital_returns_clean_tensors(processed_dir, cfg):
    d = load_hospital(processed_dir, "H3", cfg["data"])
    assert d.n_features == 278 and d.n_train == 167
    for X, y in ((d.X_train, d.y_train), (d.X_val, d.y_val), (d.X_test, d.y_test)):
        assert X.dtype == torch.float32 and y.shape == (X.shape[0], 1)
        assert torch.isfinite(X).all() and set(y.unique().tolist()) <= {0.0, 1.0}


def test_tampered_artifact_is_rejected(processed_dir, cfg, tmp_path):
    copy = tmp_path / "processed"
    shutil.copytree(processed_dir, copy)
    target = copy / "splits" / "H3_test.csv"
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Hash"):
        load_hospital(copy, "H3", cfg["data"])


def test_prepare_data_is_deterministic(processed_dir, tmp_path):
    from scripts.prepare_data import main
    assert main(["--silo-dir", str(SILO_DIR), "--out", str(tmp_path / "again")]) == 0
    a, b = load_manifest(processed_dir), load_manifest(tmp_path / "again")
    split_files = [k for k in a["outputs"] if k.startswith("splits/")]
    assert {k: a["outputs"][k]["sha256"] for k in split_files} == {k: b["outputs"][k]["sha256"] for k in split_files}
    assert sha256_file(processed_dir / "splits/H1_train.csv") == a["outputs"]["splits/H1_train.csv"]["sha256"]
