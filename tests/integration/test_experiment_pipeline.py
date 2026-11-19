"""E4: manifiesto → ejecutor → recogida → evaluador fuera de línea (criterios de aceptación de E4)."""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from core.data import load_hospital
from core.model import build_model
from evaluation.metrics import best_f1_threshold, binary_metrics, predict_proba
from experiments.collect import CollectError, collect_kpis
from experiments.evaluate_run import evaluate_run
from experiments.manifest import build_manifest
from experiments.runner import launch_local, prepare_run
from node.checkpoint import save_checkpoint

DUMMY = Path(__file__).with_name("dummy_node.py")
NODES = ["CN1", "CN2", "CN3", "CN4", "CN5", "infra"]
HOSPITALS = ["H1", "H2", "H3", "H4", "H5"]


def new_manifest(cfg, processed_dir, topology="centralized"):
    return build_manifest(cfg, experiment_id=f"e4-{uuid.uuid4().hex[:8]}", topology=topology, method="fedavg",
                          seed=42, processed_dir=processed_dir, study="e4-test")


def dummy(extra=()):
    return lambda run_dir, node: [sys.executable, str(DUMMY), "--run-dir", str(run_dir), "--node", node, *extra]


def test_runner_collect_and_evaluate(cfg, processed_dir, tmp_path):
    m = new_manifest(cfg, processed_dir)
    run_dir = prepare_run(m, NODES, runs_root=tmp_path)
    result = launch_local(run_dir, NODES, timeout_s=120, command=dummy())
    assert result.ok, (result.exit_codes, [(run_dir / "logs" / f"{n}.log").read_text() for n in NODES])

    kpis = collect_kpis(run_dir, expected_nodes=NODES)
    assert len(kpis) == 6 * len(NODES) and set(kpis["node_id"]) == set(NODES)
    assert (run_dir / "kpis_all.csv").exists()

    out = evaluate_run(run_dir, processed_dir)
    table, summary = out["table"], out["summary"]
    assert sorted(table["round"]) == [1, 2, 3, 4, 5, 6] and set(table["kind"]) == {"global"}
    assert summary["global"]["rounds"] == 6
    assert summary["global"]["mean_last_5"]["net_f1"] == pytest.approx(table["net_f1"].iloc[1:].mean())
    assert summary["legacy_local"]["mean_last_5"]["legacy_local_recall"] == pytest.approx(
        np.mean([0.5 + 0.01 * r for r in range(2, 7)]))
    assert (run_dir / "evaluation_summary.json").exists()


def test_evaluator_reproduces_known_model_metrics(cfg, processed_dir, tmp_path):
    """El evaluador da exactamente las métricas calculadas a mano para un modelo conocido."""
    m = new_manifest(cfg, processed_dir)
    run_dir = prepare_run(m, NODES, runs_root=tmp_path)
    data = {h: load_hospital(processed_dir, h, cfg["data"]) for h in HOSPITALS}
    model = build_model(cfg, data["H1"].n_features)
    with torch.no_grad():                                    # modelo conocido: pesos fijados con otra semilla
        torch.manual_seed(7)
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    save_checkpoint(run_dir, "infra", 1, 0, "global", {k: v.clone() for k, v in model.state_dict().items()},
                    {"aggregator_id": "CN4"})

    X_val = torch.cat([data[h].X_val for h in HOSPITALS]); y_val = torch.cat([data[h].y_val for h in HOSPITALS]).numpy()
    X_te = torch.cat([data[h].X_test for h in HOSPITALS]); y_te = torch.cat([data[h].y_test for h in HOSPITALS]).numpy()
    t = best_f1_threshold(y_val, predict_proba(model, X_val))
    expected = binary_metrics(y_te, predict_proba(model, X_te), t)
    d4 = data["H4"]
    t4 = best_f1_threshold(d4.y_val.numpy(), predict_proba(model, d4.X_val))
    expected_agg = binary_metrics(d4.y_test.numpy(), predict_proba(model, d4.X_test), t4)

    row = evaluate_run(run_dir, processed_dir)["table"].iloc[0]
    assert row["threshold"] == pytest.approx(t)
    for k in ("accuracy", "precision", "recall", "f1"):
        assert row[f"net_{k}"] == pytest.approx(expected[k])
        assert row[f"agg_{k}"] == pytest.approx(expected_agg[k])          # alcance SD † (agregador = CN4 → H4)


def test_p2p_mixed_models_are_averaged_per_round(cfg, processed_dir, tmp_path):
    m = new_manifest(cfg, processed_dir, topology="p2p")
    run_dir = prepare_run(m, NODES[:5], runs_root=tmp_path)
    for r in (1, 2):
        for i, n in enumerate(NODES[:5]):
            c = dict(cfg); c["data"] = {**cfg["data"], "seed": 10 * r + i}
            save_checkpoint(run_dir, n, r, 0, "mixed", build_model(c, 278).state_dict())
    out = evaluate_run(run_dir, processed_dir)
    table = out["table"]
    r2 = table[table["round"] == 2]
    assert len(r2) == 5
    assert out["summary"]["mixed"]["mean_last_5"]["net_f1"] == pytest.approx(
        table.groupby("round")["net_f1"].mean().mean())


def test_collect_detects_rows_from_another_manifest(cfg, processed_dir, tmp_path):
    m = new_manifest(cfg, processed_dir)
    run_dir = prepare_run(m, NODES, runs_root=tmp_path)
    assert launch_local(run_dir, NODES, timeout_s=120, command=dummy()).ok
    path = run_dir / "nodes" / "CN2" / "kpi.csv"
    df = pd.read_csv(path, dtype=str)
    df.loc[3, "manifest_sha256"] = "f" * 64
    df.to_csv(path, index=False)
    with pytest.raises(CollectError, match="CN2.*manifest_sha256"):
        collect_kpis(run_dir)
    with pytest.raises(CollectError, match="Faltan"):
        collect_kpis(run_dir, expected_nodes=NODES + ["CN6"], write=False)


def test_runner_kills_hung_nodes(cfg, processed_dir, tmp_path):
    m = new_manifest(cfg, processed_dir)
    run_dir = prepare_run(m, ["CN1"], runs_root=tmp_path)
    result = launch_local(run_dir, ["CN1"], timeout_s=1.5, command=dummy(["--sleep", "30"]))
    assert result.timed_out == ["CN1"] and not result.ok
    with pytest.raises(FileExistsError):
        prepare_run(m, ["CN1"], runs_root=tmp_path)


@pytest.mark.broker
def test_prepare_run_provisions_and_teardown_removes_broker_vhost(cfg, processed_dir, tmp_path, broker_admin):
    from experiments.runner import teardown
    m = new_manifest(cfg, processed_dir)
    run_dir = prepare_run(m, NODES, runs_root=tmp_path, broker_admin=broker_admin)
    import json
    creds = json.loads((run_dir / "credentials.json").read_text(encoding="utf-8"))
    assert set(creds) == set(NODES) and creds["CN1"]["vhost"] == f"fl-{m['experiment_id']}"
    teardown(m, broker_admin)
    assert broker_admin.queue_stats.__self__ is broker_admin
    vhosts = [v["name"] for v in broker_admin._req("GET", "/vhosts")]
    assert f"fl-{m['experiment_id']}" not in vhosts
