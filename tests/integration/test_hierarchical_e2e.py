"""E6: topología jerárquica completa sobre RabbitMQ (6 procesos reales + broker de docker/compose.yaml).

Criterios de aceptación de E6 (06_implementation_roadmap.md):
  (1) 2 niveles con todos presentes ≡ media global ponderada (FedAvg, Eq. 1) y FedNova coherente con la
      Eq. 3 (Σn, Σnτ por clúster) — sobre los modelos locales reales de cada ronda;
  (2) jerárquica plana ≡ centralizada (misma semilla ⇒ mismos modelos globales);
  (3) la caída de CN5 no bloquea al clúster B más de `cluster_round_timeout_s`;
  + las dos redistribuciones (vía agregadores y directa) y el ROUND_START falso descartado.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from core.aggregation import fedavg_weighted, fednova_d0
from core.model import build_model
from core.training import Update
from node.checkpoint import list_checkpoints, load_checkpoint
from scripts.run_experiment import run_experiment

pytestmark = pytest.mark.broker
CLIENTS = ["CN1", "CN2", "CN3", "CN4", "CN5"]
SABOTEUR = Path(__file__).with_name("saboteur_client.py")


def checkpoints_by(run_dir):
    return {(r.node_id, r.round, r.kind): (r, load_checkpoint(r)) for r in list_checkpoints(run_dir)}


def assert_states_close(a, b, tol=1e-5):
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=tol, atol=tol)


def locals_of(ck, r):
    return [Update(ck[(c, r, "local")][1], ck[(c, r, "local")][0].meta["n_samples"],
                   ck[(c, r, "local")][0].meta["tau"], 0.0) for c in CLIENTS]


@pytest.mark.parametrize("method, redistribution", [("fedavg", "via_cluster_aggregators"),
                                                    ("fednova", "direct")])
def test_two_level_tree_equals_flat_rule(method, redistribution, broker_admin, processed_dir, tmp_path, cfg):
    out = run_experiment("hierarchical", method, rounds=2, epochs=1, threads=2, processed_dir=processed_dir,
                         runs_root=tmp_path, timeout_s=600,
                         config_overrides={"checkpoints.local": True,
                                           "topologies.hierarchical.redistribution": redistribution})
    res, run_dir = out["result"], out["run_dir"]
    assert res.ok, res.exit_codes
    ck = checkpoints_by(run_dir)
    previous = {k: v.detach().clone() for k, v in build_model(cfg, 278).state_dict().items()}
    for r in (1, 2):
        loc = locals_of(ck, r)
        _, global_r = ck[("infra", r, "global")]
        expected = fedavg_weighted(loc) if method == "fedavg" else fednova_d0(previous, loc)
        assert_states_close(global_r, expected)
        a_ref, a_state = ck[("CN1", r, "cluster")]
        b_ref, b_state = ck[("CN4", r, "cluster")]
        assert a_ref.meta["members"] == ["CN1", "CN2", "CN3"] and b_ref.meta["members"] == ["CN4", "CN5"]
        assert b_ref.meta["n_samples"] == 714 + 6139
        if method == "fedavg":
            assert_states_close(b_state, fedavg_weighted(loc[3:]))
        previous = global_r

    kpis = out["kpis"]
    assert len(kpis) == 2 * 6
    roles = kpis.groupby("node_id")["role"].first().to_dict()
    assert roles == {"infra": "root", "CN1": "cluster_aggregator", "CN2": "leaf", "CN3": "leaf",
                     "CN4": "cluster_aggregator", "CN5": "leaf"}
    aggs = kpis[kpis["role"] != "leaf"]
    assert aggs["straggler_s"].notna().all() and (aggs["messages_rejected"] == 0).all()
    assert set(kpis[kpis["role"] == "cluster_aggregator"]["cluster_n_samples"]) == {4643 + 11785 + 167, 714 + 6139}
    assert (kpis[kpis["role"] != "root"]["tau"].notna()).all()
    summary = out["evaluation"]["summary"]
    assert summary["global"]["rounds"] == 2 and summary["cluster"]["rounds"] == 2


def test_flat_hierarchy_equals_centralized(broker_admin, processed_dir, tmp_path):
    common = dict(rounds=2, epochs=1, threads=2, processed_dir=processed_dir, runs_root=tmp_path, timeout_s=600)
    flat = run_experiment("hierarchical", "fedavg", variants=["hierarchical_flat"], **common)
    cen = run_experiment("centralized", "fedavg", **common)
    assert flat["result"].ok and cen["result"].ok, (flat["result"].exit_codes, cen["result"].exit_codes)
    ck_f, ck_c = checkpoints_by(flat["run_dir"]), checkpoints_by(cen["run_dir"])
    for r in (1, 2):
        assert_states_close(ck_f[("infra", r, "global")][1], ck_c[("infra", r, "global")][1], tol=1e-6)


def test_dead_leaf_does_not_block_its_cluster(broker_admin, processed_dir, tmp_path):
    cluster_timeout = 12.0
    out = run_experiment(
        "hierarchical", "fedavg", rounds=2, epochs=1, threads=2, processed_dir=processed_dir, runs_root=tmp_path,
        timeout_s=600, config_overrides={"topologies.hierarchical.cluster_round_timeout_s": cluster_timeout,
                                         "topologies.hierarchical.round_timeout_s": 60.0},
        command_overrides={"CN5": lambda rd, n: [sys.executable, str(SABOTEUR), "--run-dir", str(rd), "--node", n]})
    res = out["result"]
    assert res.exit_codes["infra"] == 0 and all(res.exit_codes[c] == 0 for c in CLIENTS), res.exit_codes
    ck = checkpoints_by(out["run_dir"])
    for r in (1, 2):
        g_ref, _ = ck[("infra", r, "global")]
        assert g_ref.meta["clusters"] == ["A", "B"] and g_ref.meta["missing"] == ["CN5"]
        b_ref, _ = ck[("CN4", r, "cluster")]
        assert b_ref.meta["members"] == ["CN4"] and b_ref.meta["missing"] == ["CN5"]
    kpis = out["kpis"].set_index(["node_id", "round"])
    # el clúster B espera como máximo su timeout (t_wait_model incluye la espera del ROUND_START)
    assert all(kpis.loc[("CN4", r), "t_wait_model_s"] <= cluster_timeout + 10 for r in (1, 2))
    # la raíz nunca agota su propio timeout (60 s): el clúster B llega tras el suyo
    assert all(kpis.loc[("infra", r), "t_wait_model_s"] < 60 for r in (1, 2))
    assert res.durations_s["infra"] < 2 * cluster_timeout + 90
    # el UPDATE suplantando a CN1 que el saboteador envía a la raíz se descarta (tipo/rol no autorizado)
    assert kpis.loc[("infra", 1), "messages_rejected"] >= 1
