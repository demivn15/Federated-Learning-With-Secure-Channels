"""E5: topología centralizada completa sobre RabbitMQ (6 procesos reales + broker de docker/compose.yaml).

Criterios de aceptación de E5:
  (1) rondas completas con 5 clientes;
  (2) el modelo global de cada ronda = la regla de D0 aplicada a los modelos locales (Eq. 1 / Eq. 3);
  (3) straggler medido en el servidor con las horas de llegada;
  (4) un cliente caído no bloquea al servidor más de `round_timeout_s`;
  + suplantación del emisor descartada.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
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
    out = {}
    for ref in list_checkpoints(run_dir):
        out[(ref.node_id, ref.round, ref.kind)] = (ref, load_checkpoint(ref))
    return out


def assert_states_close(a, b, tol=1e-5):
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=tol, atol=tol)


@pytest.mark.parametrize("method", ["fedavg", "fedprox"])
def test_centralized_rounds_and_weighted_aggregation(method, broker_admin, processed_dir, tmp_path):
    out = run_experiment("centralized", method, rounds=3, epochs=1, threads=2, processed_dir=processed_dir,
                         runs_root=tmp_path, timeout_s=600, config_overrides={"checkpoints.local": True})
    res, run_dir = out["result"], out["run_dir"]
    assert res.ok, res.exit_codes

    ck = checkpoints_by(run_dir)
    for r in (1, 2, 3):
        _, global_r = ck[("infra", r, "global")]
        locals_r = []
        for c in CLIENTS:
            ref, state = ck[(c, r, "local")]
            locals_r.append(Update(state, ref.meta["n_samples"], ref.meta["tau"], 0.0))
        assert_states_close(global_r, fedavg_weighted(locals_r))            # D0 Eq. 1 (FedProx: Eq. 2 → misma regla)

    kpis = out["kpis"]
    assert len(kpis) == 3 * 6
    server = kpis[kpis["node_id"] == "infra"]
    assert (server["straggler_s"] > 0).all() and server["arrivals"].notna().all()
    assert (server["messages_rejected"] == 0).all()
    clients = kpis[kpis["node_id"] != "infra"]
    assert set(clients.groupby("node_id")["n_samples"].first()) == {4643, 11785, 167, 714, 6139}
    assert (clients["tau"] == (clients["n_samples"] + 15) // 16).all()        # 1 época, batch 16
    assert clients["e_comp_j"].notna().all() and clients["payload_bytes_tx"].gt(450_000).all()
    assert out["evaluation"]["summary"]["global"]["rounds"] == 3


def test_centralized_fednova_follows_eq3(broker_admin, processed_dir, tmp_path, cfg):
    out = run_experiment("centralized", "fednova", rounds=2, epochs=1, threads=2, processed_dir=processed_dir,
                         runs_root=tmp_path, timeout_s=600, config_overrides={"checkpoints.local": True})
    assert out["result"].ok, out["result"].exit_codes
    ck = checkpoints_by(out["run_dir"])
    previous = {k: v.detach().clone() for k, v in build_model(cfg, 278).state_dict().items()}   # modelo inicial
    for r in (1, 2):
        locals_r = [Update(ck[(c, r, "local")][1], ck[(c, r, "local")][0].meta["n_samples"],
                           ck[(c, r, "local")][0].meta["tau"], 0.0) for c in CLIENTS]
        _, global_r = ck[("infra", r, "global")]
        assert_states_close(global_r, fednova_d0(previous, locals_r))
        assert any(not torch.allclose(global_r[k], fedavg_weighted(locals_r)[k], atol=1e-6) for k in global_r)
        previous = global_r


def test_dead_and_forging_client_do_not_block_server(broker_admin, processed_dir, tmp_path):
    timeout = 12.0
    out = run_experiment(
        "centralized", "fedavg", rounds=2, epochs=1, threads=2, processed_dir=processed_dir, runs_root=tmp_path,
        timeout_s=600, config_overrides={"topologies.centralized.round_timeout_s": timeout},
        command_overrides={"CN5": lambda rd, n: [sys.executable, str(SABOTEUR), "--run-dir", str(rd), "--node", n]})
    res = out["result"]
    assert res.exit_codes["infra"] == 0 and res.exit_codes["CN5"] == 0, res.exit_codes
    ck = checkpoints_by(out["run_dir"])
    for r in (1, 2):
        ref, _ = ck[("infra", r, "global")]
        assert ref.meta["missing"] == ["CN5"] and ref.meta["clients"] == CLIENTS[:4]
    server = out["kpis"][out["kpis"]["node_id"] == "infra"].set_index("round")
    assert server.loc[1, "messages_rejected"] >= 1                         # UPDATE suplantando a CN1
    # Cada ronda espera como máximo T_round por el cliente ausente (más el entrenamiento de los demás)
    assert (server["t_wait_model_s"] <= timeout + 5).all()
    assert res.durations_s["infra"] < 2 * timeout + 60
