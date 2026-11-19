"""E8: topología P2P (DFedAvg) sobre RabbitMQ con 5 pares reales.

Criterios de aceptación de E8 (06_implementation_roadmap.md):
  (1) malla completa + recepción completa ≡ FedAvg ponderado de la centralizada en la misma ronda;
  (2) anillo: la discrepancia decae según λ₂ (tests/unit/test_p2p.py) y aquí se mide el consenso real;
  (3) arranque en orden arbitrario ⇒ misma vista de membresía;
  (4) un par desconectado varias rondas recupera la ronda con SYNC y se reincorpora;
  (5) terminación con DONE sin bloqueos (pares que paran en rondas distintas);
  (6) ningún proceso aplica lógica FL central: la 6.ª Pi no se lanza ni tiene cola.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from core.aggregation import fedavg_weighted
from core.training import Update
from node.checkpoint import list_checkpoints, load_checkpoint
from scripts.run_experiment import run_experiment

pytestmark = pytest.mark.broker
PEERS = ["CN1", "CN2", "CN3", "CN4", "CN5"]
FAULT = Path(__file__).with_name("fault_node.py")
P = "topologies.p2p."


def fault(kind, processed_dir):
    return lambda rd, n: [sys.executable, str(FAULT), "--fault", kind, "--run-dir", str(rd), "--node", n,
                          "--processed", str(processed_dir), "--threads", "2"]


def run_p2p(processed_dir, tmp_path, rounds=2, overrides=None, commands=None):
    return run_experiment("p2p", "fedavg", rounds=rounds, epochs=1, threads=2, processed_dir=processed_dir,
                          runs_root=tmp_path, timeout_s=900, config_overrides=overrides,
                          command_overrides=commands)


def ckpts(run_dir):
    return {(r.node_id, r.round, r.kind): (r, load_checkpoint(r)) for r in list_checkpoints(run_dir)}


def test_full_mesh_equals_centralized_and_no_central_role(broker_admin, processed_dir, tmp_path):
    common = dict(rounds=2, epochs=1, threads=2, processed_dir=processed_dir, runs_root=tmp_path, timeout_s=900)
    p2p = run_experiment("p2p", "fedavg", config_overrides={"checkpoints.local": True}, **common)
    cen = run_experiment("centralized", "fedavg", **common)
    assert p2p["result"].ok and cen["result"].ok, (p2p["result"].exit_codes, cen["result"].exit_codes)
    # (6) solo los 5 pares se lanzan; la 6.ª Pi no ejecuta ningún rol
    assert set(p2p["result"].exit_codes) == set(PEERS)
    ck, cc = ckpts(p2p["run_dir"]), ckpts(cen["run_dir"])
    for r in (1, 2):
        loc = [Update(ck[(p, r, "local")][1], ck[(p, r, "local")][0].meta["n_samples"], 1, 0.0) for p in PEERS]
        expected = fedavg_weighted(loc)
        for p in PEERS:
            ref, mixed = ck[(p, r, "mixed")]
            assert ref.meta["received"] == [q for q in PEERS if q != p]
            for k in mixed:                                  # (1) ≡ FedAvg ponderado y ≡ centralizada
                torch.testing.assert_close(mixed[k], expected[k], rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(mixed[k], cc[("infra", r, "global")][1][k], rtol=1e-5, atol=1e-6)
    kpis = p2p["kpis"]
    assert set(kpis["role"]) == {"peer"} and len(kpis) == 2 * 5
    assert (kpis["messages_rejected"] == 0).all()
    assert kpis[kpis["round"] == 2]["p2p_consensus_distance"].notna().all()
    assert p2p["evaluation"]["summary"]["mixed"]["rounds"] == 2


def test_ring_with_staggered_start(broker_admin, processed_dir, tmp_path):
    out = run_p2p(processed_dir, tmp_path, rounds=3, overrides={f"{P}graph": "ring"},
                  commands={"CN1": fault("start_delay=8", processed_dir),
                            "CN4": fault("start_delay=4", processed_dir)})
    assert out["result"].ok, out["result"].exit_codes
    kpis = out["kpis"]
    # (3) misma vista: todos reciben de sus 2 vecinos en todas las rondas, pese al arranque escalonado
    neigh = {"CN1": ["CN2", "CN5"], "CN2": ["CN1", "CN3"], "CN3": ["CN2", "CN4"], "CN4": ["CN3", "CN5"],
             "CN5": ["CN1", "CN4"]}
    for row in kpis.itertuples():
        assert json.loads(row.p2p_received) == neigh[row.node_id] and json.loads(row.p2p_peers_missing) == []
    # (2) consenso aproximado medido: distinto de cero en anillo
    assert (kpis["p2p_consensus_distance"] > 0).all()


def test_partitioned_peer_resyncs_and_rejoins(broker_admin, processed_dir, tmp_path):
    # Corte de 15 s en la ronda 2: los demás tardan 2 × T_round en declararlo DEAD y luego ≈ 2 s por ronda,
    # así que CN5 vuelve a mitad de la corrida (si volviera después del final, encontraría los DONE y
    # terminaría con `all_neighbors_done`, que también es correcto pero no prueba el SYNC).
    out = run_p2p(processed_dir, tmp_path, rounds=12,
                  overrides={f"{P}round_timeout_s": 6.0, f"{P}dead_after_rounds": 2, f"{P}sync_timeout_s": 20.0},
                  commands={"CN5": fault("partition=2:15", processed_dir)})
    res, kpis = out["result"], out["kpis"]
    assert res.ok, res.exit_codes
    cn5 = kpis[kpis["node_id"] == "CN5"].sort_values("round")
    synced = cn5[cn5["p2p_synced_to"].notna()]
    # (4) CN5 vuelve, sincroniza a una ronda avanzada (saltándose las de su ausencia) y se reincorpora
    assert len(synced) == 1
    to = int(synced["p2p_synced_to"].iloc[0])
    assert to >= 2 and not set(range(2, to + 1)) & set(cn5["round"])     # no hay rondas solo-locales
    assert cn5["round"].max() == 12 and len(cn5) < 12
    others = kpis[(kpis["node_id"] != "CN5")]
    missing_cn5 = others[others["p2p_peers_missing"].map(lambda s: "CN5" in json.loads(s))]
    assert not missing_cn5.empty                               # los demás siguieron sin él
    last = others[others["round"] == 12]
    assert all("CN5" in json.loads(s) for s in last["p2p_received"])   # y lo vuelven a recibir al final
    # sin bloqueo: la ausencia no detiene a los demás más que T_round por ronda (antes de declararlo DEAD)
    assert (others["t_wait_model_s"] < 6.0 + 5).all()


def test_early_stopping_peers_finish_with_done(broker_admin, processed_dir, tmp_path):
    out = run_experiment("p2p", "fedavg", epochs=1, threads=2, processed_dir=processed_dir, runs_root=tmp_path,
                         timeout_s=900, config_overrides={f"{P}rounds": {"policy": "early_stopping", "patience": 1,
                                                                          "min_improvement": 0.5,
                                                                          "max_global_rounds": 6}})
    res, kpis = out["result"], out["kpis"]
    # (5) todos terminan (código 0) sin bloquearse, aunque paren en rondas distintas
    assert res.ok, res.exit_codes
    last = kpis.groupby("node_id")["round"].max()
    assert last.max() <= 6 and max(res.durations_s.values()) < 300
