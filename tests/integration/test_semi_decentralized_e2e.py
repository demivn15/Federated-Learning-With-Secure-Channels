"""E7: topología semi-descentralizada completa sobre RabbitMQ (6 procesos reales + broker de docker/compose.yaml).

Criterios de aceptación de E7 (06_implementation_roadmap.md):
  (1) los 5 CN calculan el mismo ganador y el coordinador lo verifica;
  (2) `lottery_d0` uniforme (tests/unit/test_sd_selectors.py, 10 000 sorteos);
  (3) el agregador no entrena en su ronda; rondas internas completas y agregación = Eq. 1 / Eq. 3;
  (4) el coordinador nunca recibe payloads de modelo;
  (5) con salvaguardas, la caída del agregador provoca una re-selección sin bloqueo (y sin ellas, la ronda
      se pierde con la métrica congelada, como el fallo en cascada de D0 §4.3);
  (6) `sd_score_hybrid` evita al nodo con straggler inyectado.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from core.aggregation import fedavg_weighted, fednova_d0
from core.model import build_model
from core.training import Update
from node.checkpoint import list_checkpoints, load_checkpoint
from scripts.run_experiment import run_experiment
from topologies.semi_decentralized.selectors import lottery_d0, round_robin

pytestmark = pytest.mark.broker
CLIENTS = ["CN1", "CN2", "CN3", "CN4", "CN5"]
FAULT = Path(__file__).with_name("fault_node.py")
SD = "topologies.semi_decentralized."


def fault(kind, processed_dir):
    return lambda rd, n: [sys.executable, str(FAULT), "--fault", kind, "--run-dir", str(rd), "--node", n,
                          "--processed", str(processed_dir), "--threads", "2"]


def run_sd(method, processed_dir, tmp_path, rounds, internal=2, overrides=None, commands=None):
    cfg = {f"{SD}internal_rounds": internal, **(overrides or {})}
    return run_experiment("semi_decentralized", method, rounds=rounds, epochs=1, threads=2,
                          processed_dir=processed_dir, runs_root=tmp_path, timeout_s=900,
                          config_overrides=cfg, command_overrides=commands)


def by_role(kpis, role):
    return kpis[kpis["role"] == role]


@pytest.mark.parametrize("method", ["fedavg", "fednova"])
def test_sd_rounds_lottery_and_aggregation(method, broker_admin, processed_dir, tmp_path, cfg):
    out = run_sd(method, processed_dir, tmp_path, rounds=2, overrides={"checkpoints.local": True})
    res, run_dir, kpis = out["result"], out["run_dir"], out["kpis"]
    assert res.ok, res.exit_codes

    coord = by_role(kpis, "coordinator").set_index("round")
    # (1) ganador = lottery_d0 con la semilla compartida, verificado por consenso de los CLAIM
    for r in (1, 2):
        assert coord.loc[r, "sd_aggregator_id"] == lottery_d0(CLIENTS, 42, r)
    assert coord["sd_claims_agree"].astype(bool).all() and coord["sd_round_completed"].astype(bool).all()
    # (4) el coordinador no recibe modelos (un modelo son ≈ 453 KB) ni rechaza nada
    assert (coord["payload_bytes_rx"] < 20_000).all() and (coord["messages_rejected"] == 0).all()

    # (3) el agregador no entrena en su ronda; 2 rondas internas; agregación = regla de D0
    ck = {(r.node_id, r.round, r.sub_round, r.kind): (r, load_checkpoint(r)) for r in list_checkpoints(run_dir)}
    previous = {k: v.detach().clone() for k, v in build_model(cfg, 278).state_dict().items()}
    for r in (1, 2):
        agg = coord.loc[r, "sd_aggregator_id"]
        workers = [c for c in CLIENTS if c != agg]
        rows = kpis[kpis["round"] == r]
        assert set(rows[rows["role"] == "worker"]["node_id"]) == set(workers)
        assert set(rows[rows["role"] == "aggregator"]["node_id"]) == {agg}
        assert rows[rows["role"] == "aggregator"]["tau"].isna().all()
        assert not any(key[0] == agg and key[1] == r and key[3] == "local" for key in ck)
        for k in (1, 2):
            ref, glob = ck[(agg, r, k, "global")]
            assert ref.meta["workers"] == workers and ref.meta["missing"] == []
            loc = [Update(ck[(w, r, k, "local")][1], ck[(w, r, k, "local")][0].meta["n_samples"],
                          ck[(w, r, k, "local")][0].meta["tau"], 0.0) for w in workers]
            expected = fedavg_weighted(loc) if method == "fedavg" else fednova_d0(previous, loc)
            for name in glob:
                torch.testing.assert_close(glob[name], expected[name], rtol=1e-5, atol=1e-5)
            previous = glob
    aggs = by_role(kpis, "aggregator")
    assert len(aggs) == 2 * 2 and aggs["straggler_s"].notna().all()
    assert out["evaluation"]["summary"]["global"]["mean_last_5"]["agg_f1"] is not None     # alcance SD †


def test_aggregator_crash_with_safeguards_triggers_reselection(broker_admin, processed_dir, tmp_path):
    t_outer = 15.0
    out = run_sd("fedavg", processed_dir, tmp_path, rounds=3,
                 overrides={f"{SD}selector": "round_robin", f"{SD}safeguards.enabled": True,
                            f"{SD}outer_round_timeout_s": t_outer, f"{SD}election_timeout_s": 10.0,
                            f"{SD}arrival_window_s": 10.0},
                 commands={"CN1": fault("crash_as_aggregator", processed_dir)})
    res, kpis = out["result"], out["kpis"]
    assert res.exit_codes["infra"] == 0, res.exit_codes
    coord = by_role(kpis, "coordinator").set_index("round")
    assert round_robin(CLIENTS, 0, 1) == "CN1"                       # CN1 es elegido y cae
    assert coord.loc[1, "sd_attempts"] == 2 and coord.loc[1, "sd_aggregator_id"] != "CN1"
    assert coord["sd_round_completed"].astype(bool).all()
    assert "CN1" not in set(coord["sd_aggregator_id"])               # el health check lo excluye después
    assert coord.loc[1, "t_round_s"] < t_outer + 60                  # sin bloqueo: T_outer + una ronda normal
    # tras la re-selección, CN1 (sin CLAIM) ya no figura entre los trabajadores de nadie
    later = kpis[(kpis["round"] >= 2) & (kpis["role"] == "aggregator")]
    assert all("CN1" not in json.loads(a) for a in later["arrivals"])
    assert "CN1" not in set(kpis["node_id"])                          # murió antes de registrar ninguna ronda


def test_aggregator_crash_without_safeguards_loses_the_round(broker_admin, processed_dir, tmp_path):
    """Reproducción de D0 §4.3: sin salvaguardas, la ronda cuyo agregador cae se pierde y la métrica se congela."""
    out = run_sd("fedavg", processed_dir, tmp_path, rounds=2,
                 # T_outer > 2 rondas internas × (entrenamiento + arrival_window): en la ronda 2 CN1 sigue en la
                 # lista de trabajadores (sin health check) y cada ronda interna lo espera arrival_window_s
                 overrides={f"{SD}selector": "round_robin", f"{SD}outer_round_timeout_s": 45.0,
                            f"{SD}election_timeout_s": 10.0, f"{SD}arrival_window_s": 10.0,
                            f"{SD}internal_round_timeout_s": 60.0},
                 commands={"CN1": fault("crash_as_aggregator", processed_dir)})
    res, kpis = out["result"], out["kpis"]
    assert res.exit_codes["infra"] == 0, res.exit_codes
    coord = by_role(kpis, "coordinator").set_index("round")
    assert not bool(coord.loc[1, "sd_round_completed"]) and coord.loc[1, "sd_attempts"] == 1
    assert coord.loc[1, "legacy_local_recall"] == 0.0                # métrica congelada (sin modelo nuevo)
    assert bool(coord.loc[2, "sd_round_completed"]) and coord.loc[2, "sd_aggregator_id"] == "CN2"
    agg2 = kpis[(kpis["round"] == 2) & (kpis["role"] == "aggregator")]
    assert all("CN1" not in json.loads(a) for a in agg2["arrivals"])  # CN1 sigue en la lista pero no llega


def test_sd_score_avoids_the_injected_straggler(broker_admin, processed_dir, tmp_path):
    out = run_sd("fedavg", processed_dir, tmp_path, rounds=4,
                 overrides={f"{SD}selector": "sd_score_hybrid"},
                 commands={"CN3": fault("train_delay=6", processed_dir)})
    res, kpis = out["result"], out["kpis"]
    assert res.ok, res.exit_codes
    coord = by_role(kpis, "coordinator").set_index("round")
    aggs = coord["sd_aggregator_id"].tolist()
    first_observed = next(r for r in range(1, 5) if aggs[r - 1] != "CN3")   # 1.ª ronda con CN3 como trabajador
    assert "CN3" not in aggs[first_observed:], aggs
    agg_rows = by_role(kpis, "aggregator")
    late = agg_rows["arrivals"].map(json.loads)
    assert all(max(a, key=a.get) == "CN3" for a in late if "CN3" in a)     # CN3 siempre llega el último
