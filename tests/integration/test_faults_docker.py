"""E9: fallos inyectados con nodos en contenedores (04 ADR-12; roadmap E9).

Criterios de aceptación:
  * Ningún nodo sano queda bloqueado más que su timeout (el ejecutor nunca tiene que matar contenedores).
  * La caída del broker termina el experimento de forma limpia y registrada en las 4 topologías
    (código 6 y eventos `broker_connection_lost` → `broker_lost` → `node_exit`), y se mide el tiempo hasta
    detectarla.
Además: un reinicio corto del broker se recupera (reconexión pasiva; en S2, nuevo handshake), y los fallos de
nodo de ADR-12 con primitivas de Docker (kill, pause, partición de red, netem, cola llena).

Requiere Docker, la imagen `fl-node:dev`, la red `fl-net` y el broker de docker/compose.yaml.
    python -m pytest -m docker tests/integration/test_faults_docker.py
"""
from __future__ import annotations

import subprocess

import pytest

from experiments.docker_launcher import DockerOptions, wait_broker_healthy
from experiments.fault_report import detection_report, node_events
from experiments.faults import Fault
from node.checkpoint import list_checkpoints
from scripts.run_experiment import run_experiment

pytestmark = [pytest.mark.broker, pytest.mark.docker]

FAST = {
    "communication.broker.loss_grace_s": 10,
    "topologies.centralized.round_timeout_s": 20, "topologies.centralized.client_round_wait_s": 90,
    "topologies.hierarchical.cluster_round_timeout_s": 15, "topologies.hierarchical.round_timeout_s": 30,
    "topologies.hierarchical.client_round_wait_s": 120,
    "topologies.semi_decentralized.election_timeout_s": 10,
    "topologies.semi_decentralized.internal_round_timeout_s": 20,
    "topologies.semi_decentralized.arrival_window_s": 15,
    "topologies.semi_decentralized.outer_round_timeout_s": 120,
    "topologies.semi_decentralized.client_round_wait_s": 150,
    "topologies.semi_decentralized.rounds.max_global_rounds": 4,
    "topologies.p2p.round_timeout_s": 20, "topologies.p2p.client_round_wait_s": 120,
    "topologies.p2p.rounds.max_global_rounds": 8,
}
# Disparo a mitad de la corrida, sobre una línea de log de cada topología
MID_RUN = {
    "centralized": ("infra", r"\[server\] ronda 2:"),
    "hierarchical": ("infra", r"\[root\] ronda 2:"),
    "semi_decentralized": ("infra", r"\[coordinator\] ronda 1: "),
    "p2p": ("CN1", r"\[CN1\] ronda 2: mezcla"),
}
TOPOLOGIES = list(MID_RUN)


@pytest.fixture(scope="module")
def docker_ready(broker_admin):
    opts = DockerOptions(threads=1)
    try:
        subprocess.run(["docker", "version"], capture_output=True, check=True, timeout=30)
        from experiments.docker_launcher import check_environment
        check_environment(opts)
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        pytest.skip(f"Docker no disponible: {exc}")
    return opts


@pytest.fixture(autouse=True)
def broker_healthy_after(docker_ready):
    yield
    assert wait_broker_healthy(docker_ready), "el broker no volvió a estar sano"


def run(topology, faults, tmp_path, processed_dir, docker_ready, overrides=None, rounds=10, profile="S0",
        timeout_s=900):
    out = run_experiment(topology, "fedavg", rounds=rounds if topology in ("centralized", "hierarchical") else None,
                         epochs=1, threads=1, processed_dir=processed_dir, runs_root=tmp_path, launcher="docker",
                         docker_opts=docker_ready, timeout_s=timeout_s, config_overrides={**FAST, **(overrides or {})},
                         faults=faults, security_profile=profile)
    res = out["result"]
    assert not res.timed_out, f"nodos bloqueados hasta el timeout del ejecutor: {res.timed_out}"
    return out


# ---------------------------------------------------------------------------- broker
@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_broker_down_ends_experiment_cleanly(topology, tmp_path, processed_dir, docker_ready):
    grace = FAST["communication.broker.loss_grace_s"]
    out = run(topology, [Fault("stop", "broker", when_log=MID_RUN[topology], duration_s=40)], tmp_path,
              processed_dir, docker_ready)
    codes = out["result"].exit_codes
    assert set(codes.values()) == {6}, codes
    for row in detection_report(out["run_dir"]):
        assert row["detect_s"] is not None and row["detect_s"] < 2.0, row       # cierre TCP: inmediato
        assert row["lost_s"] == pytest.approx(row["detect_s"] + grace, abs=3), row    # recepción o publicación
        assert row["exit_s"] <= grace + 15, row                                  # + entrenamiento en curso
        assert row["exit_code"] == 6


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_broker_restart_recovers(topology, tmp_path, processed_dir, docker_ready):
    out = run(topology, [Fault("restart", "broker", when_log=MID_RUN[topology])], tmp_path, processed_dir,
              docker_ready, overrides={"communication.broker.loss_grace_s": 60})
    codes = out["result"].exit_codes
    assert all(c == 0 for c in codes.values()), codes
    for row in detection_report(out["run_dir"]):
        if row["exit_s"] is not None and row["exit_s"] < 0:
            continue                                    # el nodo ya había terminado (p. ej. P2P con DONE)
        assert row["detect_s"] < 2.0 and row["reconnect_s"] is not None and row["reconnect_s"] < 45, row


def test_broker_restart_rekeys_in_s2(tmp_path, processed_dir, docker_ready):
    """Tras reconectar, cada nodo hace un handshake nuevo (nueva época) y el experimento termina en S2."""
    out = run("centralized", [Fault("restart", "broker", when_log=MID_RUN["centralized"])], tmp_path,
              processed_dir, docker_ready, overrides={"communication.broker.loss_grace_s": 60}, profile="S2")
    assert all(c == 0 for c in out["result"].exit_codes.values()), out["result"].exit_codes
    kpis = out["kpis"]
    server = kpis[kpis["node_id"] == "infra"]
    assert server["handshakes"].sum() >= 10                     # 5 iniciales + 5 tras la reconexión
    assert kpis["sec_rejected"].fillna(0).sum() == 0


# ---------------------------------------------------------------------------- nodos
def test_centralized_killed_client(tmp_path, processed_dir, docker_ready):
    out = run("centralized", [Fault("kill", "CN5", when_log=MID_RUN["centralized"])], tmp_path, processed_dir,
              docker_ready, rounds=6)
    codes = out["result"].exit_codes
    assert codes["infra"] == 0 and all(codes[c] == 0 for c in ("CN1", "CN2", "CN3", "CN4")), codes
    assert codes["CN5"] == 137                                  # SIGKILL
    globals_ = sorted((r for r in list_checkpoints(out["run_dir"]) if r.kind == "global"), key=lambda r: r.round)
    assert globals_[-1].meta["missing"] == ["CN5"]
    server = out["kpis"][out["kpis"]["node_id"] == "infra"]
    assert (server["t_wait_model_s"] <= FAST["topologies.centralized.round_timeout_s"] + 5).all()


def test_hierarchical_paused_leaf(tmp_path, processed_dir, docker_ready):
    """CN5 congelado 30 s (sin CPU ni heartbeats): el clúster B agrega sin él al vencer su timeout."""
    out = run("hierarchical", [Fault("pause", "CN5", when_log=MID_RUN["hierarchical"], duration_s=30)], tmp_path,
              processed_dir, docker_ready, rounds=8)
    codes = out["result"].exit_codes
    assert codes["infra"] == 0 and codes["CN1"] == 0 and codes["CN4"] == 0, codes
    cn4 = out["kpis"][out["kpis"]["node_id"] == "CN4"]
    assert (cn4["t_wait_model_s"].fillna(0) <= FAST["topologies.hierarchical.cluster_round_timeout_s"] + 30).all()


def test_semi_decentralized_latency_and_loss(tmp_path, processed_dir, docker_ready):
    """netem en CN2 (300 ms ± 50 ms, 2 % de pérdida): la corrida termina; el straggler lo refleja."""
    out = run("semi_decentralized", [Fault("netem", "CN2", when_log=("CN2", r"rol agent"),
                                           netem="delay 300ms 50ms loss 2%")], tmp_path, processed_dir, docker_ready)
    assert all(c == 0 for c in out["result"].exit_codes.values()), out["result"].exit_codes


def test_p2p_network_partition_and_sync(tmp_path, processed_dir, docker_ready):
    """CN3 desconectado de la red 40 s: lo detecta por heartbeat, se reconecta y se reincorpora."""
    hb = 10
    out = run("p2p", [Fault("disconnect", "CN3", when_log=MID_RUN["p2p"], duration_s=40)], tmp_path, processed_dir,
              docker_ready, overrides={"communication.broker.heartbeat_s": hb, "communication.broker.loss_grace_s": 90,
                                       "topologies.p2p.rounds.max_global_rounds": 12,
                                       "topologies.p2p.rounds.patience": 1000})        # sin parada temprana: CN3 debe ponerse al día
    codes = out["result"].exit_codes
    assert all(c == 0 for c in codes.values()), codes
    evs = [e["event"] for e in node_events(out["run_dir"])["CN3"]]
    assert "broker_connection_lost" in evs and "broker_reconnected" in evs, evs
    row = next(r for r in detection_report(out["run_dir"]) if r["node"] == "CN3")
    assert row["detect_s"] <= 4 * hb + 5, row                   # heartbeat AMQP (~3 intervalos sin tramas)
    # Reincorporación: tras reconectar, los demás vuelven a mezclar con CN3. Con un atraso <= buffer_rounds se
    # pone al día con los MODEL que esperaban en su cola; el SYNC (atraso mayor) lo cubre test_p2p_e2e (E8).
    p2p = out["kpis"]
    last = p2p[(p2p["node_id"] == "CN1") & (p2p["round"] == p2p["round"].max())]
    assert "CN3" in str(last["p2p_received"].iloc[0])


def test_queue_overflow_does_not_crash_sender(tmp_path, processed_dir, docker_ready):
    """Cola de CN5 limitada a 1 mensaje (prefetch 1: con prefetch 8 el broker entrega hasta 8 mensajes al socket
    del nodo pausado y no cuentan para x-max-length) y CN5 en pausa: el servidor recibe nack, cuenta el envío fallido,
    trata a CN5 como ausente y sigue (antes de E9, el TransportError tumbaba al servidor)."""
    out = run("centralized", [Fault("pause", "CN5", when_log=MID_RUN["centralized"], duration_s=45)], tmp_path,
              processed_dir, docker_ready, rounds=8,
              overrides={"communication.broker.queue.max_length": 1, "communication.broker.prefetch": 1,
                         "topologies.centralized.round_timeout_s": 8})
    codes = out["result"].exit_codes
    assert codes["infra"] == 0, codes
    server = out["kpis"][out["kpis"]["node_id"] == "infra"]
    assert server["messages_failed"].sum() >= 1
