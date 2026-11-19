"""E10, criterios (5) y (6): las 4 topologías funcionan en S0, S1 y S2 sin cambios en los orquestadores, con
los nodos en contenedores, y el coste de la seguridad queda desglosado por fase en los KPIs.

  * S1: los nodos solo pueden hablar con el listener AMQPS (5671) y verifican el certificado del broker.
  * S2: además, sesión E2E por par (handshakes > 0), ningún rechazo del canal seguro y columnas de coste
    (`t_encrypt_s`, `t_decrypt_s`, `t_handshake_s`) rellenas; en S0/S1 esas columnas quedan vacías.

    python -m pytest -m docker tests/integration/test_security_matrix_docker.py
"""
from __future__ import annotations

import subprocess

import pytest

from experiments.docker_launcher import DockerOptions, check_environment
from scripts.run_experiment import run_experiment

pytestmark = [pytest.mark.broker, pytest.mark.docker]

SHORT = {
    "topologies.semi_decentralized.rounds.max_global_rounds": 2,
    "topologies.semi_decentralized.internal_rounds": 2,
    "topologies.p2p.rounds.max_global_rounds": 3,
}
SECURITY_COLS = ["t_encrypt_s", "t_decrypt_s", "t_handshake_s", "handshakes", "sec_rejected"]


@pytest.fixture(scope="module")
def docker_ready(broker_admin):
    opts = DockerOptions(threads=1)
    try:
        subprocess.run(["docker", "version"], capture_output=True, check=True, timeout=30)
        check_environment(opts)
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        pytest.skip(f"Docker no disponible: {exc}")
    return opts


@pytest.mark.parametrize("profile", ["S0", "S1", "S2"])
@pytest.mark.parametrize("topology", ["centralized", "hierarchical", "semi_decentralized", "p2p"])
def test_topology_profile_matrix(topology, profile, tmp_path, processed_dir, docker_ready):
    out = run_experiment(topology, "fedavg", rounds=3 if topology in ("centralized", "hierarchical") else None,
                         epochs=1, threads=1, processed_dir=processed_dir, runs_root=tmp_path, launcher="docker",
                         docker_opts=docker_ready, timeout_s=900, config_overrides=SHORT, security_profile=profile)
    res = out["result"]
    assert res.ok, (res.exit_codes, res.timed_out)
    manifest_profile = (out["run_dir"] / "manifest.json").read_text(encoding="utf-8")
    assert f'"security_profile": "{profile}"' in manifest_profile
    kpis = out["kpis"]
    assert (kpis["messages_rejected"] == 0).all()
    if profile == "S2":
        assert kpis["handshakes"].sum() > 0
        assert kpis["sec_rejected"].fillna(0).sum() == 0
        assert kpis[["t_encrypt_s", "t_decrypt_s"]].notna().all().all()
        logs = "".join(p.read_text(encoding="utf-8") for p in (out["run_dir"] / "logs").glob("*.log"))
        assert "sesión S2" in logs
    else:
        assert kpis[SECURITY_COLS].isna().all().all()
    if profile in ("S1", "S2"):
        assert (out["run_dir"] / "security" / "ca.pem").exists()
