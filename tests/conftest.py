from __future__ import annotations

from pathlib import Path

import pytest

from core.config import PROJECT_ROOT, load_config

REPO_ROOT = PROJECT_ROOT.parent
P1 = REPO_ROOT / "FL_Centralized_Hospital"
P2 = REPO_ROOT / "FL_Hierarchical_Hospital"
P3 = REPO_ROOT / "FL_Semi_Descentralized_Hospital"
SILO_DIR = P2 / "data" / "silos"


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="session")
def broker_admin():
    """Administrador del broker de docker/compose.yaml; omite la prueba si no está disponible."""
    from broker.provision_broker import BrokerAdminError, admin_from_env
    try:
        admin = admin_from_env()
        admin.overview()
    except (BrokerAdminError, OSError) as exc:
        pytest.skip(f"Broker no disponible: {exc}")
    return admin


@pytest.fixture
def experiment(broker_admin, cfg):
    """Fábrica de experimentos aislados (vhost + usuarios propios); se eliminan al terminar."""
    import copy
    import uuid
    created = []

    def make(node_ids=("CN1", "CN2", "CN3", "CN4", "CN5", "infra"), **queue_overrides):
        bcfg = copy.deepcopy(cfg["communication"]["broker"])
        bcfg["queue"].update(queue_overrides)
        exp_id = f"t{uuid.uuid4().hex[:10]}"
        creds = broker_admin.provision_experiment(exp_id, node_ids, bcfg)
        created.append(exp_id)
        return exp_id, creds, bcfg

    yield make
    for exp_id in created:
        broker_admin.delete_experiment(exp_id)


@pytest.fixture(scope="session")
def processed_dir(tmp_path_factory) -> Path:
    """Ejecuta scripts/prepare_data.py una vez por sesión en un directorio temporal."""
    if not SILO_DIR.exists():
        pytest.skip("Silos de P2 no disponibles")
    from scripts.prepare_data import main
    out = tmp_path_factory.mktemp("processed")
    assert main(["--silo-dir", str(SILO_DIR), "--out", str(out)]) == 0
    return out
