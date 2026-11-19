"""Equivalencia del núcleo con el código original (oráculos), etapa E1.

Criterios de aceptación de E1 (analisis_arquitectonico/06):
  (2) misma semilla ⇒ mismos pesos, y modelo idéntico a MLP/MLPLarger de P1
  (4) fednova_legacy = FedNova de P2/P3 = FedAvg ponderado
  (5) entrenamiento = train_local de P3
  + fedavg_uniform = federated_averaging de P1; RecallPatience = check_early_stopping de P1
"""
from __future__ import annotations

import importlib.util
import time

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from core.aggregation import fedavg_uniform, fedavg_weighted, fednova_legacy
from core.model import MLP
from core.stopping import RecallPatience
from core.training import LocalTrainer, Update
from tests.conftest import P1, P2, P3
from tests.equivalence.oracle import Stub, load_function


def _p1_mlp_module():
    spec = importlib.util.spec_from_file_location("p1_mlp", P1 / "mlp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _states(n_models=4, seed=0, n_features=7):
    g = torch.Generator().manual_seed(seed)
    base = MLP(n_features, (5, 3), seed=1).state_dict()
    return [{k: torch.randn(v.shape, generator=g) for k, v in base.items()} for _ in range(n_models)]


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_features", [105, 278])
def test_reference_model_equals_p1_mlplarger(n_features):
    p1 = _p1_mlp_module()
    ours, theirs = MLP(n_features, (256, 128, 64), 0.3, seed=42), p1.MLPLarger(n_features, seed=42)
    sd_o, sd_t = ours.state_dict(), theirs.state_dict()
    assert list(sd_o) == list(sd_t) == [f"fc{i}.{p}" for i in range(1, 5) for p in ("weight", "bias")]
    for k in sd_o:
        assert torch.equal(sd_o[k], sd_t[k]), k
    x = torch.randn(8, n_features)
    ours.eval(), theirs.eval()
    assert torch.equal(ours(x), theirs(x))


def test_legacy_model_equals_p1_mlp():
    p1 = _p1_mlp_module()
    sd_o = MLP(105, (120, 84), 0.3, seed=42).state_dict()
    sd_t = p1.MLP(105, seed=42).state_dict()
    assert list(sd_o) == list(sd_t)
    assert all(torch.equal(sd_o[k], sd_t[k]) for k in sd_o)


# ---------------------------------------------------------------------------
# Agregación
# ---------------------------------------------------------------------------
def test_fedavg_uniform_equals_p1_federated_averaging():
    fn = load_function(P1 / "server_centralized.py", "federated_averaging", "CentralizedServer",
                       {"torch": torch})
    states = _states()
    oracle = fn(Stub(received_models={f"a{i}": s for i, s in enumerate(states)}))
    ours = fedavg_uniform([Update(s, n_samples=10 * (i + 1), tau=1, train_loss=0) for i, s in enumerate(states)])
    for k in oracle:
        torch.testing.assert_close(ours[k], oracle[k], rtol=1e-6, atol=1e-6)


def _nova_inputs(seed=3):
    g = torch.Generator().manual_seed(seed)
    w0 = _states(1, seed=99)[0]
    states = _states(4, seed=seed)
    n = [100, 7, 2500, 40]
    tau = [7, 1, 157, 3]
    grads = [{k: (w0[k] - s[k]) / t for k in w0} for s, t in zip(states, tau)]
    return w0, states, n, tau, grads


def test_fednova_legacy_equals_p2_fednova_aggregate():
    fn = load_function(P2 / "model" / "fed_model.py", "fednova_aggregate",
                       globals_={"torch": torch, "logger": Stub(warning=print),
                                 "federated_average": None})
    w0, states, n, tau, grads = _nova_inputs()
    ids = [f"Nodo_{i}" for i in range(len(states))]
    oracle = fn(model_paths=[], nova_grads=dict(zip(ids, grads)),
                sample_counts={i: (ni, ti) for i, ni, ti in zip(ids, n, tau)}, global_params=w0)
    updates = [Update(s, ni, ti, 0.0) for s, ni, ti in zip(states, n, tau)]
    ours = fednova_legacy(w0, updates)
    for k in w0:
        torch.testing.assert_close(ours[k], oracle[k], rtol=1e-5, atol=1e-5)
    # D1 P2-H4: la FedNova del código es FedAvg ponderado por muestras
    weighted = fedavg_weighted(updates)
    for k in w0:
        torch.testing.assert_close(ours[k], weighted[k], rtol=1e-5, atol=1e-5)


def test_fednova_legacy_equals_p3_fednova():
    fn = load_function(P3 / "agent_csv.py", "_fednova", "FederatedAgentCSV", {"torch": torch})
    w0, states, n, tau, grads = _nova_inputs(seed=5)
    ids = [f"agent_{i}" for i in range(len(states))]
    self_ = Stub(get_model_parameters=lambda: {k: v.clone() for k, v in w0.items()},
                 received_models=dict(zip(ids, states)),
                 received_metrics={i: {"local_samples": ni, "fednova_tau": ti} for i, ni, ti in zip(ids, n, tau)},
                 received_nova_grads=dict(zip(ids, grads)), agent_id="agg")
    oracle = fn(self_)
    ours = fednova_legacy(w0, [Update(s, ni, ti, 0.0) for s, ni, ti in zip(states, n, tau)])
    for k in w0:
        torch.testing.assert_close(ours[k], oracle[k], rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Entrenamiento local
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["FedAvg", "FedProx"])
def test_training_equals_p3_train_local(method):
    torch.manual_seed(0)
    X, y = torch.randn(93, 11), (torch.rand(93, 1) > 0.7).float()
    epochs, batch, lr, mu = 3, 16, 0.01, 0.01
    global_ref = MLP(11, (16, 8), seed=7).state_dict()

    cfg_stub = Stub(LOCAL_EPOCHS=epochs, AGGREGATION_METHOD=method, FEDPROX_MU=mu)
    train_local = load_function(P3 / "agent_csv.py", "train_local", "FederatedAgentCSV",
                                {"Config": cfg_stub, "time": time, "torch": torch})
    model_o = MLP(11, (16, 8), seed=7)
    oracle_self = Stub(model=model_o, optimizer=torch.optim.Adam(model_o.parameters(), lr=lr),
                       criterion=torch.nn.BCEWithLogitsLoss(),
                       train_data=DataLoader(TensorDataset(X, y), batch_size=batch, shuffle=True,
                                             generator=torch.Generator().manual_seed(42)),
                       global_model_params={k: v.clone() for k, v in global_ref.items()} if method == "FedProx" else None)
    torch.manual_seed(1234)                       # mismo estado de RNG para el dropout
    train_local(oracle_self, epochs)

    model_n = MLP(11, (16, 8), seed=7)
    trainer = LocalTrainer(model_n, X, y, batch_size=batch, learning_rate=lr, seed=42)
    torch.manual_seed(1234)
    upd = trainer.train(epochs, mu=mu if method == "FedProx" else 0.0,
                        proximal_reference=global_ref if method == "FedProx" else None)

    assert upd.tau == oracle_self.fednova_tau == epochs * -(-93 // batch)
    assert upd.n_samples == 93
    for k, v in model_o.state_dict().items():
        torch.testing.assert_close(upd.state[k], v, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Parada temprana
# ---------------------------------------------------------------------------
def test_recall_patience_equals_p1_check_early_stopping():
    cfg_stub = Stub(MAX_ROUNDS=50, MIN_RECALL_IMPROVEMENT=0.001, EARLY_STOPPING_PATIENCE=3)
    fn = load_function(P1 / "server_centralized.py", "check_early_stopping", "CentralizedServer",
                       {"Config": cfg_stub, "time": time, "print": lambda *a, **k: None})
    srv = Stub(recall_history=[], best_recall=0.0, rounds_without_improvement=0, best_model_params=None,
               best_model_round=None, convergence_time=None, experiment_start_time=time.time(),
               get_model_parameters=lambda: {})
    ours = RecallPatience(patience=3, min_improvement=0.001, max_rounds=50)
    seq = [0.0005, 0.6, 0.65, 0.6505, 0.70, 0.7005, 0.69, 0.7008, 0.71]
    for r, rec in enumerate(seq, start=1):
        srv.current_round = r
        stop_o, reason_o = fn(srv, rec)
        d = ours.update(r, rec)
        assert (d.stop, d.reason) == (stop_o, reason_o), f"ronda {r}"
        assert ours.best == pytest.approx(srv.best_recall)
        if stop_o:
            break
