from __future__ import annotations

import pytest
import torch

from core.config import load_config
from core.model import MLP, arch_hash, build_model, parameter_count
from core.stopping import FixedRounds, RecallPatience, build_round_policy
from core.training import LocalTrainer


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
def test_reference_model_from_config(cfg):
    m = build_model(cfg, 278)
    assert [n for n, _ in m.named_parameters()][::2] == ["fc1.weight", "fc2.weight", "fc3.weight", "fc4.weight"]
    assert parameter_count(m) == 278 * 256 + 256 + 256 * 128 + 128 + 128 * 64 + 64 + 64 + 1 == 112_641
    assert m(torch.randn(4, 278)).shape == (4, 1)


def test_legacy_variant_builds_120_84():
    cfg = load_config(legacy_variants=["model_120_84"])
    assert parameter_count(build_model(cfg, 105)) == 22_969      # D1 §1: 22 969 parámetros ≈ 92 KB


def test_same_seed_same_weights_and_arch_hash():
    a, b = MLP(278, seed=42), MLP(278, seed=42)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    c = MLP(278, seed=43)
    assert arch_hash(a) == arch_hash(c)                           # mismos nombres y formas
    assert arch_hash(a) != arch_hash(MLP(277, seed=42))          # otra dimensión de entrada
    assert arch_hash(a) == arch_hash(a.state_dict())


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
def _data(n=50, f=6):
    g = torch.Generator().manual_seed(0)
    return torch.randn(n, f, generator=g), (torch.rand(n, 1, generator=g) > 0.5).float()


def _trainer(seed_model=1, **kw):
    X, y = _data()
    return LocalTrainer(MLP(6, (8, 4), seed=seed_model), X, y, batch_size=16, learning_rate=0.01, seed=42, **kw)


def test_training_is_deterministic_and_counts_steps():
    torch.manual_seed(5); u1 = _trainer().train(3)
    torch.manual_seed(5); u2 = _trainer().train(3)
    assert u1.tau == u2.tau == 3 * 4                              # ceil(50/16) = 4 lotes por época
    assert u1.n_samples == 50
    assert all(torch.equal(u1.state[k], u2.state[k]) for k in u1.state)


def test_fedprox_changes_update_and_requires_reference():
    t = _trainer()
    ref = {k: v.clone() for k, v in t.model.state_dict().items()}
    torch.manual_seed(5); plain = _trainer().train(3)
    torch.manual_seed(5); prox = _trainer().train(3, mu=1.0, proximal_reference=ref)
    assert any(not torch.equal(plain.state[k], prox.state[k]) for k in plain.state)
    with pytest.raises(ValueError):
        _trainer().train(1, mu=0.01)


def test_optimizer_persistent_vs_recreated_differ_in_second_round():
    def two_rounds(mode):
        torch.manual_seed(5)
        t = _trainer(optimizer_state=mode)
        t.train(1)
        return t.train(1)
    a, b = two_rounds("persistent_per_node"), two_rounds("recreate_each_round")
    assert any(not torch.equal(a.state[k], b.state[k]) for k in a.state)


# ---------------------------------------------------------------------------
# Rondas
# ---------------------------------------------------------------------------
def test_round_policies_from_reference(cfg):
    topo = cfg["topologies"]
    c, h, sd = (build_round_policy(topo[t]["rounds"]) for t in ("centralized", "hierarchical", "semi_decentralized"))
    assert isinstance(c, FixedRounds) and c.rounds == 12
    assert isinstance(h, FixedRounds) and h.rounds == 30
    assert isinstance(sd, RecallPatience) and (sd.patience, sd.max_rounds) == (3, 50)
    assert [c.update(r).stop for r in (11, 12)] == [False, True]


def test_recall_patience_stops_after_patience_rounds_without_improvement():
    p = RecallPatience(patience=3, min_improvement=0.001, max_rounds=50)
    decisions = [p.update(r, m) for r, m in enumerate([0.6, 0.7, 0.7005, 0.69, 0.6999], start=1)]
    assert [d.stop for d in decisions] == [False, False, False, False, True]
    assert (p.best, p.best_round) == (0.7, 2)
