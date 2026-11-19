"""E8 sin red: grafos lógicos, velocidad de consenso (λ₂), membresía y adopción tras SYNC (RF-06, RF-22; ADR-06)."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from core.aggregation import fedavg_weighted, metropolis_weights, mix
from core.training import Update
from topologies.p2p.graph import build_graph, is_connected, mixing_matrix, second_eigenvalue
from topologies.p2p.membership import ALIVE, DEAD, FINISHED, SUSPECT, Membership
from topologies.p2p.sync import SyncResponse, adopt

PEERS = ["CN1", "CN2", "CN3", "CN4", "CN5"]


# ---------------------------------------------------------------- grafos
def test_graphs_are_symmetric_connected_and_as_specified():
    mesh, ring = build_graph("full_mesh", PEERS), build_graph("ring", PEERS)
    assert all(len(v) == 4 for v in mesh.values())
    assert ring["CN1"] == {"CN2", "CN5"} and all(len(v) == 2 for v in ring.values())
    for k in (2, 3, 4):
        g = build_graph("random_k", PEERS, k=k, seed=7)
        assert is_connected(g) and all(len(v) >= k for v in g.values())
        assert all(p in g[q] for p in g for q in g[p])
        assert g == build_graph("random_k", PEERS, k=k, seed=7)                 # determinista
    with pytest.raises(ValueError):
        build_graph("star", PEERS)
    with pytest.raises(ValueError):
        build_graph("random_k", PEERS, k=9)


def test_lambda2_matches_theory():
    assert second_eigenvalue(build_graph("full_mesh", PEERS)) == pytest.approx(0.0, abs=1e-12)
    assert second_eigenvalue(build_graph("ring", PEERS)) == pytest.approx((1 + 2 * math.cos(2 * math.pi / 5)) / 3)


def test_ring_mixing_disagreement_decays_as_lambda2():
    """Criterio (2) de E8: solo mezcla en anillo ⇒ ‖x_t − x̄‖ ≤ λ₂^t ‖x_0 − x̄‖ y la media se conserva."""
    g = build_graph("ring", PEERS)
    order, M = mixing_matrix(g)
    lam = second_eigenvalue(g)
    x = np.random.default_rng(0).normal(size=(5, 3))
    mean0, d0 = x.mean(axis=0), np.linalg.norm(x - x.mean(axis=0))
    for t in range(1, 15):
        x = M @ x
        np.testing.assert_allclose(x.mean(axis=0), mean0, atol=1e-12)
        assert np.linalg.norm(x - mean0) <= lam ** t * d0 + 1e-12
    assert np.linalg.norm(x - mean0) < 1e-3 * d0


def test_full_mesh_mix_with_full_reception_is_weighted_fedavg():
    """Criterio (1) de E8, en el núcleo: malla completa + recepción completa ≡ FedAvg ponderado (Eq. 1)."""
    g = torch.Generator().manual_seed(1)
    ups = {p: Update({"w": torch.randn(4, 3, generator=g)}, n, 1, 0.0)
           for p, n in zip(PEERS, [4643, 11785, 167, 714, 6139])}
    W = metropolis_weights(build_graph("full_mesh", PEERS))
    expected = fedavg_weighted([ups[p] for p in PEERS])["w"]
    for p in PEERS:
        torch.testing.assert_close(mix(p, ups, W[p])["w"], expected)


# ---------------------------------------------------------------- membresía
def test_membership_suspect_dead_and_recovery():
    m = Membership(("CN2", "CN3"), dead_after_rounds=3)
    m.end_round({"CN2"})
    assert m.state == {"CN2": ALIVE, "CN3": SUSPECT} and m.expected() == {"CN2", "CN3"}
    m.end_round({"CN2"}); m.end_round({"CN2"})
    assert m.state["CN3"] == DEAD and m.expected() == {"CN2"}
    m.heard("CN3", 7)
    assert m.state["CN3"] == ALIVE and m.ahead_by(4) == 3


def test_membership_termination_rule():
    m = Membership(("CN2", "CN3"), dead_after_rounds=1)
    m.end_round(set())
    assert not m.all_done()                    # todos caídos y ninguno terminado ⇒ aislado, no terminado
    m.finished("CN2", 5)
    assert m.all_done() and m.state["CN2"] == FINISHED
    m.heard("CN2", 6)
    assert m.state["CN2"] == FINISHED          # un terminado no revive


# ---------------------------------------------------------------- SYNC
def test_adopt_takes_the_most_advanced_round():
    a = {"w": torch.zeros(2)}
    b = {"w": torch.ones(2)}
    c = {"w": torch.full((2,), 3.0)}
    r, s, used = adopt([SyncResponse("CN2", 5, 1, a), SyncResponse("CN3", 7, 1, b), SyncResponse("CN4", 7, 3, c)])
    assert r == 7 and used == ["CN3", "CN4"]
    torch.testing.assert_close(s["w"], torch.full((2,), 2.5))
    with pytest.raises(ValueError):
        adopt([])
