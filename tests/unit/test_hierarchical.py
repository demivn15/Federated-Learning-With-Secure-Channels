"""E6 sin red: árbol de D0 y equivalencias de la agregación en dos niveles (ADR-09, 04 §7)."""
from __future__ import annotations

import pytest
import torch

from core.aggregation import aggregate, aggregate_cluster, fedavg_uniform, fedavg_weighted, fednova_d0
from core.training import Update
from topologies.hierarchical.tree import build_tree

CLIENTS = ["CN1", "CN2", "CN3", "CN4", "CN5"]
D0_CLUSTERS = {"A": {"aggregator": "CN1", "members": ["CN1", "CN2", "CN3"]},
               "B": {"aggregator": "CN4", "members": ["CN4", "CN5"]}}


def topo(**kw):
    return {"root": "infra", "clusters": D0_CLUSTERS, "redistribution": "via_cluster_aggregators",
            "cluster_aggregators_train": True, **kw}


# ---------------------------------------------------------------- árbol
def test_d0_tree_roles_and_routes(cfg):
    t = build_tree(cfg["topologies"]["hierarchical"], CLIENTS)          # la referencia D0 tal cual
    assert [c.aggregator for c in t.clusters] == ["CN1", "CN4"] and not t.flat
    assert {n: t.role_of(n) for n in ["infra", *CLIENTS]} == {
        "infra": "root", "CN1": "cluster_aggregator", "CN2": "leaf", "CN3": "leaf",
        "CN4": "cluster_aggregator", "CN5": "leaf"}
    assert [t.parent_of(n) for n in CLIENTS] == ["infra", "CN1", "CN1", "infra", "CN4"]
    assert [t.round_start_sender(n) for n in CLIENTS] == ["infra", "CN1", "CN1", "infra", "CN4"]
    direct = build_tree(topo(redistribution="direct"), CLIENTS)
    assert [direct.round_start_sender(n) for n in CLIENTS] == ["infra"] * 5
    assert [direct.parent_of(n) for n in CLIENTS] == ["infra", "CN1", "CN1", "infra", "CN4"]


def test_flat_tree_is_one_singleton_cluster_per_client():
    t = build_tree(topo(clusters="flat"), CLIENTS)
    assert t.flat and all(t.role_of(n) == "cluster_aggregator" for n in CLIENTS)
    assert all(t.parent_of(n) == "infra" for n in CLIENTS)


@pytest.mark.parametrize("bad, match", [
    ({"clusters": {"A": {"aggregator": "CN9", "members": ["CN1", "CN2", "CN3", "CN4", "CN5"]}}}, "miembro"),
    ({"clusters": {"A": {"aggregator": "CN1", "members": ["CN1", "CN2", "CN3"]},
                   "B": {"aggregator": "CN3", "members": ["CN3", "CN4", "CN5"]}}}, "clústeres A y B"),
    ({"clusters": {"A": {"aggregator": "CN1", "members": ["CN1", "CN2", "CN3"]}}}, "faltan"),
    ({"clusters": {"A": {"aggregator": "CN1", "members": ["CN1", "CN2", "CN3", "CN4", "CN5", "infra"]}}}, "raíz"),
    ({"redistribution": "gossip"}, "Redistribución"),
])
def test_invalid_trees_are_rejected(bad, match):
    with pytest.raises(ValueError, match=match):
        build_tree(topo(**bad), CLIENTS)


# ---------------------------------------------------------------- equivalencias
def random_round(seed=0):
    g = torch.Generator().manual_seed(seed)
    start = {"w": torch.randn(4, 3, generator=g), "b": torch.randn(3, generator=g)}
    ns, taus = [4643, 11785, 167, 714, 6139], [291, 737, 11, 45, 384]       # n y τ reales (1 época, batch 16)
    ups = {c: Update({k: v + 0.1 * torch.randn(v.shape, generator=g) for k, v in start.items()}, n, t, 0.5)
           for c, n, t in zip(CLIENTS, ns, taus)}
    return start, ups


def two_level(method, start, ups, clusters, agg_cfg):
    cl = [aggregate_cluster(method, [ups[m] for m in members], start, agg_cfg) for members in clusters]
    return aggregate(method, cl, start, agg_cfg)


@pytest.mark.parametrize("method, formula", [("fedavg", "d0_eq3"), ("fedprox", "d0_eq3"),
                                             ("fednova", "d0_eq3"), ("fednova", "code_equivalent_weighted_fedavg")])
def test_two_level_equals_flat_aggregation(method, formula):
    """2 niveles con todos presentes ≡ la regla aplicada a los 5 nodos (Eq. 1, Eq. 3 y la FedNova heredada)."""
    agg_cfg = {"fedavg": {"weighting": "n_samples"}, "fednova": {"formula": formula}}
    start, ups = random_round()
    flat = aggregate(method, [ups[c] for c in CLIENTS], start, agg_cfg)
    for k, v in two_level(method, start, ups, [["CN1", "CN2", "CN3"], ["CN4", "CN5"]], agg_cfg).items():
        torch.testing.assert_close(v, flat[k], rtol=1e-5, atol=1e-6)
    if method == "fednova" and formula == "d0_eq3":       # y Eq. 3 ≠ FedAvg ponderado cuando los τ difieren
        assert not torch.allclose(flat["w"], fedavg_weighted([ups[c] for c in CLIENTS])["w"], atol=1e-4)


def test_cluster_update_carries_n_and_tau_bar():
    start, ups = random_round()
    cu = aggregate_cluster("fednova", [ups["CN4"], ups["CN5"]], start, {"fednova": {"formula": "d0_eq3"}})
    assert cu.n_samples == 714 + 6139
    assert cu.tau == pytest.approx((714 * 45 + 6139 * 384) / (714 + 6139))
    torch.testing.assert_close(cu.state["w"], fednova_d0(start, [ups["CN4"], ups["CN5"]])["w"])


def test_singleton_cluster_is_passthrough():
    start, ups = random_round()
    cu = aggregate_cluster("fednova", [ups["CN3"]], start, {"fednova": {"formula": "d0_eq3"}})
    assert cu.state is ups["CN3"].state and cu.n_samples == 167 and cu.tau == 11


def test_uniform_legacy_variant_is_not_invariant_to_the_tree():
    """Documentado en aggregate_cluster: la media de medias no es la media (variante heredada)."""
    agg_cfg = {"fedavg": {"weighting": "uniform"}}
    start, ups = random_round()
    flat = fedavg_uniform([ups[c] for c in CLIENTS])
    tree = two_level("fedavg", start, ups, [["CN1", "CN2", "CN3"], ["CN4", "CN5"]], agg_cfg)
    assert not torch.allclose(tree["w"], flat["w"], atol=1e-4)
