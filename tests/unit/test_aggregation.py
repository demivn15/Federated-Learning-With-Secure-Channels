from __future__ import annotations

import pytest
import torch

from core.aggregation import (aggregate, coordinate_median, fedavg_uniform, fedavg_weighted, fednova_d0,
                              fednova_legacy, metropolis_weights, mix, trimmed_mean, weighted_average)
from core.config import load_config
from core.training import Update


def U(values, n, tau=1):
    return Update({"w": torch.tensor(values, dtype=torch.float32)}, n_samples=n, tau=tau, train_loss=0.0)


def test_fedavg_weighted_is_eq1():
    out = fedavg_weighted([U([0.0, 2.0], 1), U([4.0, 6.0], 3)])
    torch.testing.assert_close(out["w"], torch.tensor([3.0, 5.0]))


def test_fedavg_uniform_is_simple_mean():
    out = fedavg_uniform([U([0.0, 2.0], 1), U([4.0, 6.0], 3)])
    torch.testing.assert_close(out["w"], torch.tensor([2.0, 4.0]))


def test_weighted_average_validates_inputs():
    with pytest.raises(ValueError):
        weighted_average([{"w": torch.zeros(2)}, {"w": torch.zeros(3)}], [1, 1])
    with pytest.raises(ValueError):
        weighted_average([{"w": torch.zeros(2)}, {"v": torch.zeros(2)}], [1, 1])
    with pytest.raises(ValueError):
        weighted_average([{"w": torch.zeros(2, dtype=torch.long)}], [1])
    with pytest.raises(ValueError):
        weighted_average([], [])


def test_fednova_d0_matches_manual_eq3():
    g = {"w": torch.tensor([1.0])}
    ups = [U([3.0], n=1, tau=2), U([0.0], n=3, tau=6)]
    p = [0.25, 0.75]
    tau_bar = 0.25 * 2 + 0.75 * 6
    expected = 1.0 + tau_bar * (p[0] * (3.0 - 1.0) / 2 + p[1] * (0.0 - 1.0) / 6)
    torch.testing.assert_close(fednova_d0(g, ups)["w"], torch.tensor([expected]))


def test_fednova_d0_equal_taus_reduces_to_weighted_fedavg():
    g = {"w": torch.tensor([1.0, -1.0])}
    ups = [U([3.0, 0.0], n=10, tau=5), U([0.0, 2.0], n=30, tau=5)]
    torch.testing.assert_close(fednova_d0(g, ups)["w"], fedavg_weighted(ups)["w"])


def test_fednova_d0_differs_from_legacy_when_taus_differ():
    g = {"w": torch.tensor([1.0])}
    ups = [U([3.0], n=10, tau=1), U([0.0], n=30, tau=50)]
    legacy, ref = fednova_legacy(g, ups)["w"], fednova_d0(g, ups)["w"]
    torch.testing.assert_close(legacy, fedavg_weighted(ups)["w"])
    assert not torch.allclose(legacy, ref)


def test_aggregate_dispatch_follows_config_and_legacy_variants():
    g = {"w": torch.tensor([1.0])}
    ups = [U([3.0], n=10, tau=1), U([0.0], n=30, tau=50)]
    ref = load_config()["aggregation"]
    torch.testing.assert_close(aggregate("FedAvg", ups, g, ref)["w"], fedavg_weighted(ups)["w"])
    torch.testing.assert_close(aggregate("FedProx", ups, g, ref)["w"], fedavg_weighted(ups)["w"])
    torch.testing.assert_close(aggregate("FedNova", ups, g, ref)["w"], fednova_d0(g, ups)["w"])
    legacy = load_config(legacy_variants=["fedavg_uniform", "fednova_code"])["aggregation"]
    torch.testing.assert_close(aggregate("FedAvg", ups, g, legacy)["w"], fedavg_uniform(ups)["w"])
    torch.testing.assert_close(aggregate("FedNova", ups, g, legacy)["w"], fednova_legacy(g, ups)["w"])
    with pytest.raises(ValueError):
        aggregate("Scaffold", ups, g, ref)


def full_mesh(n):
    return {i: {j for j in range(n) if j != i} for i in range(n)}


def ring(n):
    return {i: {(i - 1) % n, (i + 1) % n} for i in range(n)}


@pytest.mark.parametrize("graph", [full_mesh(5), ring(5), {0: {1}, 1: {0, 2}, 2: {1}}])
def test_metropolis_rows_sum_to_one_and_symmetric(graph):
    W = metropolis_weights(graph)
    for i, row in W.items():
        assert sum(row.values()) == pytest.approx(1.0)
        assert all(v >= 0 for v in row.values())
        for j, v in row.items():
            if j != i:
                assert W[j][i] == pytest.approx(v)


def test_metropolis_known_values():
    assert all(v == pytest.approx(1 / 5) for v in metropolis_weights(full_mesh(5))[0].values())
    assert all(v == pytest.approx(1 / 3) for v in metropolis_weights(ring(5))[0].values())


def test_metropolis_rejects_asymmetric_graph():
    with pytest.raises(ValueError):
        metropolis_weights({0: {1}, 1: set()})


def test_full_mesh_mix_equals_weighted_fedavg():
    ups = {i: U([float(i), float(2 * i)], n=n) for i, n in enumerate([7739, 19642, 279, 1191, 10232])}
    W = metropolis_weights(full_mesh(5))
    expected = fedavg_weighted(list(ups.values()))["w"]
    for i in ups:
        torch.testing.assert_close(mix(i, ups, W[i])["w"], expected)


def test_full_mesh_mix_unweighted_equals_mean():
    ups = {i: U([float(i)], n=10 * (i + 1)) for i in range(5)}
    W = metropolis_weights(full_mesh(5))
    torch.testing.assert_close(mix(0, ups, W[0], weight_by_samples=False)["w"], torch.tensor([2.0]))


def test_partial_mix_renormalizes_and_identical_models_unchanged():
    W = metropolis_weights(full_mesh(5))
    same = {i: U([1.5, -2.0], n=10) for i in (0, 2, 4)}            # faltan 1 y 3
    torch.testing.assert_close(mix(0, same, W[0])["w"], torch.tensor([1.5, -2.0]))


def test_mix_requires_own_model_and_neighbors_only():
    W = metropolis_weights(ring(5))
    with pytest.raises(ValueError):
        mix(0, {1: U([1.0], 1)}, W[0])
    with pytest.raises(ValueError):
        mix(0, {0: U([1.0], 1), 2: U([1.0], 1)}, W[0])             # 2 no es vecino de 0 en el anillo


def test_robust_aggregators():
    ups = [U([1.0], 1), U([2.0], 1), U([100.0], 1), U([3.0], 1), U([2.5], 1)]
    assert float(coordinate_median(ups)["w"]) == pytest.approx(2.5)
    assert float(trimmed_mean(ups, 0.2)["w"]) == pytest.approx((2.0 + 2.5 + 3.0) / 3)
    assert float(trimmed_mean(ups, 0.5)["w"]) == pytest.approx(2.5)      # 5 modelos, recorta 2 por extremo
    with pytest.raises(ValueError):
        trimmed_mean(ups, 0.6)                                         # recortaría 3 por extremo: no queda nada
