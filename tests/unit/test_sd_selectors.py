"""E7 sin red: selectores del agregador, `SD-score` híbrido y salvaguardas (RF-04, RF-05; 04 ADR-08)."""
from __future__ import annotations

import random
from collections import Counter

import pytest

from topologies.semi_decentralized.safeguards import Observations, SafeguardState
from topologies.semi_decentralized.selectors import (hybrid_scores, lottery_d0, lottery_legacy, round_robin,
                                                     score_winner, select, uniform_draw)

CN = ["CN1", "CN2", "CN3", "CN4", "CN5"]
CHI2_DF4_P001 = 18.467                     # valor crítico χ² con 4 g.l. y p = 0,001


# ---------------------------------------------------------------- loterías
def test_lottery_d0_is_the_five_subintervals_of_a_uniform_draw():
    for r in range(1, 50):
        u = uniform_draw(42, r, 1)
        assert 0.0 <= u < 1.0
        assert lottery_d0(CN, 42, r) == CN[int(u * 5)]
    assert lottery_d0(list(reversed(CN)), 42, 7) == lottery_d0(CN, 42, 7)      # el orden de entrada no importa


@pytest.mark.parametrize("k", [5, 4])
def test_lottery_d0_is_uniform_over_10000_draws(k):
    """Criterio (2) de E7: distribución uniforme sobre ≥ 10 000 sorteos."""
    counts = Counter(lottery_d0(CN[:k], 42, r) for r in range(1, 10_001))
    expected = 10_000 / k
    chi2 = sum((counts[c] - expected) ** 2 / expected for c in CN[:k])
    assert set(counts) == set(CN[:k])
    assert chi2 < (CHI2_DF4_P001 if k == 5 else 16.266)                        # 3 g.l.: 16,266


def test_lottery_depends_on_seed_round_and_attempt_only():
    assert lottery_d0(CN, 42, 3, 1) == lottery_d0(CN, 42, 3, 1)
    draws = {(s, r, a): uniform_draw(s, r, a) for s in (42, 43) for r in (1, 2) for a in (1, 2)}
    assert len(set(draws.values())) == len(draws)


def test_lottery_legacy_is_max_of_randint_per_candidate():
    """Mismo mecanismo que `select_aggregator` de P3: randint(1, 100) por nodo y gana el máximo."""
    for r in range(1, 30):
        rng = random.Random(f"lottery_legacy|42|{r}|1")
        draws = {n: rng.randint(1, 100) for n in CN}
        assert lottery_legacy(CN, 42, r) == max(draws, key=draws.get)


def test_round_robin_rotates_and_reselection_moves_on():
    assert [round_robin(CN, 0, r) for r in range(1, 7)] == CN + ["CN1"]
    assert round_robin(CN, 0, 1, attempt=2) == "CN2"


def test_unknown_selector_and_bad_candidates_fail_loudly():
    with pytest.raises(ValueError, match="Selector"):
        select("vrf", CN, 42, 1)
    with pytest.raises(ValueError):
        select("lottery_d0", [], 42, 1)
    with pytest.raises(ValueError):
        select("lottery_d0", ["CN1", "CN1"], 42, 1)
    with pytest.raises(ValueError, match="puntuaciones"):
        select("sd_score_hybrid", CN, 42, 1)


# ---------------------------------------------------------------- SD-score híbrido
def test_hybrid_score_penalizes_the_observed_straggler():
    observed = {n: {"lateness_s": 0.2, "rtt_s": 0.01} for n in CN}
    observed["CN3"] = {"lateness_s": 6.0, "rtt_s": 0.01}
    declared = {n: {"cpu_percent": 10.0, "ram_available_mb": 2000.0, "temperature_c": 45.0} for n in CN}
    w = {"lateness": 1.0, "rtt": 0.5, "cpu_idle": 0.2, "ram_free": 0.1, "temperature": 0.2}
    s = hybrid_scores(CN, observed, declared, w)
    assert min(s, key=s.get) == "CN3"
    assert select("sd_score_hybrid", CN, 42, 2, scores=s) != "CN3"


def test_hybrid_score_lying_about_self_declared_values_does_not_outweigh_observed_lateness():
    """R-02: un nodo lento que declara CPU 0 %, RAM máxima y 20 °C sigue por debajo (pesos observados > declarados)."""
    observed = {n: {"lateness_s": 0.1, "rtt_s": 0.01} for n in CN}
    observed["CN3"]["lateness_s"] = 5.0
    declared = {n: {"cpu_percent": 50.0, "ram_available_mb": 1000.0, "temperature_c": 60.0} for n in CN}
    declared["CN3"] = {"cpu_percent": 0.0, "ram_available_mb": 4000.0, "temperature_c": 20.0}
    s = hybrid_scores(CN, observed, declared,
                      {"lateness": 1.0, "rtt": 0.5, "cpu_idle": 0.2, "ram_free": 0.1, "temperature": 0.2})
    assert max(s, key=s.get) != "CN3"


def test_without_history_everyone_ties_and_lottery_d0_decides():
    s = hybrid_scores(CN, {}, {}, {"lateness": 1.0, "rtt": 0.5})
    assert set(s.values()) == {0.0}
    assert score_winner(CN, s, 42, 1) == lottery_d0(CN, 42, 1)


def test_missing_values_are_neutral():
    observed = {"CN1": {"lateness_s": 0.0}, "CN2": {"lateness_s": 4.0}}       # CN3–CN5 sin historial
    s = hybrid_scores(CN, observed, {}, {"lateness": 1.0})
    assert s["CN1"] > s["CN3"] == s["CN4"] == s["CN5"] > s["CN2"]


# ---------------------------------------------------------------- salvaguardas
def test_without_safeguards_candidates_are_always_all_registered():
    g = SafeguardState.from_config(CN, {"enabled": False})
    g.observe_claims({"CN2"}); g.mark_failed("CN1")
    for _ in range(3):
        g.record_aggregator("CN5")
    assert g.candidates({"CN5"}) == CN


def test_health_check_and_consecutive_limit():
    g = SafeguardState.from_config(CN, {"enabled": True, "max_consecutive_selections": 2})
    g.observe_claims({"CN1", "CN2", "CN3", "CN4"})                  # CN5 no respondió
    assert g.candidates(set()) == ["CN1", "CN2", "CN3", "CN4"]
    g.record_aggregator("CN2"); g.record_aggregator("CN2")         # dos seguidas ⇒ CN2 descansa
    assert g.candidates(set()) == ["CN1", "CN3", "CN4"]
    assert g.candidates({"CN1"}) == ["CN3", "CN4"]                  # re-selección sin el que falló
    g.mark_failed("CN3")
    assert g.candidates(set()) == ["CN1", "CN4"]
    g.observe_claims(set(CN))                                      # vuelven todos
    g.record_aggregator("CN1")
    assert g.candidates(set()) == CN


def test_consecutive_limit_never_empties_the_pool():
    g = SafeguardState.from_config(["CN1"], {"enabled": True, "max_consecutive_selections": 1})
    g.record_aggregator("CN1")
    assert g.candidates(set()) == ["CN1"]


def test_observations_use_exponential_average():
    o = Observations(alpha=0.5)
    o.add_lateness({"CN1": 4.0}); o.add_lateness({"CN1": 0.0})
    o.add_rtt("CN2", 1.0)
    assert o.observed() == {"CN1": {"lateness_s": 2.0, "rtt_s": None}, "CN2": {"lateness_s": None, "rtt_s": 1.0}}
