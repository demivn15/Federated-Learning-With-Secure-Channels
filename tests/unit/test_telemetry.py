"""E4: telemetría (reloj, sistema, energía, registro de KPIs)."""
from __future__ import annotations

import csv
import math
import time

import pytest

from telemetry.clock import ArrivalClock, PhaseTimer
from telemetry.energy import e_comm_eq5, e_comm_measured, e_comp, round_energy, tcomm_shannon
from telemetry.kpi_log import COMMON_COLUMNS, KpiLogger, columns_for
from telemetry.system import ProcessCpu, cpu_freq_mhz, cpu_percent_legacy, ram_rss_mb, temperature_c
from tests.conftest import P1
from tests.equivalence.oracle import load_function

P1_CONST = {"ALPHA_CAPACITANCE": 2e-28, "C_CYCLES_PER_BIT": 20, "P_TRANSMISSION_W": 0.5}


# ---------------------------------------------------------------------------
# Energía = fórmulas del código base (oráculo P1) y de D0
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n, f, mhz", [(4643, 278, 1500.0), (167, 278, 1500.0), (11785, 105, 600.0), (1, 1, 1.0)])
def test_e_comp_equals_p1_calc_ecomp(n, f, mhz):
    oracle = load_function(P1 / "metrics_utils.py", "calc_ecomp", globals_=dict(P1_CONST))
    assert e_comp(n, f, mhz * 1e6, 2e-28, 20) == pytest.approx(oracle(n, f, mhz), abs=1e-9)


def test_e_comp_reference_value_h1():
    # H1: 4643 muestras de entrenamiento × 278 features × 32 bits a 1,5 GHz
    assert e_comp(4643, 278, 1.5e9, 2e-28, 20) == pytest.approx(2 * 2e-28 * 20 * 4643 * 278 * 32 * 1.5e9 ** 2)


@pytest.mark.parametrize("t", [0.0123, 1.5, 35.317])
def test_e_comm_equals_p1_calc_ecomm(t):
    oracle = load_function(P1 / "metrics_utils.py", "calc_ecomm", globals_=dict(P1_CONST))
    assert e_comm_measured(t, 0.5) == pytest.approx(oracle(t), abs=1e-6)


def test_missing_inputs_give_none_not_negative():
    assert e_comp(10, 10, None, 2e-28, 20) is None
    assert e_comm_measured(0.0, 0.5) is None
    assert e_comm_eq5(453_904, 0.5, None, 1.0, 1.0) is None                       # Q-14: sin B no se calcula


def test_eq5_shannon_formula():
    t = tcomm_shannon(453_904, 1e6, 1.0, 1.0, 0.5)
    assert t == pytest.approx(453_904 * 8 / (1e6 * math.log2(1.5)))
    assert e_comm_eq5(453_904, 0.5, 1e6, 1.0, 1.0) == pytest.approx(0.5 * t)


def test_round_energy_uses_reference_config(cfg):
    out = round_energy(cfg["energy"], n_samples=4643, n_features=278, f_hz=1.5e9,
                       tcomm_legacy_s=6.0, tcomm_transfer_s=0.05, model_bytes=453_904)
    assert out["e_comm_legacy_j"] == pytest.approx(3.0) and out["e_comm_transfer_j"] == pytest.approx(0.025)
    assert out["e_comm_eq5_j"] is None                                           # sin B, h, σ² confirmados
    assert out["e_comp_j"] == pytest.approx(e_comp(4643, 278, 1.5e9, 2e-28, 20))


# ---------------------------------------------------------------------------
# Reloj y sistema
# ---------------------------------------------------------------------------
def test_phase_timer_accumulates():
    t = PhaseTimer()
    for _ in range(2):
        with t.phase("train"):
            time.sleep(0.02)
    t.add("wait", 0.5)
    cols = t.as_columns()
    assert cols["t_train_s"] >= 0.04 and cols["t_wait_s"] == pytest.approx(0.5)


def test_arrival_clock_straggler():
    c = ArrivalClock()
    c.mark("CN1"); time.sleep(0.05); c.mark("CN2"); c.mark("CN1")         # la 2.ª llegada de CN1 no cuenta
    assert c.straggler_s() >= 0.05
    assert list(c.relative_s()) == ["CN1", "CN2"] and c.relative_s()["CN1"] == 0.0
    assert ArrivalClock().straggler_s() is None


def test_system_metrics_types():
    cpu = ProcessCpu(); cpu.start()
    x = 0
    for i in range(2_000_000):
        x += i
    m = cpu.stop()
    assert m["cpu_process_s"] > 0 and m["cpu_process_percent_core"] > 0
    assert 0 <= cpu_percent_legacy(0.05) <= 100
    assert ram_rss_mb() > 10
    assert temperature_c() is None or isinstance(temperature_c(), float)            # None en Windows
    assert cpu_freq_mhz() is None or cpu_freq_mhz() > 0


# ---------------------------------------------------------------------------
# Registro de KPIs
# ---------------------------------------------------------------------------
def test_kpi_logger_schema_and_validation(tmp_path):
    path = tmp_path / "nodes" / "CN1" / "kpi.csv"
    log = KpiLogger(path, topology="p2p", experiment_id="e1", manifest_sha256="a" * 64, node_id="CN1")
    log.log({"round": 1, "sub_round": 0, "n_samples": 4643, "arrivals": {"CN2": 0.0}, "p2p_isolated": False,
             "temperature_c": None})
    with pytest.raises(KeyError):
        log.log({"round": 2, "typo_column": 1})
    with pytest.raises(ValueError):
        log.log({"round": 2, "node_id": "CN9"})
    with pytest.raises(ValueError):
        log.log({"round": 2, "train_loss": float("nan")})
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert list(rows[0]) == list(columns_for("p2p")) and len(rows) == 1
    assert rows[0]["node_id"] == "CN1" and rows[0]["arrivals"] == '{"CN2":0.0}' and rows[0]["p2p_isolated"] == "0"
    assert rows[0]["temperature_c"] == ""
    assert set(COMMON_COLUMNS) <= set(columns_for("centralized"))
    with pytest.raises(ValueError):
        columns_for("star")
