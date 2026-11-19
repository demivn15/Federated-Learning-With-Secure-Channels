"""Energía analítica de D0 §3.6 (no se mide: decisión Q-07).

  E_comp = 2 · α · c · D · f²          (Eq. 4)   D = n_muestras_train · n_features · 32 bits (código base)
  E_comm = p · T_comm                  (Eq. 5)
     T_comm medido: dos fuentes (las dos del código base, ver configs: energy.e_comm.tcomm_sources)
     T_comm de Shannon: M / (B · log2(1 + p·h²/σ²)), solo si B, h y σ² tienen valor (Q-14)

`e_comp` y `e_comm_measured` reproducen `calc_ecomp` / `calc_ecomm` de P1 (verificado en tests/equivalence),
salvo que devuelven None en lugar de −1 cuando falta un dato.
"""
from __future__ import annotations

import math
from typing import Any, Mapping


def e_comp(n_samples: int, n_features: int, f_hz: float | None, alpha: float, c_cycles_per_bit: float) -> float | None:
    if not f_hz or f_hz <= 0:
        return None
    D = n_samples * n_features * 32
    return 2 * alpha * c_cycles_per_bit * D * f_hz ** 2


def e_comm_measured(tcomm_s: float | None, p_watts: float) -> float | None:
    if tcomm_s is None or tcomm_s <= 0:
        return None
    return p_watts * tcomm_s


def tcomm_shannon(model_bytes: int, bandwidth_hz: float, channel_gain: float, noise_power: float,
                  p_watts: float) -> float:
    rate = bandwidth_hz * math.log2(1 + p_watts * channel_gain ** 2 / noise_power)
    return model_bytes * 8 / rate


def e_comm_eq5(model_bytes: int, p_watts: float, bandwidth_hz: float | None, channel_gain: float | None,
               noise_power: float | None) -> float | None:
    if None in (bandwidth_hz, channel_gain, noise_power):
        return None                                      # Q-14: sin valores confirmados no se calcula
    return p_watts * tcomm_shannon(model_bytes, bandwidth_hz, channel_gain, noise_power, p_watts)


def round_energy(energy_cfg: Mapping[str, Any], *, n_samples: int, n_features: int, f_hz: float | None,
                 tcomm_legacy_s: float | None, tcomm_transfer_s: float | None,
                 model_bytes: int | None) -> dict[str, float | None]:
    """Columnas de energía de una ronda para un nodo que entrena."""
    ec, em = energy_cfg["e_comp"], energy_cfg["e_comm"]
    eq5 = em.get("variant_eq5", {})
    comp = e_comp(n_samples, n_features, f_hz, ec["alpha"], ec["c_cycles_per_bit"])
    legacy = e_comm_measured(tcomm_legacy_s, em["p_watts"])
    transfer = e_comm_measured(tcomm_transfer_s, em["p_watts"])
    shannon = None
    if eq5.get("enabled") and model_bytes:
        shannon = e_comm_eq5(model_bytes, em["p_watts"], eq5.get("bandwidth_B"), eq5.get("channel_gain_h"),
                             eq5.get("noise_sigma2"))
    return {"e_comp_j": comp, "e_comm_legacy_j": legacy, "e_comm_transfer_j": transfer, "e_comm_eq5_j": shannon}
