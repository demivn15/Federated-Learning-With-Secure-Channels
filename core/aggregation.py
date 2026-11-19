"""Agregadores como funciones puras (analisis_arquitectonico/04 ADR-02 y ADR-06).

Referencia (D0 §3.5):
  * `fedavg_weighted`  — Eq. 1: M = Σ (n_k/N) M_k
  * FedProx            — Eq. 2: la regla del servidor es FedAvg ponderado (el término proximal va en el cliente)
  * `fednova_d0`       — Eq. 3: M = M_t + τ̄ Σ (n_k/N)(M_k − M_t)/τ_k,  τ̄ = Σ (n_k/N) τ_k
Variantes heredadas del código local (solo para explicar diferencias, no son la referencia):
  * `fedavg_uniform`   — media simple (P1/P2/P3)
  * `fednova_legacy`   — M_t − Σ p_k τ_k d_k con d_k = (M_t − M_k)/τ_k (P2/P3) ≡ FedAvg ponderado
P2P (D5 §3.3, 03 Q-09a): `metropolis_weights` + `mix` (renormalizado sobre lo recibido, ponderado por n).
Robustos opcionales (RF-29): `coordinate_median`, `trimmed_mean`.
"""
from __future__ import annotations

from typing import Hashable, Mapping, Sequence

import torch

from core.model import StateDict
from core.training import Update


# ---------------------------------------------------------------------------
# Validación y combinación lineal
# ---------------------------------------------------------------------------
def _check_compatible(states: Sequence[Mapping[str, torch.Tensor]]) -> None:
    if not states:
        raise ValueError("No hay modelos que agregar")
    ref = states[0]
    for s in states[1:]:
        if s.keys() != ref.keys():
            raise ValueError("Los modelos tienen parámetros distintos")
        for k in ref:
            if s[k].shape != ref[k].shape:
                raise ValueError(f"Forma distinta en {k}: {tuple(s[k].shape)} vs {tuple(ref[k].shape)}")
    for k, t in ref.items():
        if not torch.is_floating_point(t):
            raise ValueError(f"Parámetro no flotante {k}: no se puede promediar")


def weighted_average(states: Sequence[Mapping[str, torch.Tensor]],
                     weights: Sequence[float]) -> StateDict:
    """Σ w_k M_k con los pesos normalizados a 1. Acumula en float64 y devuelve el dtype original."""
    _check_compatible(states)
    if len(weights) != len(states):
        raise ValueError("Número de pesos distinto del número de modelos")
    w = torch.tensor(weights, dtype=torch.float64)
    if (w < 0).any() or float(w.sum()) <= 0:
        raise ValueError("Los pesos deben ser no negativos y sumar > 0")
    w = w / w.sum()
    out: StateDict = {}
    for k, ref in states[0].items():
        acc = torch.zeros_like(ref, dtype=torch.float64)
        for wk, s in zip(w, states):
            acc += wk * s[k].to(torch.float64)
        out[k] = acc.to(ref.dtype)
    return out


# ---------------------------------------------------------------------------
# Referencia D0
# ---------------------------------------------------------------------------
def fedavg_weighted(updates: Sequence[Update]) -> StateDict:
    """D0 Eq. 1 (también la regla de servidor de FedProx, Eq. 2)."""
    return weighted_average([u.state for u in updates], [u.n_samples for u in updates])


def fednova_d0(global_state: Mapping[str, torch.Tensor], updates: Sequence[Update]) -> StateDict:
    """D0 Eq. 3 (Wang et al., 2020)."""
    _check_compatible([global_state, *[u.state for u in updates]])
    if any(u.tau <= 0 for u in updates):
        raise ValueError("FedNova necesita tau > 0 en todas las actualizaciones")
    N = float(sum(u.n_samples for u in updates))
    p = [u.n_samples / N for u in updates]
    tau_bar = sum(pk * u.tau for pk, u in zip(p, updates))
    out: StateDict = {}
    for k, g in global_state.items():
        g64 = g.to(torch.float64)
        step = torch.zeros_like(g64)
        for pk, u in zip(p, updates):
            step += pk * (u.state[k].to(torch.float64) - g64) / u.tau
        out[k] = (g64 + tau_bar * step).to(g.dtype)
    return out


# ---------------------------------------------------------------------------
# Variantes heredadas
# ---------------------------------------------------------------------------
def fedavg_uniform(updates: Sequence[Update]) -> StateDict:
    """Media simple, como `federated_averaging` de P1 (D1 P1-H2)."""
    return weighted_average([u.state for u in updates], [1.0] * len(updates))


def fednova_legacy(global_state: Mapping[str, torch.Tensor], updates: Sequence[Update]) -> StateDict:
    """Fórmula de P2/P3: M_t − Σ p_k τ_k d_k con d_k = (M_t − M_k)/τ_k (≡ FedAvg ponderado, D1 P2-H4)."""
    _check_compatible([global_state, *[u.state for u in updates]])
    N = float(sum(u.n_samples for u in updates))
    out: StateDict = {}
    for k, g in global_state.items():
        g64 = g.to(torch.float64)
        corr = torch.zeros_like(g64)
        for u in updates:
            tau = max(u.tau, 1)
            d = (g64 - u.state[k].to(torch.float64)) / tau
            corr += (u.n_samples / N) * tau * d
        out[k] = (g64 - corr).to(g.dtype)
    return out


# ---------------------------------------------------------------------------
# Despacho para C / H / SD
# ---------------------------------------------------------------------------
def aggregate(method: str, updates: Sequence[Update], global_state: Mapping[str, torch.Tensor],
              aggregation_cfg: Mapping) -> StateDict:
    """Aplica el método configurado (`cfg['aggregation']`)."""
    method = method.lower()
    if method in ("fedavg", "fedprox"):
        weighting = aggregation_cfg.get("fedavg", {}).get("weighting", "n_samples")
        return fedavg_weighted(updates) if weighting == "n_samples" else fedavg_uniform(updates)
    if method == "fednova":
        formula = aggregation_cfg.get("fednova", {}).get("formula", "d0_eq3")
        if formula == "d0_eq3":
            return fednova_d0(global_state, updates)
        if formula == "code_equivalent_weighted_fedavg":
            return fednova_legacy(global_state, updates)
        raise ValueError(f"Fórmula FedNova desconocida: {formula}")
    raise ValueError(f"Método de agregación desconocido: {method}")


def aggregate_cluster(method: str, updates: Sequence[Update], global_state: Mapping[str, torch.Tensor],
                      aggregation_cfg: Mapping) -> Update:
    """Modelo de clúster (jerárquica, ADR-09) expresado como una actualización para el nivel superior.

    state = la regla del método sobre los miembros; n = Σ n_k; τ = τ̄_c = Σ n_k τ_k / Σ n_k.
    Con FedAvg ponderado (Eq. 1) y FedNova (Eq. 3), aplicar la misma regla en la raíz a estas
    actualizaciones de clúster es **idéntico** a aplicarla a todos los nodos a la vez:
      Eq. 1: Σ_c (N_c/N) Σ_{k∈c} (n_k/N_c) M_k = Σ_k (n_k/N) M_k;
      Eq. 3: (M_c − M_t)/τ̄_c = Σ_{k∈c} (n_k/N_c)(M_k − M_t)/τ_k  y  Σ_c (N_c/N) τ̄_c = τ̄.
    (No se cumple con la variante heredada `fedavg_uniform`: la media de medias no es la media.)
    Un clúster de un solo modelo lo devuelve sin tocar (sin redondeos), como la topología centralizada.
    """
    if not updates:
        raise ValueError("Clúster sin actualizaciones")
    N = sum(u.n_samples for u in updates)
    tau_bar = sum(u.n_samples * u.tau for u in updates) / N
    loss = sum(u.n_samples * u.train_loss for u in updates) / N
    if len(updates) == 1:
        u = updates[0]
        return Update(u.state, u.n_samples, u.tau, u.train_loss)
    return Update(aggregate(method, updates, global_state, aggregation_cfg), N, tau_bar, loss)


# ---------------------------------------------------------------------------
# P2P: mezcla de Metropolis-Hastings
# ---------------------------------------------------------------------------
def metropolis_weights(neighbors: Mapping[Hashable, set]) -> dict[Hashable, dict[Hashable, float]]:
    """a_ij = 1/(1+max(d_i,d_j)) para vecinos; a_ii = 1 − Σ_j a_ij (grafo no dirigido)."""
    for i, nbrs in neighbors.items():
        for j in nbrs:
            if i == j or i not in neighbors.get(j, set()):
                raise ValueError(f"Grafo no simétrico o con bucle en ({i}, {j})")
    deg = {i: len(n) for i, n in neighbors.items()}
    W: dict[Hashable, dict[Hashable, float]] = {}
    for i, nbrs in neighbors.items():
        row = {j: 1.0 / (1 + max(deg[i], deg[j])) for j in nbrs}
        row[i] = 1.0 - sum(row.values())
        W[i] = row
    return W


def mix(own_id: Hashable, received: Mapping[Hashable, Update],
        weights_row: Mapping[Hashable, float], weight_by_samples: bool = True) -> StateDict:
    """w_i = Σ_{j ∈ R_i ∪ {i}} a_ij w_j, renormalizado sobre los modelos efectivamente disponibles.

    `received` debe incluir el propio modelo (`own_id`). Con `weight_by_samples`, a_ij ∝ W_ij · n_j,
    de modo que en malla completa con recepción completa la mezcla es FedAvg ponderado (D0 Eq. 1).
    """
    if own_id not in received:
        raise ValueError("La mezcla debe incluir el propio modelo")
    ids = [j for j in received if j in weights_row]
    if set(received) - set(ids):
        raise ValueError(f"Modelos de nodos que no son vecinos: {sorted(map(str, set(received) - set(ids)))}")
    raw = [weights_row[j] * (received[j].n_samples if weight_by_samples else 1.0) for j in ids]
    return weighted_average([received[j].state for j in ids], raw)


# ---------------------------------------------------------------------------
# Robustos (opcionales, desactivados en la referencia)
# ---------------------------------------------------------------------------
def coordinate_median(updates: Sequence[Update]) -> StateDict:
    states = [u.state for u in updates]
    _check_compatible(states)
    return {k: torch.stack([s[k] for s in states]).median(dim=0).values for k in states[0]}


def trimmed_mean(updates: Sequence[Update], trim_ratio: float = 0.2) -> StateDict:
    states = [u.state for u in updates]
    _check_compatible(states)
    n = len(states)
    k_trim = int(n * trim_ratio)
    if n - 2 * k_trim <= 0:
        raise ValueError("trim_ratio demasiado grande para el número de modelos")
    out: StateDict = {}
    for k in states[0]:
        sorted_vals = torch.stack([s[k] for s in states]).sort(dim=0).values
        out[k] = sorted_vals[k_trim:n - k_trim].mean(dim=0)
    return out
