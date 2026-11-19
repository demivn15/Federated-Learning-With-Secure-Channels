"""Grafo lógico de la topología P2P (04 ADR-06; D5 §4). Independiente del broker: todo el tráfico pasa por
RabbitMQ, el grafo solo decide a quién envía cada par su modelo y con quién mezcla.

  * `full_mesh` — todos con todos. Con pesos de Metropolis ponderados por n y recepción completa, la mezcla
                  es exactamente FedAvg ponderado (D0 Eq. 1) en una ronda (λ₂ = 0).
  * `ring`      — anillo sobre los pares ordenados. λ₂ = (1 + 2 cos(2π/N))/3 con pesos 1/3 (N ≥ 3).
  * `random_k`  — anillo sobre una permutación aleatoria (conexo por construcción) + aristas aleatorias hasta
                  que cada par tenga grado ≥ k. Determinista con `graph_seed`.
"""
from __future__ import annotations

import random
from typing import Iterable, Mapping

import numpy as np

GRAPHS = ("full_mesh", "ring", "random_k")
Graph = dict[str, set[str]]


def _ring(order: list[str]) -> Graph:
    g: Graph = {p: set() for p in order}
    n = len(order)
    if n < 2:
        return g
    for i, p in enumerate(order):
        q = order[(i + 1) % n]
        if q != p:
            g[p].add(q)
            g[q].add(p)
    return g


def build_graph(kind: str, peers: Iterable[str], k: int | None = None, seed: int = 0) -> Graph:
    peers = sorted(peers)
    if len(set(peers)) != len(peers) or not peers:
        raise ValueError("Pares vacíos o repetidos")
    if kind == "full_mesh":
        return {p: set(peers) - {p} for p in peers}
    if kind == "ring":
        return _ring(peers)
    if kind == "random_k":
        if k is None or not 2 <= k <= len(peers) - 1:
            raise ValueError(f"random_k necesita 2 ≤ k ≤ {len(peers) - 1}")
        rng = random.Random(f"random_k|{seed}")
        order = peers[:]
        rng.shuffle(order)
        g = _ring(order)
        for p in peers:
            others = [q for q in peers if q != p and q not in g[p]]
            rng.shuffle(others)
            while len(g[p]) < k and others:
                q = others.pop()
                g[p].add(q)
                g[q].add(p)
        return g
    raise ValueError(f"Grafo desconocido: {kind!r} (opciones: {GRAPHS})")


def is_connected(g: Mapping[str, set[str]]) -> bool:
    if not g:
        return True
    start = next(iter(g))
    seen, stack = {start}, [start]
    while stack:
        for q in g[stack.pop()]:
            if q not in seen:
                seen.add(q)
                stack.append(q)
    return len(seen) == len(g)


def mixing_matrix(g: Mapping[str, set[str]]) -> tuple[list[str], np.ndarray]:
    """Matriz de Metropolis (sin ponderar por n) en el orden de los pares; doblemente estocástica."""
    from core.aggregation import metropolis_weights
    order = sorted(g)
    W = metropolis_weights({p: set(g[p]) for p in order})
    return order, np.array([[W[i].get(j, 0.0) for j in order] for i in order])


def second_eigenvalue(g: Mapping[str, set[str]]) -> float:
    """λ₂ = mayor |valor propio| distinto del de la media: velocidad de consenso de la mezcla pura."""
    _, M = mixing_matrix(g)
    ev = sorted(np.abs(np.linalg.eigvalsh(M)), reverse=True)        # simétrica ⇒ valores propios reales
    return float(ev[1]) if len(ev) > 1 else 0.0
