"""Reincorporación de un par atrasado (RF-22; D5 §5.2–5.3; 04 ADR-06).

Un par que vuelve tras estar aislado del broker encuentra en su cola los `MODEL` que sus vecinos le
siguieron enviando (la cola nombrada los retiene hasta el TTL). Si el vecino más adelantado le saca más de
`buffer_rounds` rondas, no intenta ponerse al día ronda a ronda (nunca alcanzaría a pares igual de rápidos):
envía `SYNC_REQ`, recibe `SYNC_RESP(ronda, modelo)` de sus vecinos y adopta el modelo de la ronda más
avanzada (media ponderada por n si varios están en esa ronda). Se reincorpora en la siguiente frontera de
ronda; su primera contribución es ya posterior a la sincronización (no arrastra al grupo hacia atrás).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from core.aggregation import weighted_average


@dataclass(frozen=True)
class SyncResponse:
    sender: str
    round: int
    n_samples: int
    state: dict[str, torch.Tensor]


def adopt(responses: Sequence[SyncResponse]) -> tuple[int, dict[str, torch.Tensor], list[str]]:
    """(ronda adoptada, modelo, vecinos cuyo modelo se usó)."""
    if not responses:
        raise ValueError("Sin respuestas de sincronización")
    top = max(r.round for r in responses)
    best = sorted((r for r in responses if r.round == top), key=lambda r: r.sender)
    if len(best) == 1:
        return top, best[0].state, [best[0].sender]
    return top, weighted_average([r.state for r in best], [r.n_samples for r in best]), [r.sender for r in best]
