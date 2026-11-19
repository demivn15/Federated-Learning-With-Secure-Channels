"""Membresía de un par P2P (RF-22; D5 §5.2; 04 ADR-06, ADR-12): estado de cada vecino del grafo.

Detección por rondas (D5 §5.2 (2)): una ronda sin su `MODEL` ⇒ `SUSPECT` (se le sigue esperando hasta
`round_timeout_s`); `dead_after_rounds` rondas seguidas ⇒ `DEAD` (ya no se le espera: la ronda no consume
T_round por su causa). Cualquier mensaje suyo lo devuelve a `ALIVE`. `DONE` ⇒ terminado: no se le espera
y cuenta para la terminación.

No se usan *heartbeats* periódicos por `fl.broadcast` (04 ADR-06 los preveía). Motivo, observado al
diseñar E8: con `x-overflow = reject-publish`, una difusión `fanout` que alcanza la cola llena de un par
caído recibe un *nack* del broker, así que los latidos de los pares vivos empezarían a fallar justo
cuando hay un fallo. El tráfico de cada ronda (`MODEL`) ya es la señal de vida, y todo se envía en unicast.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

ALIVE, SUSPECT, DEAD, FINISHED = "ALIVE", "SUSPECT", "DEAD", "DONE"


@dataclass
class Membership:
    neighbors: tuple[str, ...]
    dead_after_rounds: int = 3
    state: dict[str, str] = field(default_factory=dict)
    misses: dict[str, int] = field(default_factory=dict)
    last_round: dict[str, int] = field(default_factory=dict)       # mayor ronda vista de cada vecino

    def __post_init__(self) -> None:
        self.neighbors = tuple(sorted(self.neighbors))
        for n in self.neighbors:
            self.state.setdefault(n, ALIVE)
            self.misses.setdefault(n, 0)

    def heard(self, node: str, round_: int | None = None) -> None:
        if node not in self.state:
            return
        if round_ is not None:
            self.last_round[node] = max(round_, self.last_round.get(node, 0))
        if self.state[node] != FINISHED:
            self.state[node], self.misses[node] = ALIVE, 0

    def finished(self, node: str, round_: int | None = None) -> None:
        if node in self.state:
            self.heard(node, round_)
            self.state[node] = FINISHED

    def end_round(self, received: Iterable[str]) -> None:
        received = set(received)
        for n in self.neighbors:
            if self.state[n] == FINISHED:
                continue
            if n in received:
                self.state[n], self.misses[n] = ALIVE, 0
            else:
                self.misses[n] += 1
                self.state[n] = DEAD if self.misses[n] >= self.dead_after_rounds else SUSPECT

    def expected(self) -> set[str]:
        """Vecinos cuyo MODEL se espera en la ronda."""
        return {n for n in self.neighbors if self.state[n] in (ALIVE, SUSPECT)}

    def all_done(self) -> bool:
        """Terminación (D5 §3.4): ningún vecino vivo pendiente y al menos uno terminado."""
        states = [self.state[n] for n in self.neighbors]
        return bool(states) and FINISHED in states and all(s in (FINISHED, DEAD) for s in states)

    def ahead_by(self, round_: int) -> int:
        """Cuántas rondas va por delante el vecino más adelantado (para decidir un SYNC)."""
        return max((r - round_ for r in self.last_round.values()), default=0)

    def snapshot(self) -> dict[str, str]:
        return dict(self.state)
