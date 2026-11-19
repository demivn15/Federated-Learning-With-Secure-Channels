"""Salvaguardas de la lotería SD (RF-05; D0 §4.3; 04 ADR-08, ADR-12) y observaciones para `SD-score`.

D0 §4.3 observó que, sin salvaguardas, la lotería puede elegir repetidamente un nodo lento o caído y
provocar un fallo en cascada que congela las métricas. Recomienda *health checks*, un límite de
selecciones consecutivas y un *fallback*. Aquí, con `safeguards.enabled`:
  * **health check**: un Client Node que no responde a `ELECTION` (no envía `CLAIM` en
    `election_timeout_s`) deja de ser candidato hasta que vuelva a responder;
  * **límite de selecciones consecutivas** (`max_consecutive_selections`): el agregador de las últimas N
    rondas queda excluido de la siguiente elección (salvo que sea el único candidato);
  * **re-selección**: si el ganador no responde o el agregador no envía `AGG_COMPLETE` en
    `outer_round_timeout_s`, el coordinador repite la elección de la MISMA ronda (intento + 1) sin él.
Sin salvaguardas (reproducción fiel de D0) los candidatos son siempre todos los registrados y una ronda
cuyo agregador falla se da por perdida (métrica congelada).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class SafeguardState:
    registered: tuple[str, ...]
    enabled: bool = False
    max_consecutive: int = 2
    alive: set[str] = field(default_factory=set)
    history: list[str] = field(default_factory=list)      # agregador efectivo de cada ronda completada

    def __post_init__(self) -> None:
        self.alive = set(self.registered)

    @classmethod
    def from_config(cls, registered, cfg: Mapping[str, Any] | bool | None) -> "SafeguardState":
        cfg = cfg if isinstance(cfg, Mapping) else {"enabled": bool(cfg)}
        return cls(tuple(sorted(registered)), bool(cfg.get("enabled", False)),
                   int(cfg.get("max_consecutive_selections", 2)))

    def candidates(self, excluded: set[str]) -> list[str]:
        """Candidatos del intento actual."""
        if not self.enabled:
            return list(self.registered)
        pool = sorted(self.alive - excluded)
        last = self.history[-self.max_consecutive:]
        if len(last) == self.max_consecutive and len(set(last)) == 1 and len(pool) > 1:
            pool = [n for n in pool if n != last[0]]
        return pool

    def observe_claims(self, responders: set[str]) -> None:
        if self.enabled:
            self.alive = set(responders)

    def mark_failed(self, node: str) -> None:
        if self.enabled:
            self.alive.discard(node)

    def record_aggregator(self, node: str) -> None:
        self.history.append(node)


@dataclass
class Observations:
    """Variables OBSERVADAS por otros nodos (no autodeclaradas), con media exponencial (R-02)."""
    alpha: float = 0.5
    lateness_s: dict[str, float] = field(default_factory=dict)
    rtt_s: dict[str, float] = field(default_factory=dict)
    declared: dict[str, dict[str, float | None]] = field(default_factory=dict)

    def _ewma(self, table: dict[str, float], node: str, value: float) -> None:
        table[node] = value if node not in table else self.alpha * value + (1 - self.alpha) * table[node]

    def add_rtt(self, node: str, seconds: float) -> None:
        self._ewma(self.rtt_s, node, seconds)

    def add_lateness(self, lateness: Mapping[str, float]) -> None:
        for node, s in lateness.items():
            self._ewma(self.lateness_s, node, float(s))

    def add_declared(self, node: str, telemetry: Mapping[str, float | None]) -> None:
        self.declared[node] = dict(telemetry)

    def observed(self) -> dict[str, dict[str, float | None]]:
        nodes = set(self.lateness_s) | set(self.rtt_s)
        return {n: {"lateness_s": self.lateness_s.get(n), "rtt_s": self.rtt_s.get(n)} for n in nodes}
