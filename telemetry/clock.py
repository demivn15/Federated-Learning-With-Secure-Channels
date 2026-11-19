"""Reloj monótono local y temporizadores por fase (RNF-07, ADR-11).

Nunca se restan marcas de tiempo de nodos distintos: cada intervalo se mide en el nodo que lo observa.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


def now_ns() -> int:
    # perf_counter_ns: monótono y de alta resolución en todas las plataformas. monotonic_ns en Windows
    # tiene ≈ 15,6 ms de resolución (detectado en E4), demasiado gruesa para medir stragglers.
    return time.perf_counter_ns()


class PhaseTimer:
    """Acumula segundos por fase (`train`, `serialize`, `wait`, …) dentro de una ronda."""

    def __init__(self) -> None:
        self._acc: dict[str, int] = {}
        self._start = now_ns()

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        t0 = now_ns()
        try:
            yield
        finally:
            self._acc[name] = self._acc.get(name, 0) + (now_ns() - t0)

    def add(self, name: str, seconds: float) -> None:
        self._acc[name] = self._acc.get(name, 0) + int(seconds * 1e9)

    def seconds(self, name: str) -> float:
        return self._acc.get(name, 0) / 1e9

    def as_columns(self, prefix: str = "t_", suffix: str = "_s") -> dict[str, float]:
        return {f"{prefix}{k}{suffix}": v / 1e9 for k, v in self._acc.items()}

    def elapsed(self) -> float:
        return (now_ns() - self._start) / 1e9


class ArrivalClock:
    """Horas de llegada de las actualizaciones, medidas en el agregador (straggler válido; corrige P2-H1)."""

    def __init__(self) -> None:
        self.arrivals: dict[str, int] = {}

    def mark(self, sender: str, t_ns: int | None = None) -> None:
        """Registra la llegada de `sender` (por defecto, ahora). `t_ns`: hora de llegada al transporte."""
        self.arrivals.setdefault(sender, now_ns() if t_ns is None else t_ns)

    def straggler_s(self) -> float | None:
        if len(self.arrivals) < 2:
            return None
        return (max(self.arrivals.values()) - min(self.arrivals.values())) / 1e9

    def relative_s(self) -> dict[str, float]:
        if not self.arrivals:
            return {}
        first = min(self.arrivals.values())
        return {k: round((v - first) / 1e9, 6) for k, v in sorted(self.arrivals.items(), key=lambda kv: kv[1])}
