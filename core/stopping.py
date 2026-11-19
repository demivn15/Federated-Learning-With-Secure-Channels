"""Políticas de rondas (analisis_arquitectonico/03 RF-13, I-29).

* `FixedRounds`  — centralizada (12) y jerárquica (30): presupuesto fijo (D0 Tabla 2).
* `RecallPatience` — SD y P2P: misma lógica que `check_early_stopping` de P1/P3
  (mejora si recall > mejor + min_improvement; parar tras `patience` rondas sin mejora o al llegar al máximo).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class StopDecision:
    stop: bool
    reason: str | None = None
    improved: bool = False


class FixedRounds:
    def __init__(self, rounds: int) -> None:
        if rounds <= 0:
            raise ValueError("rounds > 0")
        self.rounds = rounds

    def update(self, round_idx: int, metric: float | None = None) -> StopDecision:
        if round_idx >= self.rounds:
            return StopDecision(True, f"fixed_rounds_{self.rounds}")
        return StopDecision(False)


@dataclass
class RecallPatience:
    patience: int
    min_improvement: float
    max_rounds: int
    best: float = 0.0                      # P1/P3 inician best_recall = 0.0
    best_round: int | None = None
    rounds_without_improvement: int = 0
    history: list[tuple[int, float]] = field(default_factory=list)

    def update(self, round_idx: int, metric: float) -> StopDecision:
        self.history.append((round_idx, metric))
        if round_idx >= self.max_rounds:
            return StopDecision(True, f"max_rounds_{self.max_rounds}")
        improved = metric > self.best + self.min_improvement
        if improved:
            self.best, self.best_round, self.rounds_without_improvement = metric, round_idx, 0
        else:
            self.rounds_without_improvement += 1
        if self.rounds_without_improvement >= self.patience:
            return StopDecision(True, f"early_stopping_patience_{self.patience}", improved)
        return StopDecision(False, None, improved)


def build_round_policy(rounds_cfg: Mapping[str, Any]) -> FixedRounds | RecallPatience:
    policy = rounds_cfg["policy"]
    if policy == "fixed":
        return FixedRounds(int(rounds_cfg["value"]))
    if policy == "early_stopping":
        return RecallPatience(int(rounds_cfg["patience"]), float(rounds_cfg["min_improvement"]),
                              int(rounds_cfg.get("max_global_rounds", rounds_cfg.get("max_rounds"))))
    raise ValueError(f"Política de rondas desconocida: {policy}")
