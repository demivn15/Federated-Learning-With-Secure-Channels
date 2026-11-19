"""Entrenamiento local de un Client Node (FedAvg / FedProx / FedNova).

Reproduce el bucle de `train_local` de P3 (`agent_csv.py`), equivalente al de P1:
para cada época y lote → `zero_grad` → pérdida (+ término proximal de FedProx) → `backward` → `step`.
`tau` = nº de pasos del optimizador, que FedNova (D0 Eq. 3) necesita.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from core.model import StateDict, clone_state


@dataclass
class Update:
    """Resultado de una ronda de entrenamiento local: lo que el nodo envía para agregar."""
    state: StateDict
    n_samples: int
    tau: float                         # pasos del optimizador (int) de un nodo; τ̄ (real) de un clúster
    train_loss: float


class LocalTrainer:
    """Mantiene el modelo, el optimizador y el cargador de datos de un nodo entre rondas."""

    def __init__(self, model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor, *,
                 batch_size: int, learning_rate: float, seed: int,
                 optimizer_state: str = "persistent_per_node") -> None:
        if optimizer_state not in ("persistent_per_node", "recreate_each_round"):
            raise ValueError(f"optimizer_state desconocido: {optimizer_state}")
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer_state = optimizer_state
        self.criterion = nn.BCEWithLogitsLoss()
        self.n_samples = int(X_train.shape[0])
        self.loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size,
                                 shuffle=True, generator=torch.Generator().manual_seed(seed))
        self.optimizer = self._new_optimizer()

    def _new_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def load_global(self, state: Mapping[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state)

    def train(self, epochs: int, *, mu: float = 0.0,
              proximal_reference: Mapping[str, torch.Tensor] | None = None) -> Update:
        """Entrena `epochs` épocas. Si `mu > 0` aplica FedProx contra `proximal_reference` (modelo global)."""
        if mu > 0 and proximal_reference is None:
            raise ValueError("FedProx necesita el modelo global de referencia")
        if self.optimizer_state == "recreate_each_round":  # variante P2
            self.optimizer = self._new_optimizer()
        ref = None if proximal_reference is None else clone_state(proximal_reference)

        self.model.train()
        tau, loss_sum = 0, 0.0
        for _ in range(epochs):
            for bx, by in self.loader:
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(bx), by)
                if mu > 0:
                    prox = sum(((p - ref[n]) ** 2).sum() for n, p in self.model.named_parameters())
                    loss = loss + (mu / 2) * prox
                loss.backward()
                self.optimizer.step()
                tau += 1
                loss_sum += float(loss.detach())
        return Update(state=clone_state(self.model.state_dict()), n_samples=self.n_samples,
                      tau=tau, train_loss=loss_sum / max(tau, 1))
