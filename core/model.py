"""MLP del paper base (D0 §3.3) con nombres canónicos de parámetros.

Referencia: capas ocultas 256/128/64, ReLU, dropout 0,3 tras cada capa oculta, salida de 1 logit.
La construcción reproduce `MLPLarger` / `MLP` de P1/P3 (`mlp.py`): misma semilla ⇒ mismos pesos
iniciales y mismas claves (`fc1`, `fc2`, …). Se verifica en tests/equivalence.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

StateDict = dict[str, torch.Tensor]


class MLP(nn.Module):
    """Perceptrón multicapa para clasificación binaria; devuelve logits."""

    def __init__(self, in_features: int, hidden_layers: Sequence[int] = (256, 128, 64),
                 dropout: float = 0.3, seed: int = 42) -> None:
        super().__init__()
        if in_features <= 0 or not hidden_layers:
            raise ValueError("in_features > 0 y al menos una capa oculta")
        # Igual que P1/P3: la semilla global se fija aquí, antes de crear las capas.
        torch.manual_seed(seed)
        dims = [in_features, *hidden_layers, 1]
        self._layer_names: list[str] = []
        for i in range(len(dims) - 1):
            name = f"fc{i + 1}"
            setattr(self, name, nn.Linear(dims[i], dims[i + 1]))
            self._layer_names.append(name)
        self.dropout = nn.Dropout(p=dropout)
        self.in_features = in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *hidden, last = self._layer_names
        for name in hidden:
            x = self.dropout(F.relu(getattr(self, name)(x)))
        return getattr(self, last)(x)


def build_model(cfg: Mapping[str, Any], in_features: int) -> MLP:
    """Construye el MLP desde la configuración (`cfg['model']`, semilla de `cfg['data']`)."""
    m = cfg["model"]
    if m.get("type", "mlp") != "mlp":
        raise ValueError(f"Tipo de modelo no soportado: {m.get('type')}")
    return MLP(in_features, m["hidden_layers"], m["dropout"], seed=cfg["data"]["seed"])


def arch_hash(state: nn.Module | Mapping[str, torch.Tensor]) -> str:
    """Huella de la arquitectura: nombres, formas y tipos de los parámetros (no sus valores)."""
    sd = state.state_dict() if isinstance(state, nn.Module) else state
    spec = [[name, list(t.shape), str(t.dtype)] for name, t in sd.items()]
    return hashlib.sha256(json.dumps(spec).encode("utf-8")).hexdigest()


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def clone_state(state: Mapping[str, torch.Tensor]) -> StateDict:
    return {k: v.detach().clone() for k, v in state.items()}
