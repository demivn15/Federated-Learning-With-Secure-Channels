"""Serialización de parámetros con el formato del proyecto base: `pickle` de `dict[nombre → Tensor]`.

Justificación del formato y mitigaciones: docs/serialization_rationale.md (04 ADR-04).

Envío (`dumps_state`): idéntico a P1/P3 (`pickle.dumps(dict)`), con protocolo fijo ⇒ mismos bytes en la red.

Recepción (`loads_state`): *unpickler* restringido.
  * Solo se resuelven tres `globals`, los únicos que produce `pickle.dumps` de tensores:
      - `collections.OrderedDict`
      - `torch._utils._rebuild_tensor_v2`
      - `torch.storage._load_from_bytes` → **sustituido** por `_safe_load_storage_bytes`.
        El original llama a `torch.load(..., weights_only=False)`, que deserializaría sin restricciones
        un payload anidado; el sustituto usa `weights_only=True` (cargador restringido de PyTorch).
  * Cualquier otro `global` ⇒ `UnsafePayloadError`, **antes** de ejecutar nada.
  * Después se valida la estructura: dict de str → Tensor flotante denso, finito, sin gradiente.
"""
from __future__ import annotations

import collections
import io
import pickle
from typing import Any, Mapping

import torch
import torch._utils

PICKLE_PROTOCOL = 4


class UnsafePayloadError(pickle.UnpicklingError):
    """El payload intenta usar algo fuera de la lista blanca o no tiene la estructura esperada."""


def _safe_load_storage_bytes(b: bytes) -> Any:
    obj = torch.load(io.BytesIO(b), weights_only=True)
    if not isinstance(obj, (torch.UntypedStorage, torch.storage.TypedStorage)):
        raise UnsafePayloadError(f"El almacenamiento anidado no es un storage: {type(obj).__name__}")
    return obj


_ALLOWED: dict[tuple[str, str], Any] = {
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
    ("torch.storage", "_load_from_bytes"): _safe_load_storage_bytes,
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        try:
            return _ALLOWED[(module, name)]
        except KeyError:
            raise UnsafePayloadError(f"global no permitido en el payload: {module}.{name}") from None

    def persistent_load(self, pid: Any) -> Any:
        raise UnsafePayloadError("persistent_load no permitido")


def dumps_state(state: Mapping[str, torch.Tensor]) -> bytes:
    """Mismo formato que P1/P3: `pickle.dumps(dict[str, Tensor])`."""
    plain = {}
    for k, v in state.items():
        if not isinstance(k, str) or not isinstance(v, torch.Tensor):
            raise TypeError("El estado debe ser dict[str, Tensor]")
        plain[k] = v.detach().cpu().contiguous()
    return pickle.dumps(plain, protocol=PICKLE_PROTOCOL)


def loads_state(data: bytes) -> dict[str, torch.Tensor]:
    """Deserializa con el *unpickler* restringido y valida la estructura. No llama a `pickle.loads`."""
    try:
        obj = RestrictedUnpickler(io.BytesIO(data)).load()
    except UnsafePayloadError:
        raise
    except Exception as exc:  # incluye errores de torch.load(weights_only=True) y pickle mal formado
        raise UnsafePayloadError(f"payload no deserializable de forma segura: {type(exc).__name__}: {exc}") from exc
    if not isinstance(obj, dict) or not obj:
        raise UnsafePayloadError("El payload debe ser un dict no vacío")
    for k, v in obj.items():
        if not isinstance(k, str):
            raise UnsafePayloadError("Clave no textual en el payload")
        if not isinstance(v, torch.Tensor) or v.layout != torch.strided or not torch.is_floating_point(v):
            raise UnsafePayloadError(f"{k}: se esperaba un tensor flotante denso")
        if v.requires_grad:
            raise UnsafePayloadError(f"{k}: tensor con requires_grad")
        if not torch.isfinite(v).all():
            raise UnsafePayloadError(f"{k}: contiene NaN o Inf")
    return obj
