"""Trama del protocolo y ruta de recepción (04 §4.6 y §4.8; perfiles S0/S1).

Formato:  u32 big-endian (longitud de la cabecera) | cabecera JSON UTF-8 | payload pickle (opcional)

Ruta de recepción (`decode_message`), en este orden y sin deserializar nada antes de tiempo:
  1. tamaño total ≤ límite                     5. descriptor = modelo esperado (si se indica)
  2. longitud de cabecera ≤ límite             6. unpickler restringido (messaging.serialization)
  3. JSON válido + contrato del msg_type       7. tensores recibidos = descriptor
  4. tamaño y sha256 del payload = descriptor
El sobre E2E (perfil S2: cabecera externa + AEAD) y la ventana anti-replay se añaden en E10.
"""
from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from messaging.contracts import ContractError, ContractValidator
from messaging.serialization import PICKLE_PROTOCOL, UnsafePayloadError, dumps_state, loads_state

_LEN = struct.Struct("!I")
DTYPE_NAMES = {torch.float32: "float32"}


class FrameError(ValueError):
    """Trama mal formada o fuera de límites."""


@dataclass(frozen=True)
class Limits:
    max_frame_bytes: int = 4 * 1024 * 1024
    max_header_bytes: int = 64 * 1024
    max_payload_bytes: int = 4 * 1024 * 1024

    @classmethod
    def from_config(cls, communication_cfg: Mapping[str, Any]) -> "Limits":
        return cls(**communication_cfg.get("limits", {}))


@dataclass
class Message:
    header: dict[str, Any]
    state: dict[str, torch.Tensor] | None
    nbytes: int


# ---------------------------------------------------------------------------
# Trama
# ---------------------------------------------------------------------------
def _reject_constant(name: str) -> Any:
    raise FrameError(f"Constante JSON no permitida: {name}")


def encode_frame(header: Mapping[str, Any], payload: bytes = b"", limits: Limits = Limits()) -> bytes:
    raw = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                     allow_nan=False).encode("utf-8")
    if len(raw) > limits.max_header_bytes:
        raise FrameError(f"Cabecera de {len(raw)} B > {limits.max_header_bytes}")
    if len(payload) > limits.max_payload_bytes:
        raise FrameError(f"Payload de {len(payload)} B > {limits.max_payload_bytes}")
    frame = _LEN.pack(len(raw)) + raw + payload
    if len(frame) > limits.max_frame_bytes:
        raise FrameError(f"Trama de {len(frame)} B > {limits.max_frame_bytes}")
    return frame


def decode_frame(data: bytes, limits: Limits = Limits()) -> tuple[dict[str, Any], bytes]:
    if len(data) > limits.max_frame_bytes:
        raise FrameError(f"Trama de {len(data)} B > {limits.max_frame_bytes}")
    if len(data) < _LEN.size:
        raise FrameError("Trama truncada")
    (hlen,) = _LEN.unpack_from(data)
    if hlen > limits.max_header_bytes or _LEN.size + hlen > len(data):
        raise FrameError(f"Longitud de cabecera inválida: {hlen}")
    try:
        header = json.loads(data[_LEN.size:_LEN.size + hlen].decode("utf-8"),
                            parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrameError(f"Cabecera JSON inválida: {exc}") from exc
    if not isinstance(header, dict):
        raise FrameError("La cabecera debe ser un objeto JSON")
    payload = bytes(data[_LEN.size + hlen:])
    if len(payload) > limits.max_payload_bytes:
        raise FrameError(f"Payload de {len(payload)} B > {limits.max_payload_bytes}")
    return header, payload


# ---------------------------------------------------------------------------
# Mensajes con modelo
# ---------------------------------------------------------------------------
def tensor_specs(state: Mapping[str, torch.Tensor]) -> list[dict[str, Any]]:
    specs = []
    for name, t in state.items():
        if t.dtype not in DTYPE_NAMES:
            raise TypeError(f"{name}: dtype {t.dtype} no soportado en la red")
        specs.append({"name": name, "shape": list(t.shape), "dtype": DTYPE_NAMES[t.dtype]})
    return specs


@dataclass(frozen=True)
class PreparedPayload:
    """Modelo serializado una sola vez, reutilizable en varias tramas (p. ej. difusión del modelo global)."""
    data: bytes
    descriptor: dict[str, Any]


def prepare_payload(state: Mapping[str, torch.Tensor]) -> PreparedPayload:
    data = dumps_state(state)
    return PreparedPayload(data, {"format": "pickle", "protocol": PICKLE_PROTOCOL, "nbytes": len(data),
                                  "sha256": hashlib.sha256(data).hexdigest(), "tensors": tensor_specs(state)})


def encode_message(header: Mapping[str, Any], state: Mapping[str, torch.Tensor] | PreparedPayload | None = None,
                   limits: Limits = Limits()) -> bytes:
    """Construye la trama. Si hay `state` (o un payload ya preparado), añade el descriptor `payload`."""
    header = dict(header)
    if "payload" in header:
        raise FrameError("La cabecera no debe traer el descriptor: lo añade encode_message")
    if state is None:
        return encode_frame(header, b"", limits)
    prepared = state if isinstance(state, PreparedPayload) else prepare_payload(state)
    header["payload"] = prepared.descriptor
    return encode_frame(header, prepared.data, limits)


def decode_message(data: bytes, validator: ContractValidator, limits: Limits = Limits(),
                   expected_tensors: list[dict[str, Any]] | None = None) -> Message:
    """Ruta de recepción. Lanza FrameError, ContractError o UnsafePayloadError; nunca ejecuta el payload."""
    header, payload = decode_frame(data, limits)
    validator.validate(header)
    desc = header.get("payload")
    if desc is None:
        if payload:
            raise FrameError("Payload presente sin descriptor")
        return Message(header, None, len(data))
    if len(payload) != desc.get("nbytes"):
        raise FrameError(f"Tamaño del payload {len(payload)} ≠ descriptor {desc.get('nbytes')}")
    if hashlib.sha256(payload).hexdigest() != desc.get("sha256"):
        raise FrameError("sha256 del payload no coincide con el descriptor")
    if expected_tensors is not None and desc.get("tensors") != expected_tensors:
        raise ContractError("Los tensores anunciados no coinciden con el modelo esperado")
    state = loads_state(payload)
    if tensor_specs(state) != desc["tensors"]:
        raise UnsafePayloadError("El contenido deserializado no coincide con el descriptor")
    return Message(header, state, len(data))
