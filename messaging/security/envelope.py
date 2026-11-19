"""Trama del perfil S2 (04 §4.8):

    u32 len_o | cabecera externa JSON | AES-256-GCM( trama interna S0 = u32 len_h | cabecera | payload )

La cabecera externa va en claro (el broker necesita poco más que el destino) y es el AAD exacto del AEAD:
`schema_version`, `msg_type`, `experiment_id`, `sender_id`, `recipient_id`, `epoch` (hex de 16) y `seq`.
Cambiar cualquier byte de la cabecera o del cifrado ⇒ la etiqueta no verifica.
"""
from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass
from typing import Any

from messaging.security.handshake import Session
from messaging.security.primitives import TAG_LEN, aead_open, aead_seal

E2E_CONTENT_TYPE = "application/vnd.fl-e2e"
_LEN = struct.Struct("!I")
MAX_OUTER_HEADER = 1024
OUTER_KEYS = ("schema_version", "msg_type", "experiment_id", "sender_id", "recipient_id", "epoch", "seq")
_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_TYPE = re.compile(r"^[A-Z_]{1,32}$")
_EPOCH = re.compile(r"^[0-9a-f]{16}$")


class EnvelopeError(ValueError):
    """Trama S2 mal formada (antes de cualquier descifrado)."""


@dataclass(frozen=True)
class Outer:
    header: dict[str, Any]
    aad: bytes
    ciphertext: bytes

    @property
    def epoch(self) -> int:
        return int(self.header["epoch"], 16)

    @property
    def seq(self) -> int:
        return self.header["seq"]


def outer_header(msg_type: str, experiment_id: str, sender: str, recipient: str, epoch: int, seq: int) -> bytes:
    h = {"schema_version": 1, "msg_type": msg_type, "experiment_id": experiment_id, "sender_id": sender,
         "recipient_id": recipient, "epoch": f"{epoch:016x}", "seq": seq}
    return json.dumps(h, sort_keys=True, separators=(",", ":")).encode()


def seal(session: Session, inner_frame: bytes, msg_type: str, experiment_id: str) -> bytes:
    seq, key, nonce = session.next_send()
    aad = outer_header(msg_type, experiment_id, session.me, session.peer, session.epoch, seq)
    return _LEN.pack(len(aad)) + aad + aead_seal(key, nonce, inner_frame, aad)


def parse(data: bytes, max_frame_bytes: int) -> Outer:
    """Solo estructura y tipos de la cabecera externa: no descifra ni consulta sesiones."""
    if len(data) > max_frame_bytes:
        raise EnvelopeError(f"Trama S2 de {len(data)} B > {max_frame_bytes}")
    if len(data) < _LEN.size:
        raise EnvelopeError("Trama S2 truncada")
    (hlen,) = _LEN.unpack_from(data)
    if hlen > MAX_OUTER_HEADER or _LEN.size + hlen + TAG_LEN > len(data):
        raise EnvelopeError(f"Longitud de cabecera externa inválida: {hlen}")
    aad = bytes(data[_LEN.size:_LEN.size + hlen])
    try:
        h = json.loads(aad.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EnvelopeError(f"Cabecera externa no es JSON: {exc}") from exc
    if not isinstance(h, dict) or sorted(h) != sorted(OUTER_KEYS):
        raise EnvelopeError("Cabecera externa con campos inesperados")
    ok = (h["schema_version"] == 1 and isinstance(h["msg_type"], str) and _TYPE.match(h["msg_type"])
          and all(isinstance(h[k], str) and _ID.match(h[k]) for k in ("experiment_id", "sender_id", "recipient_id"))
          and isinstance(h["epoch"], str) and _EPOCH.match(h["epoch"])
          and type(h["seq"]) is int and 1 <= h["seq"] < 2 ** 63)
    if not ok:
        raise EnvelopeError("Cabecera externa con valores inválidos")
    return Outer(h, aad, bytes(data[_LEN.size + hlen:]))


def open_(session: Session, outer: Outer) -> bytes:
    """Descifra con la clave derivada de (época, seq). Lanza CryptoError si la etiqueta no verifica.
    La ventana anti-replay la consulta y marca el canal (antes y después de esta llamada)."""
    key, nonce = session.recv_key(outer.seq)
    return aead_open(key, nonce, outer.ciphertext, outer.aad)
