"""Telemetría SD firmada (perfil S2; 04 ADR-05/ADR-08, R-02).

El AEAD por par ya autentica el `CLAIM` ante el coordinador; la firma ECDSA de larga duración añade
**no repudio**: la telemetría autodeclarada queda atribuible a su nodo aunque se reenvíe o se audite después
(ataque F5 de D3, `SD-score` falsificado). La firma cubre experimento, emisor, ronda, intento y telemetría.
Firmar no hace verdadera la telemetría (D3 "firma ≠ veracidad"): solo impide negarla o alterarla.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

from messaging.security.identity import Identity, TrustStore
from messaging.security.primitives import CryptoError, sign, verify

SIGNED_FIELDS = ("experiment_id", "sender_id", "round", "attempt", "telemetry")


def _payload(header: Mapping[str, Any]) -> bytes:
    body = {k: header.get(k) for k in SIGNED_FIELDS}
    return b"telemetry" + json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class TelemetrySigner:
    def __init__(self, identity: Identity, trust: TrustStore) -> None:
        self.identity, self.trust = identity, trust

    def sign_header(self, header: dict[str, Any]) -> None:
        if "telemetry" in header:
            header["telemetry_sig"] = sign(self.identity.private_key, _payload(header)).hex()

    def verify_header(self, header: Mapping[str, Any]) -> None:
        if "telemetry" not in header:
            return
        sig = header.get("telemetry_sig")
        if not isinstance(sig, str):
            raise CryptoError("Telemetría sin firma (obligatoria en S2)")
        verify(self.trust.get(header["sender_id"]).public_key, bytes.fromhex(sig), _payload(header))
