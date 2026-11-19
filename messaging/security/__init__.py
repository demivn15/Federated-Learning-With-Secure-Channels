"""Seguridad de extremo a extremo sobre el broker (E10; 04 ADR-05, RF-19…RF-21).

Perfiles (`communication.security_profile`):
  * S0 — sin seguridad (reproducción de D0).
  * S1 — AMQPS (TLS nodo↔broker) + usuario y permisos por nodo + *unpickler* restringido.
  * S2 — S1 + sobre E2E: *handshake* firmado por par y época, derivación indexada de claves,
         AES-256-GCM con la cabecera externa como AAD, ventana anti-replay y telemetría SD firmada.

Módulos: `primitives` (HKDF, AEAD, ECDH, ECDSA), `identity` (claves de larga duración y almacén de
confianza), `handshake` (máquina de estados sin E/S), `envelope` (trama S2), `replay` (ventana) y
`channel` (transporte seguro que envuelve a `AmqpTransport`).
"""
PROFILES = ("S0", "S1", "S2")


def uses_tls(profile: str) -> bool:
    return profile in ("S1", "S2")


def uses_e2e(profile: str) -> bool:
    return profile == "S2"
