"""Primitivas criptográficas de D3 (ECDSA P-256, ECDH, HKDF-SHA256, AES-256-GCM) — 04 ADR-05.

Solo envuelve a `cryptography` (OpenSSL): no hay criptografía implementada a mano. La conformidad con
vectores públicos (RFC 5869, GCM, RFC 5903, RFC 6979) se prueba en `tests/conformance/`.
"""
from __future__ import annotations

import hashlib
import hmac

from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature, encode_dss_signature
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDFExpand

CURVE = ec.SECP256R1()
KEY_LEN = 32           # AES-256 y longitud de las claves derivadas
NONCE_LEN = 12         # GCM
TAG_LEN = 16
SIG_LEN = 64           # firma ECDSA P-256 en formato r‖s (32 + 32 bytes)


class CryptoError(ValueError):
    """Fallo de verificación (firma, etiqueta AEAD) o material criptográfico inválido."""


# ---------------------------------------------------------------------------- HKDF (RFC 5869)
def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    """PRK = HMAC-SHA256(salt, IKM). Con `salt` vacío se usa un bloque de ceros (RFC 5869 §2.2)."""
    return hmac.new(salt or b"\x00" * hashlib.sha256().digest_size, ikm, hashlib.sha256).digest()


def hkdf_expand(prk: bytes, info: bytes, length: int = KEY_LEN) -> bytes:
    return HKDFExpand(hashes.SHA256(), length, info).derive(prk)


# ---------------------------------------------------------------------------- AEAD
def aead_seal(key: bytes, nonce: bytes, plaintext: bytes, aad: bytes) -> bytes:
    """AES-256-GCM: devuelve ciphertext ‖ etiqueta (16 bytes)."""
    return AESGCM(key).encrypt(nonce, plaintext, aad)


def aead_open(key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes:
    try:
        return AESGCM(key).decrypt(nonce, ciphertext, aad)
    except InvalidTag as exc:
        raise CryptoError("Etiqueta AEAD inválida") from exc


# ---------------------------------------------------------------------------- ECDH / ECDSA
def generate_ephemeral() -> ec.EllipticCurvePrivateKey:
    return ec.generate_private_key(CURVE)


def encode_point(pk: ec.EllipticCurvePublicKey) -> bytes:
    """Punto público sin comprimir (65 bytes, 0x04‖X‖Y)."""
    return pk.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)


def decode_point(data: bytes) -> ec.EllipticCurvePublicKey:
    """Valida que el punto está en la curva (rechaza puntos inválidos: ataque de curva inválida)."""
    try:
        return ec.EllipticCurvePublicKey.from_encoded_point(CURVE, data)
    except ValueError as exc:
        raise CryptoError("Punto público inválido") from exc


def ecdh(sk: ec.EllipticCurvePrivateKey, peer: ec.EllipticCurvePublicKey) -> bytes:
    return sk.exchange(ec.ECDH(), peer)


def sign(sk: ec.EllipticCurvePrivateKey, data: bytes) -> bytes:
    r, s = decode_dss_signature(sk.sign(data, ec.ECDSA(hashes.SHA256())))
    return r.to_bytes(32, "big") + s.to_bytes(32, "big")


def verify(pk: ec.EllipticCurvePublicKey, signature: bytes, data: bytes) -> None:
    if len(signature) != SIG_LEN:
        raise CryptoError("Longitud de firma inválida")
    der = encode_dss_signature(int.from_bytes(signature[:32], "big"), int.from_bytes(signature[32:], "big"))
    try:
        pk.verify(der, data, ec.ECDSA(hashes.SHA256()))
    except InvalidSignature as exc:
        raise CryptoError("Firma inválida") from exc
