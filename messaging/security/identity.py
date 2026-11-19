"""Identidades de larga duración de los nodos (perfil S2) y almacén de confianza.

`trust.json` = {"curve": "P-256", "nodes": {node_id: {"kid": hex, "public_key": PEM}}}.
La clave privada de cada nodo está en `<node_id>.key` (PKCS#8 PEM). `kid` = SHA-256(DER)[:16] en hex.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from messaging.security.primitives import CryptoError

TRUST_FILE = "trust.json"


def public_key_pem(pk: ec.EllipticCurvePublicKey) -> bytes:
    return pk.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)


def private_key_pem(sk: ec.EllipticCurvePrivateKey) -> bytes:
    return sk.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption())


def key_id(pk: ec.EllipticCurvePublicKey) -> str:
    der = pk.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return hashlib.sha256(der).hexdigest()[:32]


@dataclass(frozen=True)
class PeerKey:
    node_id: str
    kid: str
    public_key: ec.EllipticCurvePublicKey


class TrustStore:
    def __init__(self, peers: Mapping[str, PeerKey]) -> None:
        self._peers = dict(peers)

    @classmethod
    def load(cls, path: str | Path) -> "TrustStore":
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        if doc.get("curve") != "P-256":
            raise CryptoError("Curva no soportada en el almacén de confianza")
        peers = {}
        for n, e in doc["nodes"].items():
            pk = serialization.load_pem_public_key(e["public_key"].encode())
            if not isinstance(pk, ec.EllipticCurvePublicKey) or pk.curve.name != "secp256r1":
                raise CryptoError(f"{n}: la clave pública no es P-256")
            if key_id(pk) != e["kid"]:
                raise CryptoError(f"{n}: el kid no corresponde a la clave pública")
            peers[n] = PeerKey(n, e["kid"], pk)
        return cls(peers)

    def get(self, node_id: str) -> PeerKey:
        try:
            return self._peers[node_id]
        except KeyError:
            raise CryptoError(f"Nodo sin identidad registrada: {node_id}") from None

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._peers

    @property
    def node_ids(self) -> list[str]:
        return sorted(self._peers)


@dataclass(frozen=True)
class Identity:
    node_id: str
    kid: str
    private_key: ec.EllipticCurvePrivateKey

    @classmethod
    def load(cls, node_id: str, key_path: str | Path, trust: TrustStore) -> "Identity":
        sk = serialization.load_pem_private_key(Path(key_path).read_bytes(), password=None)
        if not isinstance(sk, ec.EllipticCurvePrivateKey):
            raise CryptoError("La clave del nodo no es ECDSA")
        kid = key_id(sk.public_key())
        if trust.get(node_id).kid != kid:
            raise CryptoError(f"{node_id}: la clave privada no corresponde al almacén de confianza")
        return cls(node_id, kid, sk)
