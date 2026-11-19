"""*Handshake* E2E del perfil S2 (04 §4.6 y ADR-05): máquina de estados sin E/S.

Protocolo simétrico (no hay iniciador fijo; ambos extremos pueden empezar a la vez):

    A → B : HS_HELLO(kid_A, N_A, ePK_A, reply_to)      reply_to = N del HELLO al que responde, o null
    B → A : HS_HELLO(kid_B, N_B, ePK_B, reply_to = N_A)
    A → B : HS_AUTH(N_A, N_B, Sig_A(transcript ‖ "A"))
    B → A : HS_AUTH(N_B, N_A, Sig_B(transcript ‖ "B"))

    transcript = "EdgeFed-v3" ‖ exp_id ‖ id_X ‖ kid_X ‖ id_Y ‖ kid_Y ‖ N_X ‖ N_Y ‖ ePK_X ‖ ePK_Y
                 (X, Y = los dos nodos ordenados por node_id; cada campo con prefijo de longitud u16)
    K_master   = HKDF-Extract(N_X ‖ N_Y, ECDH(eSK, ePK))
    K_dir(P→Q) = HKDF-Expand(K_master, "dir" ‖ H(transcript) ‖ P ‖ "->" ‖ Q)
    K_msg(seq) = HKDF-Expand(K_dir, "msg" ‖ epoch ‖ seq)              (derivación INDEXADA, R-01)
    epoch      = primeros 8 bytes de SHA-256(transcript)

Diferencias con D3 (02 §3.1, W-12): identidades e IDs de clave dentro del transcript; derivación indexada
en lugar del ratchet encadenado (perder mensajes no desincroniza); nueva época en cada *handshake*.

Reglas de emparejamiento de HELLO (garantizan que los dos extremos eligen el mismo par de HELLO):
  * HELLO con `reply_to` = mi HELLO pendiente ⇒ emparejar.
  * HELLO "fresco" (`reply_to` null) y tengo un HELLO fresco pendiente ⇒ inicio simultáneo: emparejar.
  * HELLO fresco sin pendiente (o con un pendiente que ya respondía a otro) ⇒ responder con un HELLO
    nuevo (`reply_to` = su nonce) y emparejar.
  * Cualquier otro (nonce repetido, respuesta a un HELLO que ya no está pendiente) ⇒ ignorar.
"""
from __future__ import annotations

import hashlib
import os
import struct
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping

from cryptography.hazmat.primitives.asymmetric import ec

from messaging.security.identity import Identity, TrustStore
from messaging.security.primitives import (KEY_LEN, CryptoError, decode_point, ecdh, encode_point,
                                           generate_ephemeral, hkdf_expand, hkdf_extract, sign, verify)
from messaging.security.replay import ReplayWindow

LABEL = b"EdgeFed-v3"
NONCE_BYTES = 32
HELLO, AUTH = "HS_HELLO", "HS_AUTH"
HANDSHAKE_TYPES = frozenset({HELLO, AUTH})


def _lp(b: bytes) -> bytes:
    return struct.pack("!H", len(b)) + b


def build_transcript(experiment_id: str, a: Mapping[str, Any], b: Mapping[str, Any]) -> bytes:
    """`a` y `b`: {node_id, kid, nonce, epk} de cada extremo (en cualquier orden; se ordenan por node_id)."""
    x, y = sorted((a, b), key=lambda h: h["node_id"])
    parts = [LABEL, experiment_id.encode(), x["node_id"].encode(), bytes.fromhex(x["kid"]),
             y["node_id"].encode(), bytes.fromhex(y["kid"]), x["nonce"], y["nonce"], x["epk"], y["epk"]]
    return b"".join(_lp(p) for p in parts)


def message_key(k_dir: bytes, epoch: int, seq: int) -> bytes:
    return hkdf_expand(k_dir, b"msg" + epoch.to_bytes(8, "big") + seq.to_bytes(8, "big"), KEY_LEN)


def message_nonce(epoch: int, seq: int) -> bytes:
    """Único por clave (cada mensaje tiene su propia clave); se fija igualmente a `epoch[4:] ‖ seq`."""
    return epoch.to_bytes(8, "big")[4:] + seq.to_bytes(8, "big")


@dataclass
class Session:
    """Claves de una época entre este nodo y `peer`."""
    me: str
    peer: str
    epoch: int
    k_send: bytes
    k_recv: bytes
    generation: int = 0                    # generación de conexión del transporte en que se creó
    send_seq: int = 0
    window: ReplayWindow = field(default_factory=ReplayWindow)

    def next_send(self) -> tuple[int, bytes, bytes]:
        self.send_seq += 1
        return self.send_seq, message_key(self.k_send, self.epoch, self.send_seq), message_nonce(self.epoch, self.send_seq)

    def recv_key(self, seq: int) -> tuple[bytes, bytes]:
        return message_key(self.k_recv, self.epoch, seq), message_nonce(self.epoch, seq)


def derive_session(me: str, peer: str, transcript: bytes, shared: bytes, nonces_sorted: bytes,
                   replay_window: int = 64) -> Session:
    th = hashlib.sha256(transcript).digest()
    k_master = hkdf_extract(nonces_sorted, shared)

    def k_dir(src: str, dst: str) -> bytes:
        return hkdf_expand(k_master, b"dir" + th + src.encode() + b"->" + dst.encode(), KEY_LEN)

    return Session(me, peer, int.from_bytes(th[:8], "big"), k_dir(me, peer), k_dir(peer, me),
                   window=ReplayWindow(replay_window))


def auth_payload(transcript: bytes, signer: str) -> bytes:
    return b"auth" + hashlib.sha256(transcript).digest() + signer.encode()


@dataclass
class _Hello:
    nonce: bytes
    esk: ec.EllipticCurvePrivateKey | None
    reply_to: bytes | None


@dataclass
class _Paired:
    mine: bytes
    theirs: bytes
    transcript: bytes
    session: Session


class Handshaker:
    """Estado de los *handshakes* de un nodo con todos sus pares. No hace E/S: devuelve los mensajes a enviar
    como `(msg_type, campos)`; el canal (`channel.SecureTransport`) los envía y recibe."""

    def __init__(self, identity: Identity, trust: TrustStore, experiment_id: str, replay_window: int = 64) -> None:
        self.identity, self.trust, self.experiment_id = identity, trust, experiment_id
        self.replay_window = replay_window
        self._pending: dict[str, _Hello] = {}
        self._paired: dict[str, _Paired] = {}
        self._seen: dict[str, deque[bytes]] = {}

    # -------------------------------------------------------------- HELLO propio
    def _hello(self, peer: str, reply_to: bytes | None) -> tuple[str, dict[str, Any]]:
        h = _Hello(os.urandom(NONCE_BYTES), generate_ephemeral(), reply_to)
        self._pending[peer] = h
        return HELLO, {"hs_kid": self.identity.kid, "hs_nonce": h.nonce.hex(),
                       "hs_epk": encode_point(h.esk.public_key()).hex(),
                       "hs_reply_to": reply_to.hex() if reply_to else None}

    def initiate(self, peer: str) -> tuple[str, dict[str, Any]] | None:
        """HELLO fresco hacia `peer`, o None si ya hay un *handshake* en curso con él."""
        self.trust.get(peer)
        if peer in self._pending or peer in self._paired:
            return None
        return self._hello(peer, None)

    def abort(self, peer: str) -> None:
        self._pending.pop(peer, None)
        self._paired.pop(peer, None)

    def in_progress(self, peer: str) -> bool:
        return peer in self._pending or peer in self._paired

    # -------------------------------------------------------------- recepción
    def on_hello(self, peer: str, fields: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        pk = self.trust.get(peer)
        if fields["hs_kid"] != pk.kid:
            raise CryptoError(f"HELLO de {peer} con un kid distinto del registrado")
        nonce, epk = bytes.fromhex(fields["hs_nonce"]), bytes.fromhex(fields["hs_epk"])
        reply_to = bytes.fromhex(fields["hs_reply_to"]) if fields.get("hs_reply_to") else None
        peer_point = decode_point(epk)
        seen = self._seen.setdefault(peer, deque(maxlen=128))
        if nonce in seen:
            return []                                           # duplicado (p. ej. reentrega tras reconexión)
        seen.append(nonce)

        out: list[tuple[str, dict[str, Any]]] = []
        mine = self._pending.get(peer)
        if reply_to is not None:
            if mine is None or mine.nonce != reply_to:
                return []                                       # respuesta a un HELLO que ya no está pendiente
        elif mine is None or mine.reply_to is not None:
            out.append(self._hello(peer, nonce))                # responder a un HELLO fresco
            mine = self._pending[peer]
        # (si no: inicio simultáneo, los dos HELLO frescos se emparejan)

        me = {"node_id": self.identity.node_id, "kid": self.identity.kid, "nonce": mine.nonce,
              "epk": encode_point(mine.esk.public_key())}
        them = {"node_id": peer, "kid": pk.kid, "nonce": nonce, "epk": epk}
        transcript = build_transcript(self.experiment_id, me, them)
        shared = ecdh(mine.esk, peer_point)
        nonces = b"".join(h["nonce"] for h in sorted((me, them), key=lambda h: h["node_id"]))
        session = derive_session(self.identity.node_id, peer, transcript, shared, nonces, self.replay_window)
        mine.esk = None                                         # la clave efímera ya no se necesita
        del self._pending[peer]
        self._paired[peer] = _Paired(mine.nonce, nonce, transcript, session)
        out.append((AUTH, {"hs_nonce": mine.nonce.hex(), "hs_peer_nonce": nonce.hex(),
                           "hs_sig": sign(self.identity.private_key,
                                          auth_payload(transcript, self.identity.node_id)).hex()}))
        return out

    def on_auth(self, peer: str, fields: Mapping[str, Any]) -> Session | None:
        """Sesión establecida si la firma del par sobre el transcript es válida; None si el AUTH es de un
        *handshake* que ya no está en curso. Lanza CryptoError si la firma no es válida."""
        p = self._paired.get(peer)
        if p is None or bytes.fromhex(fields["hs_nonce"]) != p.theirs or bytes.fromhex(fields["hs_peer_nonce"]) != p.mine:
            return None
        verify(self.trust.get(peer).public_key, bytes.fromhex(fields["hs_sig"]), auth_payload(p.transcript, peer))
        del self._paired[peer]
        return p.session
