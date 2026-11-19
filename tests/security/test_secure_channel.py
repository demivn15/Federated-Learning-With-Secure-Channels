"""E10: canal seguro S2 (`messaging/security/channel.py`) sobre un broker simulado en memoria.

Comprueba el protocolo completo sin RabbitMQ: handshake (iniciado por un extremo y simultáneo), ida y vuelta
cifrada, difusión, y que se RECHAZAN replay, manipulación, mensajes en claro (downgrade), suplantación de la
identidad del broker y un MITM que sustituye la clave efímera del HELLO. También la nueva época al reconectar
(los mensajes de la época anterior siguen siendo descifrables) y la telemetría firmada.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict

import pytest

from messaging.contracts import ContractValidator
from messaging.frame import Limits, decode_frame, encode_frame
from messaging.security.channel import SecureTransport
from messaging.security.identity import TRUST_FILE, Identity, TrustStore
from messaging.security.primitives import CryptoError, encode_point, generate_ephemeral
from messaging.security.telemetry_sig import TelemetrySigner
from messaging.transport_amqp import CONTENT_TYPE, Delivery, PublishStats, TransportError
from scripts.provision_identities import provision_node_identities

EXP = "exp-s2"
NODES = ["CN1", "CN2", "infra"]


class FakeMsg:
    def __init__(self) -> None:
        self.acked = self.rejected = False

    async def ack(self) -> None:
        self.acked = True

    async def reject(self, requeue: bool = False) -> None:
        self.rejected = True


class FakeBroker:
    """Enruta por destinatario; `user_id` = usuario autenticado del emisor (como RabbitMQ). `hook` permite a
    un atacante con acceso al broker modificar o descartar mensajes (ataque S6 de D3)."""

    def __init__(self) -> None:
        self.queues: dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        self.tap: list[tuple[str, tuple]] = []
        self.hook = None

    def deliver(self, recipient: str, body: bytes, msg_type: str, sender_user: str, content_type: str) -> None:
        item = (body, sender_user, msg_type, content_type)
        if self.hook is not None:
            item = self.hook(recipient, item)
            if item is None:
                return
        self.tap.append((recipient, item))
        self.queues[recipient].put_nowait(item)


class FakeTransport:
    def __init__(self, broker: FakeBroker, node_id: str) -> None:
        self.broker, self.node_id, self.username = broker, node_id, f"{EXP}.{node_id}"
        self.generation = 1

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def send(self, recipient, frame, msg_type, message_id=None, content_type=CONTENT_TYPE):
        self.broker.deliver(recipient, frame, msg_type, self.username, content_type)
        return PublishStats(len(frame), 0.0)

    async def broadcast(self, frame, msg_type, message_id=None, content_type=CONTENT_TYPE):
        for n in NODES:
            if n != self.node_id:
                self.broker.deliver(n, frame, msg_type, self.username, content_type)
        return PublishStats(len(frame), 0.0)

    async def recv(self, timeout=None):
        body, user, msg_type, ctype = await asyncio.wait_for(self.broker.queues[self.node_id].get(), timeout)
        return Delivery(body, user, msg_type, None, time.perf_counter_ns(), FakeMsg(), ctype)


@pytest.fixture
def identities(tmp_path):
    provision_node_identities(tmp_path, NODES)
    trust = TrustStore.load(tmp_path / TRUST_FILE)
    return {n: Identity.load(n, tmp_path / f"{n}.key", trust) for n in NODES}, trust


@pytest.fixture
def net(identities):
    ids, trust = identities
    broker = FakeBroker()
    validator = ContractValidator()

    def make(node, timeout=5.0):
        return SecureTransport(FakeTransport(broker, node), ids[node], trust, experiment_id=EXP,
                               topology="centralized", validator=validator, limits=Limits(),
                               handshake_timeout_s=timeout)
    return broker, make


def frame(sender, recipient, msg_type="UPDATE", **extra):
    return encode_frame({"schema_version": 1, "msg_type": msg_type, "experiment_id": EXP, "topology": "centralized",
                         "sender_id": sender, "recipient_id": recipient, "seq": 1, **extra}, b"")


async def started(*transports):
    for t in transports:
        await t.connect()
    return transports


async def stop(*transports):
    for t in transports:
        await t.close()


def app_items(broker, recipient):
    return [it for r, it in broker.tap if r == recipient and it[2] not in ("HS_HELLO", "HS_AUTH")]


# ---------------------------------------------------------------------------- funcionamiento
async def test_handshake_and_encrypted_roundtrip(net):
    broker, make = net
    a, b = await started(make("CN1"), make("infra"))
    f = frame("CN1", "infra", note="hola")
    await a.send("infra", f, "UPDATE")
    d = await b.recv(5)
    assert d.body == f and d.content_type == CONTENT_TYPE
    wire = app_items(broker, "infra")[0][0]
    assert b"hola" not in wire                                    # el broker no ve el contenido
    await b.send("CN1", frame("infra", "CN1", "ROUND_START"), "ROUND_START")
    assert decode_frame((await a.recv(5)).body)[0]["msg_type"] == "ROUND_START"
    sa, sb = a.take_stats(), b.take_stats()
    assert sa["handshakes"] == 1 and sb["handshakes"] == 1 and sa["sec_rejected"] == sb["sec_rejected"] == 0
    assert a._send["infra"].epoch == b._send["CN1"].epoch
    await stop(a, b)


async def test_simultaneous_initiation_converges(net):
    _, make = net
    a, b = await started(make("CN1"), make("infra"))
    await asyncio.gather(a.send("infra", frame("CN1", "infra"), "UPDATE"),
                         b.send("CN1", frame("infra", "CN1", "ROUND_START"), "ROUND_START"))
    await a.recv(5), await b.recv(5)
    assert a._send["infra"].epoch == b._send["CN1"].epoch
    await stop(a, b)


async def test_broadcast_reaches_peers_with_session(net):
    _, make = net
    s, c1, c2 = await started(make("infra"), make("CN1"), make("CN2"))
    for c in (c1, c2):
        await c.send("infra", frame(c.node_id, "infra", "REGISTER"), "REGISTER")
        await s.recv(5)
    await s.broadcast(frame("infra", "*", "STOP"), "STOP")
    for c in (c1, c2):
        assert decode_frame((await c.recv(5)).body)[0]["msg_type"] == "STOP"
    await stop(s, c1, c2)


async def test_losing_messages_does_not_desynchronize(net):
    broker, make = net
    a, b = await started(make("CN1"), make("infra"))
    await a.send("infra", frame("CN1", "infra", n=0), "UPDATE")
    await b.recv(5)
    drop = {2, 3, 5, 8, 9}
    counter = {"i": 0}

    def lossy(recipient, item):
        if item[2] == "UPDATE":
            counter["i"] += 1
            if counter["i"] in drop:
                return None
        return item
    broker.hook = lossy
    for i in range(1, 11):
        await a.send("infra", frame("CN1", "infra", n=i), "UPDATE")
    got = [decode_frame((await b.recv(5)).body)[0]["n"] for _ in range(10 - len(drop))]
    assert got == [i for i in range(1, 11) if i not in drop]
    await stop(a, b)


async def test_new_epoch_after_reconnection_keeps_old_messages_readable(net):
    _, make = net
    a, b = await started(make("CN1"), make("infra"))
    await a.send("infra", frame("CN1", "infra", n=1), "UPDATE")
    old_epoch = a._send["infra"].epoch
    a.inner.generation += 1                                       # reconexión al broker
    await a.send("infra", frame("CN1", "infra", n=2), "UPDATE")
    assert a._send["infra"].epoch != old_epoch
    assert [decode_frame((await b.recv(5)).body)[0]["n"] for _ in range(2)] == [1, 2]
    await stop(a, b)


# ---------------------------------------------------------------------------- ataques
async def _expect_rejected(t: SecureTransport, timeout=0.5):
    with pytest.raises(asyncio.TimeoutError):
        await t.recv(timeout)
    return t.take_stats()["sec_rejected"]


async def test_replay_of_captured_message_rejected(net):
    broker, make = net
    a, b = await started(make("CN1"), make("infra"))
    await a.send("infra", frame("CN1", "infra"), "UPDATE")
    await b.recv(5)
    b.take_stats()
    captured = app_items(broker, "infra")[0]
    for _ in range(3):
        broker.queues["infra"].put_nowait(captured)               # P1: replay ×3
    assert await _expect_rejected(b) == 3
    await stop(a, b)


async def test_tampered_ciphertext_and_header_rejected(net):
    broker, make = net
    a, b = await started(make("CN1"), make("infra"))
    await a.send("infra", frame("CN1", "infra"), "UPDATE")
    await b.recv(5)
    body, user, mt, ct = app_items(broker, "infra")[0]
    b.take_stats()
    for pos in (10, len(body) - 5):
        bad = bytearray(body)
        bad[pos] ^= 1
        broker.queues["infra"].put_nowait((bytes(bad), user, mt, ct))
    assert await _expect_rejected(b) == 2
    await stop(a, b)


async def test_plaintext_downgrade_rejected(net):
    broker, make = net
    b, = await started(make("infra"))
    broker.queues["infra"].put_nowait((frame("CN1", "infra"), f"{EXP}.CN1", "UPDATE", CONTENT_TYPE))
    assert await _expect_rejected(b) == 1
    await stop(b)


async def test_broker_identity_mismatch_rejected(net):
    broker, make = net
    a, b, c2 = await started(make("CN1"), make("infra"), make("CN2"))
    await a.send("infra", frame("CN1", "infra"), "UPDATE")
    await b.recv(5)
    body, _, mt, ct = app_items(broker, "infra")[0]
    b.take_stats()
    broker.queues["infra"].put_nowait((body, f"{EXP}.CN2", mt, ct))    # CN2 reenvía un mensaje de CN1
    assert await _expect_rejected(b) == 1
    await stop(a, b, c2)


async def test_mitm_replacing_ephemeral_key_fails(net):
    """P3: un atacante en el broker cambia ePK en el HELLO. Las firmas del transcript no verifican: no hay
    sesión y el envío falla por timeout del handshake (sin aceptar nunca una clave del atacante)."""
    broker, make = net
    attacker_epk = encode_point(generate_ephemeral().public_key()).hex()

    def mitm(recipient, item):
        body, user, mt, ct = item
        if mt == "HS_HELLO":
            h, _ = decode_frame(body)
            h["hs_epk"] = attacker_epk
            body = encode_frame(h, b"")
        return body, user, mt, ct
    broker.hook = mitm
    a, b = await started(make("CN1", timeout=1.5), make("infra", timeout=1.5))
    with pytest.raises(TransportError):
        await a.send("infra", frame("CN1", "infra"), "UPDATE")
    await asyncio.sleep(0.2)
    assert not a._send and not b._send
    assert a.take_stats()["sec_rejected"] + b.take_stats()["sec_rejected"] >= 1   # AUTH con firma inválida
    await stop(a, b)


async def test_handshake_from_unregistered_or_wrong_kid_rejected(net, identities):
    broker, make = net
    b, = await started(make("infra"))
    h = {"schema_version": 1, "msg_type": "HS_HELLO", "experiment_id": EXP, "topology": "centralized",
         "sender_id": "CN1", "recipient_id": "infra", "seq": 1, "hs_kid": "00" * 16, "hs_nonce": "11" * 32,
         "hs_epk": encode_point(generate_ephemeral().public_key()).hex(), "hs_reply_to": None}
    broker.queues["infra"].put_nowait((encode_frame(h, b""), f"{EXP}.CN1", "HS_HELLO", CONTENT_TYPE))
    await asyncio.sleep(0.2)
    assert b.take_stats()["sec_rejected"] == 1 and not b._send
    await stop(b)


# ---------------------------------------------------------------------------- telemetría firmada
def test_signed_telemetry(identities):
    ids, trust = identities
    signer, coordinator = TelemetrySigner(ids["CN1"], trust), TelemetrySigner(ids["infra"], trust)
    h = {"experiment_id": EXP, "sender_id": "CN1", "round": 3, "attempt": 1,
         "telemetry": {"cpu_percent": 10.0, "ram_available_mb": 900.0, "temperature_c": 51.0}}
    signer.sign_header(h)
    coordinator.verify_header(json.loads(json.dumps(h)))
    forged = {**h, "telemetry": {**h["telemetry"], "temperature_c": 30.0}}       # F5: SD-score falsificado
    with pytest.raises(CryptoError):
        coordinator.verify_header(forged)
    with pytest.raises(CryptoError):
        coordinator.verify_header({k: v for k, v in h.items() if k != "telemetry_sig"})
    with pytest.raises(CryptoError):
        coordinator.verify_header({**h, "sender_id": "CN2"})                  # atribuida a otro nodo
