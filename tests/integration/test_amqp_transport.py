"""E3: transporte AMQP contra un RabbitMQ real (docker/compose.yaml). Criterios de aceptación 1–6 de E3.

Cada prueba crea su propio vhost y sus usuarios (fixture `experiment`) y los elimina al terminar.
"""
from __future__ import annotations

import asyncio

import aio_pika
import pytest
import torch

from core.model import MLP
from messaging.contracts import ContractValidator
from messaging.frame import Limits, decode_message, encode_message, tensor_specs
from messaging.transport_amqp import AmqpTransport, TransportError

pytestmark = pytest.mark.broker
CNS = ("CN1", "CN2", "CN3", "CN4", "CN5")


def transport(creds, node, bcfg, max_frame_bytes=4 * 1024 * 1024):
    c = creds[node]
    return AmqpTransport(c.amqp_url(), node, c.username, bcfg, max_frame_bytes)


def header(exp_id, msg_type, sender, recipient, seq=0, **extra):
    h = {"schema_version": 1, "msg_type": msg_type, "experiment_id": exp_id, "topology": "centralized",
         "sender_id": sender, "recipient_id": recipient, "seq": seq}
    h.update(extra)
    return h


@pytest.fixture(scope="module")
def state():
    return {k: v.detach().clone() for k, v in MLP(278).state_dict().items()}


# ---------------------------------------------------------------------------
# (1) 6 participantes intercambian tramas reales por q.<id>
# ---------------------------------------------------------------------------
async def test_six_nodes_exchange_real_frames(experiment, state):
    exp_id, creds, bcfg = experiment()
    validator, specs = ContractValidator(), tensor_specs(state)
    nodes = {n: transport(creds, n, bcfg) for n in (*CNS, "infra")}
    await asyncio.gather(*(t.connect() for t in nodes.values()))
    try:
        # Subida: CN_i → infra (UPDATE con el modelo de referencia, 453 KB)
        for i, n in enumerate(CNS, start=1):
            frame = encode_message(header(exp_id, "UPDATE", n, "infra", i, round=1, n_samples=100 * i, tau=5,
                                          metrics={"train_loss": 0.5, "train_time_s": 1.0}), state)
            stats = await nodes[n].send("infra", frame, "UPDATE")
            assert stats.nbytes == len(frame)
        received = {}
        for _ in CNS:
            d = await nodes["infra"].recv(timeout=10)
            msg = decode_message(d.body, validator, expected_tensors=specs)
            # Identidad del emisor validada por el broker = la que declara la cabecera
            assert d.sender_user == f"{exp_id}.{msg.header['sender_id']}"
            received[msg.header["sender_id"]] = msg
            await d.ack()
        assert set(received) == set(CNS)
        assert all(torch.equal(received["CN3"].state[k], state[k]) for k in state)

        # Bajada: infra → CN_i (ROUND_START) y difusión de STOP a todos
        for n in CNS:
            frame = encode_message(header(exp_id, "ROUND_START", "infra", n, round=2,
                                          training={"method": "fedavg", "local_epochs": 3, "mu": 0.0}), state)
            await nodes["infra"].send(n, frame, "ROUND_START")
        await nodes["infra"].broadcast(encode_message(header(exp_id, "STOP", "infra", "*", round=2,
                                                             reason="fixed_rounds_12")), "STOP")
        for n in CNS:
            types = []
            for _ in range(2):
                d = await nodes[n].recv(timeout=10)
                types.append(decode_message(d.body, validator).header["msg_type"])
                await d.ack()
            assert sorted(types) == ["ROUND_START", "STOP"]
        d = await nodes["infra"].recv(timeout=10)                    # infra también recibe la difusión
        assert d.msg_type == "STOP"
        await d.ack()
    finally:
        await asyncio.gather(*(t.close() for t in nodes.values()))


# ---------------------------------------------------------------------------
# (2) Permisos mínimos y suplantación de identidad
# ---------------------------------------------------------------------------
async def _raw_channel(creds, node):
    conn = await aio_pika.connect(creds[node].amqp_url())
    return conn, await conn.channel(publisher_confirms=True)


async def test_node_cannot_read_other_queue_or_configure(experiment):
    exp_id, creds, bcfg = experiment(("CN1", "CN2"))
    conn, ch = await _raw_channel(creds, "CN1")
    try:
        async def ignore(message):
            pass

        with pytest.raises(Exception, match="ACCESS_REFUSED"):
            q = await ch.get_queue("q.CN2", ensure=False)
            await q.consume(ignore)
        ch = await conn.channel()
        with pytest.raises(Exception, match="ACCESS_REFUSED"):
            await ch.declare_queue("q.evil")
        ch = await conn.channel()
        with pytest.raises(Exception, match="ACCESS_REFUSED"):
            await ch.declare_exchange("fl.evil", aio_pika.ExchangeType.FANOUT)
    finally:
        await conn.close()


async def test_user_id_spoofing_is_rejected_by_broker(experiment):
    exp_id, creds, bcfg = experiment(("CN1", "CN2", "infra"))
    conn, ch = await _raw_channel(creds, "CN1")
    try:
        ex = await ch.get_exchange(bcfg["unicast_exchange"], ensure=False)
        with pytest.raises(Exception, match="PRECONDITION_FAILED|user_id"):
            await ex.publish(aio_pika.Message(b"x", user_id=creds["CN2"].username), routing_key="infra", timeout=10)
    finally:
        await conn.close()


async def test_message_without_user_id_has_no_validated_sender(experiment):
    exp_id, creds, bcfg = experiment(("CN1", "infra"))
    conn, ch = await _raw_channel(creds, "CN1")
    infra = transport(creds, "infra", bcfg)
    await infra.connect()
    try:
        ex = await ch.get_exchange(bcfg["unicast_exchange"], ensure=False)
        await ex.publish(aio_pika.Message(b"x"), routing_key="infra", timeout=10)
        d = await infra.recv(timeout=10)
        assert d.sender_user is None             # el receptor debe rechazar mensajes sin identidad validada
        await d.reject()
    finally:
        await infra.close()
        await conn.close()


# ---------------------------------------------------------------------------
# (3) Cola llena, destino inexistente y tamaño máximo ⇒ el emisor lo detecta
# ---------------------------------------------------------------------------
async def test_full_queue_rejects_publish(experiment):
    exp_id, creds, bcfg = experiment(("CN1", "infra"), max_length=3)
    async with transport(creds, "CN1", bcfg) as cn1:              # infra no consume
        for i in range(3):
            await cn1.send("infra", b"m%d" % i, "HEARTBEAT")
        with pytest.raises(TransportError, match="rechazada"):
            await cn1.send("infra", b"m3", "HEARTBEAT")


async def test_unknown_recipient_raises(experiment):
    exp_id, creds, bcfg = experiment(("CN1", "infra"))
    async with transport(creds, "CN1", bcfg) as cn1:
        with pytest.raises(TransportError):
            await cn1.send("CN9", b"x", "HEARTBEAT")


async def test_oversize_frames_rejected_by_client_and_broker(experiment):
    exp_id, creds, bcfg = experiment(("CN1", "infra"))
    async with transport(creds, "CN1", bcfg, max_frame_bytes=1024) as small:
        with pytest.raises(TransportError, match="Trama"):
            await small.send("infra", b"x" * 2048, "UPDATE")
    # Con el límite del cliente desactivado, el broker (max_message_size = 4 MiB) rechaza igualmente
    async with transport(creds, "CN1", bcfg, max_frame_bytes=64 * 1024 * 1024) as big:
        with pytest.raises(Exception):
            await big.send("infra", b"x" * (5 * 1024 * 1024), "UPDATE")


# ---------------------------------------------------------------------------
# (4) Reconexión: los mensajes pendientes esperan en la cola nombrada (hasta el TTL)
# ---------------------------------------------------------------------------
async def test_pending_messages_survive_disconnection(experiment):
    exp_id, creds, bcfg = experiment(("CN1", "CN2"))
    async with transport(creds, "CN2", bcfg) as cn2:
        pass                                                      # CN2 se conecta y se desconecta
    async with transport(creds, "CN1", bcfg) as cn1:
        await cn1.send("CN2", b"pendiente", "HEARTBEAT")
    async with transport(creds, "CN2", bcfg) as cn2:              # reconexión
        d = await cn2.recv(timeout=10)
        assert d.body == b"pendiente"
        await d.ack()


async def test_expired_messages_are_dropped_by_ttl(experiment):
    exp_id, creds, bcfg = experiment(("CN1", "CN2"), message_ttl_ms=500)
    async with transport(creds, "CN1", bcfg) as cn1:
        await cn1.send("CN2", b"caduca", "HEARTBEAT")
    await asyncio.sleep(1.2)
    async with transport(creds, "CN2", bcfg) as cn2:
        with pytest.raises(asyncio.TimeoutError):
            await cn2.recv(timeout=1.5)


async def test_rejected_message_is_not_requeued(experiment):
    exp_id, creds, bcfg = experiment(("CN1", "infra"))
    async with transport(creds, "CN1", bcfg) as cn1, transport(creds, "infra", bcfg) as infra:
        await cn1.send("infra", b"basura", "UPDATE")
        d = await infra.recv(timeout=10)
        await d.reject()
        with pytest.raises(asyncio.TimeoutError):
            await infra.recv(timeout=1.0)


# ---------------------------------------------------------------------------
# (6) Memoria del broker por debajo de la marca de agua tras el intercambio
# ---------------------------------------------------------------------------
async def test_broker_memory_below_watermark(broker_admin, experiment, state):
    exp_id, creds, bcfg = experiment(("CN1", "infra"))
    async with transport(creds, "CN1", bcfg) as cn1:
        frame = encode_message(header(exp_id, "UPDATE", "CN1", "infra", round=1, n_samples=10, tau=1,
                                      metrics={"train_loss": 0.1, "train_time_s": 0.1}), state)
        for _ in range(20):                                       # 20 × 453 KB encolados (infra no consume)
            await cn1.send("infra", frame, "UPDATE")
    # Las estadísticas de la API de gestión son muestras periódicas (≈ 5 s): se espera a que se actualicen
    for _ in range(30):
        stats = broker_admin.queue_stats(exp_id)
        if stats["q.infra"]["messages"] == 20:
            break
        await asyncio.sleep(0.5)
    mem = broker_admin.node_memory()
    assert stats["q.infra"]["messages"] == 20
    assert stats["q.infra"]["message_bytes"] == 20 * len(frame)
    assert mem["mem_alarm"] is False and mem["mem_used"] < mem["mem_limit"]
