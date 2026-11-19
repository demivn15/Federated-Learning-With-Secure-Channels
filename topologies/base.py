"""Contexto común de un nodo: envío y recepción validada de mensajes, contadores y registro de KPIs.

Solo contiene lo que comparten todas las topologías; la lógica de ronda vive en cada orquestador.

Ruta de recepción (`NodeContext.receive`), además de la de `messaging.frame.decode_message`:
  * identidad: el `user_id` validado por el broker debe ser `<experiment_id>.<sender_id>` de la cabecera;
  * la cabecera debe ser de este experimento y esta topología, y dirigida a este nodo (o a `*`);
  * el tipo de mensaje debe ser uno de los esperados en ese momento.
Lo que no cumple se descarta (`reject`, sin reencolar) y se cuenta en `messages_rejected`.

Perfiles de seguridad (E10): `transport` es `AmqpTransport` (S0, S1) o `messaging.security.channel.
SecureTransport` (S2), con la misma interfaz; los orquestadores no cambian. En S2, `security` firma y
verifica la telemetría autodeclarada (`CLAIM` de SD) y los costes del canal seguro (`take_stats`) se suman
a la fila de KPIs de la ronda.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from messaging.contracts import ContractError, ContractValidator
from messaging.security.primitives import CryptoError
from messaging.frame import FrameError, Limits, PreparedPayload, decode_message, encode_message
from messaging.serialization import UnsafePayloadError
from messaging.transport_amqp import AmqpTransport, BrokerLostError, Delivery, PublishStats, TransportError
from telemetry.kpi_log import KpiLogger

log = logging.getLogger("fl.node")


class ProtocolError(RuntimeError):
    pass


@dataclass
class Received:
    header: dict[str, Any]
    state: dict[str, torch.Tensor] | None
    nbytes: int
    t_deserialize_s: float
    delivery: Delivery

    @property
    def sender(self) -> str:
        return self.header["sender_id"]

    @property
    def arrived_ns(self) -> int:
        """Hora de llegada al nodo (reloj local), no la de desencolado."""
        return self.delivery.received_ns

    async def accept(self) -> None:
        await self.delivery.ack()

    async def discard(self) -> None:
        await self.delivery.reject()


@dataclass
class Sent:
    nbytes: int
    t_serialize_s: float
    publish: PublishStats


@dataclass
class NodeContext:
    cfg: dict[str, Any]
    manifest: dict[str, Any]
    run_dir: Path
    node_id: str
    transport: AmqpTransport
    validator: ContractValidator
    limits: Limits
    kpi: KpiLogger
    security: Any = None                 # TelemetrySigner en S2
    seq: int = 0
    counters: dict[str, int] = field(default_factory=lambda: {"messages_tx": 0, "messages_rx": 0,
                                                              "messages_rejected": 0, "messages_failed": 0,
                                                              "payload_bytes_tx": 0, "payload_bytes_rx": 0})

    @property
    def experiment_id(self) -> str:
        return self.manifest["experiment_id"]

    @property
    def topology(self) -> str:
        return self.manifest["topology"]

    def take_counters(self) -> dict[str, int]:
        """Contadores acumulados desde la última llamada (una fila de KPIs por ronda)."""
        out, self.counters = self.counters, {k: 0 for k in self.counters}
        take_stats = getattr(self.transport, "take_stats", None)
        if take_stats is not None:                         # S2: cifrado, descifrado, handshakes, rechazos
            out.update(take_stats())
        return out

    # -------------------------------------------------------------- envío
    def header(self, msg_type: str, recipient: str, **fields: Any) -> dict[str, Any]:
        self.seq += 1
        return {"schema_version": 1, "msg_type": msg_type, "experiment_id": self.experiment_id,
                "topology": self.topology, "sender_id": self.node_id, "recipient_id": recipient,
                "seq": self.seq, **fields}

    async def send(self, recipient: str, msg_type: str,
                   state: Mapping[str, torch.Tensor] | PreparedPayload | None = None, **fields: Any) -> Sent:
        t0 = time.perf_counter()
        header = self.header(msg_type, recipient, **fields)
        if self.security is not None:
            self.security.sign_header(header)
        frame = encode_message(header, state, self.limits)
        t_ser = time.perf_counter() - t0
        stats = await self.transport.send(recipient, frame, msg_type)
        self.counters["messages_tx"] += 1
        self.counters["payload_bytes_tx"] += len(frame)
        return Sent(len(frame), t_ser, stats)

    @staticmethod
    def unsent() -> Sent:
        """`Sent` vacío para un envío fallido (`send_or_skip` → None): las columnas de la ronda siguen siendo válidas."""
        return Sent(0, 0.0, PublishStats(0, 0.0))

    async def send_or_skip(self, recipient: str, msg_type: str,
                           state: Mapping[str, torch.Tensor] | PreparedPayload | None = None, **fields: Any) -> Sent | None:
        """Como `send`, pero un destinatario inalcanzable (cola llena de un nodo caído o en pausa, *handshake* sin
        respuesta) no tumba al emisor: se registra en `messages_failed` y devuelve None; el orquestador lo trata
        como ausente en la ronda (ADR-12). La pérdida del broker sí se propaga (fin del experimento)."""
        try:
            return await self.send(recipient, msg_type, state, **fields)
        except BrokerLostError:
            raise
        except TransportError as exc:
            self.counters["messages_failed"] += 1
            log.warning("[%s] %s → %s no entregado: %s", self.node_id, msg_type, recipient, exc)
            return None

    async def broadcast(self, msg_type: str, **fields: Any) -> Sent:
        t0 = time.perf_counter()
        frame = encode_message(self.header(msg_type, "*", **fields), None, self.limits)
        t_ser = time.perf_counter() - t0
        try:
            stats = await self.transport.broadcast(frame, msg_type)
        except BrokerLostError:
            raise
        except TransportError as exc:
            # fanout + `reject-publish`: si la cola de UN nodo (caído o en pausa) está llena, el broker entrega
            # al resto y devuelve nack al emisor (E9). La difusión es parcial, no un fallo del emisor.
            self.counters["messages_failed"] += 1
            log.warning("[%s] difusión de %s parcial: %s", self.node_id, msg_type, exc)
            stats = PublishStats(len(frame), 0.0)
        self.counters["messages_tx"] += 1
        self.counters["payload_bytes_tx"] += len(frame)
        return Sent(len(frame), t_ser, stats)

    # -------------------------------------------------------------- recepción
    async def receive(self, timeout_s: float, expected: Iterable[str],
                      expected_tensors: list[dict[str, Any]] | None = None) -> Received | None:
        """Siguiente mensaje válido de un tipo esperado, o None si vence el plazo."""
        expected = set(expected)
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                d = await self.transport.recv(timeout=remaining)
            except asyncio.TimeoutError:          # en Python 3.10 ≠ TimeoutError (se unificaron en 3.11)
                return None
            t0 = time.perf_counter()
            try:
                msg = decode_message(d.body, self.validator, self.limits, expected_tensors)
                h = msg.header
                problem = None
                if d.sender_user != f"{self.experiment_id}.{h['sender_id']}":
                    problem = f"identidad del emisor no validada ({d.sender_user!r} vs {h['sender_id']!r})"
                elif h["experiment_id"] != self.experiment_id or h["topology"] != self.topology:
                    problem = "experimento o topología ajenos"
                elif h["recipient_id"] not in (self.node_id, "*"):
                    problem = f"destinatario {h['recipient_id']}"
                elif h["msg_type"] not in expected:
                    problem = f"tipo inesperado {h['msg_type']} (se esperaba {sorted(expected)})"
                elif self.security is not None:
                    self.security.verify_header(h)
            except (FrameError, ContractError, UnsafePayloadError, CryptoError) as exc:
                problem, msg = f"{type(exc).__name__}: {exc}", None
            if problem is not None:
                self.counters["messages_rejected"] += 1
                log.warning("[%s] mensaje descartado: %s", self.node_id, problem)
                await d.reject()
                continue
            self.counters["messages_rx"] += 1
            self.counters["payload_bytes_rx"] += msg.nbytes
            return Received(msg.header, msg.state, msg.nbytes, time.perf_counter() - t0, d)

    async def receive_from(self, timeout_s: float, allowed: Mapping[str, Iterable[str]],
                           expected_tensors: list[dict[str, Any]] | None = None) -> Received | None:
        """Como `receive`, pero además exige que el emisor tenga el ROL que corresponde al tipo de mensaje.

        `allowed` = {msg_type: emisores permitidos}. La identidad la valida el broker (`user_id`); esto
        valida la autorización: p. ej. un `ROUND_START` solo puede venir del servidor/raíz/agregador del
        nodo, aunque cualquier nodo tenga permiso de publicar en `fl.unicast`.
        """
        allowed = {t: set(s) for t, s in allowed.items()}
        deadline = time.monotonic() + timeout_s
        while True:
            rec = await self.receive(deadline - time.monotonic(), allowed, expected_tensors)
            if rec is None or rec.sender in allowed[rec.header["msg_type"]]:
                return rec
            self.counters["messages_rejected"] += 1
            log.warning("[%s] %s de %s descartado: emisor no autorizado", self.node_id,
                        rec.header["msg_type"], rec.sender)
            await rec.discard()
