"""Transporte seguro del perfil S2: misma interfaz que `AmqpTransport` (`send`, `broadcast`, `recv`, `connect`,
`close`), de modo que `NodeContext` y los orquestadores no cambian entre S0, S1 y S2 (criterio 5 de E10).

  * Una tarea de fondo (*pump*) consume la cola del nodo: atiende los `HS_HELLO`/`HS_AUTH` en cuanto llegan
    (aunque el orquestador esté entrenando o esperando otro tipo de mensaje) y deja el resto en una cola de
    aplicación. Sin ella, un `send` que espera un *handshake* se bloquearía: nadie leería la respuesta.
  * `send` obtiene la sesión con el destinatario (*handshake* si no hay una de la generación de conexión
    actual), cifra la trama S0 ya construida y la publica con `content_type = application/vnd.fl-e2e`.
  * `recv` descifra en este orden: estructura de la cabecera externa → identidad validada por el broker =
    emisor → sesión (emisor, época) → ventana anti-replay → AEAD → coherencia cabecera interna/externa →
    marcar la ventana. Un mensaje de aplicación en claro se rechaza (sin *downgrade* a S0/S1).
  * Cada reconexión al broker (nueva generación) ⇒ nuevo *handshake* y nueva época para enviar. Las
    sesiones anteriores se conservan para RECIBIR (los mensajes que esperaban en la cola durante el corte
    siguen siendo descifrables).
  * `broadcast` (solo control, p. ej. `STOP`) se cifra por destinatario a cada par con sesión vigente.
Las rutas de rechazo nunca reencolan el mensaje. Los costes (cifrado, descifrado, *handshake*) se acumulan en
`take_stats()` y `NodeContext` los añade a la fila de KPIs de la ronda (RNF-05).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Mapping

from messaging.contracts import ContractError, ContractValidator
from messaging.frame import FrameError, Limits, decode_frame, encode_frame
from messaging.security import envelope
from messaging.security.handshake import AUTH, HANDSHAKE_TYPES, HELLO, Handshaker, Session
from messaging.security.identity import Identity, TrustStore
from messaging.security.primitives import CryptoError
from messaging.transport_amqp import CONTENT_TYPE, AmqpTransport, BrokerLostError, Delivery, PublishStats, TransportError

log = logging.getLogger("fl.security")
MAX_EPOCHS_PER_PEER = 4
_STATS0 = {"t_encrypt_s": 0.0, "t_decrypt_s": 0.0, "t_handshake_s": 0.0, "handshakes": 0, "sec_rejected": 0}


class SecureTransport:
    def __init__(self, inner: AmqpTransport, identity: Identity, trust: TrustStore, *, experiment_id: str,
                 topology: str, validator: ContractValidator, limits: Limits, handshake_timeout_s: float = 60.0,
                 replay_window: int = 64) -> None:
        self.inner = inner
        self.identity, self.trust = identity, trust
        self.experiment_id, self.topology = experiment_id, topology
        self.validator, self.limits = validator, limits
        self.handshake_timeout_s = float(handshake_timeout_s)
        self.hs = Handshaker(identity, trust, experiment_id, replay_window)
        self._send: dict[str, Session] = {}
        self._recv: dict[str, dict[int, Session]] = {}
        self._waiters: dict[str, asyncio.Future[Session]] = {}
        self._app: asyncio.Queue[Delivery | BaseException] = asyncio.Queue()
        self._pump_task: asyncio.Task | None = None
        self._hs_seq = 0
        self.stats: dict[str, float] = dict(_STATS0)

    # -------------------------------------------------------------- interfaz de transporte
    @property
    def node_id(self) -> str:
        return self.inner.node_id

    @property
    def username(self) -> str:
        return self.inner.username

    async def connect(self) -> None:
        await self.inner.connect()
        if self._pump_task is None or self._pump_task.done():
            self._pump_task = asyncio.create_task(self._pump(), name=f"s2-pump-{self.node_id}")

    async def close(self) -> None:
        if self._pump_task is not None:
            self._pump_task.cancel()
            try:
                await self._pump_task
            except (asyncio.CancelledError, Exception):
                pass
            self._pump_task = None
        await self.inner.close()

    async def __aenter__(self) -> "SecureTransport":
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    def take_stats(self) -> dict[str, float]:
        out, self.stats = self.stats, dict(_STATS0)
        return out

    # -------------------------------------------------------------- envío
    async def send(self, recipient: str, frame: bytes, msg_type: str, message_id: str | None = None) -> PublishStats:
        session = await self._session_for(recipient)
        t0 = time.perf_counter()
        sealed = envelope.seal(session, frame, msg_type, self.experiment_id)
        self.stats["t_encrypt_s"] += time.perf_counter() - t0
        stats = await self.inner.send(recipient, sealed, msg_type, message_id, content_type=envelope.E2E_CONTENT_TYPE)
        return PublishStats(len(sealed), stats.confirm_s)

    async def broadcast(self, frame: bytes, msg_type: str, message_id: str | None = None) -> PublishStats:
        total, confirm = 0, 0.0
        for peer in sorted(self._send):
            try:
                st = await self.send(peer, frame, msg_type, message_id)
            except TransportError as exc:                     # un par caído no impide avisar a los demás
                log.warning("[%s] difusión de %s a %s fallida: %s", self.node_id, msg_type, peer, exc)
                continue
            total, confirm = total + st.nbytes, confirm + st.confirm_s
        return PublishStats(total, confirm)

    async def _session_for(self, peer: str) -> Session:
        s = self._send.get(peer)
        if s is not None and s.generation == self.inner.generation:
            return s
        fut = self._waiters.get(peer)
        if fut is None or fut.done():
            fut = asyncio.get_running_loop().create_future()
            self._waiters[peer] = fut
            if self.hs.in_progress(peer) and s is not None:
                self.hs.abort(peer)                       # handshake de una generación anterior
            first = self.hs.initiate(peer)
            if first is not None:
                await self._send_hs(peer, *first)
        t0 = time.perf_counter()
        try:
            return await asyncio.wait_for(asyncio.shield(fut), self.handshake_timeout_s)
        except asyncio.TimeoutError:
            self._waiters.pop(peer, None)
            self.hs.abort(peer)
            raise TransportError(f"Handshake con {peer} sin respuesta en {self.handshake_timeout_s:.0f} s") from None
        finally:
            self.stats["t_handshake_s"] += time.perf_counter() - t0

    async def _send_hs(self, peer: str, msg_type: str, fields: Mapping[str, Any]) -> None:
        self._hs_seq += 1
        header = {"schema_version": 1, "msg_type": msg_type, "experiment_id": self.experiment_id,
                  "topology": self.topology, "sender_id": self.node_id, "recipient_id": peer,
                  "seq": self._hs_seq, **fields}
        await self.inner.send(peer, encode_frame(header, b"", self.limits), msg_type, content_type=CONTENT_TYPE)

    def _install(self, s: Session) -> None:
        s.generation = self.inner.generation
        self._send[s.peer] = s
        epochs = self._recv.setdefault(s.peer, {})
        epochs[s.epoch] = s
        while len(epochs) > MAX_EPOCHS_PER_PEER:
            epochs.pop(next(iter(epochs)))
        self.stats["handshakes"] += 1
        fut = self._waiters.pop(s.peer, None)
        if fut is not None and not fut.done():
            fut.set_result(s)
        log.info("[%s] sesión S2 con %s, época %016x", self.node_id, s.peer, s.epoch)

    # -------------------------------------------------------------- recepción
    async def _pump(self) -> None:
        while True:
            try:
                d = await self.inner.recv(None)
            except BrokerLostError as exc:
                for fut in self._waiters.values():
                    if not fut.done():
                        fut.set_exception(exc)
                await self._app.put(exc)
                return
            if d.msg_type in HANDSHAKE_TYPES and d.content_type == CONTENT_TYPE:
                await self._handle_hs(d)
            else:
                await self._app.put(d)

    def _reject(self, reason: str) -> None:
        self.stats["sec_rejected"] += 1
        log.warning("[%s] mensaje S2 descartado: %s", self.node_id, reason)

    async def _handle_hs(self, d: Delivery) -> None:
        try:
            header, payload = decode_frame(d.body, self.limits)
            self.validator.validate(header)
            peer = header["sender_id"]
            if payload or d.sender_user != f"{self.experiment_id}.{peer}" or header["recipient_id"] != self.node_id \
                    or header["experiment_id"] != self.experiment_id or peer not in self.trust or peer == self.node_id:
                raise CryptoError("handshake con identidad, destino o experimento no válidos")
            if header["msg_type"] == HELLO:
                for msg_type, fields in self.hs.on_hello(peer, header):
                    await self._send_hs(peer, msg_type, fields)
            else:
                s = self.hs.on_auth(peer, header)
                if s is not None:
                    self._install(s)
        except (FrameError, ContractError, CryptoError, ValueError, KeyError) as exc:
            self._reject(f"{d.msg_type}: {exc}")
        except TransportError as exc:
            log.warning("[%s] no se pudo responder al handshake: %s", self.node_id, exc)
        await d.ack()

    async def recv(self, timeout: float | None = None) -> Delivery:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            item = await asyncio.wait_for(self._app.get(), remaining)
            if isinstance(item, BaseException):
                await self._app.put(item)                 # las siguientes llamadas también fallan
                raise item
            t0 = time.perf_counter()
            plain, problem = self._open(item)
            self.stats["t_decrypt_s"] += time.perf_counter() - t0
            if problem is None:
                return Delivery(plain, item.sender_user, item.msg_type, item.message_id, item.received_ns,
                                item._message, CONTENT_TYPE)
            self._reject(problem)
            await item.reject()

    def _open(self, d: Delivery) -> tuple[bytes, str | None]:
        if d.content_type != envelope.E2E_CONTENT_TYPE:
            return b"", f"{d.msg_type} sin sobre E2E (S2 no admite mensajes en claro)"
        try:
            outer = envelope.parse(d.body, self.limits.max_frame_bytes + envelope.MAX_OUTER_HEADER + 64)
        except envelope.EnvelopeError as exc:
            return b"", str(exc)
        h = outer.header
        sender = h["sender_id"]
        if d.sender_user != f"{self.experiment_id}.{sender}":
            return b"", f"identidad del broker {d.sender_user!r} ≠ emisor {sender!r}"
        if h["recipient_id"] != self.node_id or h["experiment_id"] != self.experiment_id or h["msg_type"] != d.msg_type:
            return b"", "cabecera externa de otro destino, experimento o tipo"
        session = self._recv.get(sender, {}).get(outer.epoch)
        if session is None:
            return b"", f"época desconocida {h['epoch']} de {sender}"
        why = session.window.check(outer.seq)
        if why is not None:
            return b"", f"{why}: {sender} época {h['epoch']} seq {outer.seq}"
        try:
            plain = envelope.open_(session, outer)
            inner, _ = decode_frame(plain, self.limits)
        except (CryptoError, FrameError) as exc:
            return b"", f"{type(exc).__name__}: {exc}"
        if inner.get("sender_id") != sender or inner.get("msg_type") != h["msg_type"] \
                or inner.get("recipient_id") not in (self.node_id, "*"):
            return b"", "cabecera interna incoherente con la externa"
        session.window.mark(outer.seq)
        return plain, None
