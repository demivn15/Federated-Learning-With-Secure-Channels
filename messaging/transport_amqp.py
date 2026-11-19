"""Transporte AMQP sobre RabbitMQ (04 ADR-03, §4.5; RF-31).

Cada nodo:
  * abre UNA conexión con su usuario propio (`<experiment>.<node_id>`) al vhost del experimento; en los
    perfiles S1/S2 la conexión es AMQPS y verifica el certificado del broker con la CA del despliegue;
  * consume SOLO su cola `q.<node_id>` con ack manual y `prefetch` acotado;
  * publica en `fl.unicast` (clave = destinatario) o `fl.broadcast`, con mensajes NO persistentes,
    `mandatory=True` (destino inexistente ⇒ error) y confirmación del publicador (cola llena ⇒ nack ⇒ error);
  * fija `user_id` = su usuario AMQP. RabbitMQ rechaza un `user_id` distinto del usuario autenticado,
    de modo que el receptor obtiene una identidad del emisor validada por el broker (`Delivery.sender_user`).
El nodo no declara exchanges ni colas (permiso `configure` vacío): solo los usa.

Caída del broker (E9; 04 ADR-12): al perder la conexión se emite `broker_connection_lost` y el transporte
reintenta conectarse (de forma pasiva: conexión, canal y consumidor, sin declarar nada). Si no lo logra en
`loss_grace_s`, `recv` lanza `BrokerLostError` y el nodo termina de forma limpia y registrada (código 6). Si
vuelve, se emite `broker_reconnected` y aumenta `generation` (el perfil S2 hace un *handshake* nuevo ⇒
nueva época). Una publicación durante el corte espera la reconexión dentro del mismo plazo.
"""
from __future__ import annotations

import asyncio
import logging
import ssl
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import aio_pika
from aio_pika import DeliveryMode
from aio_pika.abc import AbstractIncomingMessage
from aio_pika.exceptions import AMQPError, DeliveryError, PublishError

CONTENT_TYPE = "application/vnd.fl-frame"
log = logging.getLogger("fl.transport")
EventHook = Callable[[str, Mapping[str, Any]], None]
_FATAL = object()


class TransportError(RuntimeError):
    """Fallo al publicar (cola llena, destino inexistente, sin confirmación) o al recibir."""


class BrokerLostError(TransportError):
    """La conexión con el broker no se recuperó dentro de `loss_grace_s` (ADR-12: fin del experimento)."""


@dataclass
class Delivery:
    body: bytes
    sender_user: str | None          # validado por el broker
    msg_type: str | None
    message_id: str | None
    received_ns: int                 # llegada a este nodo (callback de aio-pika), reloj monótono local (RNF-07)
    _message: AbstractIncomingMessage = field(repr=False)
    content_type: str | None = None

    async def ack(self) -> None:
        try:
            await self._message.ack()
        except Exception as exc:     # canal de una conexión anterior: el broker lo reentregará
            log.debug("ack sin efecto: %s", exc)

    async def reject(self) -> None:
        """Descarta el mensaje (sin reencolar): se usa cuando no supera la validación."""
        try:
            await self._message.reject(requeue=False)
        except Exception as exc:
            log.debug("reject sin efecto: %s", exc)


@dataclass
class PublishStats:
    nbytes: int
    confirm_s: float                 # tiempo hasta la confirmación del broker (t_publish_confirm)


def tls_context(ca_file: str) -> ssl.SSLContext:
    """Contexto TLS de cliente: verifica el certificado del broker contra la CA del despliegue (S1/S2)."""
    ctx = ssl.create_default_context(cafile=ca_file)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    return ctx


class AmqpTransport:
    def __init__(self, url: str, node_id: str, username: str, broker_cfg: Mapping[str, Any],
                 max_frame_bytes: int, *, ssl_context: ssl.SSLContext | None = None,
                 on_event: EventHook | None = None) -> None:
        self.url, self.node_id, self.username = url, node_id, username
        self.cfg = broker_cfg
        self.max_frame_bytes = max_frame_bytes
        self.ssl_context = ssl_context
        self.on_event = on_event
        self.loss_grace_s = float(broker_cfg.get("loss_grace_s", 30))
        self.generation = 0                      # aumenta en cada (re)conexión
        self._conn: aio_pika.abc.AbstractConnection | None = None
        self._inbox: asyncio.Queue[Any] = asyncio.Queue()
        self._consumer_tag: str | None = None
        self._up = asyncio.Event()
        self._closing = False
        self._lost_at: float | None = None
        self._fatal: BrokerLostError | None = None
        self._watchdog: asyncio.Task | None = None

    def _event(self, name: str, **details: Any) -> None:
        log.warning("[%s] %s %s", self.node_id, name, details or "")
        if self.on_event is not None:
            self.on_event(name, details)

    # -------------------------------------------------------------- ciclo de vida
    async def connect(self) -> None:
        """Primera conexión (o reconexión explícita tras `close`). Lanza si el broker no responde."""
        self._inbox = asyncio.Queue()
        self._closing, self._fatal = False, None
        await self._open()
        self.generation += 1
        self._up.set()

    async def _open(self) -> None:
        # Reconexión PROPIA en vez de la restauración de `connect_robust`: aio-pika vuelve a declarar la cola
        # al restaurar el canal, y con el permiso `configure` vacío (ADR-03) el broker cierra el canal ⇒ el
        # nodo quedaba conectado pero sin consumir (detectado en E9 al reiniciar el broker). Aquí todo es
        # pasivo: se reabre la conexión, el canal, los exchanges y el consumidor, sin declarar nada.
        conn = await aio_pika.connect(
            self.url, heartbeat=int(self.cfg["heartbeat_s"]), ssl_context=self.ssl_context,
            client_properties={"connection_name": f"fl-node-{self.node_id}"})
        try:
            channel = await conn.channel(publisher_confirms=True, on_return_raises=True)
            await channel.set_qos(prefetch_count=int(self.cfg["prefetch"]))
            unicast = await channel.get_exchange(self.cfg["unicast_exchange"], ensure=False)
            broadcast = await channel.get_exchange(self.cfg["broadcast_exchange"], ensure=False)
            queue = await channel.get_queue(f"q.{self.node_id}", ensure=False)
            self._consumer_tag = await queue.consume(self._on_message, no_ack=False)
        except BaseException:
            await conn.close()
            raise
        self._conn, self._channel, self._unicast, self._broadcast, self._queue = conn, channel, unicast, broadcast, queue
        conn.close_callbacks.add(self._on_close)

    def _on_close(self, sender: Any, exc: BaseException | None = None) -> None:
        if self._closing or sender is not self._conn or not self._up.is_set():
            return                                  # cierre propio o de una conexión anterior
        self._up.clear()
        self._lost_at = time.perf_counter()
        self._event("broker_connection_lost", error=f"{type(exc).__name__}: {exc}" if exc else None,
                    grace_s=self.loss_grace_s)
        self._watchdog = asyncio.get_running_loop().create_task(self._recover())

    async def _recover(self) -> None:
        """Reintenta la conexión cada `reconnect_interval_s` hasta `loss_grace_s`. Los mensajes ya recibidos
        del canal anterior se conservan: si el broker sigue vivo los reentrega (duplicados que los receptores
        descartan por ronda/seq); si se reinició, los no persistentes se han perdido y solo quedan esos."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.loss_grace_s
        interval = float(self.cfg.get("reconnect_interval_s", 2))
        while not self._closing:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(interval, remaining))
            try:
                await asyncio.wait_for(self._open(), max(1.0, deadline - loop.time()))
            except asyncio.CancelledError:
                raise
            except Exception as exc:                # broker aún caído: se reintenta
                log.debug("[%s] reconexión fallida: %s", self.node_id, exc)
                continue
            self.generation += 1
            self._up.set()
            self._event("broker_reconnected", downtime_s=time.perf_counter() - self._lost_at,
                        generation=self.generation)
            return
        if not self._closing:
            self._fatal = BrokerLostError(f"Sin broker durante {self.loss_grace_s:.0f} s")
            self._event("broker_lost", waited_s=self.loss_grace_s)
            self._inbox.put_nowait(_FATAL)

    async def _on_message(self, m: AbstractIncomingMessage) -> None:
        # Se sella la llegada aquí y no al desencolar: un nodo que entrena (agregador de clúster, par P2P)
        # desencola después de entrenar, y el straggler saldría subestimado.
        await self._inbox.put((m, time.perf_counter_ns()))

    async def close(self) -> None:
        self._closing = True
        if self._watchdog is not None:
            self._watchdog.cancel()
        self._up.clear()
        if self._conn is not None and not self._conn.is_closed:
            try:
                await self._conn.close()
            except Exception as exc:          # broker ya caído
                log.debug("cierre de la conexión: %s", exc)

    async def __aenter__(self) -> "AmqpTransport":
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # -------------------------------------------------------------- envío
    def _message(self, frame: bytes, msg_type: str, message_id: str | None, content_type: str) -> aio_pika.Message:
        if len(frame) > self.max_frame_bytes:
            raise TransportError(f"Trama de {len(frame)} B > {self.max_frame_bytes}")
        return aio_pika.Message(
            frame, content_type=content_type, type=msg_type, message_id=message_id,
            user_id=self.username,
            delivery_mode=DeliveryMode.PERSISTENT if self.cfg.get("message_persistent") else DeliveryMode.NOT_PERSISTENT)

    async def _wait_up(self) -> None:
        if self._fatal is not None:
            raise self._fatal
        if not self._up.is_set():
            try:
                await asyncio.wait_for(self._up.wait(), self.loss_grace_s)
            except asyncio.TimeoutError:
                if self._fatal is None:          # detectado al publicar, antes que el vigilante de recepción
                    self._fatal = BrokerLostError(f"Sin broker durante {self.loss_grace_s:.0f} s (publicación)")
                    self._event("broker_lost", waited_s=self.loss_grace_s, path="publish")
                    self._inbox.put_nowait(_FATAL)
                raise self._fatal from None
            if self._fatal is not None:
                raise self._fatal

    async def _connection_cancelled(self) -> bool:
        """¿La `CancelledError` viene de aiormq (cancela la confirmación pendiente al cerrar una conexión colgada,
        p. ej. heartbeat perdido) y no de una cancelación de esta tarea? Solo entonces es un corte (E9).

        Python ≥ 3.11: una cancelación de la tarea incrementa `cancelling()`; la del future de aiormq no.
        Python 3.10 (Pi): sin `cancelling()`, se espera hasta 1 s a que la conexión conste como cerrada (el
        aviso de cierre puede llegar un poco después de la cancelación)."""
        if self._closing:
            return False
        cancelling = getattr(asyncio.current_task(), "cancelling", None)
        if cancelling is not None:
            return cancelling() == 0
        for _ in range(10):
            if self._conn is None or self._conn.is_closed or not self._up.is_set():
                return True
            await asyncio.sleep(0.1)
        return False

    async def _publish(self, which: str, routing_key: str, msg: aio_pika.Message) -> PublishStats:
        """`which` = "unicast" | "broadcast": el exchange se resuelve en cada intento, porque tras una
        reconexión pertenece al canal nuevo."""
        await self._wait_up()
        t0 = time.perf_counter()
        attempts = 0
        while True:
            exchange = self._unicast if which == "unicast" else self._broadcast
            try:
                await exchange.publish(msg, routing_key=routing_key, mandatory=True,
                                       timeout=float(self.cfg["publish_timeout_s"]))
                break
            except (DeliveryError, PublishError) as exc:
                raise TransportError(f"Publicación rechazada hacia {routing_key or '*'}: {exc}") from exc
            except asyncio.TimeoutError as exc:
                raise TransportError(f"Sin confirmación del broker hacia {routing_key or '*'}") from exc
            except (AMQPError, ConnectionError, RuntimeError, asyncio.CancelledError) as exc:
                # La conexión cayó durante la publicación: se espera la reconexión y se reintenta
                # (el receptor descarta duplicados por ronda/seq). El plazo total lo acota `loss_grace_s`.
                if isinstance(exc, asyncio.CancelledError) and not await self._connection_cancelled():
                    raise
                attempts += 1
                if attempts > 3:
                    raise TransportError(f"Publicación fallida hacia {routing_key or '*'}: {exc}") from exc
                log.warning("[%s] publicación interrumpida (%s); se reintenta tras la reconexión", self.node_id, exc)
                await asyncio.sleep(0.5)
                await self._wait_up()
        return PublishStats(len(msg.body), time.perf_counter() - t0)

    async def send(self, recipient: str, frame: bytes, msg_type: str,
                   message_id: str | None = None, content_type: str = CONTENT_TYPE) -> PublishStats:
        return await self._publish("unicast", recipient, self._message(frame, msg_type, message_id, content_type))

    async def broadcast(self, frame: bytes, msg_type: str, message_id: str | None = None,
                        content_type: str = CONTENT_TYPE) -> PublishStats:
        return await self._publish("broadcast", "", self._message(frame, msg_type, message_id, content_type))

    # -------------------------------------------------------------- recepción
    async def recv(self, timeout: float | None = None) -> Delivery:
        """Siguiente mensaje de la cola propia. El llamador debe hacer `ack()` o `reject()`.

        Lanza `asyncio.TimeoutError` si vence el plazo y `BrokerLostError` si el broker no volvió."""
        if self._fatal is not None:
            raise self._fatal
        item = await asyncio.wait_for(self._inbox.get(), timeout)
        if item is _FATAL:
            self._inbox.put_nowait(_FATAL)
            raise self._fatal
        m, arrived_ns = item
        return Delivery(m.body, m.user_id, m.type, m.message_id, arrived_ns, m, m.content_type)
