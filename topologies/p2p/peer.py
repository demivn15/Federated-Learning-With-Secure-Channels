"""Par de la topología P2P: DFedAvg con mezcla de Metropolis ponderada por n (RF-06; D5 §3; 04 ADR-06).

No hay servidor, raíz ni coordinador: el broker solo enruta y la 6.ª Pi no ejecuta ningún rol. Cada par:

  0. arranque: `HELLO(arch_hash, n)` a sus vecinos del grafo y espera los suyos (`hello_timeout_s`); como
     las colas existen antes de arrancar, el orden de arranque no cambia la vista inicial;
  ronda r (barrera con timeout, D5 §3.3):
  1. procesa lo pendiente (`drain`): modelos adelantados (r+1…r+s) al buffer, `DONE`, `SYNC_REQ`;
     si un vecino le saca más de s rondas ⇒ `SYNC` (`topologies/p2p/sync.py`);
  2. entrena desde su modelo actual y envía `MODEL(r)` a cada vecino (unicast; también a los `DEAD`, cuya
     cola lo guarda por si vuelven);
  3. espera los `MODEL(r)` de los vecinos `ALIVE`/`SUSPECT` hasta tenerlos o hasta `round_timeout_s`;
  4. mezcla w_i = Σ a_ij w_j sobre lo recibido ∪ {i} (`core.aggregation.mix`, renormalizado; malla
     completa + recepción completa ≡ FedAvg ponderado, D0 Eq. 1); actualiza la membresía;
  5. checkpoint `mixed`, KPIs (distancia de consenso, vecinos ausentes, atraso) y parada local: *patience*
     sobre el recall del modelo MEZCLADO en su validación (D5 §3.4). Al parar envía `DONE(r)`.
  Un par termina cuando su parada se cumple o cuando todos sus vecinos vivos han terminado (D5 §3.4).
  Si en una ronda no recibe ningún modelo de un vecino (aislado), la ronda no cuenta: espera tráfico y se
  sincroniza (`min_peers` = 1, 03 Q-09c), en lugar de avanzar con rondas solo-locales.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import torch

from core.aggregation import metropolis_weights, mix
from core.model import clone_state
from core.stopping import build_round_policy
from core.training import Update
from evaluation.metrics import binary_metrics, predict_proba
from messaging.transport_amqp import BrokerLostError, TransportError
from node.checkpoint import save_checkpoint
from telemetry.clock import PhaseTimer
from telemetry.system import NicCounters, ProcessCpu
from topologies.base import NodeContext, Received
from topologies.centralized.server import training_params
from topologies.local_worker import LocalWorker
from topologies.p2p.graph import build_graph, is_connected
from topologies.p2p.membership import FINISHED, Membership
from topologies.p2p.sync import SyncResponse, adopt

log = logging.getLogger("fl.p2p")
P2P_TYPES = ("HELLO", "MODEL", "SYNC_REQ", "SYNC_RESP", "DONE")
DRAIN_S = 0.05


def consensus_distance(own: dict[str, torch.Tensor], others: list[dict[str, torch.Tensor]]) -> float | None:
    """Media de ‖w_i − w_j‖₂ con los vecinos recibidos (D5 §3.3)."""
    if not others:
        return None
    flat = torch.cat([own[k].flatten().double() for k in own])
    return float(sum(torch.linalg.vector_norm(flat - torch.cat([o[k].flatten().double() for k in own]))
                     for o in others) / len(others))


class Peer:
    def __init__(self, ctx: NodeContext, processed_dir: Path) -> None:
        self.ctx, self.cfg = ctx, ctx.cfg
        topo = self.topo = ctx.cfg["topologies"]["p2p"]
        self.me = ctx.node_id
        self.graph = build_graph(topo["graph"], topo["peers"], topo.get("k"), int(topo.get("graph_seed", 0)))
        if not is_connected(self.graph):
            raise ValueError("El grafo P2P no es conexo")
        self.neighbors = sorted(self.graph[self.me])
        self.weights = metropolis_weights(self.graph)[self.me]
        self.member = Membership(tuple(self.neighbors), int(topo["dead_after_rounds"]))
        self.allowed = {t: set(self.neighbors) for t in P2P_TYPES}
        self.worker = LocalWorker(ctx.cfg, self.me, processed_dir)
        self.current = clone_state(self.worker.model.state_dict())
        self.round = 0                                                   # última ronda completada
        self.window = int(topo["buffer_rounds"])
        self.buffer: dict[int, dict[str, Update]] = {}
        self.sync_responses: list[SyncResponse] = []
        self.synced_to: int | None = None
        self.training = training_params(ctx.cfg, ctx.manifest["method"])
        self.policy = build_round_policy(topo["rounds"])
        self.publish_failures = 0

    # ================================================================ envío y recepción
    async def send_to(self, targets, msg_type: str, state=None, **fields) -> list:
        sent = []
        for n in targets:
            try:
                sent.append(await self.ctx.send(n, msg_type, state, **fields))
            except BrokerLostError:
                raise
            except TransportError as exc:          # cola de un par caído llena: no debe tumbar a este par
                self.publish_failures += 1
                log.warning("[%s] %s → %s no entregado: %s", self.me, msg_type, n, exc)
        return sent

    def unfinished(self) -> list[str]:
        return [n for n in self.neighbors if self.member.state[n] != FINISHED]

    async def handle(self, rec: Received) -> None:
        """Mensaje que no es el MODEL de la ronda en curso."""
        h, s, t = rec.header, rec.sender, rec.header["msg_type"]
        await rec.accept()
        if t == "DONE":
            self.member.finished(s, h["round"])
            return
        self.member.heard(s, h["round"])
        if t == "MODEL":
            r = h["round"]
            if self.round < r <= self.round + 1 + self.window:
                self.buffer.setdefault(r, {}).setdefault(
                    s, Update(rec.state, h["n_samples"], h["tau"], h["metrics"]["train_loss"]))
            # r ≤ self.round: obsoleto; r más allá de la ventana: solo cuenta para detectar el atraso
        elif t == "SYNC_REQ":
            await self.send_to([s], "SYNC_RESP", self.current, round=self.round, n_samples=self.worker.n_samples)
        elif t == "SYNC_RESP":
            self.sync_responses.append(SyncResponse(s, h["round"], h["n_samples"], rec.state))

    async def drain(self) -> None:
        while (rec := await self.ctx.receive_from(DRAIN_S, self.allowed, self.worker.specs)) is not None:
            await self.handle(rec)

    # ================================================================ fases
    async def hello(self) -> list[str]:
        await self.send_to(self.neighbors, "HELLO", round=0, n_samples=self.worker.n_samples,
                           arch_hash=self.worker.register_fields()["arch_hash"])
        deadline = time.monotonic() + float(self.topo["hello_timeout_s"])
        seen: set[str] = set()
        while seen < set(self.neighbors):
            rec = await self.ctx.receive_from(deadline - time.monotonic(), self.allowed, self.worker.specs)
            if rec is None:
                break
            if rec.header["msg_type"] == "HELLO" and rec.header["arch_hash"] != self.worker.register_fields()["arch_hash"]:
                log.error("[%s] HELLO de %s con otra arquitectura: se ignora", self.me, rec.sender)
                await rec.discard()
                continue
            seen.add(rec.sender)
            await self.handle(rec)
        if set(self.neighbors) - seen:
            self.member.end_round(seen)                    # quien no saludó empieza como SUSPECT
        log.info("[%s] vista inicial: %s (vecinos %s)", self.me, sorted(seen), self.neighbors)
        return sorted(seen)

    async def sync(self) -> bool:
        self.sync_responses = []
        targets = self.unfinished()
        await self.send_to(targets, "SYNC_REQ", round=self.round)
        wanted = self.member.expected()
        deadline = time.monotonic() + float(self.topo["sync_timeout_s"])
        while {r.sender for r in self.sync_responses} < wanted:
            rec = await self.ctx.receive_from(deadline - time.monotonic(), self.allowed, self.worker.specs)
            if rec is None:
                break
            await self.handle(rec)
        if not self.sync_responses:
            return False
        top, state, used = adopt(self.sync_responses)
        log.info("[%s] SYNC: ronda %d → %d con el modelo de %s", self.me, self.round, top, used)
        self.current, self.round, self.synced_to = state, top, top
        self.buffer = {r: m for r, m in self.buffer.items() if r > top}
        return True

    async def wait_for_traffic(self) -> bool:
        """Aislado: no cuenta rondas solo-locales; espera cualquier mensaje de un vecino."""
        rec = await self.ctx.receive_from(float(self.topo["client_round_wait_s"]), self.allowed, self.worker.specs)
        if rec is None:
            return False
        await self.handle(rec)
        return True

    # ================================================================ ronda
    async def step(self) -> str | None:
        """Una ronda. Devuelve el motivo de parada o None para seguir."""
        ctx, worker = self.ctx, self.worker
        await self.drain()
        if self.member.all_done():
            return "all_neighbors_done"
        if self.member.ahead_by(self.round) > self.window:
            if not await self.sync() and not await self.wait_for_traffic():
                return "isolated_timeout"
            return None
        r = self.round + 1
        staleness = max(self.member.ahead_by(self.round) - 1, 0)

        cpu_before, ram_before = worker.cpu_before()
        timer, proc, nic0 = PhaseTimer(), ProcessCpu(), NicCounters.snapshot()
        proc.start()
        t0 = time.perf_counter()
        res = await worker.train(self.current, self.training, timer)
        own = res.update
        sent = await self.send_to(self.unfinished(), "MODEL", own.state, round=r, n_samples=own.n_samples,
                                  tau=own.tau, metrics=worker.update_metrics(res, timer))
        for s in sent:
            timer.add("serialize", s.t_serialize_s)
            timer.add("publish_confirm", s.publish.confirm_s)
        if self.cfg.get("checkpoints", {}).get("local"):
            save_checkpoint(ctx.run_dir, self.me, r, 0, "local", own.state,
                            {"n_samples": own.n_samples, "tau": own.tau})

        received: dict[str, Update] = dict(self.buffer.pop(r, {}))
        deadline = time.monotonic() + float(self.topo["round_timeout_s"])
        with timer.phase("wait_model"):
            while not self.member.expected() <= set(received):
                rec = await ctx.receive_from(deadline - time.monotonic(), self.allowed, worker.specs)
                if rec is None:
                    break
                h = rec.header
                if h["msg_type"] == "MODEL" and h["round"] == r and rec.sender not in received:
                    await rec.accept()
                    self.member.heard(rec.sender, r)
                    timer.add("deserialize", rec.t_deserialize_s)
                    received[rec.sender] = Update(rec.state, h["n_samples"], h["tau"], h["metrics"]["train_loss"])
                else:
                    await self.handle(rec)
        missing = sorted(self.member.expected() - set(received))
        self.member.end_round(received)

        if not received:
            if self.member.all_done():
                return "all_neighbors_done"
            log.warning("[%s] ronda %d sin modelos de vecinos: aislado, la ronda no cuenta", self.me, r)
            if not await self.wait_for_traffic():
                return "isolated_timeout"
            return None

        with timer.phase("aggregate"):
            mixed = mix(self.me, {self.me: own, **received}, self.weights,
                        weight_by_samples=bool(self.topo.get("weight_by_samples", True)))
        distance = consensus_distance(own.state, [u.state for u in received.values()])
        self.current, self.round = mixed, r
        save_checkpoint(ctx.run_dir, self.me, r, 0, "mixed", mixed,
                        {"received": sorted(received), "missing": missing, "synced_to": self.synced_to})
        worker.model.load_state_dict(mixed)
        mixed_recall = binary_metrics(worker.data.y_val.numpy(), predict_proba(worker.model, worker.data.X_val),
                                      0.5)["recall"]
        decision = self.policy.update(r, mixed_recall)

        cpu = proc.stop()
        tx, rx = nic0.delta(NicCounters.snapshot())
        upload_s = sum(s.publish.confirm_s for s in sent)
        ctx.kpi.log({
            "role": "peer", "method": self.training["method"], "round": r, "sub_round": 0,
            "t_round_s": time.perf_counter() - t0, **timer.as_columns(),
            "nic_bytes_tx": tx, "nic_bytes_rx": rx, **ctx.take_counters(), **cpu,
            **worker.kpi_columns(res, cpu_before=cpu_before, ram_before=ram_before, upload_confirm_s=upload_s,
                                 upload_bytes=sent[0].nbytes if sent else None),
            "p2p_neighbors": self.neighbors, "p2p_received": sorted(received), "p2p_peers_missing": missing,
            "p2p_consensus_distance": distance, "p2p_staleness": staleness, "p2p_isolated": False,
            "p2p_mixed_val_recall": mixed_recall, "p2p_synced_to": self.synced_to,
        })
        log.info("[%s] ronda %d: mezcla con %s, faltan %s, consenso %.4f, recall mezclado %.3f", self.me, r,
                 sorted(received), missing, distance or 0.0, mixed_recall)
        self.synced_to = None
        if decision.stop:
            return decision.reason
        if self.member.all_done():
            return "all_neighbors_done"
        return None

    async def run(self) -> int:
        await self.hello()
        while True:
            reason = await self.step()
            if reason is not None:
                await self.send_to(self.unfinished(), "DONE", round=self.round, reason=reason)
                log.info("[%s] fin en la ronda %d: %s", self.me, self.round, reason)
                return 0 if reason != "isolated_timeout" else 4


async def run_peer(ctx: NodeContext, processed_dir: Path) -> int:
    return await Peer(ctx, processed_dir).run()
