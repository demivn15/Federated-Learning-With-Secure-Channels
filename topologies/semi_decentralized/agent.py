"""Client Node de la semi-descentralizada (RF-03, RF-04; D0 §3.3.3, §4.3; 04 ADR-08).

El nodo obedece al coordinador (`ELECTION`, `ROLE_ASSIGN`, `STOP`) y, en cada ronda externa, hace uno de
dos papeles durante `internal_rounds` rondas internas (5 en D0):

  trabajador: entrena desde su modelo actual ──UPDATE(r, k)──▶ agregador ──AGG_MODEL(r, k)──▶ nuevo modelo
  agregador:  NO entrena (D0 §4.3: "remaining four worker Client Nodes"); espera los UPDATE hasta tenerlos
              todos o hasta el plazo (`internal_round_timeout_s` desde el inicio de la ronda interna y
              `arrival_window_s` desde la 1.ª llegada, ADR-12); agrega con la regla del método (Eq. 1–3,
              con M_t = su modelo actual); guarda el checkpoint `global` (r, k) y devuelve AGG_MODEL a TODOS
              los trabajadores asignados (también a los que llegaron tarde, para que se resincronicen).
              Al final envía `AGG_COMPLETE` al coordinador: recall local medio de los trabajadores en la
              última ronda interna (métrica de parada de P3) y retraso de llegada de cada uno (`SD-score`).

Continuidad del modelo (como P3): no hay difusión inicial del modelo; cada nodo empieza la ronda externa con
el último modelo agregado que recibió (el inicial es idéntico en todos: misma semilla). Un mensaje del
coordinador interrumpe el papel en curso (re-selección o fin) y se procesa a continuación.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from core.aggregation import aggregate
from core.model import clone_state
from core.training import Update
from messaging.frame import prepare_payload
from node.checkpoint import save_checkpoint
from telemetry.clock import ArrivalClock, PhaseTimer
from telemetry.system import NicCounters, ProcessCpu, cpu_freq_mhz, declared_telemetry, ram_rss_mb, temperature_c
from topologies.base import NodeContext, Received
from topologies.local_worker import LocalWorker
from topologies.semi_decentralized.selectors import select

log = logging.getLogger("fl.sd.agent")
State = dict[str, torch.Tensor]


class Agent:
    def __init__(self, ctx: NodeContext, processed_dir: Path) -> None:
        self.ctx, self.cfg = ctx, ctx.cfg
        self.topo = ctx.cfg["topologies"]["semi_decentralized"]
        self.coord = self.topo["coordinator"]
        self.worker = LocalWorker(ctx.cfg, ctx.node_id, processed_dir)
        self.current: State = clone_state(self.worker.model.state_dict())
        self.from_coord = {"ELECTION": {self.coord}, "ROLE_ASSIGN": {self.coord}, "STOP": {self.coord}}
        self.wait_s = float(self.topo["client_round_wait_s"])

    # ================================================================ bucle principal
    async def run(self) -> int:
        ctx = self.ctx
        await ctx.send(self.coord, "REGISTER", **self.worker.register_fields())
        ack = await ctx.receive_from(float(self.topo["register_timeout_s"]),
                                     {"REGISTER_ACK": {self.coord}, "STOP": {self.coord}})
        if ack is None or ack.header["msg_type"] == "STOP" or not ack.header["accepted"]:
            if ack is not None:
                await ack.accept()
            log.error("[%s] registro rechazado o sin respuesta", ctx.node_id)
            return 2
        await ack.accept()

        pending: Received | None = None
        while True:
            msg = pending or await ctx.receive_from(self.wait_s, self.from_coord)
            pending = None
            if msg is None:
                log.error("[%s] sin noticias del coordinador", ctx.node_id)
                return 4
            await msg.accept()
            h = msg.header
            if h["msg_type"] == "STOP":
                log.info("[%s] STOP: %s", ctx.node_id, h["reason"])
                return 0
            if h["msg_type"] == "ELECTION":
                try:
                    winner = select(h["selector"], h["candidates"], int(self.topo["selector_seed"]), h["round"],
                                    h["attempt"], h.get("scores"))
                except ValueError as exc:            # p. ej. sd_score_hybrid sin puntuaciones: no se responde
                    log.warning("[%s] ELECTION inválida: %s", ctx.node_id, exc)
                    continue
                await ctx.send_or_skip(self.coord, "CLAIM", round=h["round"], attempt=h["attempt"], winner=winner,
                               telemetry=declared_telemetry())
                continue
            if h["role"] == "aggregator":
                pending = await self.aggregate_round(h)
            else:
                pending = await self.work_round(h)

    # ================================================================ trabajador
    async def work_round(self, assign: Mapping[str, Any]) -> Received | None:
        ctx, worker = self.ctx, self.worker
        agg, r, training = assign["aggregator_id"], assign["round"], assign["training"]
        allowed = {"AGG_MODEL": {agg}, **self.from_coord}
        for k in range(1, int(assign["internal_rounds"]) + 1):
            cpu_before, ram_before = worker.cpu_before()
            timer, proc, nic0 = PhaseTimer(), ProcessCpu(), NicCounters.snapshot()
            proc.start()
            t0 = time.perf_counter()
            res = await worker.train(self.current, training, timer)
            upd = res.update
            sent = await ctx.send_or_skip(agg, "UPDATE", upd.state, round=r, sub_round=k, n_samples=upd.n_samples,
                                          tau=upd.tau, metrics=worker.update_metrics(res, timer)) or ctx.unsent()
            timer.add("serialize", sent.t_serialize_s)
            timer.add("publish_confirm", sent.publish.confirm_s)
            if self.cfg.get("checkpoints", {}).get("local"):
                save_checkpoint(ctx.run_dir, ctx.node_id, r, k, "local", upd.state,
                                {"n_samples": upd.n_samples, "tau": upd.tau, "aggregator_id": agg})

            interrupt = None
            t_wait = time.perf_counter()
            while True:
                rec = await ctx.receive_from(self.wait_s, allowed, worker.specs)
                if rec is None or rec.header["msg_type"] != "AGG_MODEL":
                    interrupt = rec
                    break
                if rec.header["round"] == r and rec.header["sub_round"] == k:
                    await rec.accept()
                    self.current = rec.state
                    timer.add("deserialize", rec.t_deserialize_s)
                    break
                await rec.discard()                                  # modelo de una ronda interna anterior
            timer.add("wait_model", time.perf_counter() - t_wait)

            cpu = proc.stop()
            tx, rx = nic0.delta(NicCounters.snapshot())
            ctx.kpi.log({
                "role": "worker", "method": training["method"], "round": r, "sub_round": k,
                "t_round_s": time.perf_counter() - t0, **timer.as_columns(),
                "nic_bytes_tx": tx, "nic_bytes_rx": rx, **ctx.take_counters(), **cpu,
                **worker.kpi_columns(res, cpu_before=cpu_before, ram_before=ram_before,
                                     upload_confirm_s=sent.publish.confirm_s, upload_bytes=sent.nbytes),
                "sd_aggregator_id": agg,
            })
            if interrupt is not None or rec is None:
                log.warning("[%s] ronda %d.%d interrumpida (%s)", ctx.node_id, r, k,
                            interrupt.header["msg_type"] if interrupt else "sin respuesta del agregador")
                return interrupt
        return None

    # ================================================================ agregador
    async def aggregate_round(self, assign: Mapping[str, Any]) -> Received | None:
        ctx, cfg, worker = self.ctx, self.cfg, self.worker
        r, attempt, training = assign["round"], assign["attempt"], assign["training"]
        workers = list(assign["workers"])
        method = training["method"]
        allowed = {"UPDATE": set(workers), **self.from_coord}
        p_watts = float(cfg["energy"]["e_comm"]["p_watts"])
        M = self.current
        lateness: dict[str, list[float]] = {w: [] for w in workers}
        stragglers: list[float] = []
        last_recalls: dict[str, float] = {}
        missing: list[str] = list(workers)
        done = 0
        for k in range(1, int(assign["internal_rounds"]) + 1):
            timer, proc, nic0, arrivals = PhaseTimer(), ProcessCpu(), NicCounters.snapshot(), ArrivalClock()
            proc.start()
            ram_before = ram_rss_mb()
            start = time.monotonic()
            hard = start + float(self.topo["internal_round_timeout_s"])
            first: float | None = None
            updates: dict[str, Update] = {}
            metrics: dict[str, Mapping[str, float]] = {}
            with timer.phase("wait_model"):
                while len(updates) < len(workers):
                    deadline = hard if first is None else min(hard, first + float(self.topo["arrival_window_s"]))
                    rec = await ctx.receive_from(deadline - time.monotonic(), allowed, worker.specs)
                    if rec is None:
                        break
                    h = rec.header
                    if h["msg_type"] != "UPDATE":
                        log.warning("[%s] agregación de la ronda %d interrumpida en %d/%s por %s", ctx.node_id, r,
                                    k, assign["internal_rounds"], h["msg_type"])
                        self.current = M
                        return rec
                    if h["round"] != r or h.get("sub_round") != k or rec.sender in updates:
                        await rec.discard()                          # tardío de una ronda interna anterior
                        continue
                    await rec.accept()
                    arrivals.mark(rec.sender, rec.arrived_ns)
                    first = first or time.monotonic()
                    timer.add("deserialize", rec.t_deserialize_s)
                    updates[rec.sender] = Update(rec.state, h["n_samples"], h["tau"], h["metrics"]["train_loss"])
                    metrics[rec.sender] = h["metrics"]
            if not updates:
                log.error("[%s] ronda %d.%d sin modelos de los trabajadores: se abandona", ctx.node_id, r, k)
                break
            with timer.phase("aggregate"):
                M = aggregate(method, [updates[w] for w in sorted(updates)], M, cfg["aggregation"])
            t_agg_ns = time.perf_counter_ns()
            missing = sorted(set(workers) - set(updates))
            save_checkpoint(ctx.run_dir, ctx.node_id, r, k, "global", M,
                            {"aggregator_id": ctx.node_id, "workers": sorted(updates), "missing": missing,
                             "attempt": attempt, "method": method})
            with timer.phase("serialize"):
                payload = prepare_payload(M)
            publish_s = 0.0
            for w in workers:
                sent = await ctx.send_or_skip(w, "AGG_MODEL", payload, round=r, sub_round=k)
                publish_s += sent.publish.confirm_s if sent is not None else 0.0
            timer.add("publish_confirm", publish_s)

            t_first = min(arrivals.arrivals.values())
            for w in workers:                       # un ausente cuenta como llegado al momento de agregar
                lateness[w].append((arrivals.arrivals.get(w, t_agg_ns) - t_first) / 1e9)
            if arrivals.straggler_s() is not None:
                stragglers.append(arrivals.straggler_s())
            last_recalls = {w: m["recall"] for w, m in metrics.items() if "recall" in m}
            done = k

            cpu = proc.stop()
            tx, rx = nic0.delta(NicCounters.snapshot())
            ctx.kpi.log({
                "role": "aggregator", "method": method, "round": r, "sub_round": k,
                "t_round_s": timer.elapsed(), **timer.as_columns(),
                "nic_bytes_tx": tx, "nic_bytes_rx": rx, **ctx.take_counters(), **cpu,
                "ram_rss_mb_before": ram_before, "ram_rss_mb_after": ram_rss_mb(),
                "temperature_c": temperature_c(), "cpu_freq_mhz": cpu_freq_mhz(),
                "e_comm_transfer_j": p_watts * publish_s,
                "straggler_s": arrivals.straggler_s(), "arrivals": arrivals.relative_s(),
                "sd_aggregator_id": ctx.node_id,
            })
            log.info("[%s] agregador ronda %d.%d: %d/%d modelos, faltan %s, straggler %.3fs", ctx.node_id, r, k,
                     len(updates), len(workers), missing, arrivals.straggler_s() or 0.0)

        self.current = M
        recall = sum(last_recalls.values()) / len(last_recalls) if last_recalls else None
        await ctx.send_or_skip(self.coord, "AGG_COMPLETE", round=r, attempt=attempt, internal_rounds_done=done,
                       recall=recall, straggler_s=sum(stragglers) / len(stragglers) if stragglers else None,
                       lateness={w: sum(v) / len(v) for w, v in lateness.items() if v},
                       workers_missing=missing)
        return None


async def run_agent(ctx: NodeContext, processed_dir: Path) -> int:
    return await Agent(ctx, processed_dir).run()
