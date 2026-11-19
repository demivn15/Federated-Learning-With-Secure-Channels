"""Client Node de la topología centralizada (RF-01; D0 §3.3.2). Entrena solo con su hospital.

1. Se registra (`REGISTER` con nº de entradas, huella de arquitectura y nº de muestras) y espera `REGISTER_ACK`.
2. Por cada `ROUND_START`: carga el modelo global, entrena con los hiperparámetros recibidos (FedProx usa
   el global como referencia proximal), calcula la métrica heredada (modelo LOCAL en su validación con
   umbral 0,5, como `calculate_all_metrics` de P1; D1 H1) y envía `UPDATE` con el modelo, n_k y τ_k.
3. Termina con `STOP` o si no llega la siguiente ronda en `client_round_wait_s`.
KPIs: semántica heredada de CPU (antes de recibir/después de evaluar) y CPU del proceso en la ronda (I-31);
energía de cómputo (Eq. 4) y de comunicación por la subida (I-32). El entrenamiento está en
`topologies/local_worker.py`, común a todas las topologías.

El mismo bucle es la **hoja** de la jerárquica (`topologies/hierarchical/node.py`): solo cambian a quién
envía el `UPDATE` (`upstream`) y de quién acepta el `ROUND_START` (`round_start_from`). Cada mensaje de
control se acepta solo de su emisor legítimo (`NodeContext.receive_from`).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Mapping

from node.checkpoint import save_checkpoint
from telemetry.clock import PhaseTimer
from telemetry.system import NicCounters, ProcessCpu
from topologies.base import NodeContext
from topologies.local_worker import LocalWorker

log = logging.getLogger("fl.centralized.client")


async def run_client(ctx: NodeContext, processed_dir: Path, *, topology_key: str = "centralized",
                     registrar: str = "infra", upstream: str = "infra", round_start_from: str = "infra",
                     role: str = "client", extra_kpi: Mapping[str, Any] | None = None) -> int:
    cfg, topo = ctx.cfg, ctx.cfg["topologies"][topology_key]
    worker = LocalWorker(cfg, ctx.node_id, processed_dir)

    await ctx.send(registrar, "REGISTER", **worker.register_fields())
    ack = await ctx.receive_from(float(topo["register_timeout_s"]), {"REGISTER_ACK": {registrar},
                                                                      "STOP": {registrar}})
    if ack is None or ack.header["msg_type"] == "STOP" or not ack.header["accepted"]:
        if ack is not None:
            await ack.accept()
        log.error("[%s] registro rechazado o sin respuesta", ctx.node_id)
        return 2
    await ack.accept()

    waiting_since = time.perf_counter()
    while True:
        cpu_before, ram_before = worker.cpu_before()        # P1: antes de recibir el modelo
        rec = await ctx.receive_from(float(topo["client_round_wait_s"]),
                                     {"ROUND_START": {round_start_from}, "STOP": {registrar}}, worker.specs)
        if rec is None:
            log.error("[%s] no llegó la siguiente ronda", ctx.node_id)
            return 4
        await rec.accept()
        if rec.header["msg_type"] == "STOP":
            log.info("[%s] STOP: %s", ctx.node_id, rec.header["reason"])
            return 0

        round_, training = rec.header["round"], rec.header["training"]
        timer, proc, nic0 = PhaseTimer(), ProcessCpu(), NicCounters.snapshot()
        timer.add("wait_model", time.perf_counter() - waiting_since)
        timer.add("deserialize", rec.t_deserialize_s)
        proc.start()
        t_round0 = time.perf_counter()

        res = await worker.train(rec.state, training, timer)
        upd = res.update
        sent = await ctx.send(upstream, "UPDATE", upd.state, round=round_, n_samples=upd.n_samples, tau=upd.tau,
                              metrics=worker.update_metrics(res, timer))
        timer.add("serialize", sent.t_serialize_s)
        timer.add("publish_confirm", sent.publish.confirm_s)
        waiting_since = time.perf_counter()
        if cfg.get("checkpoints", {}).get("local"):
            save_checkpoint(ctx.run_dir, ctx.node_id, round_, 0, "local", upd.state,
                            {"n_samples": upd.n_samples, "tau": upd.tau})

        cpu = proc.stop()
        tx, rx = nic0.delta(NicCounters.snapshot())
        ctx.kpi.log({
            "role": role, "method": training["method"], "round": round_, "sub_round": 0,
            "t_round_s": time.perf_counter() - t_round0, **timer.as_columns(),
            "nic_bytes_tx": tx, "nic_bytes_rx": rx, **ctx.take_counters(), **cpu,
            **worker.kpi_columns(res, cpu_before=cpu_before, ram_before=ram_before,
                                 upload_confirm_s=sent.publish.confirm_s, upload_bytes=sent.nbytes),
            **(extra_kpi or {}),
        })
        log.info("[%s] ronda %d: τ=%d, pérdida %.4f, recall local %.3f", ctx.node_id, round_, upd.tau,
                 upd.train_loss, res.legacy["recall"])
