"""Servidor de la topología centralizada (RF-01; D0 §3.3.2). No entrena.

1. Registro: espera `REGISTER` de los clientes (hasta `register_timeout_s`); comprueba que todos anuncian
   el mismo nº de entradas y la huella de arquitectura del modelo de referencia; responde `REGISTER_ACK`.
2. Rondas: difunde `ROUND_START` con el modelo global y los hiperparámetros; espera los `UPDATE` hasta
   tenerlos todos o hasta `round_timeout_s` (barrera acotada, corrige P1-H3); agrega con el método del
   manifiesto (D0 Eqs. 1–3); guarda el checkpoint global; aplica la política de rondas.
3. Fin: difunde `STOP`.
El straggler se mide aquí, con las horas de llegada de cada `UPDATE` (ADR-11; corrige P2-H1).
"""
from __future__ import annotations

import logging
import time

import torch

from core.aggregation import aggregate
from core.model import arch_hash, build_model, clone_state
from core.stopping import build_round_policy
from core.training import Update
from messaging.frame import prepare_payload, tensor_specs
from node.checkpoint import save_checkpoint
from telemetry.clock import ArrivalClock, PhaseTimer
from telemetry.system import NicCounters, ProcessCpu, cpu_freq_mhz, cpu_percent_legacy, ram_rss_mb, temperature_c
from topologies.base import NodeContext

log = logging.getLogger("fl.centralized.server")


def training_params(cfg: dict, method: str) -> dict:
    mu = float(cfg["aggregation"]["fedprox"]["mu"]) if method == "fedprox" else 0.0
    return {"method": method, "local_epochs": int(cfg["training"]["local_epochs"]), "mu": mu}


async def accept_registrations(ctx: NodeContext, clients: list[str], timeout_s: float,
                               reference: torch.nn.Module) -> dict[str, int]:
    """Espera `REGISTER` de `clients` hasta `timeout_s`; acepta los compatibles con el modelo de referencia.

    Devuelve {nodo: n_samples}. Común a la centralizada, la raíz jerárquica y el coordinador SD.
    """
    expected_features = ctx.cfg["data"]["preprocessing"]["expected_n_features"]
    ref_hash = arch_hash(reference)
    registered: dict[str, int] = {}
    deadline = time.monotonic() + timeout_s
    while len(registered) < len(clients):
        rec = await ctx.receive(deadline - time.monotonic(), {"REGISTER"})
        if rec is None:
            break
        h = rec.header
        ok = h["sender_id"] in clients and h["n_features"] == expected_features and h["arch_hash"] == ref_hash
        reason = None if ok else "arquitectura, entradas o nodo no esperados"
        await rec.accept()
        await ctx.send(h["sender_id"], "REGISTER_ACK", accepted=ok, **({"reason": reason} if reason else {}))
        if ok:
            registered[h["sender_id"]] = h["n_samples"]
        log.info("[%s] registro de %s: %s", ctx.node_id, h["sender_id"], "aceptado" if ok else reason)
    return registered


async def run_server(ctx: NodeContext) -> int:
    cfg, topo = ctx.cfg, ctx.cfg["topologies"]["centralized"]
    method = ctx.manifest["method"]
    clients = sorted(n for n in cfg["platform"]["node_assignment"] if n != "infra")
    expected_features = cfg["data"]["preprocessing"]["expected_n_features"]

    # ------------------------------------------------------------------ registro
    reference = build_model(cfg, expected_features)
    registered = await accept_registrations(ctx, clients, float(topo["register_timeout_s"]), reference)
    if len(registered) < int(topo["min_clients"]):
        log.error("[server] solo %d/%d clientes registrados", len(registered), topo["min_clients"])
        await ctx.broadcast("STOP", round=0, reason="insufficient_clients")
        return 2

    # ------------------------------------------------------------------ rondas
    global_state = clone_state(reference.state_dict())
    specs = tensor_specs(global_state)
    params = training_params(cfg, method)
    policy = build_round_policy(topo["rounds"])
    active = sorted(registered)
    p_watts = float(cfg["energy"]["e_comm"]["p_watts"])
    round_ = 0
    while True:
        round_ += 1
        timer, proc, arrivals = PhaseTimer(), ProcessCpu(), ArrivalClock()
        cpu_before, ram_before, nic0 = cpu_percent_legacy(cfg["kpis"]["cpu"]["legacy_interval_s"]), ram_rss_mb(), NicCounters.snapshot()
        proc.start()

        with timer.phase("serialize"):
            payload = prepare_payload(global_state)
        publish_s = {}
        for c in active:
            sent = await ctx.send_or_skip(c, "ROUND_START", payload, round=round_, training=params)
            if sent is not None:
                publish_s[c] = sent.publish.confirm_s
        timer.add("publish_confirm", sum(publish_s.values()))

        updates: dict[str, Update] = {}
        legacy: dict[str, dict] = {}
        deadline = time.monotonic() + float(topo["round_timeout_s"])
        with timer.phase("wait_model"):
            while len(updates) < len(active):
                rec = await ctx.receive(deadline - time.monotonic(), {"UPDATE"}, specs)
                if rec is None:
                    break
                h = rec.header
                if h["round"] != round_ or h["sender_id"] not in active or h["sender_id"] in updates:
                    log.warning("[server] UPDATE descartado de %s (ronda %s)", h["sender_id"], h["round"])
                    ctx.counters["messages_rejected"] += 1
                    await rec.discard()
                    continue
                arrivals.mark(h["sender_id"], rec.arrived_ns)
                timer.add("deserialize", rec.t_deserialize_s)
                await rec.accept()
                updates[h["sender_id"]] = Update(rec.state, h["n_samples"], h["tau"], h["metrics"]["train_loss"])
                legacy[h["sender_id"]] = h["metrics"]
        missing = sorted(set(active) - set(updates))
        if not updates:
            log.error("[server] ronda %d sin actualizaciones", round_)
            await ctx.broadcast("STOP", round=round_, reason="no_updates")
            return 3

        with timer.phase("aggregate"):
            ordered = [updates[c] for c in sorted(updates)]
            global_state = aggregate(method, ordered, global_state, cfg["aggregation"])
        save_checkpoint(ctx.run_dir, ctx.node_id, round_, 0, "global", global_state,
                        {"clients": sorted(updates), "missing": missing, "method": method})

        recalls = [m["recall"] for m in legacy.values() if "recall" in m]
        legacy_recall = sum(recalls) / len(recalls) if recalls else None
        decision = policy.update(round_, legacy_recall if legacy_recall is not None else 0.0)

        cpu = proc.stop()
        nic1 = NicCounters.snapshot()
        tx, rx = nic0.delta(nic1)
        ctx.kpi.log({
            "role": "server", "method": method, "round": round_, "sub_round": 0,
            "t_round_s": timer.elapsed(), **{k: v for k, v in timer.as_columns().items() if k in ctx.kpi.columns},
            "nic_bytes_tx": tx, "nic_bytes_rx": rx, **ctx.take_counters(),
            "cpu_percent_legacy_before": cpu_before,
            "cpu_percent_legacy_after": cpu_percent_legacy(cfg["kpis"]["cpu"]["legacy_interval_s"]),
            **cpu, "ram_rss_mb_before": ram_before, "ram_rss_mb_after": ram_rss_mb(),
            "temperature_c": temperature_c(), "cpu_freq_mhz": cpu_freq_mhz(),
            "legacy_local_recall": legacy_recall,
            "e_comm_legacy_j": p_watts * sum(publish_s.values()),      # ≈ descargas de P1 (I-32)
            "straggler_s": arrivals.straggler_s(), "arrivals": arrivals.relative_s(),
        })
        log.info("[server] ronda %d: %d/%d actualizaciones, faltan %s, straggler %.3fs",
                 round_, len(updates), len(active), missing, arrivals.straggler_s() or 0.0)
        if decision.stop:
            await ctx.broadcast("STOP", round=round_, reason=decision.reason)
            log.info("[server] fin: %s", decision.reason)
            return 0
