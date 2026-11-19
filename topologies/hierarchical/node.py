"""Orquestadores de la topología jerárquica (RF-02; D0 §3.3.4; 04 ADR-09, ADR-12).

Ronda r (redistribución `via_cluster_aggregators`; con `direct` la raíz envía el ROUND_START a todos):

    raíz ──ROUND_START(M_t, members)──▶ agregador de clúster ──ROUND_START(M_t)──▶ hojas
    hojas: entrenan ──UPDATE(M_k, n_k, τ_k)──▶ agregador
    agregador: entrena también (doble rol de D0), espera a sus hojas hasta `cluster_round_timeout_s`
               (contado desde que recibe M_t), agrega el clúster (core.aggregation.aggregate_cluster)
               ──CLUSTER_UPDATE(M_c, N_c, τ̄_c)──▶ raíz
    raíz: espera los clústeres hasta `round_timeout_s`, aplica la regla del método (Eq. 1–3) ⇒ M_{t+1}

Fallos (ADR-12): una hoja caída retrasa a su clúster como máximo `cluster_round_timeout_s` y el clúster
agrega con lo recibido; un agregador de clúster caído deja a su clúster sin aportar en esa ronda; la raíz
es SPOF (como el servidor de la centralizada). `round_timeout_s` debe ser mayor que
`cluster_round_timeout_s` para que un clúster con una hoja caída llegue a tiempo.

Straggler (ADR-11): la raíz mide la dispersión de llegada de los modelos de clúster; cada agregador, la de
los modelos de su clúster (incluido el suyo, al terminar de entrenar). Ambas con el reloj del que mide.
Checkpoints: `global` en la raíz; `cluster` en cada agregador (D0: "1 global + 2 de clúster").
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from core.aggregation import aggregate, aggregate_cluster
from core.model import build_model, clone_state
from core.stopping import build_round_policy
from core.training import Update
from messaging.frame import prepare_payload, tensor_specs
from node.checkpoint import save_checkpoint
from telemetry.clock import ArrivalClock, PhaseTimer
from telemetry.system import NicCounters, ProcessCpu, cpu_freq_mhz, cpu_percent_legacy, ram_rss_mb, temperature_c
from topologies.base import NodeContext
from topologies.centralized.client import run_client
from topologies.centralized.server import accept_registrations, training_params
from topologies.hierarchical.tree import Tree
from topologies.local_worker import LocalWorker

log = logging.getLogger("fl.hierarchical")


# ===================================================================== raíz
async def run_root(ctx: NodeContext, tree: Tree) -> int:
    cfg, topo = ctx.cfg, ctx.cfg["topologies"]["hierarchical"]
    method = ctx.manifest["method"]
    clients = sorted(m for c in tree.clusters for m in c.members)
    reference = build_model(cfg, cfg["data"]["preprocessing"]["expected_n_features"])

    registered = await accept_registrations(ctx, clients, float(topo["register_timeout_s"]), reference)
    active = {c.cluster_id: [m for m in c.members if m in registered] for c in tree.clusters
              if c.aggregator in registered}
    aggregator_of = {c.cluster_id: c.aggregator for c in tree.clusters}
    if len(registered) < int(topo["min_clients"]) or not active:
        log.error("[root] registrados %d/%d (clústeres con agregador: %s)", len(registered), len(clients),
                  sorted(active))
        await ctx.broadcast("STOP", round=0, reason="insufficient_clients")
        return 2

    global_state = clone_state(reference.state_dict())
    specs = tensor_specs(global_state)
    params = training_params(cfg, method)
    policy = build_round_policy(topo["rounds"])
    p_watts = float(cfg["energy"]["e_comm"]["p_watts"])
    round_ = 0
    while True:
        round_ += 1
        timer, proc, arrivals = PhaseTimer(), ProcessCpu(), ArrivalClock()
        cpu_before, ram_before, nic0 = (cpu_percent_legacy(cfg["kpis"]["cpu"]["legacy_interval_s"]), ram_rss_mb(),
                                        NicCounters.snapshot())
        proc.start()

        with timer.phase("serialize"):
            payload = prepare_payload(global_state)
        publish_s = 0.0
        for cid, members in sorted(active.items()):
            agg = aggregator_of[cid]
            recipients = [agg] if tree.redistribution == "via_cluster_aggregators" else members
            for n in recipients:
                extra = {"members": members} if n == agg else {}
                sent = await ctx.send_or_skip(n, "ROUND_START", payload, round=round_, training=params, **extra)
                publish_s += sent.publish.confirm_s if sent is not None else 0.0
        timer.add("publish_confirm", publish_s)

        received: dict[str, Update] = {}
        missing_nodes: list[str] = []
        expected_aggs = {aggregator_of[cid]: cid for cid in active}
        deadline = time.monotonic() + float(topo["round_timeout_s"])
        with timer.phase("wait_model"):
            while len(received) < len(active):
                rec = await ctx.receive_from(deadline - time.monotonic(),
                                             {"CLUSTER_UPDATE": set(expected_aggs)}, specs)
                if rec is None:
                    break
                h = rec.header
                cid = expected_aggs[h["sender_id"]]
                if h["round"] != round_ or h["cluster_id"] != cid or cid in received \
                        or not set(h["members"]) <= set(active[cid]):
                    log.warning("[root] CLUSTER_UPDATE descartado de %s (ronda %s, clúster %s)", h["sender_id"],
                                h["round"], h["cluster_id"])
                    ctx.counters["messages_rejected"] += 1
                    await rec.discard()
                    continue
                arrivals.mark(cid, rec.arrived_ns)
                timer.add("deserialize", rec.t_deserialize_s)
                await rec.accept()
                received[cid] = Update(rec.state, h["n_samples"], h["tau"], h["metrics"]["train_loss"])
                missing_nodes += h["missing"]
        missing_clusters = sorted(set(active) - set(received))
        missing_nodes = sorted(set(missing_nodes) | {m for cid in missing_clusters for m in active[cid]})
        if not received:
            log.error("[root] ronda %d sin modelos de clúster", round_)
            await ctx.broadcast("STOP", round=round_, reason="no_updates")
            return 3

        with timer.phase("aggregate"):
            global_state = aggregate(method, [received[c] for c in sorted(received)], global_state,
                                     cfg["aggregation"])
        save_checkpoint(ctx.run_dir, ctx.node_id, round_, 0, "global", global_state,
                        {"clusters": sorted(received), "missing_clusters": missing_clusters,
                         "missing": missing_nodes, "method": method})
        decision = policy.update(round_, None)

        cpu = proc.stop()
        tx, rx = nic0.delta(NicCounters.snapshot())
        ctx.kpi.log({
            "role": "root", "method": method, "round": round_, "sub_round": 0,
            "t_round_s": timer.elapsed(), **{k: v for k, v in timer.as_columns().items() if k in ctx.kpi.columns},
            "nic_bytes_tx": tx, "nic_bytes_rx": rx, **ctx.take_counters(),
            "cpu_percent_legacy_before": cpu_before,
            "cpu_percent_legacy_after": cpu_percent_legacy(cfg["kpis"]["cpu"]["legacy_interval_s"]),
            **cpu, "ram_rss_mb_before": ram_before, "ram_rss_mb_after": ram_rss_mb(),
            "temperature_c": temperature_c(), "cpu_freq_mhz": cpu_freq_mhz(),
            "e_comm_legacy_j": p_watts * publish_s,
            "straggler_s": arrivals.straggler_s(), "arrivals": arrivals.relative_s(),
            "cluster_n_samples": sum(u.n_samples for u in received.values()),
        })
        log.info("[root] ronda %d: clústeres %s, faltan %s (nodos %s), straggler %.3fs", round_, sorted(received),
                 missing_clusters, missing_nodes, arrivals.straggler_s() or 0.0)
        if decision.stop:
            await ctx.broadcast("STOP", round=round_, reason=decision.reason)
            log.info("[root] fin: %s", decision.reason)
            return 0


# ===================================================================== agregador de clúster
async def run_cluster_aggregator(ctx: NodeContext, tree: Tree, processed_dir: Path) -> int:
    cfg, topo = ctx.cfg, ctx.cfg["topologies"]["hierarchical"]
    cluster = tree.cluster_of(ctx.node_id)
    worker = LocalWorker(cfg, ctx.node_id, processed_dir)

    await ctx.send(tree.root, "REGISTER", **worker.register_fields())
    ack = await ctx.receive_from(float(topo["register_timeout_s"]), {"REGISTER_ACK": {tree.root},
                                                                      "STOP": {tree.root}})
    if ack is None or ack.header["msg_type"] == "STOP" or not ack.header["accepted"]:
        if ack is not None:
            await ack.accept()
        log.error("[%s] registro rechazado o sin respuesta", ctx.node_id)
        return 2
    await ack.accept()

    waiting_since = time.perf_counter()
    while True:
        cpu_before, ram_before = worker.cpu_before()
        rec = await ctx.receive_from(float(topo["client_round_wait_s"]),
                                     {"ROUND_START": {tree.root}, "STOP": {tree.root}}, worker.specs)
        if rec is None:
            log.error("[%s] no llegó la siguiente ronda", ctx.node_id)
            return 4
        await rec.accept()
        if rec.header["msg_type"] == "STOP":
            log.info("[%s] STOP: %s", ctx.node_id, rec.header["reason"])
            return 0

        h = rec.header
        round_, training = h["round"], h["training"]
        members = [m for m in h.get("members", [ctx.node_id]) if m in cluster.members]
        leaves = [m for m in members if m != ctx.node_id]
        deadline = time.monotonic() + float(topo["cluster_round_timeout_s"])
        timer, proc, nic0, arrivals = PhaseTimer(), ProcessCpu(), NicCounters.snapshot(), ArrivalClock()
        timer.add("wait_model", time.perf_counter() - waiting_since)
        timer.add("deserialize", rec.t_deserialize_s)
        proc.start()
        t_round0 = time.perf_counter()

        # 1) redistribución dentro del clúster (antes de entrenar, para que las hojas trabajen en paralelo)
        publish_s = 0.0
        if tree.redistribution == "via_cluster_aggregators" and leaves:
            with timer.phase("serialize"):
                payload = prepare_payload(rec.state)
            for leaf in leaves:
                sent = await ctx.send_or_skip(leaf, "ROUND_START", payload, round=round_, training=training)
                publish_s += sent.publish.confirm_s if sent is not None else 0.0

        # 2) entrenamiento propio (doble rol de D0)
        updates: dict[str, Update] = {}
        res = None
        if tree.aggregators_train:
            res = await worker.train(rec.state, training, timer)
            updates[ctx.node_id] = res.update
            arrivals.mark(ctx.node_id)
            if cfg.get("checkpoints", {}).get("local"):
                save_checkpoint(ctx.run_dir, ctx.node_id, round_, 0, "local", res.update.state,
                                {"n_samples": res.update.n_samples, "tau": res.update.tau})

        # 3) modelos de las hojas
        with timer.phase("wait_model"):
            while len(updates) < len(members):
                got = await ctx.receive_from(deadline - time.monotonic(),
                                             {"UPDATE": set(leaves), "STOP": {tree.root}}, worker.specs)
                if got is None:
                    break
                g = got.header
                if g["msg_type"] == "STOP":
                    await got.accept()
                    log.info("[%s] STOP durante la ronda %d: %s", ctx.node_id, round_, g["reason"])
                    return 0
                if g["round"] != round_ or g["sender_id"] in updates:
                    log.warning("[%s] UPDATE descartado de %s (ronda %s)", ctx.node_id, g["sender_id"], g["round"])
                    ctx.counters["messages_rejected"] += 1
                    await got.discard()
                    continue
                arrivals.mark(g["sender_id"], got.arrived_ns)
                timer.add("deserialize", got.t_deserialize_s)
                await got.accept()
                updates[g["sender_id"]] = Update(got.state, g["n_samples"], g["tau"], g["metrics"]["train_loss"])
        missing = sorted(set(members) - set(updates))
        if not updates:
            log.error("[%s] ronda %d: clúster %s sin modelos", ctx.node_id, round_, cluster.cluster_id)
            waiting_since = time.perf_counter()
            continue

        # 4) modelo del clúster → raíz
        with timer.phase("aggregate"):
            cu = aggregate_cluster(training["method"], [updates[n] for n in sorted(updates)], rec.state,
                                   cfg["aggregation"])
        save_checkpoint(ctx.run_dir, ctx.node_id, round_, 0, "cluster", cu.state,
                        {"cluster_id": cluster.cluster_id, "members": sorted(updates), "missing": missing,
                         "n_samples": cu.n_samples, "tau": cu.tau})
        sent = await ctx.send_or_skip(tree.root, "CLUSTER_UPDATE", cu.state, round=round_,
                                      cluster_id=cluster.cluster_id, n_samples=cu.n_samples, tau=cu.tau,
                                      sum_n_tau=cu.n_samples * cu.tau, members=sorted(updates), missing=missing,
                                      metrics={"train_loss": cu.train_loss, "train_time_s": timer.seconds("train")}) \
            or ctx.unsent()
        timer.add("serialize", sent.t_serialize_s)
        publish_s += sent.publish.confirm_s
        timer.add("publish_confirm", publish_s)
        waiting_since = time.perf_counter()

        cpu = proc.stop()
        tx, rx = nic0.delta(NicCounters.snapshot())
        row = {"role": "cluster_aggregator", "method": training["method"], "round": round_, "sub_round": 0,
               "t_round_s": time.perf_counter() - t_round0, **timer.as_columns(),
               "nic_bytes_tx": tx, "nic_bytes_rx": rx, **ctx.take_counters(), **cpu,
               "straggler_s": arrivals.straggler_s(), "arrivals": arrivals.relative_s(),
               "cluster_id": cluster.cluster_id, "cluster_n_samples": cu.n_samples}
        if res is not None:
            # E_comm del nodo = todo lo que publica con modelo en la ronda (redistribución + subida)
            row.update(worker.kpi_columns(res, cpu_before=cpu_before, ram_before=ram_before,
                                          upload_confirm_s=publish_s, upload_bytes=sent.nbytes))
        ctx.kpi.log(row)
        log.info("[%s] ronda %d: clúster %s con %s, faltan %s, straggler %.3fs", ctx.node_id, round_,
                 cluster.cluster_id, sorted(updates), missing, arrivals.straggler_s() or 0.0)


# ===================================================================== hoja
async def run_leaf(ctx: NodeContext, tree: Tree, processed_dir: Path) -> int:
    cluster = tree.cluster_of(ctx.node_id)
    return await run_client(ctx, processed_dir, topology_key="hierarchical", registrar=tree.root,
                            upstream=tree.parent_of(ctx.node_id), round_start_from=tree.round_start_sender(ctx.node_id),
                            role="leaf", extra_kpi={"cluster_id": cluster.cluster_id})
