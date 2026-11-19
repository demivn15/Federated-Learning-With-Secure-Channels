"""Coordinador pasivo de la semi-descentralizada (RF-03; D0 §3.3.3, §4.3; 04 ADR-08, ADR-12). 6.ª Pi.

No entrena, no agrega y **nunca recibe modelos**: solo acepta `REGISTER`, `CLAIM` y `AGG_COMPLETE`
(contratos sin payload), así que un mensaje con modelo dirigido a él se rechaza en la ruta de recepción.

Ronda externa r:
  1. `ELECTION(r, intento, selector, candidatos[, puntuaciones])` a cada Client Node registrado.
  2. Cada CN calcula el ganador con la semilla compartida y responde `CLAIM(r, intento, ganador, telemetría)`.
     El coordinador calcula lo mismo y **verifica** que todos los `CLAIM` coinciden (`sd_claims_agree`).
     El tiempo de respuesta es un *health check* y una variable observada de `SD-score`.
  3. `ROLE_ASSIGN` a cada uno: agregador o trabajador, con la lista de trabajadores y los hiperparámetros.
  4. Espera `AGG_COMPLETE(r)` del agregador hasta `outer_round_timeout_s` (T_outer). Trae el recall local
     medio de los trabajadores en la última ronda interna (métrica de parada de P3) y su retraso de llegada.
Con salvaguardas, si el ganador no responde o el agregador no completa a tiempo, repite la elección de la
misma ronda sin él (re-selección). Sin salvaguardas, la ronda se pierde y la métrica queda congelada
(el fallo en cascada de D0 §4.3). Parada: *patience* sobre el recall (`core.stopping`).
"""
from __future__ import annotations

import logging
import time

from core.model import build_model
from core.stopping import build_round_policy
from telemetry.clock import PhaseTimer
from telemetry.system import NicCounters, ProcessCpu, cpu_freq_mhz, ram_rss_mb, temperature_c
from topologies.base import NodeContext
from topologies.centralized.server import accept_registrations, training_params
from topologies.semi_decentralized.safeguards import Observations, SafeguardState
from topologies.semi_decentralized.selectors import hybrid_scores, select

log = logging.getLogger("fl.sd.coordinator")


async def run_coordinator(ctx: NodeContext) -> int:
    cfg, topo = ctx.cfg, ctx.cfg["topologies"]["semi_decentralized"]
    method = ctx.manifest["method"]
    candidates_cfg = sorted(topo["candidates"])
    reference = build_model(cfg, cfg["data"]["preprocessing"]["expected_n_features"])
    registered = sorted(await accept_registrations(ctx, candidates_cfg, float(topo["register_timeout_s"]),
                                                   reference))
    if len(registered) < int(topo["min_clients"]):
        log.error("[coordinator] solo %d/%d Client Nodes registrados", len(registered), topo["min_clients"])
        await ctx.broadcast("STOP", round=0, reason="insufficient_clients")
        return 2

    guard = SafeguardState.from_config(registered, topo.get("safeguards"))
    obs = Observations()
    selector, seed = topo["selector"], int(topo["selector_seed"])
    weights = topo.get("sd_score", {}).get("weights", {})
    params = training_params(cfg, method)
    internal_rounds = int(topo["internal_rounds"])
    policy = build_round_policy(topo["rounds"])
    last_metric, prev_agg, round_ = 0.0, None, 0
    while True:
        round_ += 1
        timer, proc, nic0 = PhaseTimer(), ProcessCpu(), NicCounters.snapshot()
        proc.start()
        excluded: set[str] = set()
        attempt, completed, agg, agree_all = 0, None, None, True
        while completed is None:
            attempt += 1
            cands = guard.candidates(excluded)
            if not cands:
                log.error("[coordinator] ronda %d: sin candidatos", round_)
                break
            scores = hybrid_scores(cands, obs.observed(), obs.declared, weights) \
                if selector == "sd_score_hybrid" else None
            expected = select(selector, cands, seed, round_, attempt, scores)

            # 1–2) elección y verificación
            sent_at: dict[str, int] = {}
            for n in registered:
                await ctx.send_or_skip(n, "ELECTION", round=round_, attempt=attempt, selector=selector, candidates=cands,
                               **({"scores": scores} if scores is not None else {}))
                sent_at[n] = time.perf_counter_ns()
            claims: dict[str, str] = {}
            deadline = time.monotonic() + float(topo["election_timeout_s"])
            with timer.phase("election"):
                while len(claims) < len(registered):
                    rec = await ctx.receive_from(deadline - time.monotonic(), {"CLAIM": set(registered)})
                    if rec is None:
                        break
                    h = rec.header
                    if h["round"] != round_ or h["attempt"] != attempt or rec.sender in claims:
                        await rec.discard()                      # CLAIM tardío de un intento anterior
                        continue
                    await rec.accept()
                    claims[rec.sender] = h["winner"]
                    obs.add_rtt(rec.sender, (rec.arrived_ns - sent_at[rec.sender]) / 1e9)
                    obs.add_declared(rec.sender, h["telemetry"])
            agree = all(w == expected for w in claims.values())
            agree_all &= agree
            if not agree:
                log.warning("[coordinator] ronda %d.%d: CLAIM discrepantes %s (esperado %s)", round_, attempt,
                            claims, expected)
            guard.observe_claims(set(claims))
            if guard.enabled and expected not in claims:
                log.warning("[coordinator] ronda %d.%d: %s no responde ⇒ re-selección", round_, attempt, expected)
                excluded.add(expected)
                continue

            # 3) asignación de roles
            agg = expected
            pool = guard.alive if guard.enabled else set(registered)
            workers = sorted(pool - {agg})
            for n in sorted({agg, *workers}):
                await ctx.send_or_skip(n, "ROLE_ASSIGN", round=round_, attempt=attempt,
                               role="aggregator" if n == agg else "worker", aggregator_id=agg, workers=workers,
                               internal_rounds=internal_rounds, training=params)
            log.info("[coordinator] ronda %d.%d: agregador %s (%s), trabajadores %s", round_, attempt, agg,
                     selector, workers)

            # 4) espera del agregador (T_outer)
            deadline = time.monotonic() + float(topo["outer_round_timeout_s"])
            with timer.phase("wait_aggregator"):
                while True:
                    rec = await ctx.receive_from(deadline - time.monotonic(),
                                                 {"AGG_COMPLETE": {agg}, "CLAIM": set(registered)})
                    if rec is None:
                        break
                    h = rec.header
                    if h["msg_type"] == "AGG_COMPLETE" and h["round"] == round_ and h["attempt"] == attempt:
                        await rec.accept()
                        completed = h
                        break
                    await rec.discard()
            if completed is None:
                if not guard.enabled:
                    log.error("[coordinator] ronda %d: el agregador %s no completó (sin salvaguardas: ronda "
                              "perdida, métrica congelada)", round_, agg)
                    break
                log.warning("[coordinator] ronda %d.%d: %s no completó en T_outer ⇒ re-selección", round_, attempt,
                            agg)
                excluded.add(agg)
                guard.mark_failed(agg)

        if completed is not None:
            guard.record_aggregator(agg)
            obs.add_lateness(completed["lateness"])
            if completed["recall"] is not None:
                last_metric = float(completed["recall"])
        decision = policy.update(round_, last_metric)

        cpu = proc.stop()
        tx, rx = nic0.delta(NicCounters.snapshot())
        ctx.kpi.log({
            "role": "coordinator", "method": method, "round": round_, "sub_round": 0,
            "t_round_s": timer.elapsed(), "nic_bytes_tx": tx, "nic_bytes_rx": rx, **ctx.take_counters(), **cpu,
            "ram_rss_mb_after": ram_rss_mb(), "temperature_c": temperature_c(), "cpu_freq_mhz": cpu_freq_mhz(),
            "legacy_local_recall": last_metric,
            "straggler_s": completed["straggler_s"] if completed else None,
            "sd_aggregator_id": agg if completed else None,
            "sd_aggregator_changed": (agg != prev_agg) if completed and prev_agg else None,
            "sd_selector": selector, "sd_claims_agree": agree_all, "sd_attempts": attempt,
            "sd_round_completed": completed is not None,
        })
        if completed is not None:
            prev_agg = agg
        log.info("[coordinator] ronda %d: %s, recall %.4f", round_,
                 f"completada por {agg}" if completed else "PERDIDA", last_metric)
        if decision.stop or (completed is None and not guard.candidates(set())):
            reason = decision.reason or "no_candidates"
            await ctx.broadcast("STOP", round=round_, reason=reason)
            log.info("[coordinator] fin: %s", reason)
            return 0 if decision.stop else 3
