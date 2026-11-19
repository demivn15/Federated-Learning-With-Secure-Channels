"""Micro-benchmark WebSocket (transporte de D0) vs AMQP/RabbitMQ (RF-32, 06 E3).

Mide el tiempo de ida y vuelta (RTT) de la MISMA trama `UPDATE` (modelo de referencia, ≈ 453 KB) con eco:
  * WebSocket: cliente ↔ servidor de eco, conexión persistente (patrón de P1). Dos variantes:
    con `deflate` (valor por defecto de `websockets`, el que usaron P1–P3 y por tanto D0) y sin compresión.
  * AMQP: nodo A → broker → nodo B (eco) → broker → nodo A. Cuatro tramos de red en total, frente a
    los dos de WebSocket (un salto por sentido).
El RTT se mide con un único reloj (el del emisor), así que no depende de la sincronización entre máquinas (RNF-07).

Uso local (broker de docker/compose.yaml):
    python scripts/bench_transport.py --n 50
En el clúster: ejecutar el eco en una Pi y el emisor en otra (--ws-host / --amqp-host) para medir la red real.

AVISO: en el PC de desarrollo el broker corre en la VM de Docker Desktop y el servidor WebSocket en el
sistema anfitrión, así que la comparación local está sesgada. Los valores que cuentan son los de las Pi (E11).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import websockets  # noqa: E402  (solo para este instrumento de medición)

from broker.provision_broker import DEV_AMQP_PORT, admin_from_env  # noqa: E402
from core.config import load_config  # noqa: E402
from core.model import build_model  # noqa: E402
from messaging.frame import encode_message  # noqa: E402
from messaging.transport_amqp import AmqpTransport  # noqa: E402


def make_frame(cfg: dict, n_features: int) -> bytes:
    state = {k: v.detach().clone() for k, v in build_model(cfg, n_features).state_dict().items()}
    header = {"schema_version": 1, "msg_type": "UPDATE", "experiment_id": "bench", "topology": "centralized",
              "sender_id": "bench_a", "recipient_id": "bench_b", "seq": 0, "round": 1, "n_samples": 4643,
              "tau": 873, "metrics": {"train_loss": 0.4, "train_time_s": 10.0}}
    return encode_message(header, state)


def summarize(rtts: list[float]) -> dict[str, float]:
    s = sorted(rtts)
    return {"n": len(s), "median_ms": 1e3 * statistics.median(s), "mean_ms": 1e3 * statistics.fmean(s),
            "stdev_ms": 1e3 * (statistics.stdev(s) if len(s) > 1 else 0.0),
            "p95_ms": 1e3 * s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))],
            "min_ms": 1e3 * s[0], "max_ms": 1e3 * s[-1]}


async def bench_websocket(frame: bytes, n: int, warmup: int, host: str, port: int,
                          compression: str | None) -> dict:
    """`compression='deflate'` es el valor por defecto de `websockets`, el que usaron P1/P2/P3 (no lo cambian)."""
    async def echo(ws):
        async for msg in ws:
            await ws.send(msg)

    server = await websockets.serve(echo, host, port, max_size=8 * 1024 * 1024, compression=compression)
    try:
        async with websockets.connect(f"ws://{host}:{port}", max_size=8 * 1024 * 1024,
                                      compression=compression) as ws:
            rtts = []
            for i in range(n + warmup):
                t0 = time.perf_counter()
                await ws.send(frame)
                reply = await ws.recv()
                dt = time.perf_counter() - t0
                assert len(reply) == len(frame)
                if i >= warmup:
                    rtts.append(dt)
    finally:
        server.close()
        await server.wait_closed()
    return summarize(rtts)


async def bench_amqp(frame: bytes, n: int, warmup: int, host: str, port: int, cfg: dict) -> dict:
    bcfg = cfg["communication"]["broker"]
    limit = cfg["communication"]["limits"]["max_frame_bytes"]
    admin = admin_from_env()
    exp_id = f"bench{uuid.uuid4().hex[:8]}"
    creds = admin.provision_experiment(exp_id, ["bench_a", "bench_b"], bcfg)
    a = AmqpTransport(creds["bench_a"].amqp_url(host, port), "bench_a", creds["bench_a"].username, bcfg, limit)
    b = AmqpTransport(creds["bench_b"].amqp_url(host, port), "bench_b", creds["bench_b"].username, bcfg, limit)
    await a.connect(); await b.connect()

    async def echo():
        while True:
            d = await b.recv()
            await b.send("bench_a", d.body, d.msg_type or "UPDATE")
            await d.ack()

    echo_task = asyncio.create_task(echo())
    try:
        rtts, confirms = [], []
        for i in range(n + warmup):
            t0 = time.perf_counter()
            st = await a.send("bench_b", frame, "UPDATE")
            d = await a.recv(timeout=30)
            dt = time.perf_counter() - t0
            await d.ack()
            assert len(d.body) == len(frame)
            if i >= warmup:
                rtts.append(dt)
                confirms.append(st.confirm_s)
        mem = admin.node_memory()
    finally:
        echo_task.cancel()
        await a.close(); await b.close()
        admin.delete_experiment(exp_id)
    out = summarize(rtts)
    out["publish_confirm_median_ms"] = 1e3 * statistics.median(confirms)
    out["broker_mem_used_mib"] = round(mem["mem_used"] / 2 ** 20, 1)
    out["broker"] = admin.overview()
    return out


async def main_async(args) -> dict:
    cfg = load_config()
    frame = make_frame(cfg, args.n_features)
    ws_d0 = await bench_websocket(frame, args.n, args.warmup, args.ws_host, args.ws_port, "deflate")
    ws_raw = await bench_websocket(frame, args.n, args.warmup, args.ws_host, args.ws_port + 1, None)
    amqp = await bench_amqp(frame, args.n, args.warmup, args.amqp_host, args.amqp_port, cfg)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(), "platform": platform.platform(), "python": platform.python_version(),
        "frame_bytes": len(frame), "n": args.n, "warmup": args.warmup,
        "setup": args.setup,
        "websocket_deflate_d0": ws_d0, "websocket_no_compression": ws_raw, "amqp": amqp,
        "amqp_vs_websocket_d0_median_ratio": amqp["median_ms"] / ws_d0["median_ms"],
        "amqp_vs_websocket_raw_median_ratio": amqp["median_ms"] / ws_raw["median_ms"],
        "note": ("RTT con eco medido con el reloj del emisor. AMQP = 4 tramos (vía broker); WebSocket = 2. "
                 "websocket_deflate_d0 reproduce el transporte de D0 (compresión por defecto de websockets)."),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--n-features", type=int, default=278)
    ap.add_argument("--ws-host", default="127.0.0.1")
    ap.add_argument("--ws-port", type=int, default=8765)
    ap.add_argument("--amqp-host", default="127.0.0.1")
    ap.add_argument("--amqp-port", type=int, default=DEV_AMQP_PORT)
    ap.add_argument("--setup", default="dev-pc: broker en Docker Desktop; WebSocket nativo (comparación sesgada)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "bench"))
    args = ap.parse_args(argv)

    result = asyncio.run(main_async(args))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"transport_{platform.node()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Trama: {result['frame_bytes']} B · n={args.n} (+{args.warmup} de calentamiento) · {args.setup}")
    print(f"{'':10}{'mediana':>10}{'p95':>10}{'media':>10}{'desv':>10}{'mín':>9}{'máx':>9}  (ms, RTT)")
    for name, label in (("websocket_deflate_d0", "WS deflate"), ("websocket_no_compression", "WS crudo"),
                        ("amqp", "AMQP")):
        r = result[name]
        print(f"{label:10}{r['median_ms']:>10.2f}{r['p95_ms']:>10.2f}{r['mean_ms']:>10.2f}"
              f"{r['stdev_ms']:>10.2f}{r['min_ms']:>9.2f}{r['max_ms']:>9.2f}")
    a = result["amqp"]
    print(f"AMQP / WS deflate (D0): {result['amqp_vs_websocket_d0_median_ratio']:.2f}× · "
          f"AMQP / WS crudo: {result['amqp_vs_websocket_raw_median_ratio']:.2f}× · confirmación de "
          f"publicación (mediana) {a['publish_confirm_median_ms']:.2f} ms · memoria del broker "
          f"{a['broker_mem_used_mib']} MiB · {a['broker']}")
    print(f"Resultado: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
