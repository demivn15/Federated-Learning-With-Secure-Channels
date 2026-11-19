"""Punto de entrada único de un nodo (analisis_arquitectonico/05 §3.6).

    python -m node.main --run-dir results/runs/<exp> --node CN1 [--processed data/processed]
                        [--amqp-host 127.0.0.1 --amqp-port 5673] [--threads N]

Lee el manifiesto de la corrida (configuración efectiva, topología, método), verifica que los datos y el
código son los registrados, se conecta al broker con sus credenciales y ejecuta el rol que le toca.
Sustituye a los `run_*.sh` de los proyectos base, que editaban `config.py` con `sed -i` (P2-H5).

Códigos de salida: 0 bien · 2 registro fallido · 3 ronda sin actualizaciones · 4 sin noticias del servidor ·
5 error de configuración.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from broker.provision_broker import DEV_AMQP_PORT, NodeCredentials  # noqa: E402
from core.config import PROJECT_ROOT  # noqa: E402
from experiments.manifest import load_manifest, verify_inputs  # noqa: E402
from messaging.contracts import ContractValidator  # noqa: E402
from messaging.frame import Limits  # noqa: E402
from messaging.transport_amqp import AmqpTransport  # noqa: E402
from telemetry.kpi_log import KpiLogger, kpi_path  # noqa: E402
from topologies.base import NodeContext  # noqa: E402

log = logging.getLogger("fl.node")


def clients_of(cfg: dict) -> list[str]:
    return sorted(n for n in cfg["platform"]["node_assignment"] if n != "infra")


def role_of(cfg: dict, topology: str, node_id: str) -> str:
    if topology == "centralized":
        return "server" if node_id == "infra" else "client"
    if topology == "hierarchical":
        from topologies.hierarchical.tree import build_tree
        return build_tree(cfg["topologies"]["hierarchical"], clients_of(cfg)).role_of(node_id)
    if topology == "semi_decentralized":
        return "coordinator" if node_id == cfg["topologies"]["semi_decentralized"]["coordinator"] else "agent"
    if topology == "p2p":
        return "peer" if node_id in cfg["topologies"]["p2p"]["peers"] else "none"
    raise NotImplementedError(f"Topología desconocida: {topology}")


async def run_role(ctx: NodeContext, role: str, processed: Path) -> int:
    topology = ctx.topology
    if topology == "centralized":
        if role == "server":
            from topologies.centralized.server import run_server
            return await run_server(ctx)
        from topologies.centralized.client import run_client
        return await run_client(ctx, processed)
    if topology == "hierarchical":
        from topologies.hierarchical import node as h
        from topologies.hierarchical.tree import build_tree
        tree = build_tree(ctx.cfg["topologies"]["hierarchical"], clients_of(ctx.cfg))
        if role == "root":
            return await h.run_root(ctx, tree)
        if role == "cluster_aggregator":
            return await h.run_cluster_aggregator(ctx, tree, processed)
        return await h.run_leaf(ctx, tree, processed)
    if topology == "semi_decentralized":
        if role == "coordinator":
            from topologies.semi_decentralized.coordinator import run_coordinator
            return await run_coordinator(ctx)
        from topologies.semi_decentralized.agent import run_agent
        return await run_agent(ctx, processed)
    if topology == "p2p":
        if role == "none":
            return 0                                     # la 6.ª Pi solo aloja el broker
        from topologies.p2p.peer import run_peer
        return await run_peer(ctx, processed)
    raise NotImplementedError(topology)


async def run(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    manifest = load_manifest(run_dir)
    verify_inputs(manifest, args.processed, check_code=not args.skip_code_check)
    cfg = manifest["config"]
    if args.node not in cfg["platform"]["node_assignment"]:
        log.error("Nodo desconocido: %s", args.node)
        return 5
    torch.manual_seed(int(cfg["data"]["seed"]))
    creds = NodeCredentials(**json.loads((run_dir / "credentials.json").read_text(encoding="utf-8"))[args.node])
    limits = Limits.from_config(cfg["communication"])
    transport = AmqpTransport(creds.amqp_url(args.amqp_host, args.amqp_port), args.node, creds.username,
                              cfg["communication"]["broker"], limits.max_frame_bytes)
    kpi = KpiLogger(kpi_path(run_dir, args.node), topology=manifest["topology"],
                    experiment_id=manifest["experiment_id"], manifest_sha256=manifest["manifest_sha256"],
                    node_id=args.node)
    validator = ContractValidator(enabled=cfg["communication"].get("schema_validation", True))
    role = role_of(cfg, manifest["topology"], args.node)
    async with transport:
        ctx = NodeContext(cfg, manifest, run_dir, args.node, transport, validator, limits, kpi)
        log.info("[%s] rol %s · %s · %s · %s", args.node, role, manifest["topology"], manifest["method"],
                 manifest["experiment_id"])
        return await run_role(ctx, role, Path(args.processed))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--node", required=True)
    ap.add_argument("--processed", default=str(PROJECT_ROOT / "data" / "processed"))
    ap.add_argument("--amqp-host", default="127.0.0.1")
    ap.add_argument("--amqp-port", type=int, default=DEV_AMQP_PORT)
    ap.add_argument("--threads", type=int, default=None, help="hilos de PyTorch (varios nodos en un mismo PC)")
    ap.add_argument("--skip-code-check", action="store_true", help="no comprobar el hash del código (desarrollo)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("aio_pika").setLevel(logging.WARNING)
    logging.getLogger("aiormq").setLevel(logging.WARNING)
    if args.threads:
        torch.set_num_threads(args.threads)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
