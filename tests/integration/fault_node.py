"""Nodo con un fallo inyectado, para las pruebas de integración (no forma parte del código de los nodos).

    python tests/integration/fault_node.py --fault crash_as_aggregator  <argumentos de node.main>
    python tests/integration/fault_node.py --fault train_delay=6        <argumentos de node.main>

Sustituye un método del nodo real antes de arrancarlo (`node.main`), de modo que el resto del protocolo
es exactamente el de producción:
  * `crash_as_aggregator` — el proceso muere en cuanto el coordinador SD lo nombra agregador;
  * `train_delay=S`       — cada entrenamiento local tarda S segundos más (straggler; ataque F7 de D3).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fault", required=True)
    args, rest = ap.parse_known_args()
    if args.fault == "crash_as_aggregator":
        from topologies.semi_decentralized import agent

        async def crash(self, assign):          # noqa: ANN001
            print(f"[fault] {self.ctx.node_id}: caída al ser nombrado agregador", flush=True)
            os._exit(0)

        agent.Agent.aggregate_round = crash
    elif args.fault.startswith("train_delay="):
        delay = float(args.fault.split("=", 1)[1])
        from topologies import local_worker
        original = local_worker.LocalWorker.train

        async def slow(self, *a, **k):         # noqa: ANN001
            res = await original(self, *a, **k)
            await asyncio.sleep(delay)
            return res

        local_worker.LocalWorker.train = slow
    elif args.fault.startswith("partition="):                 # P2P: aislado del broker S segundos en la ronda R
        at_round, seconds = (float(x) for x in args.fault.split("=", 1)[1].split(":"))
        from topologies.p2p import peer
        original_step = peer.Peer.step

        async def step(self):                  # noqa: ANN001
            if self.round + 1 == int(at_round) and not getattr(self, "_partitioned", False):
                self._partitioned = True
                print(f"[fault] {self.me}: desconectado del broker {seconds:.0f} s", flush=True)
                await self.ctx.transport.close()
                await asyncio.sleep(seconds)
                await self.ctx.transport.connect()
            return await original_step(self)

        peer.Peer.step = step
    elif args.fault.startswith("start_delay="):               # arranque tardío
        import time
        time.sleep(float(args.fault.split("=", 1)[1]))
    else:
        raise SystemExit(f"Fallo desconocido: {args.fault}")
    from node.main import main as node_main
    return node_main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
