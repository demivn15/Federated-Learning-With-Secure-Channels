"""Cliente saboteador para las pruebas de E5.

Se registra correctamente, luego publica un UPDATE cuya cabecera dice ser de CN1 (suplantación: el broker
valida su usuario real, distinto) y después desaparece sin entrenar. El servidor debe descartar el UPDATE
falso y no quedarse bloqueado más que `round_timeout_s`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from broker.provision_broker import NodeCredentials  # noqa: E402
from core.model import arch_hash, build_model  # noqa: E402
from experiments.manifest import load_manifest  # noqa: E402
from messaging.frame import Limits, encode_message  # noqa: E402
from messaging.transport_amqp import AmqpTransport  # noqa: E402


async def main_async(run_dir: Path, node: str) -> int:
    m = load_manifest(run_dir)
    cfg = m["config"]
    creds = NodeCredentials(**json.loads((run_dir / "credentials.json").read_text(encoding="utf-8"))[node])
    t = AmqpTransport(creds.amqp_url(), node, creds.username, cfg["communication"]["broker"], 4 * 1024 * 1024)
    n_features = cfg["data"]["preprocessing"]["expected_n_features"]
    model = build_model(cfg, n_features)

    def header(msg_type, sender, seq, **f):
        return {"schema_version": 1, "msg_type": msg_type, "experiment_id": m["experiment_id"],
                "topology": m["topology"], "sender_id": sender, "recipient_id": "infra", "seq": seq, **f}

    async with t:
        await t.send("infra", encode_message(header("REGISTER", node, 1, n_features=n_features,
                                                    arch_hash=arch_hash(model), n_samples=100)), "REGISTER")
        d = await t.recv(timeout=60)                       # REGISTER_ACK
        await d.ack()
        d = await t.recv(timeout=60)                       # ROUND_START de la ronda 1
        await d.ack()
        forged = encode_message(header("UPDATE", "CN1", 2, round=1, n_samples=10**6, tau=1,
                                       metrics={"train_loss": 0.0, "train_time_s": 0.0}), model.state_dict())
        await t.send("infra", forged, "UPDATE")            # suplantación de CN1
    return 0                                               # desaparece sin enviar su propio UPDATE


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--node", required=True)
    a = ap.parse_args()
    raise SystemExit(asyncio.run(main_async(Path(a.run_dir), a.node)))
