"""Checkpoints por ronda (RF-23) y entrada del evaluador fuera de línea (ADR-07).

Formato: `<run_dir>/checkpoints/<node>/r<round>_s<sub_round>_<kind>.pkl`, con el mismo pickle restringido
de la red (`messaging.serialization`), más un índice `index.jsonl` con el sha256 de cada archivo. Los
checkpoints se copian desde las Pi al PC de control, así que se cargan como datos no confiables
(con hash y deserializador restringido).

kind:
  global  — modelo global tras la agregación (C, H: raíz; SD: modelo de la ronda interna)
  cluster — modelo de clúster (H)
  mixed   — modelo de un par tras la mezcla (P2P)
  local   — modelo local tras entrenar (opcional; métrica heredada)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from messaging.serialization import dumps_state, loads_state

KINDS = ("global", "cluster", "mixed", "local")


@dataclass(frozen=True)
class CheckpointRef:
    node_id: str
    round: int
    sub_round: int
    kind: str
    path: Path
    sha256: str
    meta: dict[str, Any]


def save_checkpoint(run_dir: str | Path, node_id: str, round_: int, sub_round: int, kind: str,
                    state: Mapping[str, torch.Tensor], meta: Mapping[str, Any] | None = None) -> CheckpointRef:
    if kind not in KINDS:
        raise ValueError(f"kind desconocido: {kind}")
    folder = Path(run_dir) / "checkpoints" / node_id
    folder.mkdir(parents=True, exist_ok=True)
    data = dumps_state(state)
    path = folder / f"r{round_:04d}_s{sub_round:02d}_{kind}.pkl"
    path.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    entry = {"node_id": node_id, "round": round_, "sub_round": sub_round, "kind": kind,
             "file": path.name, "sha256": sha, "meta": dict(meta or {})}
    with open(folder / "index.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    return CheckpointRef(node_id, round_, sub_round, kind, path, sha, entry["meta"])


def list_checkpoints(run_dir: str | Path) -> list[CheckpointRef]:
    refs = []
    for index in sorted((Path(run_dir) / "checkpoints").glob("*/index.jsonl")):
        for line in index.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            refs.append(CheckpointRef(e["node_id"], e["round"], e["sub_round"], e["kind"],
                                      index.parent / e["file"], e["sha256"], e.get("meta", {})))
    return sorted(refs, key=lambda r: (r.round, r.sub_round, r.node_id, r.kind))


def load_checkpoint(ref: CheckpointRef) -> dict[str, torch.Tensor]:
    data = ref.path.read_bytes()
    if hashlib.sha256(data).hexdigest() != ref.sha256:
        raise ValueError(f"Hash distinto en el checkpoint {ref.path.name}")
    return loads_state(data)
