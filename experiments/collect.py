"""Recogida y verificación de los KPIs de una corrida (RNF-15).

Comprueba que cada CSV tiene el esquema de su topología, que cada fila pertenece al experimento y al
manifiesto de la corrida, y que el nodo de la carpeta coincide con el de las filas. Cualquier discrepancia
⇒ `CollectError` con la lista completa (no se mezclan corridas ni se corrigen datos en silencio).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.manifest import load_manifest
from telemetry.kpi_log import columns_for


class CollectError(ValueError):
    pass


def collect_kpis(run_dir: str | Path, expected_nodes: list[str] | None = None, write: bool = True) -> pd.DataFrame:
    run_dir = Path(run_dir)
    manifest = load_manifest(run_dir)
    columns = list(columns_for(manifest["topology"]))
    problems, frames = [], []
    found = []
    for csv_path in sorted((run_dir / "nodes").glob("*/kpi.csv")):
        node = csv_path.parent.name
        found.append(node)
        df = pd.read_csv(csv_path, dtype={"experiment_id": str, "manifest_sha256": str, "node_id": str})
        if list(df.columns) != columns:
            problems.append(f"{node}: columnas distintas del esquema de '{manifest['topology']}'")
            continue
        for col, expected in (("experiment_id", manifest["experiment_id"]),
                              ("manifest_sha256", manifest["manifest_sha256"]), ("node_id", node)):
            bad = df.index[df[col] != expected].tolist()
            if bad:
                problems.append(f"{node}: {len(bad)} fila(s) con {col} distinto (filas {bad[:5]})")
        frames.append(df)
    if expected_nodes is not None:
        missing = sorted(set(expected_nodes) - set(found))
        if missing:
            problems.append(f"Faltan los KPIs de: {missing}")
    if problems:
        raise CollectError("; ".join(problems))
    if not frames:
        raise CollectError("La corrida no tiene KPIs")
    all_kpis = pd.concat(frames, ignore_index=True).sort_values(["round", "sub_round", "node_id"], kind="stable")
    if write:
        all_kpis.to_csv(run_dir / "kpis_all.csv", index=False)
    return all_kpis
