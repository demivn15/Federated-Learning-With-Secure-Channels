"""Registro de KPIs: un CSV por nodo, una fila por (ronda, sub-ronda) (RF-14; 04 §4.8).

Esquema fijo: columnas comunes + columnas de la topología. Una clave desconocida es un error (evita que
una errata cree columnas nuevas en silencio). Cada fila lleva `experiment_id` y `manifest_sha256`, que
`experiments/collect.py` verifica contra el manifiesto.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

COMMON_COLUMNS: tuple[str, ...] = (
    # identificación
    "experiment_id", "manifest_sha256", "node_id", "role", "topology", "method", "round", "sub_round",
    # tiempos (reloj monótono local, s)
    "t_round_s", "t_train_s", "t_serialize_s", "t_deserialize_s", "t_publish_confirm_s", "t_wait_model_s",
    "t_aggregate_s",
    # comunicación
    "payload_bytes_tx", "payload_bytes_rx", "nic_bytes_tx", "nic_bytes_rx", "messages_tx", "messages_rx",
    "messages_rejected",
    # seguridad E2E (perfil S2; vacías en S0/S1): coste por fase (RNF-05) y rechazos del canal seguro
    "t_encrypt_s", "t_decrypt_s", "t_handshake_s", "handshakes", "sec_rejected",
    # sistema
    "cpu_percent_legacy_before", "cpu_percent_legacy_after", "cpu_process_s", "cpu_process_percent_core",
    "cpu_process_percent_total", "ram_rss_mb_before", "ram_rss_mb_after", "temperature_c", "cpu_freq_mhz",
    # entrenamiento local
    "n_samples", "tau", "train_loss",
    # métrica heredada (media local previa a la agregación; D1 H1)
    "legacy_local_accuracy", "legacy_local_precision", "legacy_local_recall", "legacy_local_f1",
    # energía analítica (J)
    "e_comp_j", "e_comm_legacy_j", "e_comm_transfer_j", "e_comm_eq5_j", "e_cum_legacy_j", "e_cum_transfer_j",
    # agregador
    "straggler_s", "arrivals",
)

TOPOLOGY_COLUMNS: dict[str, tuple[str, ...]] = {
    "centralized": (),
    "hierarchical": ("cluster_id", "cluster_n_samples"),
    "semi_decentralized": ("sd_aggregator_id", "sd_aggregator_changed", "sd_selector", "sd_claims_agree",
                           "sd_attempts", "sd_round_completed"),
    "p2p": ("p2p_neighbors", "p2p_received", "p2p_peers_missing", "p2p_consensus_distance", "p2p_staleness",
            "p2p_isolated", "p2p_mixed_val_recall", "p2p_synced_to"),
}


def kpi_path(run_dir: str | Path, node_id: str) -> Path:
    return Path(run_dir) / "nodes" / node_id / "kpi.csv"


def columns_for(topology: str) -> tuple[str, ...]:
    if topology not in TOPOLOGY_COLUMNS:
        raise ValueError(f"Topología desconocida: {topology}")
    return COMMON_COLUMNS + TOPOLOGY_COLUMNS[topology]


def _cell(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, float) and not math.isfinite(v):
        raise ValueError("Valor no finito en el registro de KPIs")
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, sort_keys=True, separators=(",", ":"))
    if isinstance(v, bool):
        return int(v)
    return v


class KpiLogger:
    def __init__(self, path: str | Path, *, topology: str, experiment_id: str, manifest_sha256: str,
                 node_id: str) -> None:
        self.path = Path(path)
        self.columns = columns_for(topology)
        self._fixed = {"experiment_id": experiment_id, "manifest_sha256": manifest_sha256,
                       "node_id": node_id, "topology": topology}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(self.columns)

    def log(self, row: Mapping[str, Any]) -> None:
        unknown = set(row) - set(self.columns)
        if unknown:
            raise KeyError(f"Columnas desconocidas en el registro de KPIs: {sorted(unknown)}")
        clash = {k for k in self._fixed if k in row and row[k] != self._fixed[k]}
        if clash:
            raise ValueError(f"Valores que contradicen al registrador: {sorted(clash)}")
        full = {**row, **self._fixed}
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([_cell(full.get(c)) for c in self.columns])
