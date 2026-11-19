"""Entrenamiento local de un Client Node, común a todas las topologías (C, H, SD, P2P).

Agrupa lo que hace un nodo que entrena en cada ronda, para que los orquestadores solo decidan **cuándo**
entrenar y **a quién** enviar el resultado (RNF-08, ADR-01):
  * carga su hospital, construye el modelo de referencia y el `LocalTrainer` (optimizador persistente);
  * entrena en un hilo (`asyncio.to_thread`): el bucle de eventos sigue atendiendo los *heartbeats* AMQP.
    Sin esto, un entrenamiento de más de 2 × `heartbeat_s` (H2 en la Pi) haría que RabbitMQ cerrara la
    conexión. El hilo no cambia el resultado: la ronda sigue siendo secuencial;
  * calcula la métrica heredada (modelo LOCAL en su validación, umbral 0,5; D1 H1);
  * produce las columnas de KPI de un nodo que entrena (CPU heredada antes/después, energía Eq. 4 y
    E_comm por la subida, I-31/I-32).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from core.data import load_hospital
from core.model import arch_hash, build_model
from core.training import LocalTrainer, Update
from evaluation.metrics import binary_metrics, predict_proba
from messaging.frame import tensor_specs
from telemetry.clock import PhaseTimer
from telemetry.energy import round_energy
from telemetry.system import cpu_freq_mhz, cpu_percent_legacy, ram_rss_mb, temperature_c

LEGACY_KEYS = ("accuracy", "precision", "recall", "f1")


@dataclass
class TrainResult:
    update: Update
    legacy: dict[str, float]           # métrica heredada (modelo local, validación local, umbral 0,5)
    cpu_after: float                   # CPU heredada después de evaluar (P1)
    ram_after: float


class LocalWorker:
    def __init__(self, cfg: Mapping[str, Any], node_id: str, processed_dir: str | Path) -> None:
        self.cfg = cfg
        self.node_id = node_id
        self.hospital = cfg["platform"]["node_assignment"][node_id]["hospital"]
        self.data = load_hospital(processed_dir, self.hospital, cfg["data"])
        self.model = build_model(cfg, self.data.n_features)
        self.specs = tensor_specs(self.model.state_dict())
        self.trainer = LocalTrainer(self.model, self.data.X_train, self.data.y_train,
                                    batch_size=int(cfg["training"]["batch_size"]),
                                    learning_rate=float(cfg["training"]["learning_rate"]),
                                    seed=int(cfg["data"]["seed"]),
                                    optimizer_state=cfg["training"]["optimizer_state"])
        self.cpu_interval = cfg["kpis"]["cpu"]["legacy_interval_s"]
        self.e_cum = 0.0

    # -------------------------------------------------------------- registro
    def register_fields(self) -> dict[str, Any]:
        return {"n_features": self.data.n_features, "arch_hash": arch_hash(self.model),
                "n_samples": self.data.n_train}

    @property
    def n_samples(self) -> int:
        return self.data.n_train

    # -------------------------------------------------------------- ronda
    def cpu_before(self) -> tuple[float, float]:
        """CPU heredada y RAM antes de recibir el modelo (P1)."""
        return cpu_percent_legacy(self.cpu_interval), ram_rss_mb()

    async def train(self, start_state: Mapping[str, torch.Tensor], training: Mapping[str, Any],
                    timer: PhaseTimer) -> TrainResult:
        """Carga `start_state`, entrena con los hiperparámetros de la ronda y evalúa la métrica heredada.

        FedProx usa `start_state` (el modelo recibido) como referencia proximal.
        """
        self.trainer.load_global(start_state)
        mu = float(training["mu"])
        with timer.phase("train"):
            upd = await asyncio.to_thread(self.trainer.train, int(training["local_epochs"]), mu=mu,
                                          proximal_reference=start_state if mu > 0 else None)
        legacy = binary_metrics(self.data.y_val.numpy(), predict_proba(self.model, self.data.X_val), 0.5)
        return TrainResult(upd, {k: legacy[k] for k in LEGACY_KEYS},
                           cpu_percent_legacy(self.cpu_interval), ram_rss_mb())

    @staticmethod
    def update_metrics(res: TrainResult, timer: PhaseTimer) -> dict[str, float]:
        """Campo `metrics` de UPDATE (contrato `local_metrics`)."""
        return {"train_loss": res.update.train_loss, "train_time_s": timer.seconds("train"), **res.legacy}

    def kpi_columns(self, res: TrainResult, *, cpu_before: float, ram_before: float,
                    upload_confirm_s: float | None, upload_bytes: int | None) -> dict[str, Any]:
        """Columnas de KPI de un nodo que entrena (sin tiempos ni contadores, que pone el orquestador)."""
        freq = cpu_freq_mhz()
        upd = res.update
        energy = round_energy(self.cfg["energy"], n_samples=upd.n_samples, n_features=self.data.n_features,
                              f_hz=freq * 1e6 if freq else None, tcomm_legacy_s=None,
                              tcomm_transfer_s=upload_confirm_s, model_bytes=upload_bytes)
        self.e_cum += (energy["e_comp_j"] or 0.0) + (energy["e_comm_transfer_j"] or 0.0)
        return {
            "cpu_percent_legacy_before": cpu_before, "cpu_percent_legacy_after": res.cpu_after,
            "ram_rss_mb_before": ram_before, "ram_rss_mb_after": res.ram_after,
            "temperature_c": temperature_c(), "cpu_freq_mhz": freq,
            "n_samples": upd.n_samples, "tau": upd.tau, "train_loss": upd.train_loss,
            **{f"legacy_local_{k}": v for k, v in res.legacy.items()},
            **energy, "e_cum_transfer_j": self.e_cum,
        }
