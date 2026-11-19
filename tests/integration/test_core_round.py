"""Una ronda FedAvg centralizada en proceso, sin red, sobre los 5 hospitales reales de D0.

Verifica que las piezas de E1 encajan: datos → modelo → entrenamiento local → agregación ponderada →
evaluación con el protocolo de D0 (umbral F1 en la validación común, métricas en el holdout común).
No pretende reproducir cifras del paper (eso es E11).
"""
from __future__ import annotations

import numpy as np
import torch

from core.aggregation import aggregate
from core.data import load_hospital
from core.model import build_model
from core.training import LocalTrainer
from evaluation.metrics import best_f1_threshold, binary_metrics, predict_proba

HOSPITALS = ("H1", "H2", "H3", "H4", "H5")


def test_one_centralized_fedavg_round_end_to_end(processed_dir, cfg):
    data = {h: load_hospital(processed_dir, h, cfg["data"]) for h in HOSPITALS}
    n_features = data["H1"].n_features
    tr = cfg["training"]

    global_model = build_model(cfg, n_features)
    global_state = {k: v.clone() for k, v in global_model.state_dict().items()}

    updates = []
    for i, h in enumerate(HOSPITALS):
        local = build_model(cfg, n_features)                     # mismo init (semilla 42) en todos los nodos
        trainer = LocalTrainer(local, data[h].X_train, data[h].y_train, batch_size=tr["batch_size"],
                               learning_rate=tr["learning_rate"], seed=cfg["data"]["seed"] + i)
        trainer.load_global(global_state)
        updates.append(trainer.train(1))                         # 1 época para mantener la prueba rápida

    new_state = aggregate("FedAvg", updates, global_state, cfg["aggregation"])
    assert [u.n_samples for u in updates] == [data[h].n_train for h in HOSPITALS]
    global_model.load_state_dict(new_state)

    X_val = torch.cat([data[h].X_val for h in HOSPITALS]); y_val = torch.cat([data[h].y_val for h in HOSPITALS])
    X_te = torch.cat([data[h].X_test for h in HOSPITALS]); y_te = torch.cat([data[h].y_test for h in HOSPITALS])
    t = best_f1_threshold(y_val.numpy(), predict_proba(global_model, X_val))
    m = binary_metrics(y_te.numpy(), predict_proba(global_model, X_te), t)

    print(f"\n[E1 smoke] holdout común: n={len(y_te)} umbral={t:.3f} "
          + " ".join(f"{k}={v:.3f}" for k, v in m.items() if k != "threshold"))
    assert len(y_te) == sum(len(data[h].y_test) for h in HOSPITALS)
    assert all(np.isfinite(v) for v in m.values())
    assert m["f1"] > 0.5                                         # aprende algo más que la clase mayoritaria
