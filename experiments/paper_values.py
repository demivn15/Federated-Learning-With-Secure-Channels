"""Valores publicados en el paper base (D0), transcritos de sus tablas, para comparar las réplicas.

Fuente: Castro-Rodriguez et al., ICI2ST 2026 — Tabla 7 (rendimiento, comunicación, energía), Tabla 8
(hardware de los Client Nodes que entrenan, últimas 5 rondas) y Tabla 9 (straggler).
SD con † en la Tabla 7: evaluado en el nodo agregador, no con alcance de red.
"""
from __future__ import annotations

TABLE7 = {  # recall, accuracy, F1, comunicación por ronda (MB), E_cum (J)
    ("centralized", "fedavg"): {"recall": 0.699, "accuracy": 0.854, "f1": 0.661, "comm_mb": 1.69, "e_cum_j": 6.09},
    ("centralized", "fednova"): {"recall": 0.672, "accuracy": 0.848, "f1": 0.646, "comm_mb": 1.69, "e_cum_j": 0.90},
    ("centralized", "fedprox"): {"recall": 0.705, "accuracy": 0.851, "f1": 0.669, "comm_mb": 1.69, "e_cum_j": 2.23},
    ("semi_decentralized", "fedavg"): {"recall": 0.853, "accuracy": 0.840, "f1": 0.791, "comm_mb": 0.45, "e_cum_j": 0.74, "scope": "aggregator"},
    ("semi_decentralized", "fednova"): {"recall": 0.794, "accuracy": 0.828, "f1": 0.768, "comm_mb": 0.45, "e_cum_j": 0.93, "scope": "aggregator"},
    ("semi_decentralized", "fedprox"): {"recall": 0.847, "accuracy": 0.827, "f1": 0.775, "comm_mb": 0.45, "e_cum_j": 2.12, "scope": "aggregator"},
    ("hierarchical", "fedavg"): {"recall": 0.753, "accuracy": 0.924, "f1": 0.789, "comm_mb": 2.98, "e_cum_j": 21.45},
    ("hierarchical", "fednova"): {"recall": 0.766, "accuracy": 0.941, "f1": 0.808, "comm_mb": 2.82, "e_cum_j": 15.35},
    ("hierarchical", "fedprox"): {"recall": 0.726, "accuracy": 0.860, "f1": 0.725, "comm_mb": 3.44, "e_cum_j": 18.53},
}

TABLE8 = {  # CPU (%), temperatura (°C), RAM (MB)
    "centralized": {"cpu_percent": 1.2, "temperature_c": 43.1, "ram_mb": 407.2},
    "semi_decentralized": {"cpu_percent": 4.5, "temperature_c": 44.9, "ram_mb": 408.8},
    "hierarchical": {"cpu_percent": 3.4, "temperature_c": 47.2, "ram_mb": 670.4},
}

TABLE9 = {  # straggler (s): media, desviación, mediana
    ("centralized", "fedavg"): {"mean": 5.563, "std": 0.346, "median": 5.660},
    ("centralized", "fednova"): {"mean": 5.501, "std": 0.433, "median": 5.437},
    ("centralized", "fedprox"): {"mean": 7.263, "std": 0.209, "median": 7.297},
    ("semi_decentralized", "fedavg"): {"mean": 5.050, "std": 0.846, "median": 5.228},
    ("semi_decentralized", "fednova"): {"mean": 8.354, "std": 9.418, "median": 5.538},
    ("semi_decentralized", "fedprox"): {"mean": 6.323, "std": 0.759, "median": 6.464},
    ("hierarchical", "fedavg"): {"mean": 1.037, "std": 0.215, "median": 1.026},
    ("hierarchical", "fednova"): {"mean": 0.105, "std": 0.279, "median": 0.000},
    ("hierarchical", "fedprox"): {"mean": 0.992, "std": 0.355, "median": 1.023},
}
