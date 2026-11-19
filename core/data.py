"""Partición de D0 (Tabla 1), split 60/20/20 estratificado y preprocesamiento de D0 §3.1.

Flujo:
  * Fuera de línea: `scripts/prepare_data.py` usa `load_partition`, `stratified_split`,
    `build_preprocessor` y `fit_preprocessor`, y escribe los splits, el preprocesador y un manifiesto con hashes.
  * En cada nodo: `load_hospital` verifica los hashes del manifiesto y devuelve tensores.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

SPLITS = ("train", "val", "test")
MANIFEST_NAME = "manifest.json"
PREPROCESSOR_NAME = "preprocessor.joblib"


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8", low_memory=False)


def feature_columns(data_cfg: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    feats = data_cfg["features"]
    return list(feats["categorical"]), list(feats["numeric"])


def _features_frame(df: pd.DataFrame, data_cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Selecciona las 19 variables con tipos homogéneos (categóricas como object con NaN)."""
    cat, num = feature_columns(data_cfg)
    missing = [c for c in cat + num if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas de entrada: {missing}")
    out = pd.DataFrame(index=df.index)
    for c in cat:
        col = df[c].astype(object)
        out[c] = col.where(df[c].notna(), np.nan)
    for c in num:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return out


# ---------------------------------------------------------------------------
# Partición (D0 Tabla 1)
# ---------------------------------------------------------------------------
def load_partition(silo_dir: str | Path, data_cfg: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    """Reconstruye los 5 hospitales a partir de los silos de P2 y verifica sus tamaños."""
    silo_dir = Path(silo_dir)
    label = data_cfg["label_column"]
    partition: dict[str, pd.DataFrame] = {}
    for hospital, spec in data_cfg["hospitals"].items():
        frames = [read_csv(silo_dir / f) for f in spec["source_files"]]
        df = pd.concat(frames, ignore_index=True)
        if len(df) != spec["rows"]:
            raise ValueError(f"{hospital}: {len(df)} filas, se esperaban {spec['rows']} (D0 Tabla 1)")
        if df[label].isna().any():
            raise ValueError(f"{hospital}: etiquetas vacías en {label}")
        partition[hospital] = df
    return partition


def stratified_split(df: pd.DataFrame, label: str, fractions: Mapping[str, float],
                     seed: int) -> dict[str, pd.DataFrame]:
    """Split estratificado train/val/test (D0 §3.7: 60/20/20)."""
    f_train, f_val, f_test = fractions["train"], fractions["validation"], fractions["test"]
    if not np.isclose(f_train + f_val + f_test, 1.0):
        raise ValueError("Las fracciones deben sumar 1")
    train, rest = train_test_split(df, train_size=f_train, stratify=df[label], random_state=seed)
    val, test = train_test_split(rest, test_size=f_test / (f_val + f_test),
                                 stratify=rest[label], random_state=seed)
    return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------------------
# Preprocesamiento (D0 §3.1)
# ---------------------------------------------------------------------------
def build_preprocessor(data_cfg: Mapping[str, Any]) -> ColumnTransformer:
    pre = data_cfg["preprocessing"]
    cat, num = feature_columns(data_cfg)
    scalers = {"robust": RobustScaler, "standard": StandardScaler}
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy=pre["numeric"]["impute"])),
        ("scale", scalers[pre["numeric"]["scale"]]()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy=pre["categorical"]["impute"])),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    # Mismo orden que P1 (categóricas y luego numéricas).
    return ColumnTransformer([("cat", categorical, cat), ("num", numeric, num)], remainder="drop")


def fit_preprocessor(pre: ColumnTransformer, train_frames: list[pd.DataFrame],
                     data_cfg: Mapping[str, Any]) -> ColumnTransformer:
    """Ajusta solo con datos de entrenamiento (nunca con validación ni test)."""
    pre.fit(_features_frame(pd.concat(train_frames, ignore_index=True), data_cfg))
    return pre


def transform(df: pd.DataFrame, pre: ColumnTransformer,
              data_cfg: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    X = pre.transform(_features_frame(df, data_cfg)).astype(np.float32)
    y = df[data_cfg["label_column"]].to_numpy(dtype=np.float32).reshape(-1, 1)
    return torch.from_numpy(X), torch.from_numpy(y)


# ---------------------------------------------------------------------------
# Carga en el nodo
# ---------------------------------------------------------------------------
@dataclass
class HospitalSplits:
    hospital: str
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor

    @property
    def n_train(self) -> int:
        return int(self.X_train.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X_train.shape[1])


def load_manifest(processed_dir: str | Path) -> dict[str, Any]:
    with open(Path(processed_dir) / MANIFEST_NAME, encoding="utf-8") as fh:
        return json.load(fh)


def _verified(processed_dir: Path, rel: str, manifest: Mapping[str, Any]) -> Path:
    path = processed_dir / rel
    expected = manifest["outputs"][rel]["sha256"]
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"Hash distinto en {rel}: {actual} != {expected} (artefacto modificado)")
    return path


def load_preprocessor(processed_dir: str | Path, manifest: Mapping[str, Any] | None = None) -> ColumnTransformer:
    """Carga el preprocesador local tras verificar su hash. No se usa nunca sobre datos de red."""
    processed_dir = Path(processed_dir)
    manifest = manifest or load_manifest(processed_dir)
    return joblib.load(_verified(processed_dir, PREPROCESSOR_NAME, manifest))


def load_hospital(processed_dir: str | Path, hospital: str,
                  data_cfg: Mapping[str, Any]) -> HospitalSplits:
    processed_dir = Path(processed_dir)
    manifest = load_manifest(processed_dir)
    pre = load_preprocessor(processed_dir, manifest)
    tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for split in SPLITS:
        rel = f"splits/{hospital}_{split}.csv"
        tensors[split] = transform(read_csv(_verified(processed_dir, rel, manifest)), pre, data_cfg)
    return HospitalSplits(hospital, *tensors["train"], *tensors["val"], *tensors["test"])
