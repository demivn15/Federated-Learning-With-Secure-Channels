"""Carga de la configuración de referencia y de sus variantes."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = PROJECT_ROOT / "configs" / "variants" / "d0_reference.yaml"


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Fusiona `override` sobre `base` recursivamente (sin modificar ninguno)."""
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path = DEFAULT_REFERENCE,
                legacy_variants: Iterable[str] = ()) -> dict[str, Any]:
    """Carga la referencia D0 y aplica, en orden, las variantes heredadas pedidas."""
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    available = cfg.get("legacy_variants", {})
    for name in legacy_variants:
        if name not in available:
            raise KeyError(f"Variante desconocida: {name!r}. Disponibles: {sorted(available)}")
        cfg = deep_merge(cfg, available[name])
        cfg.setdefault("applied_variants", []).append(name)
    return cfg
