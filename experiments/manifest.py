"""Manifiesto de experimento (RF-24, RNF-02): todo lo necesario para regenerar un resultado.

Incluye la configuración efectiva, los hashes de los datos (manifiesto de `prepare_data`), el hash del
código del proyecto, las versiones de las dependencias y la plataforma. `manifest_sha256` es el hash
canónico del resto del manifiesto; cada fila de KPIs lo lleva para que `collect` detecte mezclas.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.config import PROJECT_ROOT
from core.data import MANIFEST_NAME, sha256_file

CODE_DIRS = ("core", "messaging", "schemas", "topologies", "node", "telemetry", "evaluation", "experiments",
             "broker", "scripts", "configs")
PACKAGES = ("torch", "numpy", "pandas", "scikit-learn", "joblib", "jsonschema", "aio-pika", "psutil", "pyyaml")


class ManifestError(ValueError):
    pass


def canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                     default=str).encode("utf-8")).hexdigest()


def code_sha256(root: Path = PROJECT_ROOT) -> str:
    """Hash del código fuente del proyecto (rutas relativas + contenido; sin datos, resultados ni caché)."""
    h = hashlib.sha256()
    for d in CODE_DIRS:
        base = root / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix in {".py", ".json", ".yaml", ".yml", ".conf"} and "__pycache__" not in p.parts:
                h.update(p.relative_to(root).as_posix().encode())
                h.update(p.read_bytes().replace(b"\r\n", b"\n"))       # mismo hash en Windows y en la Pi
    return h.hexdigest()


def package_versions(packages: Iterable[str] = PACKAGES) -> dict[str, str | None]:
    out = {}
    for name in packages:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def build_manifest(cfg: Mapping[str, Any], *, experiment_id: str, topology: str, method: str, seed: int,
                   processed_dir: str | Path, security_profile: str = "S0",
                   assignment: Mapping[str, str] | None = None, study: str | None = None,
                   extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if topology not in cfg["topologies"] and topology != "p2p":
        raise ManifestError(f"Topología sin configuración: {topology}")
    if method not in cfg["aggregation"]["methods"]:
        raise ManifestError(f"Método desconocido: {method}")
    data_manifest = Path(processed_dir) / MANIFEST_NAME
    if not data_manifest.exists():
        raise ManifestError(f"Falta el manifiesto de datos: {data_manifest} (ejecuta scripts/prepare_data.py)")
    body = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "study": study,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "topology": topology,
        "method": method,
        "seed": seed,
        "security_profile": security_profile,
        "applied_variants": list(cfg.get("applied_variants", [])),
        "assignment": dict(assignment or cfg["platform"]["node_assignment"]),
        "config": dict(cfg),
        "config_sha256": canonical_sha256(cfg),
        "data_manifest_sha256": sha256_file(data_manifest),
        "code_sha256": code_sha256(),
        "versions": {"python": platform.python_version(), **package_versions()},
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(),
                     "node": platform.node()},
        "extra": dict(extra or {}),
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def save_manifest(manifest: Mapping[str, Any], run_dir: str | Path) -> Path:
    path = Path(run_dir) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def load_manifest(run_dir: str | Path) -> dict[str, Any]:
    manifest = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if canonical_sha256(body) != manifest.get("manifest_sha256"):
        raise ManifestError("El manifiesto fue modificado después de crearse")
    return manifest


def verify_inputs(manifest: Mapping[str, Any], processed_dir: str | Path, check_code: bool = True) -> None:
    """Comprueba que los datos (y opcionalmente el código) son los mismos que registró el manifiesto."""
    actual = sha256_file(Path(processed_dir) / MANIFEST_NAME)
    if actual != manifest["data_manifest_sha256"]:
        raise ManifestError("Los datos procesados no coinciden con los del manifiesto")
    if check_code and code_sha256() != manifest["code_sha256"]:
        raise ManifestError("El código del proyecto cambió desde que se creó el manifiesto")
