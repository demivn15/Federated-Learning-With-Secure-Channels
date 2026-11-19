"""RNF-08: separación ML / comunicación, y aislamiento de `pickle` (04 ADR-04)."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.config import PROJECT_ROOT

NETWORK = ("messaging", "topologies", "node", "broker", "aio_pika", "pika", "websockets", "socket")
RULES = {
    # paquete: módulos de primer nivel prohibidos
    "core": NETWORK + ("pickle",),
    "evaluation": NETWORK + ("pickle",),
    "messaging": ("core", "topologies", "node", "evaluation"),
    "telemetry": NETWORK + ("pickle", "core"),
}
PICKLE_ALLOWED = {PROJECT_ROOT / "messaging" / "serialization.py"}


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


@pytest.mark.parametrize("package", sorted(RULES))
def test_layer_dependencies(package):
    for path in (PROJECT_ROOT / package).rglob("*.py"):
        for name in _imports(path):
            assert name.split(".")[0] not in RULES[package], f"{package}/{path.name} importa {name}"


def test_pickle_only_imported_by_serialization():
    for package in ("core", "evaluation", "messaging", "scripts", "telemetry", "experiments", "node"):
        for path in (PROJECT_ROOT / package).rglob("*.py"):
            if "pickle" in [n.split(".")[0] for n in _imports(path)]:
                assert path in PICKLE_ALLOWED, f"{path.relative_to(PROJECT_ROOT)} importa pickle"
