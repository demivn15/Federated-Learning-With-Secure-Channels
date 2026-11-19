"""Validación de cabeceras JSON contra los contratos de `schemas/` (RF-16; generaliza D4).

Cada esquema define un `msg_type` (constante) y `additionalProperties: false`. Con `enabled=False`
solo se comprueba que la cabecera sea un objeto con un `msg_type` conocido: ese modo existe para
medir el coste de la validación (RF-30), no para producción.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft7Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT7

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"


class ContractError(ValueError):
    """La cabecera no cumple el contrato de su tipo de mensaje."""


class ContractValidator:
    def __init__(self, schema_dir: str | Path = SCHEMA_DIR, enabled: bool = True) -> None:
        self.enabled = enabled
        schemas = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(schema_dir).rglob("*.json"))]
        registry = Registry().with_resources(
            (s["$id"], Resource.from_contents(s, default_specification=DRAFT7)) for s in schemas)
        self._validators: dict[str, Draft7Validator] = {}
        for s in schemas:
            msg_type = s.get("properties", {}).get("msg_type", {}).get("const")
            if msg_type is None:
                continue                                    # archivo de definiciones
            Draft7Validator.check_schema(s)
            if msg_type in self._validators:
                raise ValueError(f"msg_type duplicado en los esquemas: {msg_type}")
            self._validators[msg_type] = Draft7Validator(s, registry=registry)

    @property
    def msg_types(self) -> frozenset[str]:
        return frozenset(self._validators)

    def validate(self, header: Mapping[str, Any]) -> None:
        if not isinstance(header, Mapping):
            raise ContractError("La cabecera debe ser un objeto JSON")
        msg_type = header.get("msg_type")
        validator = self._validators.get(msg_type) if isinstance(msg_type, str) else None
        if validator is None:
            raise ContractError(f"msg_type desconocido: {msg_type!r}")
        if not self.enabled:
            return
        error = next(iter(sorted(validator.iter_errors(header), key=lambda e: list(e.path))), None)
        if error is not None:
            where = "/".join(map(str, error.path)) or "(raíz)"
            raise ContractError(f"{msg_type}: {where}: {error.message}")
