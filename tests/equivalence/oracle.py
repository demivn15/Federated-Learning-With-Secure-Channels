"""Extrae funciones o métodos del código original (P1/P2/P3) para usarlos como oráculos de prueba.

Se lee el archivo fuente y se compila **solo** la función pedida, con los globales que indique la prueba.
Así se evita importar los módulos completos (que necesitan `websockets`, `config`, etc.) y los proyectos
originales no se modifican (RNF-11).
"""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from typing import Any, Callable


def load_function(path: Path, name: str, class_name: str | None = None,
                  globals_: dict[str, Any] | None = None) -> Callable:
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    scope = tree.body
    if class_name is not None:
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
        scope = cls.body
    node = next(n for n in scope if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    code = textwrap.dedent(ast.get_source_segment(source, node))
    namespace: dict[str, Any] = dict(globals_ or {})
    exec(compile(code, f"<oracle {Path(path).name}:{name}>", "exec"), namespace)
    return namespace[name]


class Stub:
    """Objeto con atributos arbitrarios para hacer de `self` o de `Config`."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)
