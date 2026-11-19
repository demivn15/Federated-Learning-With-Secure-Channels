"""Genera docker/.env con credenciales de administrador aleatorias para el broker local (no se versiona)."""
from __future__ import annotations

import secrets
from pathlib import Path

ENV = Path(__file__).resolve().parents[1] / "docker" / ".env"


def main() -> int:
    if ENV.exists():
        print(f"{ENV} ya existe; no se sobrescribe")
        return 0
    ENV.write_text(f"RABBITMQ_DEFAULT_USER=fl_admin\nRABBITMQ_DEFAULT_PASS={secrets.token_urlsafe(24)}\n",
                   encoding="utf-8")
    print(f"Creado {ENV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
