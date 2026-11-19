"""Identidades criptográficas (E10; 04 ADR-05, RF-19…RF-21).

Dos usos:

  * **TLS del broker (perfil S1/S2):** una CA propia y el certificado del servidor RabbitMQ.
        python scripts/provision_identities.py tls [--out docker/tls] [--san fl-rabbitmq --san 10.0.0.6]
    Escribe `ca.pem` (lo reciben los nodos para verificar al broker), `server.pem` y `server.key`
    (solo el broker). Los SAN por defecto cubren el broker de Docker visto desde el PC (localhost,
    127.0.0.1) y desde la red de contenedores (fl-rabbitmq); en la 6.ª Pi se añade su nombre o IP.

  * **Identidades de los nodos (perfil S2):** un par ECDSA P-256 de larga duración por nodo.
        python scripts/provision_identities.py nodes --out <dir> --nodes CN1,CN2,CN3,CN4,CN5,infra
    Escribe `<nodo>.key` (clave privada PKCS#8; en las Pi se queda en cada Pi) y `trust.json`
    (almacén de confianza: nodo → kid + clave pública). `kid` = primeros 16 bytes del SHA-256 del DER
    de la clave pública. El ejecutor llama a `provision_node_identities` para cada corrida S2.

Nada de esto se versiona (`.gitignore`: identities/, docker/tls/).
"""
from __future__ import annotations

import argparse
import datetime as dt
import ipaddress
import json
import sys
from pathlib import Path
from typing import Iterable

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from messaging.security.identity import TRUST_FILE, key_id, private_key_pem, public_key_pem  # noqa: E402

DEFAULT_SANS = ("localhost", "127.0.0.1", "fl-rabbitmq")
TLS_DIR = Path(__file__).resolve().parents[1] / "docker" / "tls"


# ---------------------------------------------------------------------------
# TLS del broker
# ---------------------------------------------------------------------------
def _name(cn: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.ORGANIZATION_NAME, "FL-KPI-Edge-Benchmark"),
                      x509.NameAttribute(NameOID.COMMON_NAME, cn)])


def _san(entries: Iterable[str]) -> x509.SubjectAlternativeName:
    names: list[x509.GeneralName] = []
    for e in entries:
        try:
            names.append(x509.IPAddress(ipaddress.ip_address(e)))
        except ValueError:
            names.append(x509.DNSName(e))
    return x509.SubjectAlternativeName(names)


def provision_broker_tls(out: str | Path = TLS_DIR, sans: Iterable[str] = DEFAULT_SANS,
                         days: int = 825, overwrite: bool = False) -> dict[str, Path]:
    out = Path(out)
    paths = {"ca": out / "ca.pem", "cert": out / "server.pem", "key": out / "server.key"}
    if not overwrite and all(p.exists() for p in paths.values()):
        return paths
    out.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca = (x509.CertificateBuilder().subject_name(_name("fl-benchmark CA")).issuer_name(_name("fl-benchmark CA"))
          .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
          .not_valid_before(now - dt.timedelta(minutes=5)).not_valid_after(now + dt.timedelta(days=days))
          .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
          .add_extension(x509.KeyUsage(digital_signature=False, content_commitment=False, key_encipherment=False,
                                       data_encipherment=False, key_agreement=False, key_cert_sign=True,
                                       crl_sign=True, encipher_only=False, decipher_only=False), critical=True)
          .sign(ca_key, hashes.SHA256()))
    srv_key = ec.generate_private_key(ec.SECP256R1())
    srv = (x509.CertificateBuilder().subject_name(_name("fl-rabbitmq")).issuer_name(ca.subject)
           .public_key(srv_key.public_key()).serial_number(x509.random_serial_number())
           .not_valid_before(now - dt.timedelta(minutes=5)).not_valid_after(now + dt.timedelta(days=days))
           .add_extension(_san(sans), critical=False)
           .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
           .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
           .sign(ca_key, hashes.SHA256()))
    paths["ca"].write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    paths["cert"].write_bytes(srv.public_bytes(serialization.Encoding.PEM))
    paths["key"].write_bytes(srv_key.private_bytes(serialization.Encoding.PEM,
                                                   serialization.PrivateFormat.TraditionalOpenSSL,
                                                   serialization.NoEncryption()))
    return paths


# ---------------------------------------------------------------------------
# Identidades de los nodos (S2)
# ---------------------------------------------------------------------------
def provision_node_identities(out: str | Path, node_ids: Iterable[str]) -> Path:
    """Genera un par ECDSA P-256 por nodo y el almacén de confianza. Devuelve la ruta de `trust.json`."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    trust = {}
    for n in node_ids:
        sk = ec.generate_private_key(ec.SECP256R1())
        (out / f"{n}.key").write_bytes(private_key_pem(sk))
        trust[n] = {"kid": key_id(sk.public_key()), "public_key": public_key_pem(sk.public_key()).decode()}
    path = out / TRUST_FILE
    path.write_text(json.dumps({"curve": "P-256", "nodes": trust}, indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tls")
    t.add_argument("--out", default=str(TLS_DIR))
    t.add_argument("--san", action="append", default=[])
    t.add_argument("--overwrite", action="store_true")
    n = sub.add_parser("nodes")
    n.add_argument("--out", required=True)
    n.add_argument("--nodes", default="CN1,CN2,CN3,CN4,CN5,infra")
    args = ap.parse_args(argv)
    if args.cmd == "tls":
        paths = provision_broker_tls(args.out, list(DEFAULT_SANS) + args.san, overwrite=args.overwrite)
        print("TLS del broker:", ", ".join(str(p) for p in paths.values()))
    else:
        print("Almacén de confianza:", provision_node_identities(args.out, args.nodes.split(",")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
