"""E2 criterio 3: un pickle con cualquier `global` fuera de la lista blanca se rechaza SIN ejecutar código.

Cada ataque intentaría crear un archivo testigo; la prueba comprueba que el archivo no existe.
Ataques: S1 de D3 (pickle RCE) y la variante anidada a través de `torch.storage._load_from_bytes`.
"""
from __future__ import annotations

import builtins
import io
import os
import pickle
import random
import subprocess

import pytest
import torch
import torch.storage

from messaging.contracts import ContractValidator
from messaging.frame import decode_message, encode_frame
from messaging.serialization import UnsafePayloadError, loads_state


class _Reduce:
    def __init__(self, fn, args):
        self.fn, self.args = fn, args

    def __reduce__(self):
        return self.fn, self.args


def _witness_code(path):
    return f"open({str(path)!r}, 'w').write('pwned')"


@pytest.mark.parametrize("fn_factory", [
    lambda p: (builtins.exec, (_witness_code(p),)),
    lambda p: (builtins.eval, (f"__import__('builtins').exec({_witness_code(p)!r})",)),
    lambda p: (os.system, (f'python -c "{_witness_code(p)}"',)),
    lambda p: (subprocess.call, (["python", "-c", _witness_code(p)],)),
])
def test_top_level_malicious_pickle_is_rejected_without_execution(tmp_path, fn_factory):
    witness = tmp_path / "witness.txt"
    payload = pickle.dumps({"fc1.weight": _Reduce(*fn_factory(witness))})
    with pytest.raises(UnsafePayloadError, match="global no permitido"):
        loads_state(payload)
    assert not witness.exists()


class _Evil:
    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return builtins.exec, (_witness_code(self.path),)


def test_nested_attack_through_storage_loader_is_blocked(tmp_path):
    """El `_load_from_bytes` original haría torch.load(weights_only=False) del contenido anidado."""
    witness = tmp_path / "witness.txt"
    buf = io.BytesIO()
    torch.save(_Evil(witness), buf)                     # objeto malicioso en formato torch.save
    payload = pickle.dumps({"fc1.weight": _Reduce(torch.storage._load_from_bytes, (buf.getvalue(),))})
    with pytest.raises(UnsafePayloadError):
        loads_state(payload)
    assert not witness.exists()


def test_malicious_payload_rejected_through_full_receive_path(tmp_path):
    witness = tmp_path / "witness.txt"
    payload = pickle.dumps({"fc1.weight": _Reduce(builtins.exec, (_witness_code(witness),))})
    import hashlib
    header = {"schema_version": 1, "msg_type": "UPDATE", "experiment_id": "exp-001", "topology": "centralized",
              "sender_id": "CN3", "recipient_id": "infra", "seq": 1, "round": 1, "n_samples": 167, "tau": 33,
              "metrics": {"train_loss": 0.5, "train_time_s": 1.0},
              "payload": {"format": "pickle", "protocol": 4, "nbytes": len(payload),
                          "sha256": hashlib.sha256(payload).hexdigest(),
                          "tensors": [{"name": "fc1.weight", "shape": [256, 278], "dtype": "float32"}]}}
    with pytest.raises(UnsafePayloadError):
        decode_message(encode_frame(header, payload), ContractValidator())
    assert not witness.exists()


@pytest.mark.filterwarnings("ignore:Detected pickle protocol:UserWarning")
def test_fuzzed_payloads_never_escape_controlled_errors():
    """Bytes aleatorios y mutaciones de un pickle válido: solo errores controlados (D3 S2, versión mínima)."""
    rng = random.Random(0)
    valid = pickle.dumps({"w": torch.zeros(3)})
    samples = [bytes(rng.getrandbits(8) for _ in range(rng.randint(1, 200))) for _ in range(300)]
    for _ in range(300):
        b = bytearray(valid)
        for _ in range(rng.randint(1, 5)):
            b[rng.randrange(len(b))] = rng.getrandbits(8)
        samples.append(bytes(b))
    for s in samples:
        try:
            loads_state(s)
        except pickle.UnpicklingError:
            pass
