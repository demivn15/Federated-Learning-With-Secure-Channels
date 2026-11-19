"""E2: trama, serialización pickle restringida y contratos JSON (criterios de aceptación 1, 2 y 4)."""
from __future__ import annotations

import json
import pickle

import pytest
import torch

from core.model import MLP
from messaging.contracts import ContractError, ContractValidator
from messaging.frame import (FrameError, Limits, decode_frame, decode_message, encode_frame, encode_message,
                             tensor_specs)
from messaging.serialization import dumps_state, loads_state


@pytest.fixture(scope="module")
def validator():
    return ContractValidator()


@pytest.fixture(scope="module")
def state():
    return {k: v.detach().clone() for k, v in MLP(278).state_dict().items()}


def base(msg_type, **extra):
    h = {"schema_version": 1, "msg_type": msg_type, "experiment_id": "exp-001", "topology": "centralized",
         "sender_id": "CN1", "recipient_id": "infra", "seq": 7}
    h.update(extra)
    return h


def update_header(**extra):
    return base("UPDATE", round=1, n_samples=4643, tau=873,
                metrics={"train_loss": 0.41, "train_time_s": 12.5}, **extra)


# ---------------------------------------------------------------------------
# Serialización (formato del proyecto base)
# ---------------------------------------------------------------------------
def test_wire_bytes_identical_to_base_project_format(state):
    # P1/P3 envían pickle.dumps(dict[str, Tensor]) con el protocolo por defecto (4 en Python 3.8+)
    assert dumps_state(state) == pickle.dumps(state, protocol=4)
    assert pickle.DEFAULT_PROTOCOL >= 4


def test_restricted_loader_reads_base_project_bytes_exactly(state):
    loaded = loads_state(pickle.dumps(state))                   # bytes como los produce P1/P3
    assert list(loaded) == list(state)
    assert all(torch.equal(loaded[k], state[k]) for k in state)


def test_reference_model_payload_size(state):
    # Tamaño M de la Eq. 5 para el modelo de referencia (278 entradas)
    assert len(dumps_state(state)) == 453_090


@pytest.mark.parametrize("bad", [
    {"w": torch.tensor([1.0, float("nan")])},
    {"w": torch.tensor([float("inf")])},
    {"w": torch.tensor([1, 2])},                                # entero
    {"w": torch.zeros(2, requires_grad=True)},
])
def test_loads_rejects_bad_tensors(bad):
    with pytest.raises(pickle.UnpicklingError):
        loads_state(pickle.dumps(bad))


@pytest.mark.parametrize("obj", [[torch.zeros(1)], {}, {1: torch.zeros(1)}, "texto", 42])
def test_loads_rejects_wrong_structure(obj):
    with pytest.raises(pickle.UnpicklingError):
        loads_state(pickle.dumps(obj))


# ---------------------------------------------------------------------------
# Trama
# ---------------------------------------------------------------------------
def test_frame_roundtrip_and_canonical_json():
    f = encode_frame({"b": 1, "a": "ñ"}, b"xyz")
    assert decode_frame(f) == ({"a": "ñ", "b": 1}, b"xyz")
    assert f == encode_frame({"a": "ñ", "b": 1}, b"xyz")         # orden de claves canónico


@pytest.mark.parametrize("data", [b"", b"\x00\x00", b"\x00\x00\x00\x10{}", b"\xff\xff\xff\xff{}",
                                  b"\x00\x00\x00\x02[]", b"\x00\x00\x00\x03NaN", b"\x00\x00\x00\x02\xff\xfe"])
def test_decode_frame_rejects_malformed(data):
    with pytest.raises(FrameError):
        decode_frame(data)


def test_size_limits_checked_before_parsing():
    small = Limits(max_frame_bytes=100, max_header_bytes=50, max_payload_bytes=60)
    with pytest.raises(FrameError):
        decode_frame(b"\x00\x00\x00\x02{}" + b"x" * 200, small)
    with pytest.raises(FrameError):
        encode_frame({"k": "v" * 60}, b"", small)
    with pytest.raises(FrameError):
        encode_frame({}, b"x" * 61, small)


# ---------------------------------------------------------------------------
# Mensajes completos
# ---------------------------------------------------------------------------
def test_update_message_roundtrip(validator, state):
    frame = encode_message(update_header(), state)
    msg = decode_message(frame, validator, expected_tensors=tensor_specs(state))
    assert msg.header["n_samples"] == 4643 and msg.header["payload"]["nbytes"] == 453_090
    assert all(torch.equal(msg.state[k], state[k]) for k in state)
    assert encode_message(update_header(), state) == frame                 # determinista


def test_control_message_without_payload(validator):
    msg = decode_message(encode_message(base("STOP", round=12, reason="fixed_rounds_12")), validator)
    assert msg.state is None and msg.header["reason"] == "fixed_rounds_12"


def test_payload_integrity_and_expected_model_checked_before_unpickling(validator, state):
    frame = bytearray(encode_message(update_header(), state))
    frame[-10] ^= 0xFF                                                   # corrupción del payload
    with pytest.raises(FrameError, match="sha256"):
        decode_message(bytes(frame), validator)
    other = {k: v.detach().clone() for k, v in MLP(105).state_dict().items()}
    with pytest.raises(ContractError, match="modelo esperado"):
        decode_message(encode_message(update_header(), other), validator, expected_tensors=tensor_specs(state))


# ---------------------------------------------------------------------------
# Contratos
# ---------------------------------------------------------------------------
def test_schemas_cover_e2_catalog(validator):
    # Catálogo común + centralizada (E2); cada topología añade sus tipos en E6–E8 (04 §4.8)
    assert {"HEARTBEAT", "ERROR", "STOP", "BYE", "REGISTER", "REGISTER_ACK", "ROUND_START",
            "UPDATE"} <= validator.msg_types


def test_schemas_cover_hierarchical_catalog(validator):
    assert "CLUSTER_UPDATE" in validator.msg_types


def test_schemas_cover_semi_decentralized_catalog(validator):
    assert {"ELECTION", "CLAIM", "ROLE_ASSIGN", "AGG_MODEL", "AGG_COMPLETE"} <= validator.msg_types


def test_coordinator_contracts_never_carry_a_model(validator):
    """RF-03: lo que el coordinador SD recibe (REGISTER, CLAIM, AGG_COMPLETE) no admite payload."""
    import json
    from messaging.contracts import SCHEMA_DIR
    for path in ("centralized/register.json", "semi_decentralized/claim.json",
                 "semi_decentralized/agg_complete.json"):
        s = json.loads((SCHEMA_DIR / path).read_text(encoding="utf-8"))
        assert "payload" not in s["properties"] and s["additionalProperties"] is False


@pytest.mark.parametrize("mutate, where", [
    (lambda h: h.update(extra_field=1), "extra_field"),                  # additionalProperties: false
    (lambda h: h.pop("seq"), "seq"),
    (lambda h: h.update(sender_id="CN 1; DROP"), "sender_id"),
    (lambda h: h.update(topology="star"), "topology"),
    (lambda h: h.update(round=0), "round"),
    (lambda h: h["metrics"].update(train_loss=-1), "train_loss"),
    (lambda h: h["metrics"].update(recall=1.5), "recall"),
    (lambda h: h.update(schema_version=2), "schema_version"),
])
def test_contract_rejects_invalid_headers(validator, state, mutate, where):
    h = json.loads(json.dumps(update_header()))
    h["payload"] = {"format": "pickle", "protocol": 4, "nbytes": 10, "sha256": "0" * 64,
                    "tensors": [{"name": "fc1.weight", "shape": [256, 278], "dtype": "float32"}]}
    validator.validate(h)
    mutate(h)
    with pytest.raises(ContractError):
        validator.validate(h)


def test_update_requires_payload_descriptor(validator):
    with pytest.raises(ContractError, match="payload"):
        validator.validate(update_header())


def test_unknown_message_type_rejected_even_without_validation():
    for v in (ContractValidator(), ContractValidator(enabled=False)):
        with pytest.raises(ContractError):
            v.validate(base("SHUTDOWN_EVERYTHING"))


def test_disabled_validation_skips_schema_rules():
    ContractValidator(enabled=False).validate(update_header(extra_field=1))   # solo para medir RF-30
