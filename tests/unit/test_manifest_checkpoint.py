"""E4: manifiesto (hashes, verificación) y checkpoints (ida y vuelta, integridad, carga segura)."""
from __future__ import annotations

import builtins
import json
import pickle
import shutil

import pytest
import torch

from core.model import MLP
from experiments.manifest import (ManifestError, build_manifest, code_sha256, load_manifest, save_manifest,
                                  verify_inputs)
from messaging.serialization import UnsafePayloadError
from node.checkpoint import list_checkpoints, load_checkpoint, save_checkpoint


def make_manifest(cfg, processed_dir, **kw):
    return build_manifest(cfg, experiment_id=kw.pop("experiment_id", "exp-unit"), topology="centralized",
                          method="fedavg", seed=42, processed_dir=processed_dir, **kw)


def test_manifest_records_hashes_and_versions(cfg, processed_dir, tmp_path):
    m = make_manifest(cfg, processed_dir)
    for key in ("manifest_sha256", "config_sha256", "data_manifest_sha256", "code_sha256"):
        assert len(m[key]) == 64
    assert m["versions"]["torch"] and m["versions"]["python"]
    assert m["config"]["model"]["hidden_layers"] == [256, 128, 64]
    save_manifest(m, tmp_path)
    assert load_manifest(tmp_path) == json.loads(json.dumps(m, default=str))
    verify_inputs(m, processed_dir)                                   # mismos datos y mismo código


def test_modified_manifest_is_detected(cfg, processed_dir, tmp_path):
    save_manifest(make_manifest(cfg, processed_dir), tmp_path)
    p = tmp_path / "manifest.json"
    d = json.loads(p.read_text(encoding="utf-8")); d["seed"] = 43
    p.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ManifestError, match="modificado"):
        load_manifest(tmp_path)


def test_changed_data_is_detected(cfg, processed_dir, tmp_path):
    m = make_manifest(cfg, processed_dir)
    other = tmp_path / "processed"
    shutil.copytree(processed_dir, other)
    (other / "manifest.json").write_text((other / "manifest.json").read_text(encoding="utf-8") + " ",
                                         encoding="utf-8")
    with pytest.raises(ManifestError, match="datos"):
        verify_inputs(m, other)


def test_manifest_rejects_unknown_method_and_missing_data(cfg, processed_dir, tmp_path):
    with pytest.raises(ManifestError):
        build_manifest(cfg, experiment_id="x", topology="centralized", method="scaffold", seed=1,
                       processed_dir=processed_dir)
    with pytest.raises(ManifestError, match="prepare_data"):
        build_manifest(cfg, experiment_id="x", topology="centralized", method="fedavg", seed=1,
                       processed_dir=tmp_path)


def test_code_hash_is_deterministic():
    assert code_sha256() == code_sha256()


def test_checkpoint_roundtrip_listing_and_integrity(tmp_path):
    states = {r: {k: v.detach().clone() for k, v in MLP(278, seed=r).state_dict().items()} for r in (1, 2)}
    for r, s in states.items():
        save_checkpoint(tmp_path, "infra", r, 0, "global", s, {"aggregator_id": None})
    save_checkpoint(tmp_path, "CN1", 1, 0, "local", states[1])
    refs = list_checkpoints(tmp_path)
    assert [(x.node_id, x.round, x.kind) for x in refs] == [("CN1", 1, "local"), ("infra", 1, "global"),
                                                            ("infra", 2, "global")]
    loaded = load_checkpoint(refs[2])
    assert all(torch.equal(loaded[k], states[2][k]) for k in loaded)
    refs[1].path.write_bytes(refs[1].path.read_bytes()[:-1] + b"\x00")
    with pytest.raises(ValueError, match="Hash"):
        load_checkpoint(refs[1])
    with pytest.raises(ValueError):
        save_checkpoint(tmp_path, "infra", 3, 0, "weights", states[1])


def test_malicious_checkpoint_rejected_even_with_matching_hash(tmp_path, monkeypatch):
    """Un checkpoint copiado desde una Pi comprometida: hash coherente pero contenido malicioso."""
    witness = tmp_path / "witness.txt"

    class Evil:
        def __reduce__(self):
            return builtins.exec, (f"open({str(witness)!r}, 'w').write('x')",)

    import node.checkpoint as ck
    monkeypatch.setattr(ck, "dumps_state", lambda state: pickle.dumps({"fc1.weight": Evil()}))
    ref = save_checkpoint(tmp_path, "CN3", 1, 0, "global", {"w": torch.zeros(1)})
    with pytest.raises(UnsafePayloadError):
        load_checkpoint(ref)
    assert not witness.exists()
