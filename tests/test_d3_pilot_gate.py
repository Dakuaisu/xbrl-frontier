"""Offline admission and at-most-once tests for the D3 pilot."""
import json
import struct

import pytest

from scripts import d3_pilot_gate as gate


@pytest.fixture
def final_run(tmp_path):
    adapter = tmp_path / "final_adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"base_model_name_or_path":"Qwen/Qwen3-1.7B"}')
    header = json.dumps({"layer": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (adapter / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"1234")
    (tmp_path / "run.json").write_text(json.dumps({
        "complete": True, "global_step": 4716, "final_adapter": "final_adapter",
        "model": "Qwen/Qwen3-1.7B", "revision": "pinned", "seed": 1,
        "sessions": [{"resumed_from": "checkpoint-4180"}],
        "corpus_sha256": {"dev": "devhash"}, "config_hash": "confighash",
    }))
    return tmp_path


def test_old_error_waits():
    assert gate.training_ready(2, "ERROR") is False


def test_new_version_must_complete():
    assert gate.training_ready(3, "RUNNING") is False
    assert gate.training_ready(3, "COMPLETE") is True


def test_final_artifact_must_match_provenance(final_run):
    result = gate.verify_final(final_run, gate.Provenance("pinned", "devhash", "confighash"))
    assert len(result) == 64


def test_pinned_huggingface_snapshot_path_is_accepted(final_run):
    config = final_run / "final_adapter/adapter_config.json"
    config.write_text(json.dumps({"base_model_name_or_path": "/root/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/pinned"}))
    assert len(gate.verify_final(final_run, gate.Provenance("pinned", "devhash", "confighash"))) == 64


def test_wrong_huggingface_snapshot_path_is_rejected(final_run):
    config = final_run / "final_adapter/adapter_config.json"
    config.write_text(json.dumps({"base_model_name_or_path": "/root/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/wrong"}))
    with pytest.raises(gate.PilotBlocked):
        gate.verify_final(final_run, gate.Provenance("pinned", "devhash", "confighash"))


@pytest.mark.parametrize("field,value", [("complete", False), ("global_step", 4180),
                                        ("final_adapter", None), ("revision", "wrong")])
def test_incomplete_or_wrong_run_refused(final_run, field, value):
    path = final_run / "run.json"
    run = json.loads(path.read_text())
    run[field] = value
    path.write_text(json.dumps(run))
    with pytest.raises(gate.PilotBlocked):
        gate.verify_final(final_run, gate.Provenance("pinned", "devhash", "confighash"))


def test_truncated_weights_refused(final_run):
    (final_run / "final_adapter" / "adapter_model.safetensors").write_bytes(b"broken")
    with pytest.raises(gate.PilotBlocked):
        gate.verify_final(final_run, gate.Provenance("pinned", "devhash", "confighash"))


def test_wrong_dev_hash_refused(final_run):
    with pytest.raises(gate.PilotBlocked):
        gate.verify_final(final_run, gate.Provenance("pinned", "wrong", "confighash"))


def test_offsets_past_end_refused(final_run):
    header = json.dumps({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 100]}}).encode()
    (final_run / "final_adapter" / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"1234")
    with pytest.raises(gate.PilotBlocked):
        gate.verify_final(final_run, gate.Provenance("pinned", "devhash", "confighash"))


def test_tensor_shape_must_match_storage(final_run):
    header = json.dumps({"x": {"dtype": "F32", "shape": [100], "data_offsets": [0, 4]}}).encode()
    (final_run / "final_adapter" / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"1234")
    with pytest.raises(gate.PilotBlocked):
        gate.verify_final(final_run, gate.Provenance("pinned", "devhash", "confighash"))


def test_budget_timeout_and_unknown_reservations_refused():
    assert gate.budget_allows(gate.Budget(1.5, 28, 60, 30, 70, True))
    assert not gate.budget_allows(gate.Budget(2.1, 0, 0, 30, 70, True))
    assert not gate.budget_allows(gate.Budget(1.5, 29, 60, 30, 70, True))
    assert not gate.budget_allows(gate.Budget(1.5, 20, 69, 30, 70, True))
    assert not gate.budget_allows(gate.Budget(1.5, 20, 60, 30, 70, False))


def test_claim_is_durable_after_unknown_push(tmp_path):
    marker = tmp_path / "attempt.json"
    assert gate.claim_once(marker, 3, "hash") is True
    assert gate.claim_once(marker, 3, "hash") is False
    assert json.loads(marker.read_text())["state"] == "push_outcome_unknown"
