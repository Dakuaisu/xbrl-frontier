import hashlib
import json
from pathlib import Path

import pytest

from scripts import d4_collect, d4_receipt
from scripts.d4_resume_guard import Identity, ResumeBlocked, ResumePlan, Target
from xbrl_frontier import config


@pytest.fixture
def output(tmp_path):
    target = Target(Identity("model", "revision", "dataset", "shard", "config", "adapter", "generation"), (0, 1, 2))
    directory = tmp_path / "d4"
    directory.mkdir()
    pred = directory / "predictions_final_finni.jsonl"
    pred.write_text(json.dumps({"id": 0, "text": "{}", "finish_reason": "length"}) + "\n")
    meta = directory / "predictions_finni.meta.json"
    meta.write_text(json.dumps({"model": "model", "revision": "revision", "set": "finni", "config_hash": "config",
                                "generation": config.get("eval.generation"), "effective_max_model_len": config.get("eval.generation.max_model_len"),
                                "complete": False, "targets": {"final": {"written": 1, "complete": False}}}))
    (directory / "preflight.json").write_text(json.dumps({"kernel_ref": "owner/kernel/1", "visible_device_count": 1,
                                                          "gpu": "Tesla T4", "adapter_sha256": "adapter",
                                                          "dataset_sha256": "shard", "dataset_revision": "dataset",
                                                          "config_hash": "config", "generation_sha256": "generation"}))
    evidence = d4_receipt.make(pred, target, "owner/kernel/1")
    (directory / "receipt.json").write_text(json.dumps({"evidence": json.loads(d4_receipt.encode(evidence)),
                                                        "metadata_sha256": hashlib.sha256(meta.read_bytes()).hexdigest()}))
    return directory, target


def test_partial_output_is_verified_but_not_complete(output):
    directory, target = output
    plan = d4_collect.inspect_output(directory, target, "owner/kernel/1")
    assert not plan.complete and plan.missing == (1, 2)


def test_version_mismatch_is_rejected(output):
    directory, target = output
    with pytest.raises(ResumeBlocked):
        d4_collect.inspect_output(directory, target, "owner/kernel/2")


def test_metadata_tampering_is_rejected(output):
    directory, target = output
    metadata = directory / "predictions_finni.meta.json"
    metadata.write_text(metadata.read_text() + " ")
    with pytest.raises(ResumeBlocked):
        d4_collect.inspect_output(directory, target, "owner/kernel/1")


def test_complete_output_is_eligible_only_with_all_ids(output):
    directory, target = output
    pred = directory / "predictions_final_finni.jsonl"
    pred.write_text("".join(json.dumps({"id": i, "text": "{}"}) + "\n" for i in (0, 1, 2)))
    meta = directory / "predictions_finni.meta.json"
    meta.write_text(json.dumps({"model": "model", "revision": "revision", "set": "finni", "config_hash": "config",
                                "generation": config.get("eval.generation"), "effective_max_model_len": config.get("eval.generation.max_model_len"),
                                "complete": True, "targets": {"final": {"written": 3, "complete": True}}}))
    evidence = d4_receipt.make(pred, target, "owner/kernel/1")
    (directory / "receipt.json").write_text(json.dumps({"evidence": json.loads(d4_receipt.encode(evidence)),
                                                        "metadata_sha256": hashlib.sha256(meta.read_bytes()).hexdigest()}))
    assert d4_collect.inspect_output(directory, target, "owner/kernel/1").complete


def test_partial_never_invokes_oracles_or_scoring(tmp_path, monkeypatch):
    monkeypatch.setattr(d4_collect.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not score partial"))
    assert not d4_collect.score_if_complete(ResumePlan((1,), 0, False), tmp_path)


def test_inconsistent_complete_plan_cannot_score(tmp_path, monkeypatch):
    monkeypatch.setattr(d4_collect.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not score missing IDs"))
    with pytest.raises(ResumeBlocked):
        d4_collect.score_if_complete(ResumePlan((1,), 0, True), tmp_path)


def test_complete_runs_oracles_before_score(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(d4_collect.config, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(d4_collect.subprocess, "run", lambda command, **kwargs: calls.append(command) or type("Result", (), {"returncode": 0})())
    assert d4_collect.score_if_complete(ResumePlan((), 0, True), tmp_path)
    assert "tests/test_oracles.py" in calls[0] and "scripts.score_predictions" in calls[1]


def test_failed_oracles_prevent_scoring(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(d4_collect.config, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(d4_collect.subprocess, "run", lambda command, **kwargs: calls.append(command) or type("Result", (), {"returncode": 1})())
    with pytest.raises(SystemExit):
        d4_collect.score_if_complete(ResumePlan((), 0, True), tmp_path)
    assert len(calls) == 1 and "tests/test_oracles.py" in calls[0]


def test_download_timeout_is_retryable_without_exposing_output(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise d4_collect.subprocess.TimeoutExpired("kaggle", 1, output="sensitive")

    monkeypatch.setattr(d4_collect.subprocess, "run", timeout)
    assert not d4_collect.fetch_output("kaggle", "owner/kernel/1", tmp_path)
