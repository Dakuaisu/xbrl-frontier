import hashlib
import json

import pytest

from scripts import d4_resume_guard as guard


@pytest.fixture
def prior(tmp_path):
    path = tmp_path / "predictions.jsonl"
    rows = [{"id": 0, "text": "{}", "finish_reason": "length", "not_generated": False},
            {"id": 2, "text": "{}", "finish_reason": "stop", "not_generated": False}]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    identity = guard.Identity("model", "revision", "dataset", "dataset-hash", "config", "adapter", "generation")
    receipt = guard.Receipt(identity, hashlib.sha256(path.read_bytes()).hexdigest())
    return path, receipt


def test_resume_keeps_cutoffs_and_selects_only_missing(prior):
    path, receipt = prior
    result = guard.validate(path, receipt, guard.Target(receipt.identity, (0, 1, 2)))
    assert result.missing == (1,) and result.cutoffs == 1 and result.complete is False


def test_resume_rejects_hash_tampering(prior):
    path, receipt = prior
    path.write_text(path.read_text() + " ")
    with pytest.raises(guard.ResumeBlocked):
        guard.validate(path, receipt, guard.Target(receipt.identity, (0, 1, 2)))


def test_resume_rejects_changed_identity(prior):
    path, receipt = prior
    different = guard.Identity("model", "revision", "dataset", "changed", "config", "adapter", "generation")
    with pytest.raises(guard.ResumeBlocked):
        guard.validate(path, receipt, guard.Target(different, (0, 1, 2)))


@pytest.mark.parametrize("ids", [[0, 0], [0, 9]])
def test_resume_rejects_duplicate_and_extra_ids(prior, ids):
    path, receipt = prior
    path.write_text("".join(json.dumps({"id": i, "text": "{}", "finish_reason": "stop"}) + "\n" for i in ids))
    receipt = guard.Receipt(receipt.identity, hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(guard.ResumeBlocked):
        guard.validate(path, receipt, guard.Target(receipt.identity, (0, 1, 2)))


def test_empty_missing_list_only_for_exact_coverage(prior):
    path, receipt = prior
    result = guard.validate(path, receipt, guard.Target(receipt.identity, (0, 2)))
    assert result.complete is True and result.missing == ()


def test_cutoff_is_scored_as_written_not_retried():
    from scripts.score_predictions import to_parsed

    parsed, not_generated, cut_off = to_parsed({"text": "invalid", "finish_reason": "length"})
    assert parsed.parse_failed and not not_generated and cut_off
