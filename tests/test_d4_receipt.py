import json

import pytest

from scripts import d4_receipt as receipt
from scripts.d4_resume_guard import Identity, ResumeBlocked, Target


@pytest.fixture
def target():
    return Target(Identity("model", "revision", "dataset", "shard", "config", "adapter", "generation"), (0, 1, 2))


def test_partial_receipt_preserves_length_and_missing_ids(tmp_path, target):
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({"id": 0, "text": "bad", "finish_reason": "length"}) + "\n")
    record = receipt.make(predictions, target, "owner/kernel/1")
    plan = receipt.check(predictions, receipt.decode(receipt.encode(record)), receipt.Expected(target, "owner/kernel/1"))
    assert plan.missing == (1, 2) and plan.cutoffs == 1 and not plan.complete


def test_receipt_rejects_version_mismatch(tmp_path, target):
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({"id": 0, "text": "ok"}) + "\n")
    record = receipt.make(predictions, target, "owner/kernel/1")
    with pytest.raises(ResumeBlocked):
        receipt.check(predictions, record, receipt.Expected(target, "owner/kernel/2"))


def test_complete_receipt_requires_every_id(tmp_path, target):
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("".join(json.dumps({"id": i, "text": "ok"}) + "\n" for i in (0, 1, 2)))
    record = receipt.make(predictions, target, "owner/kernel/1")
    assert receipt.check(predictions, record, receipt.Expected(target, "owner/kernel/1")).complete
