import hashlib
from pathlib import Path

import pytest
from scripts.d4_contract import target


def test_target_binds_all_expected_finni_ids_and_generation():
    digest = hashlib.sha256(Path("serving/generate_predictions.py").read_bytes()).hexdigest()
    first = target(digest)
    assert first.expected_ids[0] == 0
    assert len(first.expected_ids) == len(set(first.expected_ids))
    with pytest.raises(ValueError):
        target("wrong-runner")
