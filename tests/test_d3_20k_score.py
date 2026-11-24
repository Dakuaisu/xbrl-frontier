import pytest

from scripts import d3_20k_score


def test_compatible_reports_require_identical_set_and_coverage():
    reports = [{"set": "dev", "config_hash": "same", "instances_raw": 3, "instances_dedup": 2} for _ in range(3)]
    d3_20k_score.check_reports(reports)


def test_compatible_reports_reject_different_dedup():
    reports = [{"set": "dev", "config_hash": "same", "instances_raw": 3, "instances_dedup": n} for n in (2, 3)]
    with pytest.raises(ValueError):
        d3_20k_score.check_reports(reports)
