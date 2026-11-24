import pytest

from scripts import d3_10k_score


def test_exact_coverage_accepts_matching_unique_ids():
    d3_10k_score.check_ids([{"id": "a"}, {"id": "b"}], [{"id": "b"}, {"id": "a"}])


@pytest.mark.parametrize("preds", [[{"id": "a"}, {"id": "a"}], [{"id": "a"}],
                                  [{"id": "a"}, {"id": "other"}]])
def test_exact_coverage_rejects_missing_extra_or_duplicate(preds):
    with pytest.raises(ValueError):
        d3_10k_score.check_ids([{"id": "a"}, {"id": "b"}], preds)
