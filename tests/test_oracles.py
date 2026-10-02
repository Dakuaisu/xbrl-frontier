"""Calibration: eval.oracles on the pinned FinNI set, raw and deduplicated. Must pass before any scoring."""
import pytest

from xbrl_frontier import config, eval_sets, finni, oracles, scorer


@pytest.fixture(scope="module")
def sets():
    rows = finni.load()
    golds = [scorer.gold_set(r.answer) for r in rows]
    keep = eval_sets.dedup_indices(rows)
    return {"raw": golds, "dedup": [golds[i] for i in keep]}


@pytest.fixture(scope="module")
def spec():
    return config.get("eval.oracles")


@pytest.mark.parametrize("which", ["raw", "dedup"])
def test_gold_copy(sets, spec, which):
    s = scorer.score(sets[which], oracles.gold_copy(sets[which]))
    assert abs(s.overall["finni_f1"] - spec["gold_copy_f1"]) <= spec["tolerance"]
    assert s.overall["parse_failure_rate"] == 0


@pytest.mark.parametrize("which", ["raw", "dedup"])
def test_empty(sets, spec, which):
    s = scorer.score(sets[which], oracles.empty(sets[which]))
    assert abs(s.overall["finni_f1"] - spec["empty_prediction_f1"]) <= spec["tolerance"]


@pytest.mark.parametrize("which", ["raw", "dedup"])
def test_types_swapped(sets, spec, which):
    s = scorer.score(sets[which], oracles.types_swapped(sets[which]))
    assert abs(s.overall["finni_f1"] - spec["gold_with_types_swapped_f1"]) <= spec["tolerance"]


@pytest.mark.parametrize("which", ["raw", "dedup"])
def test_half_dropped(sets, spec, which):
    s = scorer.score(sets[which], oracles.half_dropped(sets[which]))
    assert abs(s.overall["finni_recall"] - spec["gold_half_dropped_recall"]) <= spec["tolerance"]
    assert s.overall["finni_precision"] == 1.0
