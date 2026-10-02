from training import schedule
from xbrl_frontier import config


def test_curve_targets_realise_top_point_as_full_corpus():
    pts = config.get("training.curve_points")
    t = schedule.curve_targets(38788)
    assert [x["curve_point"] for x in t] == pts
    assert t[-1] == {"curve_point": 40000, "prefix": 38788, "full_corpus": True}
    assert all(not x["full_corpus"] and x["prefix"] == x["curve_point"] for x in t[:-1])


def test_curve_hits_fire_once_each_in_epoch_one_only():
    n = 38788
    b = config.get("training.global_batch_instances")
    fired = []
    for step in range(1, 2 * schedule.steps_per_epoch(n) + 1):
        fired += schedule.curve_hits(step - 1, step, n)
    assert [f["curve_point"] for f in fired] == config.get("training.curve_points")
    for f in fired:
        assert f["prefix"] <= f["instances_seen"] < f["prefix"] + b or f["instances_seen"] == n


def test_resume_checkpoint_timer():
    m = config.get("training.kaggle.checkpoint_every_minutes")
    assert not schedule.due_for_resume_checkpoint(m * 60 - 1, 0)
    assert schedule.due_for_resume_checkpoint(m * 60, 0)
