import datetime as dt

from scripts.kaggle_run import committed_estimate
from xbrl_frontier import config


def test_committed_estimate_adds_running_wall_clock_and_prior_weeks():
    now = dt.datetime(2026, 10, 3, 2, 0, tzinfo=dt.timezone.utc)
    rows = [{"action": "push", "name": "a", "at_utc": "2026-10-02T23:00:00+00:00"},
            {"action": "push", "name": "b", "at_utc": "2026-10-03T01:30:00+00:00"},
            {"action": "push", "name": "c", "at_utc": "2026-10-03T01:00:00+00:00", "note": "NOT RUN: rejected"},
            {"action": "quota_reset", "at_utc": "2026-10-03T00:00:00+00:00", "last_reading_before_reset": {"gpu_used_hours": 5.0}}]
    e = committed_estimate(1.0, rows, {"a", "b", "c"}, now)
    assert e["running_wall_clock_hours"] == {"a": 3.0, "b": 0.5}
    assert e["week_estimate"] == 4.5 and e["cumulative_estimate"] == 9.5 and e["within_caps"]
    k = config.get("training.kaggle")
    assert not committed_estimate(k["weekly_gpu_hours_quota"], rows, set(), now)["within_caps"]


def test_measured_fetch_hours_count_when_the_quota_lags():
    now = dt.datetime(2026, 10, 3, 2, 0, tzinfo=dt.timezone.utc)
    rows = [{"action": "fetch", "at_utc": "2026-10-02T10:00:00+00:00", "gpu_hours_this_run": 4.0},
            {"action": "quota_reset", "at_utc": "2026-10-03T00:18:00+00:00", "last_reading_before_reset": {"gpu_used_hours": 5.0}},
            {"action": "fetch", "at_utc": "2026-10-03T01:00:00+00:00", "gpu_hours_this_run": 9.0}]
    e = committed_estimate(0.5, rows, set(), now)
    assert e["measured_fetch_hours_this_week"] == 9.0 and e["week_estimate"] == 9.0
