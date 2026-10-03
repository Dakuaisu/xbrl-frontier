import datetime as dt

from scripts.kaggle_run import committed_estimate
from xbrl_frontier import config

NOW = dt.datetime(2026, 10, 3, 12, 0, tzinfo=dt.timezone.utc)
ROWS = [{"action": "push", "name": "old", "at_utc": "2026-10-02T21:00:00+00:00"},
        {"action": "quota_reset", "at_utc": "2026-10-03T00:18:00+00:00", "week_total_hours": 21.6,
         "last_reading_before_reset": {"gpu_used_hours": 5.6}},
        {"action": "fetch", "name": "old", "at_utc": "2026-10-03T08:00:00+00:00", "gpu_hours_this_run": 11.0},
        {"action": "push", "name": "a", "at_utc": "2026-10-03T06:00:00+00:00"},
        {"action": "fetch", "name": "a", "at_utc": "2026-10-03T07:00:00+00:00", "gpu_hours_this_run": 0.9},
        {"action": "push", "name": "b", "at_utc": "2026-10-03T09:00:00+00:00"},
        {"action": "push", "name": "c", "at_utc": "2026-10-03T10:00:00+00:00", "note": "NOT RUN: rejected"}]


def test_sessions_count_in_the_week_they_started():
    e = committed_estimate(0.0, ROWS, {"old", "b", "c"}, NOW)
    assert e["running_wall_clock_hours"] == {"b": 3.0}  # "old" started before the reset; "c" never ran
    k = config.get("training.kaggle")
    assert e["sessions_started_this_week_hours"] == 3.9
    assert e["week_estimate"] == 0.9 + k["session_hours_max"]  # "b" reserved at its full possible session
    assert e["cumulative_estimate"] == round(21.6 + 0.9 + k["session_hours_max"], 2) and e["within_caps"]


def test_quota_reading_is_a_floor_and_planned_hours_count():
    assert committed_estimate(5.0, ROWS, set(), NOW)["week_estimate"] == 5.0
    q = config.get("training.kaggle.weekly_gpu_hours_quota")
    assert committed_estimate(q - 3, [], set(), NOW, planned=3)["within_caps"]
    assert not committed_estimate(q - 3, [], set(), NOW, planned=3.5)["within_caps"]
