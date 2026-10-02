import datetime as dt

from scripts.free_tier_daily import next_wake

NOW = dt.datetime(2026, 10, 2, 13, 50, tzinfo=dt.timezone.utc)


def test_paused_wakes_at_resume_after():
    w = next_wake({"state": "paused_daily_quota", "resume_after": "2026-10-03T00:00:05-07:00"}, NOW)
    assert w == dt.datetime(2026, 10, 3, 7, 0, 5, tzinfo=dt.timezone.utc)


def test_missing_status_or_resume_waits_for_pacific_midnight():
    want = dt.datetime(2026, 10, 3, 7, 0, 5, tzinfo=dt.timezone.utc)
    assert next_wake(None, NOW) == want
    assert next_wake({"state": "paused_daily_quota"}, NOW) == want


def test_other_states_run_now():
    assert next_wake({"state": "complete", "pending": [{"id": 1}]}, NOW) is None
