import pytest
import datetime as dt

from scripts import d4_budget


@pytest.mark.parametrize("status", ["RUNNING", "QUEUED"])
def test_active_status_reserves_budget(status):
    assert d4_budget.is_active('x has status "KernelWorkerStatus.' + status + '"')


@pytest.mark.parametrize("status", ["COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"])
def test_terminal_status_needs_no_reservation(status):
    assert not d4_budget.is_active('x has status "KernelWorkerStatus.' + status + '"')


def test_unknown_status_refuses_budget_estimate():
    with pytest.raises(ValueError):
        d4_budget.is_active("status unavailable")


def test_budget_rejects_unreconciled_weekly_reset():
    rows = [{"action": "quota_reset", "next_refresh_utc": "2026-11-28T00:00:00Z"}]
    with pytest.raises(ValueError):
        d4_budget.check_week(rows, dt.datetime(2026, 11, 28, tzinfo=dt.timezone.utc))


def test_budget_accepts_week_before_recorded_refresh():
    rows = [{"action": "quota_reset", "next_refresh_utc": "2026-11-28T00:00:00Z"}]
    d4_budget.check_week(rows, dt.datetime(2026, 11, 27, tzinfo=dt.timezone.utc))
