"""Read live quota and recorded-kernel reservations before a D4 UI run."""
import argparse
import datetime as dt
import json
import re

from scripts import kaggle_run


def is_active(status: str) -> bool:
    match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", status)
    if not match:
        raise ValueError("kernel status unavailable")
    state = match.group(1)
    if state in {"RUNNING", "QUEUED"}:
        return True
    if state in {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"}:
        return False
    raise ValueError("kernel status not safely classified")


def check_week(rows: list[dict], now: dt.datetime) -> None:
    resets = [row for row in rows if row.get("action") == "quota_reset"]
    if not resets or not resets[-1].get("next_refresh_utc"):
        raise ValueError("quota reset history is incomplete")
    refresh = dt.datetime.fromisoformat(resets[-1]["next_refresh_utc"].replace("Z", "+00:00"))
    if refresh.tzinfo is None or now.tzinfo is None or now >= refresh:
        raise ValueError("weekly reset must be reconciled in the ledger before budgeting")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planned-hours", type=float, required=True)
    args = parser.parse_args()
    from training.schedule import session_hours
    planned = session_hours(args.planned_hours)
    if planned > 8:
        raise SystemExit("D4 session exceeds its requested provider bound")
    rows = [json.loads(line) for line in kaggle_run.RUNS.read_text().splitlines()]
    now = dt.datetime.now(dt.timezone.utc)
    check_week(rows, now)
    names = {row["name"] for row in rows if row.get("action") == "push"}
    running = {name for name in names if is_active(kaggle_run.status(name))}
    estimate = kaggle_run.committed_estimate(kaggle_run.gpu_used_hours(), rows, running,
                                              now, planned)
    print(json.dumps(estimate, indent=2))
    if not estimate["within_caps"]:
        raise SystemExit("D4 reservation exceeds a configured cap")


if __name__ == "__main__":
    main()
