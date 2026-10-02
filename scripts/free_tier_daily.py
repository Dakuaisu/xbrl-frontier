"""Daily free-tier collection loop for E4 (operational; T-033, T-035, T-036).

Loop: if status.json is `complete` with an empty pending list, exit 0. If it is `stopped_call_failed`, exit 4 (an id
exhausted its retries: that is a finding for docs/OPEN.md, not something to retry blindly). Otherwise sleep until
`resume_after` (or, if absent, the next midnight in frontier.free_tier.daily_reset_tz + 5 s), then run exactly
`frontier.harness run --step fresh_control --arm gemini-3.5-flash-lite --live --approval T-033`, appending its output
to daily.log. Same guard, same key from the environment.
"""
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

from frontier.freetier import next_reset
from xbrl_frontier import config

OUT = config.REPO_ROOT / "data" / "phase3" / "live" / "fresh_control_gemini-3.5-flash-lite"
CMD = [sys.executable, "-m", "frontier.harness", "run", "--step", "fresh_control", "--arm", "gemini-3.5-flash-lite",
       "--live", "--approval", "T-033"]


def next_wake(status: dict | None, now: dt.datetime) -> dt.datetime | None:
    """None = run now. A paused status wakes at its resume_after; a missing one at the next reset."""
    if status is None or status.get("state") not in ("paused_daily_quota", None):
        return None if status is not None else dt.datetime.fromisoformat(next_reset(now))
    ra = status.get("resume_after")
    return dt.datetime.fromisoformat(ra) if ra else dt.datetime.fromisoformat(next_reset(now))


def log(msg: str) -> None:
    with open(OUT / "daily.log", "a") as f:
        f.write(f"[{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}] {msg}\n")


def main() -> int:
    while True:
        sp = OUT / "status.json"
        st = json.loads(sp.read_text()) if sp.exists() else None
        if st and st.get("state") == "complete" and not st.get("pending"):
            log("status complete with an empty pending list; exiting")
            return 0
        if st and st.get("state") == "stopped_call_failed":
            log(f"stopped_call_failed: {st.get('error')}; exiting for a finding (not retried)")
            return 4
        wake = next_wake(st, dt.datetime.now(dt.timezone.utc))
        if wake:
            secs = (wake - dt.datetime.now(dt.timezone.utc)).total_seconds()
            if secs > 0:
                log(f"sleeping {secs:.0f} s until {wake.isoformat()}")
                time.sleep(secs)
        log("running: " + " ".join(CMD[1:]))
        with open(OUT / "daily.log", "a") as f:
            rc = subprocess.run(CMD, stdout=f, stderr=subprocess.STDOUT, cwd=config.REPO_ROOT).returncode
        log(f"harness exit code {rc}")
        if rc not in (0, 3, 4):
            log("unexpected exit code; exiting")
            return rc


if __name__ == "__main__":
    raise SystemExit(main())
