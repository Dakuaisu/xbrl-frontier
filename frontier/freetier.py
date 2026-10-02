"""Gemini free-tier pacing, quota learning and retries (owner decision T-033; TRADEOFFS T-034).

Google publishes no per-model free-tier numbers (they are shown only in AI Studio) and generateContent responses
carry no rate-limit headers (checked 2026-10-02). So: start at frontier.free_tier.assumed_rpm; on a 429, read the
QuotaFailure/RetryInfo details (quotaId, quotaValue, retryDelay), persist what they say in
data/phase3/gemini_quota_state.json, and pace by it. A per-minute 429 waits retryDelay (or the backoff list) and
retries the same request. A per-day 429 raises DailyQuotaExhausted with the next midnight in
frontier.free_tier.daily_reset_tz: the run stops, the instance stays pending, and a later run resumes. 5xx and
network errors use the backoff list, then raise. A failed call is never turned into a prediction.
"""
import datetime as dt
import json
import re
import time
import urllib.error
from pathlib import Path
from zoneinfo import ZoneInfo

from xbrl_frontier import config

STATE = config.REPO_ROOT / "data" / "phase3" / "gemini_quota_state.json"


class DailyQuotaExhausted(RuntimeError):
    def __init__(self, resume_at: str, detail: dict):
        super().__init__(f"free-tier daily quota exhausted; resume after {resume_at}")
        self.resume_at, self.detail = resume_at, detail


class CallFailed(RuntimeError):
    pass


def parse_429(body: bytes) -> dict:
    """{per_day, per_minute, quota_value, quota_id, retry_s} from a Google 429 error body (missing fields None)."""
    try:
        err = json.loads(body).get("error", {})
    except (ValueError, AttributeError):
        err = {}
    out = {"per_day": False, "per_minute": False, "quota_value": None, "quota_id": None, "retry_s": None,
           "message": str(err.get("message", ""))[:300]}
    for d in err.get("details", []) or []:
        t = d.get("@type", "")
        if t.endswith("QuotaFailure"):
            for v in d.get("violations", []):
                qid = v.get("quotaId", "") + " " + v.get("quotaMetric", "")
                out["quota_id"] = qid.strip()
                if v.get("quotaValue") is not None:
                    out["quota_value"] = int(v["quotaValue"])
                out["per_day"] |= bool(re.search(r"PerDay", qid, re.I))
                out["per_minute"] |= bool(re.search(r"PerMinute", qid, re.I))
        if t.endswith("RetryInfo") and d.get("retryDelay"):
            out["retry_s"] = float(str(d["retryDelay"]).rstrip("s"))
    return out


def next_reset(now: dt.datetime | None = None) -> str:
    tz = ZoneInfo(config.get("frontier.free_tier.daily_reset_tz"))
    now = (now or dt.datetime.now(dt.timezone.utc)).astimezone(tz)
    return (now + dt.timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0).isoformat()


class Pacer:
    """Spaces requests by the learned per-minute limit (or the assumed starting pace)."""

    def __init__(self, state_path: Path = STATE, clock=time.monotonic, sleep=time.sleep):
        self.path, self.clock, self.sleep = state_path, clock, sleep
        self.state = json.loads(state_path.read_text()) if state_path.exists() else {}
        self.last = None

    @property
    def rpm(self) -> float:
        return float(self.state.get("rpm") or config.get("frontier.free_tier.assumed_rpm"))

    def wait(self) -> float:
        gap = 60.0 / self.rpm
        waited = 0.0
        if self.last is not None:
            waited = max(0.0, self.last + gap - self.clock())
            if waited:
                self.sleep(waited)
        self.last = self.clock()
        return waited

    def learn(self, info: dict) -> None:
        if info.get("per_minute") and info.get("quota_value"):
            self.state["rpm"] = info["quota_value"]
        if info.get("per_day") and info.get("quota_value"):
            self.state["rpd"] = info["quota_value"]
        self.state.setdefault("events", []).append({**info, "at": dt.datetime.now(dt.timezone.utc).isoformat()})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=1))


def call_with_retries(send, pacer: Pacer, sleep=None) -> tuple[bytes, float, dict]:
    """send() performs one HTTP request and returns the body. Returns (body, request seconds, retry log)."""
    sleep = sleep or time.sleep
    backoff = list(config.get("frontier.free_tier.backoff_seconds"))
    log = {"retries_429": 0, "retries_5xx": 0, "throttle_wait_s": 0.0}
    while True:
        log["throttle_wait_s"] += pacer.wait()
        t0 = time.perf_counter()
        try:
            body = send()
            return body, time.perf_counter() - t0, log
        except urllib.error.HTTPError as e:
            raw = e.read() if hasattr(e, "read") else b""
            if e.code == 429:
                info = parse_429(raw)
                pacer.learn(info)
                if info["per_day"]:
                    raise DailyQuotaExhausted(next_reset(), info) from e
                log["retries_429"] += 1
                delay = info["retry_s"] if info["retry_s"] is not None else (backoff.pop(0) if backoff else None)
                if delay is None:
                    raise CallFailed(f"429 retries exhausted: {info['message']}") from e
                sleep(delay)
                continue
            if 500 <= e.code < 600 and backoff:
                log["retries_5xx"] += 1
                sleep(backoff.pop(0))
                continue
            raise CallFailed(f"HTTP {e.code}: {raw[:300]!r}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if backoff:
                log["retries_5xx"] += 1
                sleep(backoff.pop(0))
                continue
            raise CallFailed(f"network error: {e}") from e
