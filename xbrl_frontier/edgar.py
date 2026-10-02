"""SEC EDGAR client with fair-access throttling, declared User-Agent and an on-disk cache."""
import hashlib
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import config

USER_AGENT = "Adnan Rashid adnanrashid230@gmail.com"
CACHE = config.REPO_ROOT / "data" / "edgar"


class SecRejected(RuntimeError):
    """SEC refused the declared identity; stop and log OWNER-BLOCKED, never route around it."""


class _Throttle:
    def __init__(self):
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self):
        interval = 1.0 / config.get("edgar.max_requests_per_second")
        with self._lock:
            now = time.monotonic()
            if now < self._next:
                time.sleep(self._next - now)
            self._next = max(now, self._next) + interval


_throttle = _Throttle()
stats = {"requests": 0, "cache_hits": 0}


def _cache_path(url: str) -> Path:
    h = hashlib.sha256(url.encode()).hexdigest()
    return CACHE / "cache" / h[:2] / h


def get(url: str, use_cache: bool = True) -> bytes:
    path = _cache_path(url)
    if use_cache and path.exists():
        stats["cache_hits"] += 1
        return path.read_bytes()
    backoffs = list(config.get("edgar.retry_backoff_seconds"))
    while True:
        _throttle.wait()
        stats["requests"] += 1
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 403:
                raise SecRejected(f"403 from SEC for {url}") from e
            if e.code in (429, 500, 502, 503, 504) and backoffs:
                time.sleep(backoffs.pop(0))
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if backoffs:
                time.sleep(backoffs.pop(0))
                continue
            raise
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(body)
    tmp.replace(path)
    return body


def full_text_search(phrase: str, forms: str, startdt: str, enddt: str) -> list[dict]:
    """Exact-phrase EDGAR FTS; returns hit _source dicts with the document filename added."""
    q = urllib.parse.urlencode({"q": f'"{phrase}"', "forms": forms, "dateRange": "custom",
                                "startdt": startdt, "enddt": enddt})
    data = json.loads(get(f"https://efts.sec.gov/LATEST/search-index?{q}"))
    out = []
    for h in data["hits"]["hits"]:
        src = dict(h["_source"])
        src["filename"] = h["_id"].split(":", 1)[1]
        out.append(src)
    return out


def submissions(cik: str) -> dict:
    return json.loads(get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"))


def filing_doc(cik: str, adsh: str, filename: str) -> bytes:
    return get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{adsh.replace('-', '')}/{filename}")
