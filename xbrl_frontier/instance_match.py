"""Assign FinNI contexts to source filings: normalized exact match, then fuzzy shingle containment."""
import html
import re
from collections import defaultdict
from dataclasses import dataclass

from . import config

_TOKEN = re.compile(r"[a-z0-9]+")
_DROP = re.compile(r"<(script|style)\b.*?</\1>|<ix:header\b.*?</ix:header>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")


def tokens_from_html(s: str) -> list[str]:
    """Tokens per tag-delimited segment; a segment that is exactly two copies of itself collapses to one.

    FinNI's serializer doubles labelled cell text ("Date Declared Date Declared"); collapsing on both
    sides makes context and filing comparable without rebuilding the serializer.
    """
    s = _DROP.sub(" ", s)
    out = []
    for seg in _TAG.split(s):
        t = _TOKEN.findall(html.unescape(seg).replace("\xa0", " ").lower())
        n = len(t)
        if n >= 2 and n % 2 == 0 and t[:n // 2] == t[n // 2:]:
            t = t[:n // 2]
        out.extend(t)
    return out


def _shingles(toks: list[str], k: int) -> list[tuple]:
    if len(toks) < k:
        return [tuple(toks)] if toks else []
    return [tuple(toks[i:i + k]) for i in range(len(toks) - k + 1)]


@dataclass
class Match:
    filing: str | None
    method: str          # "exact" | "fuzzy" | "none"
    containment: float
    runner_up: float


class ContextIndex:
    """Index over distinct context strings; each keeps an evenly spaced sample of its shingles."""

    def __init__(self, contexts: dict[str, str]):
        self.k = config.get("contamination.instance_match.shingle_words")
        cap = config.get("contamination.instance_match.max_shingles_per_context")
        self.tokens: dict[str, str] = {}
        self.n: dict[str, int] = {}
        self.index: dict[tuple, list[str]] = defaultdict(list)
        for cid, ctx in contexts.items():
            toks = tokens_from_html(ctx)
            self.tokens[cid] = " ".join(toks)
            if len(toks) < self.k:
                self.n[cid] = 0
                continue
            sh = list(dict.fromkeys(_shingles(toks, self.k)))
            if len(sh) > cap:
                step = len(sh) / cap
                sh = [sh[int(i * step)] for i in range(cap)]
            self.n[cid] = len(sh)
            for s in sh:
                self.index[s].append(cid)
        self.best: dict[str, list] = {}

    def scan(self, filing_id: str, filing_html: str, only: set[str] | None = None) -> int:
        """Score every (or only the given) context against one filing; returns contexts touched."""
        toks = tokens_from_html(filing_html)
        joined = " " + " ".join(toks) + " "
        hits: dict[str, set] = defaultdict(set)
        k = self.k
        for i in range(len(toks) - k + 1):
            s = tuple(toks[i:i + k])
            for cid in self.index.get(s, ()):
                if only is None or cid in only:
                    hits[cid].add(s)
        for cid, found in hits.items():
            c = len(found) / self.n[cid]
            exact = (" " + self.tokens[cid] + " ") in joined
            self.best.setdefault(cid, []).append((exact, c, filing_id))
        return len(hits)

    def _ranked(self, cid: str, allowed: set[str] | None) -> list:
        cands = [c for c in self.best.get(cid, []) if allowed is None or c[2] in allowed]
        cands.sort(key=lambda x: (not x[0], -x[1], x[2]))
        return cands

    def unique_exact(self) -> dict[str, int]:
        """Per filing: contexts whose only exact match among all scanned filings is that filing."""
        out: dict[str, int] = defaultdict(int)
        for cid in self.tokens:
            exact = [c[2] for c in self.best.get(cid, []) if c[0]]
            if len(exact) == 1:
                out[exact[0]] += 1
        return dict(out)

    def assign(self, allowed: set[str] | None = None, company_of: dict | None = None,
               dominant: dict | None = None) -> dict[str, Match]:
        """Best filing per context among `allowed` filings (all scanned if None). Exact beats fuzzy.

        A tie at the top between filings of one company goes to that company's dominant filing
        (method suffix _tie_within_company); a tie across companies is left unassigned as
        ambiguous_cross_company. Without company_of, ties go to the lowest filing id.
        """
        thr = config.get("contamination.instance_match.fuzzy_min_containment")
        out = {}
        for cid in self.tokens:
            if self.n[cid] == 0:
                out[cid] = Match(None, "too_short", 0.0, 0.0)
                continue
            cands = self._ranked(cid, allowed)
            if not cands or not (cands[0][0] or cands[0][1] >= thr):
                out[cid] = Match(None, "none", cands[0][1] if cands else 0.0, 0.0)
                continue
            key = cands[0][:2]
            top = [c for c in cands if c[:2] == key]
            method = "exact" if key[0] else "fuzzy"
            runner = max((c[1] for c in cands if c[:2] != key), default=0.0)
            pick = top[0]
            if len(top) > 1 and company_of is not None:
                companies = {company_of[c[2]] for c in top}
                if len(companies) > 1:
                    out[cid] = Match(None, "ambiguous_cross_company", key[1], key[1])
                    continue
                dom = (dominant or {}).get(next(iter(companies)))
                pick = next((c for c in top if c[2] == dom), top[0])
                method += "_tie_within_company"
            out[cid] = Match(pick[2], method, pick[1], runner)
        return out
