"""Generation statistics for a FinNI predictions file, split table vs text (T-039 / T-042 reporting).

Per context_type and overall: rows, not_generated, finish_reason counts, completion_tokens quantiles, cap hits
(completion_tokens >= max_tokens), and per-session counts. Partial files are allowed (rows present only).
Writes <pred>.stats.json next to the predictions file.
"""
import argparse
import collections
import json
from pathlib import Path

from xbrl_frontier import eval_sets, finni


def _q(v: list[int]) -> dict:
    if not v:
        return {}
    v = sorted(v)
    return {f"p{p}": v[min(len(v) - 1, int(len(v) * p / 100))] for p in (50, 90, 99)} | {"max": v[-1], "mean": round(sum(v) / len(v), 1)}


def stats(rows: list[dict], kind: dict) -> dict:
    out = {}
    groups = collections.defaultdict(list)
    for r in rows:
        groups[kind[r["id"]]].append(r)
        groups["all"].append(r)
    for g, rs in sorted(groups.items()):
        gen = [r for r in rs if not r.get("not_generated")]
        out[g] = {"rows": len(rs), "not_generated": len(rs) - len(gen),
                  "finish_reason": dict(collections.Counter(r.get("finish_reason") for r in gen)),
                  "completion_tokens": _q([r["completion_tokens"] for r in gen if r.get("completion_tokens") is not None]),
                  "cap_hits": sum(1 for r in gen if r.get("max_tokens") and r["completion_tokens"] >= r["max_tokens"]),
                  "sessions": dict(collections.Counter(r.get("session", 1) for r in rs))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    args = ap.parse_args()
    kind = {it.idx: eval_sets.context_type(it) for it in finni.load()}
    rows = [json.loads(l) for l in open(args.pred)]
    out = {"predictions": str(args.pred), "instances_total": len(kind), "rows_present": len(rows), "by_context_type": stats(rows, kind)}
    path = args.pred.with_suffix(".stats.json")
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
