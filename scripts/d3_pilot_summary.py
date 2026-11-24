"""Summarize retained D3 prediction rows without computing accuracy."""
import collections
import json
from pathlib import Path


def summarize(rows: list[dict], expected: set[str], seconds: float | None) -> dict:
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)) or not set(ids).issubset(expected):
        raise ValueError("prediction ids are duplicated or outside the pinned dev set")
    reasons = collections.Counter(row.get("finish_reason") for row in rows if not row.get("not_generated"))
    return {"status": "complete" if set(ids) == expected else "partial", "rows": len(rows),
            "expected_rows": len(expected), "seconds": seconds, "cut_off": reasons.get("length", 0),
            "not_generated": sum(bool(row.get("not_generated")) for row in rows), "finish_reasons": dict(reasons)}


def main() -> None:
    root = Path("data/kaggle/d3-pilot-output/d3-pilot")
    pred = root / "predictions_final_dev.jsonl"
    meta = root / "predictions_dev.meta.json"
    dev = Path("data/corpus/dev.jsonl")
    expected = {json.loads(line)["id"] for line in dev.read_text().splitlines()}
    rows = [json.loads(line) for line in pred.read_text().splitlines()] if pred.exists() else []
    record = json.loads(meta.read_text()) if meta.exists() else {}
    target = record.get("targets", {}).get("final", {})
    result = summarize(rows, expected, target.get("seconds_this_session"))
    result["meta_present"] = meta.exists()
    result["timing_only"] = True
    result["provider_status"] = Path("data/kaggle/d3-pilot-terminal.txt").read_text().strip()
    (root / "timing_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
