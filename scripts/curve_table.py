"""Learning-curve table and chart data from scored curve checkpoints (PRD section 5; T-016, T-022).

Inputs: a training run's run.json (curve_checkpoints with prefix, instances_seen, global_step) and one
scorer output per checkpoint, given as --point PREFIX=SCORE_JSON. A row is produced only from a file
written by scripts/score_predictions.py (kind marker, set, raw/dedup overall and slices); anything else is
refused. Checkpoints without a score file get no row: they are listed under "missing", never filled in.
The full-corpus point is labelled "<prefix> (cap <curve point> not reached after decontamination, O-016)".
Writes data/curve/<tag>.json (rows, chart series, missing) and prints a markdown table.
"""
import argparse
import json
from pathlib import Path

from scripts.score_predictions import KIND
from xbrl_frontier import config

OUT = config.REPO_ROOT / "data" / "curve"
METRICS = ("finni_f1", "finni_precision", "finni_recall", "parse_failure_rate", "not_generated_rate")


def load_score(path: Path, expect_set: str) -> dict:
    rep = json.loads(path.read_text())
    if not isinstance(rep, dict) or rep.get("kind") != KIND:
        raise SystemExit(f"{path} is not a scripts/score_predictions.py output; refusing")
    if rep.get("set") != expect_set:
        raise SystemExit(f"{path} scores set {rep.get('set')!r}, not {expect_set!r}; refusing")
    for k in ("raw", "dedup"):
        if not all(m in rep[k]["overall"] for m in METRICS) or "context_type" not in rep[k]["slices"]:
            raise SystemExit(f"{path} lacks {k} metrics or the context_type slice; refusing")
    return rep


def label(t: dict, trained: int, skipped: int) -> str:
    if not t["full_corpus"]:
        return f"{t['prefix']:,}"
    s = f"{t['prefix']:,} (cap {t['curve_point']:,} not reached after decontamination, O-016)"
    if skipped:
        s += f"; {skipped:,} instances over max_seq_len skipped (T-018)"
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="run.json of the curve training run")
    ap.add_argument("--point", action="append", default=[], help="PREFIX=SCORE_JSON, repeatable")
    ap.add_argument("--set", default="dev", choices=["dev", "finni"])
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    run = json.loads(args.run.read_text())
    cps = {c["prefix"]: c for c in run.get("curve_checkpoints", [])}
    if not cps:
        raise SystemExit(f"{args.run} has no curve_checkpoints")
    given = {}
    for p in args.point:
        k, v = p.split("=", 1)
        if int(k) not in cps:
            raise SystemExit(f"prefix {k} is not a curve checkpoint of {args.run}")
        given[int(k)] = Path(v)
    rows, missing = [], []
    for prefix in sorted(cps):
        c = cps[prefix]
        if prefix not in given:
            missing.append({"curve_point": c["curve_point"], "prefix": prefix})
            continue
        rep = load_score(given[prefix], args.set)
        row = {"curve_point": c["curve_point"], "prefix": prefix,
               "label": label(c, run.get("trained_instances", prefix), run.get("skipped_over_max_seq_len", 0)),
               "instances_seen": c["instances_seen"], "global_step": c["global_step"],
               "score_file": str(given[prefix]), "predictions_sha256": rep["predictions_sha256"]}
        for k in ("raw", "dedup"):
            o = rep[k]["overall"]
            row[k] = {m: o[m] for m in METRICS} | {"instances": o["instances"]}
            row[k]["by_context_type"] = {t: v["finni_f1"] for t, v in rep[k]["slices"]["context_type"].items()}
        rows.append(row)
    series = {k: [{"x_instances_seen": r["instances_seen"], "label": r["label"], "f1": r[k]["finni_f1"],
                   "f1_table": r[k]["by_context_type"].get("table"), "f1_text": r[k]["by_context_type"].get("text")}
                  for r in rows] for k in ("raw", "dedup")}
    out = {"run": str(args.run), "model": run.get("model"), "seed": run.get("seed"), "set": args.set,
           "config_hash": config.config_hash(), "rows": rows, "series": series, "missing": missing,
           "note": "curve points count distinct instances of the fixed order seen in epoch 1 (T-016)"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
    print("| curve point | prefix | instances seen | F1 dedup | F1 table | F1 text | parse failure | not generated |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        d = r["dedup"]
        cell = lambda v: "" if v is None else f"{v:.4f}"
        print(f"| {r['curve_point']:,} | {r['label']} | {r['instances_seen']:,} | {cell(d['finni_f1'])} | "
              f"{cell(d['by_context_type'].get('table'))} | {cell(d['by_context_type'].get('text'))} | "
              f"{cell(d['parse_failure_rate'])} | {cell(d['not_generated_rate'])} |")
    if missing:
        print("missing (no score file, no row):", [m["curve_point"] for m in missing])


if __name__ == "__main__":
    main()
