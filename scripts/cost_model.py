"""Cost model, both regimes, from result files only (PRD section 7; T-028).

Inputs (each must be a real result file; anything else is refused):
  --load NAME=FILE     a serving/load_test.py output on kaggle_t4 (kind xbrl_frontier.load_test/v1), per variant
  --student-score FILE the student's scorer output on the shared frontier sample (kind of scripts/score_predictions.py)
  --arm NAME=SCORE,BATCH_USAGE,SYNC_USAGE  per frontier arm: its sample scorer output, its batch usage.jsonl
                       (harness collect) and its synchronous usage.jsonl (latency or probe run)
Self-host $/filing (T4, every rate key in serving.hardware.kaggle_t4.market_rate_keys, each utilization):
  offline = rate / max throughput (filings/h) / utilization; online = rate / goodput at serving.slo / utilization.
Frontier $/filing: measured usage on the sample, priced from the dated sheet (batch prices for offline, sync
prices for online), per filing of the sample. Comparator per gates.cost_comparator_rule using the student's
company-cluster bootstrap CI (paired with each arm); the gate is evaluated in gates.cost_gate_regime at the
headline rate key and the regime's utilization; the other regime is reported, not gated.
Writes data/phase4/cost_model.json; RTX 3050 rows never get dollars.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from scripts.curve_table import load_score
from serving.load_test import KIND as LOAD_KIND
from xbrl_frontier import bootstrap, config, cost, eval_sets, throughput

OUT = config.REPO_ROOT / "data" / "phase4" / "cost_model.json"


def load_points(path: Path) -> dict:
    rec = json.loads(path.read_text())
    if rec.get("kind") != LOAD_KIND or rec.get("card") != "kaggle_t4" or not rec.get("points"):
        raise SystemExit(f"{path} is not a kaggle_t4 load-test output with points; refusing")
    return rec


def usage_usd_per_filing(arm: str, usage: Path, sync: bool, filing_of: dict) -> dict:
    if not usage.exists():
        raise SystemExit(f"{usage} missing; refusing (measured usage only)")
    rows = [r for r in map(json.loads, open(usage)) if r.get("input_tokens") is not None and r.get("run", 0) == 0]
    if not rows:
        raise SystemExit(f"{usage} has no usage rows; refusing")
    p = config.prices()["api_per_million_tokens"][arm]
    usd = sum((r["input_tokens"] * (p["input"] if sync else p["batch_input"]) +
               (r["output_tokens"] or 0) * (p["output"] if sync else p["batch_output"])) / 1e6 for r in rows)
    filings = {filing_of.get(r["id"]) for r in rows} - {None}
    per_instance = usd / len(rows)
    return {"usd_total": usd, "instances": len(rows), "filings": len(filings), "usd_per_instance": per_instance}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", action="append", default=[], required=True)
    ap.add_argument("--student-score", type=Path, required=True)
    ap.add_argument("--arm", action="append", default=[], required=True)
    args = ap.parse_args()
    imap = eval_sets.load_instance_map()
    filing_of = {i: m["accession"] for i, m in imap.items()}
    company_of = {i: m["cik"] for i, m in imap.items()}
    stu = load_score(args.student_score, "finni")
    if not stu.get("ids_file"):
        raise SystemExit(f"{args.student_score} is not scored on the shared frontier sample (no ids_file); refusing")
    stu_counts = {r[0]: tuple(r[1:4]) for r in stu["dedup"]["per_instance"]}
    sample_ids = set(stu_counts)
    per_filing = defaultdict(int)
    for i in sample_ids:
        if i in filing_of:
            per_filing[filing_of[i]] += 1
    ipf = sum(per_filing.values()) / len(per_filing)
    shape = cost.TokenShape(0, 0, ipf, "measured: assigned dedup instances per sample filing")

    arms, arm_counts = {}, {}
    for spec in args.arm:
        name, files = spec.split("=", 1)
        score, batch_u, sync_u = (Path(x) for x in files.split(","))
        sc = load_score(score, "finni")
        arm_counts[name] = {r[0]: tuple(r[1:4]) for r in sc["dedup"]["per_instance"]}
        if set(arm_counts[name]) != sample_ids:
            raise SystemExit(f"{name}: scored instances differ from the student's shared sample; refusing")
        b = usage_usd_per_filing(name, batch_u, False, filing_of)
        s = usage_usd_per_filing(name, sync_u, True, filing_of)
        arms[name] = {"usd_per_filing": {"offline": b["usd_per_instance"] * ipf, "online": s["usd_per_instance"] * ipf},
                      "usage": {"batch": b, "sync": s}}
    bs = bootstrap.paired({"student": stu_counts, **arm_counts}, company_of)
    s_ci = bootstrap.interval(bs["samples"]["student"])
    for name in arms:
        arms[name]["f1"] = bs["point"][name]
        arms[name]["f1_ci"] = bootstrap.interval(bs["samples"][name])

    variants = {}
    for spec in args.load:
        name, path = spec.split("=", 1)
        rec = load_points(Path(path))
        variants[name] = {"file": path, "checkpoint_hashes": rec["checkpoint_hashes"],
                          **throughput.summarize([{k: p[k] for k in throughput.REQUIRED} for p in rec["points"]], shape)}
    rates = {k: v for k, v in cost.t4_rates().items() if v is not None}
    regimes = config.get("gates.cost_regimes")
    rows = []
    for vname, v in variants.items():
        for regime, rg in regimes.items():
            fph = v[rg["self_host_denominator"]]
            for rk, rate in rates.items():
                for util in [rg["utilization"]] + [u for u in config.get("serving.utilization_scenarios") if u != rg["utilization"]]:
                    rows.append({"variant": vname, "regime": regime, "t4_rate_key": rk, "utilization": util,
                                 "filings_per_hour": fph,
                                 "usd_per_filing": cost.self_host_usd_per_filing(rate, fph, util) if fph else None})

    inside = sorted((a for a in arms if s_ci[0] <= arms[a]["f1"] <= s_ci[1]), key=lambda a: arms[a]["usd_per_filing"]["offline"])
    above = sorted((a for a in arms if arms[a]["f1"] > s_ci[1]), key=lambda a: arms[a]["usd_per_filing"]["offline"])
    rule = config.get("gates.cost_comparator_rule")
    if inside:
        comp, which = inside[0], "primary"
    elif above:
        comp, which = above[0], "fallback_1"
    else:
        comp, which = min(arms, key=lambda a: arms[a]["usd_per_filing"]["offline"]), "fallback_2"
    gated = config.get("gates.cost_gate_regime")
    headline_key = config.get("serving.hardware.kaggle_t4.market_rate_keys")[0]
    gate = {}
    for vname in variants:
        for regime in regimes:
            r = next(x for x in rows if x["variant"] == vname and x["regime"] == regime and x["t4_rate_key"] == headline_key
                     and x["utilization"] == regimes[regime]["utilization"])
            ratio = arms[comp]["usd_per_filing"][regime] / r["usd_per_filing"] if r["usd_per_filing"] else None
            gate[f"{vname}|{regime}"] = {"self_host_usd_per_filing": r["usd_per_filing"],
                                         "comparator_usd_per_filing": arms[comp]["usd_per_filing"][regime],
                                         "cost_ratio": ratio, "threshold": config.get("gates.cost_ratio_vs_comparator_min"),
                                         "gated": regime == gated,
                                         "outcome": (None if ratio is None else ("pass" if ratio >= config.get("gates.cost_ratio_vs_comparator_min") else "fail"))
                                         if regime == gated else "reported, not gated"}
    out = {"price_sheet": config.get("pins.price_sheet"), "price_sheet_observed": config.prices()["observed"],
           "config_hash": config.config_hash(), "instances_per_filing_measured": ipf,
           "student": {"score_file": str(args.student_score), "f1": bs["point"]["student"], "f1_ci": s_ci,
                       "bootstrap_unit": bs["unit"], "companies": bs["companies"]},
           "arms": arms, "comparator": {"arm": comp, "rule_applied": which, "rule_text": rule[which],
                                        "label": "student dominates on quality" if which == "fallback_2" else None},
           "self_host_rows": rows, "variants": variants, "gate": gate, "headline_rate_key": headline_key,
           "formula": "self-host $/filing = T4 $/h / filings per hour / utilization; idle cost is not included"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({"comparator": out["comparator"], "gate": gate}, indent=1))


if __name__ == "__main__":
    main()
