"""Break-even and gate-pass goodput per arm and regime (PRD section 7, serving.phase0_throughput_benchmark).

Without --results: the price-sheet-only table. With --results DIR (the Kaggle phase0 output directory):
adds measured max throughput and goodput per base model and whether each arm is beatable on cost.
Writes data/phase0/break_even.json and prints a markdown table.
"""
import argparse
import json
from pathlib import Path

from xbrl_frontier import config, cost, throughput

OUT = config.REPO_ROOT / "data" / "phase0"


def load_measured(results_dir: Path) -> dict:
    shape = cost.planning_shape()
    out = {}
    for f in sorted(results_dir.glob("benchmark_*.json")):
        rec = json.loads(f.read_text())
        out[rec["model"]] = {"run": f.name, "env": rec.get("env", {}),
                             **throughput.summarize(rec["points"], shape)}
    if not out:
        raise FileNotFoundError(f"no benchmark_*.json under {results_dir}")
    return out


def build(measured: dict | None) -> dict:
    shape = cost.planning_shape()
    arms = list(config.get("pins.frontier_arms"))
    swap = config.get("pins.arm_swap_if_over_budget")["with"]
    rates = cost.t4_rates()
    rows, dropped = [], [k for k, v in rates.items() if v is None]
    for arm in arms + [swap]:
        for regime in ("offline", "online"):
            reg = config.get(f"gates.cost_regimes.{regime}")
            arm_usd = cost.arm_usd_per_filing(arm, regime, shape)
            for rate_key, rate in rates.items():
                if rate is None:
                    continue
                for util in [reg["utilization"]] + [u for u in config.get("serving.utilization_scenarios")
                                                    if u != reg["utilization"]]:
                    be = cost.break_even_filings_per_hour(rate, arm_usd, util)
                    row = {"arm": arm, "arm_role": "swap_candidate" if arm == swap else "pinned",
                           "regime": regime, "frontier_price": reg["frontier_price"],
                           "arm_usd_per_filing": arm_usd, "t4_rate_key": rate_key, "t4_usd_per_hour": rate,
                           "utilization": util, "gated_utilization": util == reg["utilization"],
                           "break_even_goodput_filings_per_hour": be,
                           "gate_pass_goodput_filings_per_hour": cost.gate_pass_filings_per_hour(be)}
                    if measured:
                        for model, m in measured.items():
                            have = m[reg["self_host_denominator"]]
                            row[f"measured::{model}"] = have
                            row[f"beatable_at_gate::{model}"] = have >= row["gate_pass_goodput_filings_per_hour"]
                            row[f"breaks_even::{model}"] = have >= be
                    rows.append(row)
    return {
        "price_sheet": config.get("pins.price_sheet"), "price_sheet_observed": config.prices()["observed"],
        "config_hash": config.config_hash(), "token_shape": shape.__dict__,
        "t4_rate_keys_dropped_blank": dropped, "measured": measured, "rows": rows,
    }


def markdown(report: dict) -> str:
    models = sorted(report["measured"] or {})
    head = ["arm", "regime", "T4 rate key", "util", "arm $/filing", "break-even f/h", "gate-pass f/h"]
    head += [f"{m} measured f/h" for m in models] + [f"{m} beatable" for m in models]
    lines = [f"Prices: {report['price_sheet']} (observed {report['price_sheet_observed']}). "
             f"Blank T4 sources dropped: {report['t4_rate_keys_dropped_blank'] or 'none'}.", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in report["rows"]:
        cells = [r["arm"] + (" (swap)" if r["arm_role"] == "swap_candidate" else ""), r["regime"], r["t4_rate_key"],
                 f"{r['utilization']:.2f}" + ("*" if r["gated_utilization"] else ""), f"{r['arm_usd_per_filing']:.4f}",
                 f"{r['break_even_goodput_filings_per_hour']:.1f}", f"{r['gate_pass_goodput_filings_per_hour']:.1f}"]
        cells += [f"{r[f'measured::{m}']:.1f}" for m in models] + [str(r[f"beatable_at_gate::{m}"]) for m in models]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("* = the regime's utilization in gates.cost_regimes; other rows are serving.utilization_scenarios.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, help="Kaggle phase0 output directory with benchmark_*.json")
    args = ap.parse_args()
    report = build(load_measured(args.results) if args.results else None)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "break_even.json").write_text(json.dumps(report, indent=1))
    md = markdown(report)
    (OUT / "break_even.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
