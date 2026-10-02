"""Priced plan for every live frontier step against the budget (PRD section 7; T-024). No network.

Per arm and step, priced from the dated price sheet:
  probe          frontier.probe_filings filing(s) at synchronous prices
  batch          frontier.sample_filings filings at batch prices x (1 + reasoning_token_margin)
  latency        frontier.sync_latency_instances instances x frontier.sync_runs at synchronous prices
  fresh_control  every fresh-slice instance, frontier.fresh_slice_control_arm only, batch x (1 + margin)
Per-filing and per-instance token shapes are the planning shape (xbrl_frontier.cost.planning_shape) until
the probe measures them; the fresh control uses the actual fresh instances through the harness's estimator.
Writes data/phase3/frontier_plan.json and prints the table with the cap/reserve check.
"""
import json

from frontier.harness import PHASE3, instances, price
from xbrl_frontier import config, cost


def main() -> None:
    shape = cost.planning_shape()
    prices = config.prices()["api_per_million_tokens"]
    f = config.get("frontier")
    m = f["reasoning_token_margin"]
    rows = []
    for arm in config.get("pins.frontier_arms"):
        p = prices[arm]
        per_inst_sync = (shape.input_tokens_per_instance * p["input"] + shape.output_tokens_per_instance * p["output"]) / 1e6
        per_inst_batch = (shape.input_tokens_per_instance * p["batch_input"] + shape.output_tokens_per_instance * p["batch_output"]) / 1e6
        rows.append({"arm": arm, "step": "probe", "usd": f["probe_filings"] * shape.instances_per_filing * per_inst_sync,
                     "basis": "planning shape, sync"})
        rows.append({"arm": arm, "step": "batch", "usd": f["sample_filings"] * shape.instances_per_filing * per_inst_batch * (1 + m),
                     "basis": "planning shape, batch x (1+margin)"})
        rows.append({"arm": arm, "step": "latency", "usd": f["sync_latency_instances"] * f["sync_runs"] * per_inst_sync,
                     "basis": "planning shape, sync"})
    fc = price(f["fresh_slice_control_arm"], "fresh_control", instances("fresh_control", None, None, None))
    rows.append({"arm": f["fresh_slice_control_arm"], "step": "fresh_control", "usd": fc["usd_estimate"],
                 "basis": "actual fresh instances, token proxy, batch x (1+margin)"})
    total = sum(r["usd"] for r in rows)
    cap = config.get("budget.usd_cap_frontier_api")
    reserve = config.get("frontier.usd_reserve_min")
    plan = {"price_sheet": config.get("pins.price_sheet"), "price_sheet_observed": config.prices()["observed"],
            "config_hash": config.config_hash(), "rows": rows, "total_usd": total,
            "budget_usd_cap_frontier_api": cap, "frontier_usd_reserve_min": reserve,
            "leaves_reserve": cap - total >= reserve, "unspent_usd": cap - total,
            "note": "estimates; the probe replaces the planning shape with measured tokens and N is recomputed (PRD section 7)"}
    PHASE3.mkdir(parents=True, exist_ok=True)
    (PHASE3 / "frontier_plan.json").write_text(json.dumps(plan, indent=1))
    print(f"Prices: {plan['price_sheet']} (observed {plan['price_sheet_observed']})")
    print("| arm | step | est. $ | basis |\n|---|---|---|---|")
    for r in rows:
        print(f"| {r['arm']} | {r['step']} | {r['usd']:.3f} | {r['basis']} |")
    print(f"| total | | {total:.3f} | cap {cap}, reserve {reserve}: leaves reserve = {plan['leaves_reserve']} |")


if __name__ == "__main__":
    main()
