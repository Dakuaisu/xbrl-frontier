"""Priced plan for every live frontier step against the budget (PRD section 7; T-024, T-026). No network.

Per arm and step, priced from the dated price sheet:
  probe          frontier.probe_filings filing(s) at synchronous prices
  batch          N sample filings at batch prices x (1 + reasoning_token_margin)
  latency        frontier.sync_latency_instances instances x frontier.sync_runs at synchronous prices
  fresh_control  every fresh-slice instance, frontier.fresh_slice_control_arm only, batch x (1 + margin)
Per-filing and per-instance token shapes are the planning shape (xbrl_frontier.cost.planning_shape) until
the probe measures them; the fresh control uses the actual fresh instances through the harness's estimator.
plan(arms, n) is reused by the power check and the sampler ("largest N whose priced plan leaves the reserve").
Writes data/phase3/frontier_plan.json and prints the table with the cap/reserve check.
"""
import json
from functools import lru_cache

from xbrl_frontier import config, cost


@lru_cache(maxsize=None)
def fresh_control_usd() -> float:
    from frontier.harness import instances, price
    return price(config.get("frontier.fresh_slice_control_arm"), "fresh_control",
                 instances("fresh_control", None, None, None))["usd_estimate"]


def plan(arms: list[str], n: int) -> dict:
    shape = cost.planning_shape()
    prices = config.prices()["api_per_million_tokens"]
    f = config.get("frontier")
    m = f["reasoning_token_margin"]
    rows = []
    for arm in arms:
        p = prices[arm]
        sync = (shape.input_tokens_per_instance * p["input"] + shape.output_tokens_per_instance * p["output"]) / 1e6
        batch = (shape.input_tokens_per_instance * p["batch_input"] + shape.output_tokens_per_instance * p["batch_output"]) / 1e6
        rows.append({"arm": arm, "step": "probe", "usd": f["probe_filings"] * shape.instances_per_filing * sync,
                     "basis": "planning shape, sync"})
        rows.append({"arm": arm, "step": "batch", "usd": n * shape.instances_per_filing * batch * (1 + m),
                     "basis": f"planning shape, {n} filings, batch x (1+margin)"})
        rows.append({"arm": arm, "step": "latency", "usd": f["sync_latency_instances"] * f["sync_runs"] * sync,
                     "basis": "planning shape, sync"})
    rows.append({"arm": f["fresh_slice_control_arm"], "step": "fresh_control", "usd": fresh_control_usd(),
                 "basis": "actual fresh instances, token proxy, batch x (1+margin)"})
    total = sum(r["usd"] for r in rows)
    cap = config.get("budget.usd_cap_frontier_api")
    reserve = config.get("frontier.usd_reserve_min")
    return {"arms": list(arms), "sample_filings": n, "rows": rows, "total_usd": total,
            "budget_usd_cap_frontier_api": cap, "frontier_usd_reserve_min": reserve,
            "leaves_reserve": cap - total >= reserve, "unspent_usd": cap - total}


def largest_n(arms: list[str], n_max: int) -> int:
    """Largest N <= n_max whose priced plan leaves frontier.usd_reserve_min (0 if none does)."""
    best = 0
    for n in range(1, n_max + 1):
        if plan(arms, n)["leaves_reserve"]:
            best = n
        else:
            break
    return best


def main() -> None:
    from frontier.harness import PHASE3
    out = plan(config.get("frontier.running_arms"), config.get("frontier.sample_filings"))
    shape = cost.planning_shape()
    f = config.get("frontier")
    from frontier.harness import instances
    out["free_tier_requests_planned"] = {
        "probe": int(f["probe_filings"] * shape.instances_per_filing), "batch": int(f["sample_filings"] * shape.instances_per_filing),
        "latency": f["sync_latency_instances"] * f["sync_runs"], "fresh_control": len(instances("fresh_control", None, None, None))}
    out["free_tier_requests_total"] = sum(out["free_tier_requests_planned"].values())
    for r in out["rows"]:
        r["free_tier_requests"] = out["free_tier_requests_planned"][r["step"]]
    state = PHASE3 / "gemini_quota_state.json"
    rpd = json.loads(state.read_text()).get("rpd") if state.exists() else None
    out["free_tier_rpd_learned"] = rpd
    out["free_tier_days_needed"] = (-(-out["free_tier_requests_total"] // rpd)) if rpd else None
    out["billed_usd"] = 0.0
    out["note_t033"] = "T-033: only frontier.running_arms run, on the free tier; dollars are notional (measured tokens x the dated sheet)"
    out.update(price_sheet=config.get("pins.price_sheet"), price_sheet_observed=config.prices()["observed"],
               config_hash=config.config_hash(),
               note="estimates; the probe replaces the planning shape with measured tokens and N is recomputed (PRD section 7)")
    PHASE3.mkdir(parents=True, exist_ok=True)
    (PHASE3 / "frontier_plan.json").write_text(json.dumps(out, indent=1))
    print(f"Prices: {out['price_sheet']} (observed {out['price_sheet_observed']})")
    print("| arm | step | est. $ | basis |\n|---|---|---|---|")
    for r in out["rows"]:
        print(f"| {r['arm']} | {r['step']} | {r['usd']:.3f} | {r['basis']} |")
    print(f"| total | | {out['total_usd']:.3f} | cap {out['budget_usd_cap_frontier_api']}, reserve "
          f"{out['frontier_usd_reserve_min']}: leaves reserve = {out['leaves_reserve']} |")


if __name__ == "__main__":
    main()
