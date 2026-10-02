"""Live-call guard under owner decision T-033: $0 billed, Gemini free tier only (TRADEOFFS T-034).

A live call is allowed only when all of these hold, checked before any API key is read:
  - --approval equals frontier.free_tier.policy ("T-033") and docs/TRADEOFFS.md contains that owner entry;
  - the arm is in frontier.running_arms, uses the Gemini API, and frontier.free_tier.api_key_env is the key;
  - the arm's thinking setting has a verified source (frontier.thinking_per_arm.<arm>.source, T-025);
  - the call is synchronous generateContent (mode "sync"): the Batch API is refused because its quota exists only
    on paid tiers (frontier.free_tier.batch_api false), so it could bill.
Every other arm, provider, mode or approval is refused. The ledger records billed $0 and the notional price
(measured tokens x the dated sheet) used for the cost model.
"""
import json
import re
from pathlib import Path

from xbrl_frontier import config

LEDGER = config.REPO_ROOT / "data" / "phase3" / "spend_ledger.jsonl"
TRADEOFFS = config.DOCS / "TRADEOFFS.md"
BATCH_STEPS = {"batch", "fresh_control"}


class GuardRefused(RuntimeError):
    pass


def check(step: str, arm: str, approval_id: str | None, mode: str = "sync", tradeoffs: Path = TRADEOFFS) -> dict:
    ft = config.get("frontier.free_tier")
    if approval_id != ft["policy"]:
        raise GuardRefused(f"live call refused: only --approval {ft['policy']} (free tier, $0 billed) is accepted")
    if not re.search(rf"(?m)^## {re.escape(ft['policy'])} OWNER DECISION", tradeoffs.read_text()):
        raise GuardRefused(f"live call refused: docs/TRADEOFFS.md has no '## {ft['policy']} OWNER DECISION' entry")
    if arm not in config.get("frontier.running_arms"):
        raise GuardRefused(f"live call refused: {arm} is not in frontier.running_arms (T-033: no key, $0 billed)")
    spec = config.get("frontier.thinking_per_arm").get(arm) or {}
    if spec.get("api") != "gemini_generate_content" or ft["api_key_env"] != "GEMINI_API_KEY":
        raise GuardRefused(f"live call refused: {arm} is not a Gemini free-tier arm")
    if not spec.get("source"):
        raise GuardRefused(f"live call refused: frontier.thinking_per_arm.{arm} has no verified source")
    if mode != "sync" or ft["batch_api"]:
        raise GuardRefused("live call refused: only synchronous generateContent runs on the free tier (Batch API could bill)")
    return {"approval": approval_id, "arm": arm, "step": step, "mode": mode, "billed_usd": 0.0}


def ledger_total(ledger: Path = LEDGER) -> dict:
    rows = [json.loads(l) for l in open(ledger) if l.strip()] if ledger.exists() else []
    return {"billed_usd": sum(r.get("billed_usd", 0.0) for r in rows), "priced_usd": sum(r.get("priced_usd", 0.0) for r in rows)}


def record(step: str, arm: str, priced_usd: float, detail: dict, ledger: Path = LEDGER) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "a") as f:
        f.write(json.dumps({"step": step, "arm": arm, "billed_usd": 0.0, "priced_usd": priced_usd, **detail}) + "\n")
