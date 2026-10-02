"""Spend guard: no live frontier call without a TRADEOFFS pre-approval for that exact spend (T-024).

A live step needs `--approval T-NNN`, and docs/TRADEOFFS.md must contain, inside the section headed
`## T-NNN ...`, a line
    PRE-APPROVED SPEND: step=<step> arm=<arm> usd_max=<dollars>
written by the owner. The step's priced estimate must not exceed usd_max, and the ledger total plus the
estimate must leave frontier.usd_reserve_min under budget.usd_cap_frontier_api. The arm's thinking setting
must carry a verified source (frontier.thinking_per_arm.<arm>.source), and batch-priced steps (batch,
fresh_control) must be priced from that arm's measured probe tokens, not the proxy estimate (T-025).
The guard runs before any API key is read.
"""
import json
import re
from pathlib import Path

from xbrl_frontier import config

LEDGER = config.REPO_ROOT / "data" / "phase3" / "spend_ledger.jsonl"
TRADEOFFS = config.DOCS / "TRADEOFFS.md"
_LINE = re.compile(r"PRE-APPROVED SPEND:\s*step=(\S+)\s+arm=(\S+)\s+usd_max=([0-9.]+)")


BATCH_STEPS = {"batch", "fresh_control"}


class GuardRefused(RuntimeError):
    pass


def approvals(approval_id: str, tradeoffs: Path = TRADEOFFS) -> list[dict]:
    text = tradeoffs.read_text()
    m = re.search(rf"(?m)^## {re.escape(approval_id)}\b.*?(?=^## |\Z)", text, re.S)
    if not m:
        return []
    return [{"step": s, "arm": a, "usd_max": float(u)} for s, a, u in _LINE.findall(m.group(0))]


def ledger_total(ledger: Path = LEDGER) -> float:
    if not ledger.exists():
        return 0.0
    return sum(json.loads(l)["usd"] for l in open(ledger) if l.strip())


def check(step: str, arm: str, usd_estimate: float, approval_id: str | None,
          tradeoffs: Path = TRADEOFFS, ledger: Path = LEDGER, repriced_from_probe: bool = False) -> dict:
    if not approval_id:
        raise GuardRefused("live call refused: no --approval given (dry-run is the default)")
    match = [a for a in approvals(approval_id, tradeoffs) if a["step"] == step and a["arm"] == arm]
    if not match:
        raise GuardRefused(f"live call refused: {approval_id} has no 'PRE-APPROVED SPEND: step={step} arm={arm} ...' line")
    cap = match[0]["usd_max"]
    if usd_estimate > cap:
        raise GuardRefused(f"live call refused: estimate ${usd_estimate:.4f} exceeds the approved ${cap:.4f}")
    spec = config.get("frontier.thinking_per_arm").get(arm) or {}
    if not spec.get("source"):
        raise GuardRefused(f"live call refused: frontier.thinking_per_arm.{arm} has no verified source")
    if step in BATCH_STEPS and not repriced_from_probe:
        raise GuardRefused(f"live call refused: {step} for {arm} must be priced from that arm's probe "
                           "(no probe usage found, or the estimate was not re-priced)")
    spent = ledger_total(ledger)
    limit = config.get("budget.usd_cap_frontier_api") - config.get("frontier.usd_reserve_min")
    if spent + usd_estimate > limit:
        raise GuardRefused(f"live call refused: ledger ${spent:.4f} + estimate ${usd_estimate:.4f} > "
                           f"budget.usd_cap_frontier_api - frontier.usd_reserve_min = ${limit:.2f}")
    return {"approval": approval_id, "usd_max": cap, "ledger_before": spent, "estimate": usd_estimate}


def record(step: str, arm: str, usd: float, detail: dict, ledger: Path = LEDGER) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "a") as f:
        f.write(json.dumps({"step": step, "arm": arm, "usd": usd, **detail}) + "\n")
