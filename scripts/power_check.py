"""Power check before the probe (PRD section 7; frontier.power_check; T-007, T-026).

Input: the primary student's full-set scorer output (scripts/score_predictions.py on --set finni); anything
else is refused, and no placeholder or planning SD is ever used. The SD is company-level (T-007): pooled
F1 per CIK over the dedup per-instance counts, SD across companies, stated as such in the output.
Writes data/phase3/power_check.json (kind xbrl_frontier.power_check/v1), which the sampler requires.
"""
import argparse
import hashlib
import json
from pathlib import Path

from frontier.sampling import company_f1_sd, half_width
from scripts.curve_table import load_score
from scripts.frontier_plan import largest_n
from xbrl_frontier import config, eval_sets

OUT = config.REPO_ROOT / "data" / "phase3" / "power_check.json"
KIND = "xbrl_frontier.power_check/v1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", type=Path, required=True, help="primary student's full-set scorer output (set finni)")
    args = ap.parse_args()
    rep = load_score(args.score, "finni")
    per = rep["dedup"].get("per_instance")
    if not per:
        raise SystemExit(f"{args.score} has no dedup per-instance counts; refusing")
    imap = eval_sets.load_instance_map()
    sd = company_f1_sd(per, {i: m["cik"] for i, m in imap.items()})
    pc = config.get("frontier.power_check")
    arms = list(config.get("pins.frontier_arms"))
    n_filings = len({m["accession"] for m in imap.values()})
    n_all = min(config.get("frontier.sample_filings"), largest_n(arms, n_filings))
    hw = half_width(pc["expected_half_width"], sd["sd"], n_all)
    margin = config.get(pc["must_not_exceed_key"])
    out = {"kind": KIND, "score_file": str(args.score),
           "score_sha256": hashlib.sha256(args.score.read_bytes()).hexdigest(),
           "sd_unit": "company (CIK), pooled F1 per company over dedup instances (T-007)",
           "sd": sd["sd"], "companies": sd["companies"], "expression": pc["expected_half_width"],
           "sample_filings_all_arms": n_all, "expected_half_width_all_arms": hw,
           "headline_margin_f1": margin, "exceeded": hw > margin, "config_hash": config.config_hash()}
    if hw > margin:
        kept = [a for k, a in enumerate(arms) if k != 2]
        n2 = largest_n(kept, n_filings)
        out.update(applied=pc["if_exceeded"], dropped_arm=arms[2], arms=kept, sample_filings=n2,
                   expected_half_width=half_width(pc["expected_half_width"], sd["sd"], n2),
                   headline="fraction of the best remaining arm")
    else:
        out.update(applied=None, arms=arms, sample_filings=n_all, expected_half_width=hw,
                   headline="fraction of the best frontier arm")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
