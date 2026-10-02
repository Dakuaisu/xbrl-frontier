"""Curve validation (PRD section 5; training.curve_validation_run; T-030).

Compares the independent run trained on the first training.curve_validation_run instances (kaggle_train.py
--validation) with the nested curve checkpoint at that prefix, on the dev slice and (when given) the frontier
sample. Inputs are scorer outputs only. Company-cluster paired bootstrap (T-007, T-028): gap = F1(independent)
- F1(nested) with its paired interval; the caveat flag is raised when |gap| exceeds the half-width of the nested
checkpoint's own interval ("a gap larger than the bootstrap interval is reported as a caveat on the curve").
Writes data/curve/validation.json.
"""
import argparse
import json
from pathlib import Path

from scripts.curve_table import load_score
from scripts.score_predictions import _accessions
from xbrl_frontier import bootstrap, config, eval_sets

OUT = config.REPO_ROOT / "data" / "curve" / "validation.json"


def company_map(set_name: str) -> dict:
    if set_name == "finni":
        return {i: m["cik"] for i, m in eval_sets.load_instance_map().items()}
    acc = _accessions(config.docs_path(config.get("pins.train_accessions")))
    return _DevCompany(acc)


class _DevCompany(dict):
    """dev instance id 'ACCESSION:line' -> CIK."""

    def __init__(self, acc):
        super().__init__()
        self.acc = acc

    def get(self, key, default=None):
        a = self.acc.get(str(key).split(":")[0])
        return a["cik"] if a else default


def compare(indep: Path, nested: Path, set_name: str) -> dict:
    a, b = load_score(indep, set_name), load_score(nested, set_name)
    ca = {r[0]: tuple(r[1:4]) for r in a["dedup"]["per_instance"]}
    cb = {r[0]: tuple(r[1:4]) for r in b["dedup"]["per_instance"]}
    if set(ca) != set(cb):
        raise SystemExit(f"{indep} and {nested} score different instances; refusing")
    bs = bootstrap.paired({"independent": ca, "nested": cb}, company_map(set_name))
    diff = [x - y for x, y in zip(bs["samples"]["independent"], bs["samples"]["nested"])]
    lo, hi = bootstrap.interval(bs["samples"]["nested"])
    gap = bs["point"]["independent"] - bs["point"]["nested"]
    return {"set": set_name, "independent": str(indep), "nested": str(nested),
            "f1_independent": bs["point"]["independent"], "f1_nested": bs["point"]["nested"], "gap": gap,
            "gap_ci": bootstrap.interval(diff), "nested_ci": (lo, hi), "nested_half_width": (hi - lo) / 2,
            "caveat": abs(gap) > (hi - lo) / 2, "bootstrap_unit": bs["unit"], "companies": bs["companies"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--independent-dev", type=Path, required=True)
    ap.add_argument("--nested-dev", type=Path, required=True)
    ap.add_argument("--independent-sample", type=Path)
    ap.add_argument("--nested-sample", type=Path)
    args = ap.parse_args()
    out = {"curve_validation_run": config.get("training.curve_validation_run"), "config_hash": config.config_hash(),
           "dev": compare(args.independent_dev, args.nested_dev, "dev")}
    if args.independent_sample and args.nested_sample:
        out["frontier_sample"] = compare(args.independent_sample, args.nested_sample, "finni")
    out["caveat_on_curve"] = any(v["caveat"] for k, v in out.items() if isinstance(v, dict) and "caveat" in v)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
