"""Serving-variant table from the smoke tests (serving.smoke_test.failure_action; O-005, O-006).

Reads serving/smoke_test.py outputs (per card, in order; a later file's result for a variant replaces an earlier one)
and writes data/phase0/serving_variants.json: per card, the variants that passed and the ones removed with their recorded
errors. A variant with no result, or whose quantized checkpoint was never built (setup failure, never served), is
"not tested", never assumed to pass and never removed (O-029).
"""
import json
from pathlib import Path

from xbrl_frontier import config

D = config.REPO_ROOT / "data" / "phase0"
SOURCES = {"kaggle_t4": [D / "smoke_kaggle_t4" / "smoke_kaggle_t4.json", D / "smoke_kaggle_t4_awq" / "smoke_kaggle_t4.json"],
           "rtx3050_wsl2": [D / "rtx3050_wsl2" / "smoke_rtx3050_wsl2.json"]}
NEVER_SERVED = "FileNotFoundError: missing quantized checkpoint"


def summarize(sources: dict = SOURCES) -> dict:
    from serving.smoke_test import planned
    out = {"failure_action": config.get("serving.smoke_test.failure_action"), "cards": {}}
    for card, paths in sources.items():
        labels = [v for v, _, _ in planned(card)]
        paths = [x for x in ([paths] if isinstance(paths, Path) else paths) if x.exists()]
        if not paths:
            out["cards"][card] = {"source": None, "kept": [], "removed": [], "not_tested": labels, "not_tested_reasons": {}}
            continue
        res, versions = {}, None
        for path in paths:
            rec = json.loads(path.read_text())
            if rec.get("card") != card or "results" not in rec:
                raise SystemExit(f"{path} is not a serving/smoke_test.py output for {card}; refusing")
            res.update({r["variant"]: r for r in rec["results"]})
            versions = rec.get("versions")
        never = {v: res[v]["error"] for v in res if not res[v].get("ok") and str(res[v].get("error", "")).startswith(NEVER_SERVED)}
        out["cards"][card] = {"source": [str(x) for x in paths], "versions": versions,
                              "kept": [v for v in labels if res.get(v, {}).get("ok")],
                              "removed": [{"variant": v, "error": res[v].get("error")} for v in labels
                                          if v in res and not res[v].get("ok") and v not in never],
                              "not_tested": [v for v in labels if v not in res or v in never],
                              "not_tested_reasons": {v: never[v] for v in labels if v in never}}
    return out


def main() -> None:
    out = summarize()
    (D / "serving_variants.json").parent.mkdir(parents=True, exist_ok=True)
    (D / "serving_variants.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
