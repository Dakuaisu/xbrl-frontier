"""Serving-variant table from the smoke tests (serving.smoke_test.failure_action; O-005, O-006).

Reads serving/smoke_test.py outputs (data/phase0/smoke_kaggle_t4/smoke_kaggle_t4.json,
data/phase0/rtx3050_wsl2/smoke_rtx3050_wsl2.json, whichever exist) and writes data/phase0/serving_variants.json:
per card, the variants that passed and the ones removed with their recorded errors. Planned variants with no
smoke result are listed as "not tested", never assumed to pass.
"""
import json

from xbrl_frontier import config

D = config.REPO_ROOT / "data" / "phase0"
SOURCES = {"kaggle_t4": D / "smoke_kaggle_t4" / "smoke_kaggle_t4.json",
           "rtx3050_wsl2": D / "rtx3050_wsl2" / "smoke_rtx3050_wsl2.json"}


def summarize(sources: dict = SOURCES) -> dict:
    from serving.smoke_test import planned
    out = {"failure_action": config.get("serving.smoke_test.failure_action"), "cards": {}}
    for card, path in sources.items():
        labels = [v for v, _, _ in planned(card)]
        if not path.exists():
            out["cards"][card] = {"source": None, "kept": [], "removed": [], "not_tested": labels}
            continue
        rec = json.loads(path.read_text())
        if rec.get("card") != card or "results" not in rec:
            raise SystemExit(f"{path} is not a serving/smoke_test.py output for {card}; refusing")
        res = {r["variant"]: r for r in rec["results"]}
        out["cards"][card] = {"source": str(path), "versions": rec.get("versions"),
                              "kept": [v for v in labels if res.get(v, {}).get("ok")],
                              "removed": [{"variant": v, "error": res[v].get("error")} for v in labels
                                          if v in res and not res[v].get("ok")],
                              "not_tested": [v for v in labels if v not in res]}
    return out


def main() -> None:
    out = summarize()
    (D / "serving_variants.json").parent.mkdir(parents=True, exist_ok=True)
    (D / "serving_variants.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
