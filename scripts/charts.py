"""Chart specs (Vega-Lite JSON, no plotting dependency) built from result files only (T-028).

  pareto   from data/phase4/cost_model.json (+ --variant-score NAME=SCORE for each serving variant's F1 on the
           shared sample): F1 vs $/filing, one chart per regime, frontier arms and T4 variants at the headline rate
  curve    from a scripts/curve_table.py output: dedup F1 overall, table and text vs distinct instances seen
  latency  from serving/load_test.py outputs: p95 TTFT and p95 E2E vs request rate / concurrency, per card/variant
A point without a measured value is omitted and listed under "missing"; nothing is filled in. RTX 3050 load
files carry the configured cost label and never a dollar value.
Writes data/charts/<name>.vl.json.
"""
import argparse
import json
from pathlib import Path

from scripts.curve_table import load_score
from serving.load_test import KIND as LOAD_KIND
from xbrl_frontier import config

OUT = config.REPO_ROOT / "data" / "charts"
SCHEMA = "https://vega.github.io/schema/vega-lite/v5.json"


def _write(name: str, spec: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.vl.json"
    p.write_text(json.dumps(spec, indent=1))
    return p


def pareto(cost_path: Path, variant_scores: dict) -> list[Path]:
    cm = json.loads(cost_path.read_text())
    if "self_host_rows" not in cm or "arms" not in cm or "gate" not in cm:
        raise SystemExit(f"{cost_path} is not a scripts/cost_model.py output; refusing")
    out = []
    for regime, rg in config.get("gates.cost_regimes").items():
        values, missing = [], []
        for arm, a in cm["arms"].items():
            values.append({"name": arm, "kind": "frontier arm", "f1": a["f1"], "usd_per_filing": a["usd_per_filing"][regime]})
        for v in cm["variants"]:
            r = next((x for x in cm["self_host_rows"] if x["variant"] == v and x["regime"] == regime
                      and x["t4_rate_key"] == cm["headline_rate_key"] and x["utilization"] == rg["utilization"]), None)
            if v not in variant_scores or r is None or r["usd_per_filing"] is None:
                missing.append(v)
                continue
            f1 = load_score(Path(variant_scores[v]), "finni")["dedup"]["overall"]["finni_f1"]
            values.append({"name": v, "kind": "T4 variant", "f1": f1, "usd_per_filing": r["usd_per_filing"]})
        spec = {"$schema": SCHEMA, "title": f"FinNI F1 vs $/filing ({regime}; prices {cm['price_sheet']} observed {cm['price_sheet_observed']})",
                "data": {"values": values}, "mark": {"type": "point", "filled": True},
                "encoding": {"x": {"field": "usd_per_filing", "type": "quantitative", "scale": {"type": "log"}, "title": "$ per filing"},
                             "y": {"field": "f1", "type": "quantitative", "title": "F1 (dedup, shared sample)"},
                             "color": {"field": "kind"}, "tooltip": [{"field": "name"}, {"field": "f1"}, {"field": "usd_per_filing"}]},
                "usermeta": {"missing": missing, "source": str(cost_path), "formula": cm["formula"]}}
        out.append(_write(f"pareto_{regime}", spec))
    return out


def curve(curve_path: Path) -> Path:
    c = json.loads(curve_path.read_text())
    if "series" not in c or "rows" not in c:
        raise SystemExit(f"{curve_path} is not a scripts/curve_table.py output; refusing")
    values = [{"instances_seen": p["x_instances_seen"], "label": p["label"], "series": name, "f1": p[key]}
              for p in c["series"]["dedup"] for name, key in (("overall", "f1"), ("table", "f1_table"), ("text", "f1_text"))
              if p[key] is not None]
    return _write("learning_curve", {"$schema": SCHEMA, "title": "F1 vs distinct training instances (T-016)",
                                     "data": {"values": values}, "mark": {"type": "line", "point": True},
                                     "encoding": {"x": {"field": "instances_seen", "type": "quantitative"},
                                                  "y": {"field": "f1", "type": "quantitative"},
                                                  "color": {"field": "series"}, "tooltip": [{"field": "label"}]},
                                     "usermeta": {"missing": c["missing"], "note": c["note"], "source": str(curve_path)}})


def latency(load_paths: list[Path]) -> Path:
    values = []
    for p in load_paths:
        rec = json.loads(p.read_text())
        if rec.get("kind") != LOAD_KIND:
            raise SystemExit(f"{p} is not a serving/load_test.py output; refusing")
        label = config.get(f"serving.hardware.{rec['card']}").get("cost_label")
        for pt in rec["points"]:
            x = pt["request_rate"] if pt["regime"] == "poisson_rate" else pt["concurrency"]
            for metric in ("p95_ttft_ms", "p95_e2el_ms"):
                values.append({"card": rec["card"], "variant": rec["variant"], "regime": pt["regime"], "x": x,
                               "metric": metric, "ms": pt[metric], "cost_label": label})
    return _write("latency", {"$schema": SCHEMA, "title": "p95 latency vs load (generator colocated; network excluded)",
                              "data": {"values": values}, "mark": {"type": "line", "point": True},
                              "encoding": {"x": {"field": "x", "type": "quantitative", "title": "request rate (rps) or concurrency"},
                                           "y": {"field": "ms", "type": "quantitative"},
                                           "color": {"field": "variant"}, "strokeDash": {"field": "metric"},
                                           "row": {"field": "regime"}, "column": {"field": "card"}},
                              "usermeta": {"slo": config.get("serving.slo"), "sources": [str(p) for p in load_paths]}})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("chart", choices=["pareto", "curve", "latency"])
    ap.add_argument("--cost", type=Path)
    ap.add_argument("--variant-score", action="append", default=[])
    ap.add_argument("--curve-file", type=Path)
    ap.add_argument("--load", action="append", default=[], type=Path)
    a = ap.parse_args()
    if a.chart == "pareto":
        print(pareto(a.cost, dict(s.split("=", 1) for s in a.variant_score)))
    elif a.chart == "curve":
        print(curve(a.curve_file))
    else:
        print(latency(a.load))


if __name__ == "__main__":
    main()
