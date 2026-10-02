"""Every key under `gates`, evaluated from result files only (PRD section 8; T-031).

Inputs (all optional; each must be a real result file; a gate without its inputs is "not measured", blank):
  --student-full / --zero-shot-full     scorer outputs on the full FinNI set (primary student; same base, zero-shot)
  --student-sample / --zero-shot-sample scorer outputs on the shared frontier sample (--ids)
  --arm-sample NAME=SCORE               each frontier arm on the shared sample
  --quantized NAME=FP16_SCORE,Q_SCORE,FLIP_JSON   full-set scores of the fp16 merge and a quantized variant + flip file
  --cost FILE                           data/phase4/cost_model.json
  --student-fresh / --control-fresh     fresh-slice scorer outputs (student; frontier.fresh_slice_control_arm)
  --encoder SCORE, --curve FILE         presence of the encoder baseline and the learning curve
Bootstrap statistics use the company-cluster paired bootstrap and gates.decision_rule (pass / fail /
inconclusive). gain = student - zero-shot (paired, same instances); fraction = student / best arm (paired on the
sample; best = highest point F1); quantized drop = fp16 - quantized (paired, lower is better); fresh gap =
F1(FinTagging full set) - F1(fresh) (different instance sets: independent resamples of each). The flip rate and the
cost ratio are not bootstrap statistics and are compared at their point value.
Writes data/gates/summary.json and prints a table.
"""
import argparse
import json
from pathlib import Path

from scripts.curve_table import load_score
from scripts.score_predictions import _accessions
from xbrl_frontier import bootstrap, config, eval_sets

OUT = config.REPO_ROOT / "data" / "gates" / "summary.json"


def company_of(set_name: str):
    if set_name == "finni":
        return {i: m["cik"] for i, m in eval_sets.load_instance_map().items()}
    path = config.get("pins.train_accessions" if set_name == "dev" else "pins.fresh_accessions")
    acc = _accessions(config.docs_path(path))

    class _M(dict):
        def get(self, k, d=None):
            a = acc.get(str(k).split(":")[0])
            return a["cik"] if a else d
    return _M()


def counts(path: Path, set_name: str, subset: bool | None = None) -> dict:
    rep = load_score(path, set_name)
    if subset is True and not rep.get("ids_file"):
        raise SystemExit(f"{path} is not scored on the shared sample (no ids_file); refusing")
    if subset is False and rep.get("ids_file"):
        raise SystemExit(f"{path} scores a subset; the full set is required here; refusing")
    return {r[0]: tuple(r[1:4]) for r in rep["dedup"]["per_instance"]}


def paired_stat(arms: dict, set_name: str, stat) -> tuple[float, tuple]:
    bs = bootstrap.paired(arms, company_of(set_name))
    names = list(arms)
    point = stat(*(bs["point"][n] for n in names))
    samples = [stat(*vals) for vals in zip(*(bs["samples"][n] for n in names))]
    return point, bootstrap.interval(samples)


def gate(name, threshold_key, point=None, ci=None, higher=True, inputs=None, note=None, decision=None):
    thr = config.get(threshold_key) if threshold_key else None
    if point is None and decision is None:
        return {"gate": name, "threshold_key": threshold_key, "status": "not measured", "point": None, "ci": None,
                "inputs": inputs}
    if decision is None:
        decision = bootstrap.decide(point, ci, thr, higher) if ci else ("pass" if (point >= thr if higher else point <= thr) else "fail")
    return {"gate": name, "threshold_key": threshold_key, "status": decision, "point": point, "ci": ci,
            "inputs": inputs, "note": note}


def main() -> None:
    ap = argparse.ArgumentParser()
    for a in ("student-full", "zero-shot-full", "student-sample", "zero-shot-sample", "cost", "student-fresh",
              "control-fresh", "encoder", "curve"):
        ap.add_argument(f"--{a}", type=Path)
    ap.add_argument("--arm-sample", action="append", default=[])
    ap.add_argument("--quantized", action="append", default=[])
    a = ap.parse_args()
    rows = []

    arms = {s.split("=", 1)[0]: Path(s.split("=", 1)[1]) for s in a.arm_sample}
    if a.student_sample and arms:
        stu = counts(a.student_sample, "finni", True)
        ac = {k: counts(v, "finni", True) for k, v in arms.items()}
        bs = bootstrap.paired({"student": stu, **ac}, company_of("finni"))
        best = max(ac, key=lambda k: bs["point"][k])
        p, ci = paired_stat({"student": stu, best: ac[best]}, "finni", lambda s, b: s / b if b else 0.0)
        rows.append(gate("finni_f1_fraction_of_best_frontier_min", "gates.finni_f1_fraction_of_best_frontier_min", p, ci,
                         inputs=[str(a.student_sample)] + [str(v) for v in arms.values()], note=f"best arm: {best}"))
    else:
        rows.append(gate("finni_f1_fraction_of_best_frontier_min", "gates.finni_f1_fraction_of_best_frontier_min"))

    if a.student_full and a.zero_shot_full:
        p, ci = paired_stat({"s": counts(a.student_full, "finni", False), "z": counts(a.zero_shot_full, "finni", False)},
                            "finni", lambda s, z: s - z)
        note = None
        if a.student_sample and a.zero_shot_sample:
            ps, cis = paired_stat({"s": counts(a.student_sample, "finni", True), "z": counts(a.zero_shot_sample, "finni", True)},
                                  "finni", lambda s, z: s - z)
            note = f"on the shared sample: {ps:.4f} {tuple(round(x, 4) for x in cis)} (reported)"
        rows.append(gate("finni_f1_gain_over_zero_shot_min", "gates.finni_f1_gain_over_zero_shot_min", p, ci,
                         inputs=[str(a.student_full), str(a.zero_shot_full)], note=note))
    else:
        rows.append(gate("finni_f1_gain_over_zero_shot_min", "gates.finni_f1_gain_over_zero_shot_min"))

    if a.quantized:
        for spec in a.quantized:
            name, files = spec.split("=", 1)
            f16, q, flip = (Path(x) for x in files.split(","))
            p, ci = paired_stat({"f": counts(f16, "finni", False), "q": counts(q, "finni", False)}, "finni", lambda f, q_: f - q_)
            rows.append(gate(f"quantized_f1_drop_max[{name}]", "gates.quantized_f1_drop_max", p, ci, higher=False,
                             inputs=[str(f16), str(q)]))
            fl = json.loads(flip.read_text())
            if "dedup" not in fl or "flip_rate" not in fl["dedup"]["overall"]:
                raise SystemExit(f"{flip} is not a scripts/flip_rate.py output; refusing")
            rows.append(gate(f"quantized_flip_rate_max[{name}]", "gates.quantized_flip_rate_max",
                             fl["dedup"]["overall"]["flip_rate"], None, higher=False, inputs=[str(flip)], note="point value"))
    else:
        rows.append(gate("quantized_f1_drop_max", "gates.quantized_f1_drop_max"))
        rows.append(gate("quantized_flip_rate_max", "gates.quantized_flip_rate_max"))

    if a.cost:
        cm = json.loads(a.cost.read_text())
        if "gate" not in cm or "comparator" not in cm:
            raise SystemExit(f"{a.cost} is not a scripts/cost_model.py output; refusing")
        for k, g in cm["gate"].items():
            if g["gated"]:
                rows.append(gate(f"cost_ratio_vs_comparator_min[{k}]", "gates.cost_ratio_vs_comparator_min",
                                 g["cost_ratio"], None, inputs=[str(a.cost)],
                                 note=f"comparator {cm['comparator']['arm']} ({cm['comparator']['rule_applied']}); point value"))
    else:
        rows.append(gate("cost_ratio_vs_comparator_min", "gates.cost_ratio_vs_comparator_min"))

    if a.student_full and a.student_fresh:
        bf = bootstrap.paired({"x": counts(a.student_full, "finni", False)}, company_of("finni"))
        bz = bootstrap.paired({"x": counts(a.student_fresh, "fresh")}, company_of("fresh"))
        gap = bf["point"]["x"] - bz["point"]["x"]
        ci = bootstrap.interval([x - y for x, y in zip(bf["samples"]["x"], bz["samples"]["x"])])
        note = None
        if a.control_fresh and arms.get(config.get("frontier.fresh_slice_control_arm")):
            cs = bootstrap.paired({"x": counts(arms[config.get("frontier.fresh_slice_control_arm")], "finni", True)}, company_of("finni"))
            cf = bootstrap.paired({"x": counts(a.control_fresh, "fresh")}, company_of("fresh"))
            cgap = cs["point"]["x"] - cf["point"]["x"]
            note = f"control gap {cgap:.4f}; student minus control gap {gap - cgap:.4f} (fresh_slice_control_gap_reported)"
        rows.append(gate("fresh_slice_f1_gap_max", "gates.fresh_slice_f1_gap_max", gap, ci, higher=False,
                         inputs=[str(a.student_full), str(a.student_fresh)], note=note))
    else:
        rows.append(gate("fresh_slice_f1_gap_max", "gates.fresh_slice_f1_gap_max"))

    rows.append(gate("encoder_baseline_reported", None, decision="pass" if a.encoder and load_score(a.encoder, "finni") else None,
                     inputs=[str(a.encoder)] if a.encoder else None))
    if a.curve:
        c = json.loads(a.curve.read_text())
        rows.append(gate("learning_curve_reported", None, decision="pass" if c.get("rows") else None, inputs=[str(a.curve)]))
    else:
        rows.append(gate("learning_curve_reported", None))
    charts = config.REPO_ROOT / "data" / "charts"
    rows.append(gate("pareto_chart_required", None,
                     decision="pass" if all((charts / f"pareto_{r}.vl.json").exists() for r in config.get("gates.cost_regimes")) else None))
    out = {"config_hash": config.config_hash(), "decision_rule": config.get("gates.decision_rule"), "gates": rows,
           "ships": all(r["status"] == "pass" for r in rows)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print("| gate | threshold key | status | point | CI |\n|---|---|---|---|---|")
    for r in rows:
        f = lambda v: "" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
        ci = "" if not r.get("ci") else f"[{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]"
        print(f"| {r['gate']} | {r['threshold_key'] or ''} | {r['status']} | {f(r.get('point'))} | {ci} |")


if __name__ == "__main__":
    main()
