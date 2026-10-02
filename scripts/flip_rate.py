"""Per-fact flip rate and F1 drop between an fp16 and a quantized prediction file (gates.quantized_*; T-023).

Both files predict the same set (scripts/score_predictions.py formats). Per instance, A = fp16 pair set,
B = quantized pair set (a parse failure or a not_generated row is the empty set, as in scoring).
  flip_rate (per fact)   = sum_i |A_i xor B_i| / sum_i |A_i union B_i|   (instances where both are empty add 0/0)
  instance_flip_rate     = share of instances with A_i != B_i
  correct_to_wrong       = facts in A_i and gold_i that are missing from B_i
  wrong_to_correct       = facts in B_i and gold_i that are missing from A_i
  f1_drop                = F1(fp16) - F1(quantized), pooled, same scorer
Reported raw and dedup, overall and by context type, with the gate thresholds beside them. Point values
only: when an F1-drop gate decision rests on a bootstrap, it is taken in the scoring phase under
gates.decision_rule.
Writes data/scores/flip_<tag>.json.
"""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.score_predictions import SCORES, gold_set, to_parsed
from xbrl_frontier import config, eval_sets, scorer


def load(path: Path, items) -> list:
    preds = {p["id"]: to_parsed(p)[0] for p in map(json.loads, open(path))}
    missing = [it.id for it in items if it.id not in preds]
    if missing:
        raise SystemExit(f"{path}: {len(missing)} instances without a prediction; refusing")
    return [preds[it.id] for it in items]


def compare(items, a, b) -> dict:
    golds = [scorer.gold_set(it.answer) for it in items]
    sym = uni = inst = c2w = w2c = 0
    for g, pa, pb in zip(golds, a, b):
        A, B = pa.pairs, pb.pairs
        sym += len(A ^ B)
        uni += len(A | B)
        inst += A != B
        c2w += len((A & g) - B)
        w2c += len((B & g) - A)
    fa = scorer.score(golds, a).overall["finni_f1"]
    fb = scorer.score(golds, b).overall["finni_f1"]
    return {"instances": len(items), "flip_rate": sym / uni if uni else 0.0, "flipped_facts": sym, "union_facts": uni,
            "instance_flip_rate": inst / len(items) if items else 0.0, "correct_to_wrong": c2w, "wrong_to_correct": w2c,
            "f1_fp16": fa, "f1_quantized": fb, "f1_drop": fa - fb,
            "parse_failure_rate_fp16": sum(p.parse_failed for p in a) / len(a) if a else 0.0,
            "parse_failure_rate_quantized": sum(p.parse_failed for p in b) / len(b) if b else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp16", type=Path, required=True)
    ap.add_argument("--quantized", type=Path, required=True)
    ap.add_argument("--set", choices=["finni", "dev", "fresh"], required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    items, _ = gold_set(args.set, None)
    a, b = load(args.fp16, items), load(args.quantized, items)
    keep = set(eval_sets.dedup_indices(items))
    out = {"set": args.set, "config_hash": config.config_hash(),
           "fp16": str(args.fp16), "fp16_sha256": hashlib.sha256(args.fp16.read_bytes()).hexdigest(),
           "quantized": str(args.quantized), "quantized_sha256": hashlib.sha256(args.quantized.read_bytes()).hexdigest(),
           "gates": {"quantized_flip_rate_max": config.get("gates.quantized_flip_rate_max"),
                     "quantized_f1_drop_max": config.get("gates.quantized_f1_drop_max")}}
    for name, sel in (("raw", range(len(items))), ("dedup", sorted(keep))):
        sub = [items[i] for i in sel]
        sa, sb = [a[i] for i in sel], [b[i] for i in sel]
        out[name] = {"overall": compare(sub, sa, sb), "by_context_type": {}}
        for t in ("table", "text"):
            idx = [k for k, it in enumerate(sub) if eval_sets.context_type(it) == t]
            out[name]["by_context_type"][t] = compare([sub[k] for k in idx], [sa[k] for k in idx], [sb[k] for k in idx])
    SCORES.mkdir(parents=True, exist_ok=True)
    (SCORES / f"flip_{args.tag}.json").write_text(json.dumps(out, indent=1))
    d = out["dedup"]["overall"]
    print(f"dedup: flip_rate {d['flip_rate']:.4f} (max {out['gates']['quantized_flip_rate_max']}), "
          f"f1_drop {d['f1_drop']:.4f} (max {out['gates']['quantized_f1_drop_max']}), "
          f"instance_flip_rate {d['instance_flip_rate']:.4f}")


if __name__ == "__main__":
    main()
