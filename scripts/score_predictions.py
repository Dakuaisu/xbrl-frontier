"""Score a predictions file on finni, dev or fresh: raw and dedup, overall and per slice (PRD section 3).

Predictions are one JSON object per instance id: {"text": model output} for students (parsed by the FinNI
scorer; a parse failure is an empty prediction, counted in parse_failure_rate; nothing is retried or
repaired), {"pairs": [{"Fact","Type"}]} for the encoder, or {"not_generated": true} when the prompt did not
fit eval.generation.max_model_len (an empty prediction, counted in not_generated_rate). Every instance of the
set must be present. No judge anywhere: exact match only.
Writes data/scores/<tag>.json with overall + slices for raw and dedup and compact per-instance counts.
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from xbrl_frontier import config, eval_sets, finni, scorer, sft_format, tokens

SCORES = config.REPO_ROOT / "data" / "scores"
KIND = "xbrl_frontier.score_predictions/v1"


@dataclass(frozen=True)
class Item:
    idx: int
    id: object
    context: str
    answer: str


def _accessions(path: Path) -> dict:
    lines = [l.rstrip("\n") for l in open(path) if not l.startswith("#")]
    cols = lines[0].split("\t")
    return {r["accession"]: r for r in (dict(zip(cols, l.split("\t"))) for l in lines[1:] if l)}


def gold_set(name: str, corpus: Path | None) -> tuple[list[Item], dict]:
    """Items in set order plus the instance map {idx: {"cik", "fiscal_year"}} for the company/year slices."""
    if name == "finni":
        rows = finni.load()
        return [Item(r.idx, r.idx, r.context, r.answer) for r in rows], eval_sets.load_instance_map()
    if name == "dev":
        path, acc = (corpus or config.REPO_ROOT / "data" / "corpus") / "dev.jsonl", _accessions(
            config.docs_path(config.get("pins.train_accessions")))
    else:
        path, acc = config.REPO_ROOT / "data" / "fresh" / "instances.jsonl", _accessions(
            config.docs_path(config.get("pins.fresh_accessions")))
    items, imap = [], {}
    for i, x in enumerate(map(json.loads, open(path))):
        items.append(Item(i, x["id"], x["text"], sft_format.answer(x["numeric_entities"])))
        a = acc[x.get("accession") or x["id"].split(":")[0]]
        imap[i] = {"cik": a["cik"], "fiscal_year": (a.get("period") or "")[:4] or a["filing_date"][:4]}
    return items, imap


CUT = {"length", "MAX_TOKENS"}  # vLLM finish_reason / Gemini finishReason when the output cap was hit (T-039)


def to_parsed(p: dict, finish: str | None = None) -> tuple[scorer.Parsed, bool, bool]:
    """(parsed, not_generated, cut_off). A cut-off output is scored as written (T-039)."""
    if p.get("not_generated"):
        return scorer.Parsed(frozenset(), False, 0), True, False
    cut = (p.get("finish_reason") or p.get("finish") or finish) in CUT
    if "pairs" in p:
        return scorer.Parsed(scorer.pair_set(p["pairs"]), False, 0), False, cut
    return scorer.parse_prediction(p["text"]), False, cut


def score_items(items, preds, imap, counts) -> dict:
    golds = [scorer.gold_set(it.answer) for it in items]
    parsed = [preds[it.id] for it in items]
    keys = eval_sets.slice_keys(items, imap, {it.idx: counts[it.idx] for it in items})
    s = scorer.score(golds, [p for p, _, _ in parsed], keys)
    ng = sum(flag for _, flag, _ in parsed)
    cut = sum(c for _, _, c in parsed)
    s.overall["not_generated"] = ng
    s.overall["not_generated_rate"] = ng / len(items) if items else 0.0
    s.overall["cut_off"] = cut
    s.overall["cut_off_rate"] = cut / len(items) if items else 0.0
    return {"overall": s.overall, "slices": s.slices,
            "per_instance": [[it.id, pi["tp"], pi["fp"], pi["fn"], pi["parse_failed"], flag, c]
                             for it, pi, (_, flag, c) in zip(items, s.per_instance, parsed)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--set", choices=["finni", "dev", "fresh"], required=True)
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ids", type=Path, help="score only these instance ids (e.g. the shared frontier sample)")
    args = ap.parse_args()
    items, imap = gold_set(args.set, args.corpus)
    ids_sha = None
    if args.ids:
        import hashlib
        want = {json.loads(l) for l in open(args.ids) if l.strip()}
        items = [it for it in items if it.id in want]
        if len(items) != len(want):
            raise SystemExit(f"{len(want) - len(items)} ids in {args.ids} are not in set {args.set}; refusing")
        ids_sha = hashlib.sha256(args.ids.read_bytes()).hexdigest()
    usage = args.pred.parent / "usage.jsonl"
    finish = ({u["id"]: u.get("finish") for u in map(json.loads, open(usage)) if u.get("run", 0) == 0}
              if usage.exists() else {})
    raw_preds = {}
    for p in map(json.loads, open(args.pred)):
        raw_preds[p["id"]] = to_parsed(p, finish.get(p["id"]))
    missing = [it.id for it in items if it.id not in raw_preds]
    if missing:
        raise SystemExit(f"{len(missing)} instances have no prediction (e.g. {missing[:3]}); refusing to score")
    counts = tokens.context_token_counts(items)
    keep = set(eval_sets.dedup_indices(items))
    import hashlib
    rep = {"kind": KIND, "set": args.set, "predictions": str(args.pred),
           "predictions_sha256": hashlib.sha256(args.pred.read_bytes()).hexdigest(), "config_hash": config.config_hash(),
           "ids_file": str(args.ids) if args.ids else None, "ids_sha256": ids_sha,
           "instances_raw": len(items), "instances_dedup": len(keep),
           "raw": score_items(items, raw_preds, imap, counts),
           "dedup": score_items([it for it in items if it.idx in keep], raw_preds, imap, counts)}
    SCORES.mkdir(parents=True, exist_ok=True)
    (SCORES / f"{args.tag}.json").write_text(json.dumps(rep))
    for k in ("raw", "dedup"):
        o = rep[k]["overall"]
        print(f"{k}: F1 {o['finni_f1']:.4f} P {o['finni_precision']:.4f} R {o['finni_recall']:.4f} "
              f"parse_failure_rate {o['parse_failure_rate']:.4f} cut_off_rate {o['cut_off_rate']:.4f} "
              f"not_generated_rate {o['not_generated_rate']:.4f} "
              f"n={o['instances']}")


if __name__ == "__main__":
    main()
