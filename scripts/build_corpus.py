"""Build the sampled training corpus from the serialized filings (PRD section 4; TRADEOFFS T-013).

Per split (train / dev, from pins.train_accessions): every positive; negatives drawn to
training.negatives_fraction from those passing FinTagging's own negative filters, stratified by context type
so the negatives' table share equals the eval set's (FinNI dedup negatives, measured at run time).
Each split is then put in one seeded order (training.corpus_sampling.seed); training uses the first
training.target_train_instances, dev the first training.dev_slice_instances, and the rest is listed as surplus.
--exclude FILE drops instance ids (one per line, e.g. contamination flags) before the draw.
Writes data/corpus/{train,dev}.jsonl, data/corpus/surplus_ids.txt and data/corpus_report.json (or the
--tag variant).
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from scripts.serialize_corpus import OUT_DIR, fintagging_negative_filter, read_accessions
from xbrl_frontier import config, eval_sets, finni, scorer

CORPUS = config.REPO_ROOT / "data" / "corpus"


def eval_negative_table_share() -> float:
    rows = finni.load()
    keep = set(eval_sets.dedup_indices(rows))
    neg = [r for r in rows if r.idx in keep and not scorer.gold_set(r.answer)]
    return sum(eval_sets.context_type(r) == "table" for r in neg) / len(neg)


def load_pool(exclude: set) -> dict:
    """split -> {"pos": [...], "neg_table": [...], "neg_text": [...]}; instance id = accession:line."""
    split_of = {a["accession"]: a["split"] for a in read_accessions()}
    pool = {s: {"pos": [], "neg_table": [], "neg_text": []} for s in ("train", "dev")}
    stats = Counter()
    for acc, split in sorted(split_of.items()):
        for i, line in enumerate(open(OUT_DIR / f"{acc}.jsonl")):
            x = json.loads(line)
            x["id"] = f"{acc}:{i}"
            if x["id"] in exclude:
                stats[f"{split}::excluded"] += 1
                continue
            if x["numeric_entities"]:
                pool[split]["pos"].append(x)
            elif fintagging_negative_filter(x["context_type"], x["text"]):
                pool[split]["neg_table" if x["context_type"] == "table" else "neg_text"].append(x)
            else:
                stats[f"{split}::neg_failing_fintagging_filter"] += 1
    return pool, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", type=Path)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    exclude = {l.split("\t")[0].strip() for l in open(args.exclude)
               if l.strip() and not l.startswith("#") and not l.startswith("instance_id\t")} \
        if args.exclude else set()
    seed = config.get("training.corpus_sampling.seed")
    nf = config.get("training.negatives_fraction")
    t_share = eval_negative_table_share()
    pool, stats = load_pool(exclude)
    out_dir = CORPUS / args.tag if args.tag else CORPUS
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"config_hash": config.config_hash(), "seed": seed, "negatives_fraction_target": nf,
              "eval_negative_table_share": t_share, "excluded_ids": len(exclude), "pool_stats": dict(stats),
              "splits": {}}
    surplus = []
    for split, cap_key in (("train", "training.target_train_instances"), ("dev", "training.dev_slice_instances")):
        p = pool[split]
        rng = random.Random(f"{seed}:{split}")
        n_pos = len(p["pos"])
        n_neg = round(n_pos * nf / (1 - nf))
        want_table = round(n_neg * t_share)
        n_table = min(want_table, len(p["neg_table"]))
        n_text = min(n_neg - n_table, len(p["neg_text"]))
        negs = rng.sample(p["neg_table"], n_table) + rng.sample(p["neg_text"], n_text)
        order = p["pos"] + negs
        rng.shuffle(order)
        cap = config.get(cap_key)
        used, rest = order[:cap], order[cap:]
        surplus += [x["id"] for x in rest]
        path = out_dir / f"{split}.jsonl"
        with open(path, "w") as f:
            for x in used:
                f.write(json.dumps(x) + "\n")
        u_pos = sum(bool(x["numeric_entities"]) for x in used)
        u_neg = len(used) - u_pos
        report["splits"][split] = {
            "positives_available": n_pos, "negatives_drawn": len(negs),
            "negatives_table_wanted": want_table, "negatives_table_drawn": n_table,
            "table_pool_shortfall": want_table - n_table, "negatives_text_drawn": n_text,
            "pool_negatives_table": len(p["neg_table"]), "pool_negatives_text": len(p["neg_text"]),
            "ordered_instances": len(order), "cap": cap, "used": len(used), "surplus": len(rest),
            "used_positives": u_pos, "used_negatives": u_neg,
            "used_negatives_fraction": u_neg / max(len(used), 1),
            "used_negative_table_share": sum(1 for x in used if not x["numeric_entities"] and x["context_type"] == "table") / max(u_neg, 1),
            "cap_reached": len(used) == cap,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (out_dir / "surplus_ids.txt").write_text("".join(i + "\n" for i in surplus))
    rp = config.REPO_ROOT / "data" / (f"corpus_report_{args.tag}.json" if args.tag else "corpus_report.json")
    rp.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
