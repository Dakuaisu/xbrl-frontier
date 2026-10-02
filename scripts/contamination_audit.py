"""Contamination audit of training instances against every FinNI eval context (PRD section 4; T-014).

Text normalization for both checks (lm-eval-harness decontamination style): HTML tags replaced by spaces,
entities decoded, lowercased, ASCII punctuation removed, split on whitespace.
- n-gram: a training instance is flagged if any of its contamination.ngram_n-grams occurs in any eval
  context (a text shorter than n counts as one gram of its whole length, on both sides).
- MinHash: contamination.minhash_num_perm permutations over word contamination.minhash_shingle_words-grams,
  LSH bands/rows chosen by datasketch's optimal-parameter rule for contamination.minhash_jaccard_threshold;
  a candidate is flagged if its estimated Jaccard with an eval context meets the threshold.
Default input: every serialized instance that can enter the corpus (positives and negatives passing
FinTagging's filters). --files audits given corpus jsonl files instead (the post-removal check).
Writes data/contamination_audit/{flags.tsv,summary.json} or {flags_<tag>.tsv,summary_<tag>.json}.
"""
import argparse
import hashlib
import html
import json
import random
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

from scripts.serialize_corpus import OUT_DIR, fintagging_negative_filter, read_accessions
from xbrl_frontier import config, finni

AUDIT = config.REPO_ROOT / "data" / "contamination_audit"
_TAG = re.compile(r"<[^>]+>")
_PUNCT = str.maketrans("", "", string.punctuation)
P61 = (1 << 61) - 1


def words(text: str) -> list[str]:
    return html.unescape(_TAG.sub(" ", text)).lower().translate(_PUNCT).split()


def grams(w: list[str], n: int) -> list[str]:
    if len(w) < n:
        return [" ".join(w)] if w else []
    return [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]


def _h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


def lsh_params(threshold: float, num_perm: int, steps: int = 1000) -> tuple[int, int]:
    """datasketch MinHashLSH._optimal_param with equal FP/FN weights (midpoint-rule integration)."""
    def integrate(f, a, b):
        if b <= a:
            return 0.0
        dx = (b - a) / steps
        return sum(f(a + (i + 0.5) * dx) for i in range(steps)) * dx
    best, best_err = (1, num_perm), float("inf")
    for b in range(1, num_perm + 1):
        for r in range(1, num_perm // b + 1):
            fp = integrate(lambda s: 1 - (1 - s ** r) ** b, 0.0, threshold)
            fn = integrate(lambda s: 1 - (1 - (1 - s ** r) ** b), threshold, 1.0)
            err = 0.5 * fp + 0.5 * fn
            if err < best_err:
                best, best_err = (b, r), err
    return best


class MinHasher:
    def __init__(self):
        rng = random.Random(config.get("contamination.minhash_seed"))
        self.perms = [(rng.randrange(1, P61), rng.randrange(0, P61))
                      for _ in range(config.get("contamination.minhash_num_perm"))]
        self.k = config.get("contamination.minhash_shingle_words")

    def signature(self, w: list[str]) -> tuple | None:
        hs = list({_h64(g) for g in grams(w, self.k)})
        if not hs:
            return None
        return tuple(min([(a * h + b) % P61 for h in hs]) for a, b in self.perms)


def training_instances(files: list[Path] | None):
    if files:
        for f in files:
            for line in open(f):
                x = json.loads(line)
                yield x["id"], x
        return
    for a in read_accessions():
        for i, line in enumerate(open(OUT_DIR / f"{a['accession']}.jsonl")):
            x = json.loads(line)
            if x["numeric_entities"] or fintagging_negative_filter(x["context_type"], x["text"]):
                yield f"{a['accession']}:{i}", x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", type=Path)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    n = config.get("contamination.ngram_n")
    thr = config.get("contamination.minhash_jaccard_threshold")
    num_perm = config.get("contamination.minhash_num_perm")
    bands, rows_per_band = lsh_params(thr, num_perm)

    rows = finni.load()
    eval_ctx = {}
    for r in rows:
        eval_ctx.setdefault(r.context, r.idx)
    ngram_owner: dict[int, int] = {}
    mh = MinHasher()
    buckets = defaultdict(list)
    eval_sig = {}
    for ctx, idx in eval_ctx.items():
        w = words(ctx)
        for g in grams(w, n):
            ngram_owner.setdefault(hash(g), idx)
        sig = mh.signature(w)
        if sig is not None:
            eval_sig[idx] = sig
            for b in range(bands):
                buckets[(b, sig[b * rows_per_band:(b + 1) * rows_per_band])].append(idx)
    print(f"eval contexts {len(eval_ctx)}, {len(ngram_owner)} distinct {n}-grams, LSH bands={bands} rows={rows_per_band}",
          flush=True)

    AUDIT.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    stats = Counter()
    eval_hit = set()
    with open(AUDIT / f"flags{suffix}.tsv", "w") as out:
        out.write("instance_id\tmethod\teval_instance_id\tscore\tlabel\tcontext_type\n")
        for k, (iid, x) in enumerate(training_instances(args.files)):
            stats["audited"] += 1
            w = words(x["text"])
            label = "pos" if x["numeric_entities"] else "neg"
            hit_ng = next((ngram_owner[hash(g)] for g in grams(w, n) if hash(g) in ngram_owner), None)
            hit_mh, best = None, 0.0
            sig = mh.signature(w)
            if sig is not None:
                cands = set()
                for b in range(bands):
                    cands.update(buckets.get((b, sig[b * rows_per_band:(b + 1) * rows_per_band]), ()))
                for c in cands:
                    j = sum(p == q for p, q in zip(sig, eval_sig[c])) / num_perm
                    if j >= thr and j > best:
                        hit_mh, best = c, j
            if hit_ng is not None or hit_mh is not None:
                method = "both" if hit_ng is not None and hit_mh is not None else ("ngram" if hit_ng is not None else "minhash")
                ev = hit_ng if hit_ng is not None else hit_mh
                eval_hit.add(ev)
                stats[f"flagged::{method}"] += 1
                stats[f"flagged_{label}"] += 1
                stats["flagged"] += 1
                out.write(f"{iid}\t{method}\t{ev}\t{best:.4f}\t{label}\t{x['context_type']}\n")
            if k % 20000 == 0:
                print(f"audited {k} flagged {stats['flagged']}", flush=True)
    summary = {"config_hash": config.config_hash(), "input": [str(f) for f in args.files] if args.files else
               "all serialized instances eligible for the corpus (positives + negatives passing FinTagging filters)",
               "ngram_n": n, "minhash": {"num_perm": num_perm, "threshold": thr, "shingle_words": mh.k,
                                         "bands": bands, "rows_per_band": rows_per_band},
               "eval_contexts": len(eval_ctx), "eval_contexts_matched": len(eval_hit),
               "max_flagged_train_docs_after_removal": config.get("contamination.max_flagged_train_docs_after_removal"),
               **{k: 0 for k in ("flagged", "flagged_pos", "flagged_neg", "flagged::ngram", "flagged::minhash",
                                 "flagged::both")}, **dict(stats)}
    (AUDIT / f"summary{suffix}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
