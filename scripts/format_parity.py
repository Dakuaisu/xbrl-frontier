"""Format parity: eval contexts (FinNI dedup set) vs training contexts (serialized corpus), PRD section 4.

Compares, overall and for positives only: context-type mix, context token lengths (pinned student tokenizer),
share above training.kaggle.max_seq_len_tokens, HTML tag vocabulary, facts per positive instance, datatype
mix, negatives fraction, and the doubled-cell artifact. Training negatives are also shown restricted to
those passing FinTagging's own negative filters, since the eval negatives went through them. Divergences
are reported, not fixed here: fixes go in the serializer (CLAUDE.md).
Writes data/format_parity_report.json; with --files (sampled corpus jsonl) and --tag, the train side is those
files and the report is data/format_parity_report_<tag>.json.
"""
import argparse
import html as htmlmod
import json
import re
import statistics
from collections import Counter

from scripts.serialize_corpus import OUT_DIR, fintagging_negative_filter
from xbrl_frontier import config, eval_sets, finni, scorer, tokens

REPORT = config.REPO_ROOT / "data" / "format_parity_report.json"
_TAG = re.compile(r"<\s*/?\s*([a-zA-Z][\w:-]*)")
_TOK = re.compile(r"[a-z0-9]+")
QUANTILES = (0.1, 0.5, 0.9, 0.99)


def doubled_cell_share(texts) -> float:
    cells = doubled = 0
    for t in texts:
        for seg in re.split(r"<[^>]+>", t):
            w = _TOK.findall(htmlmod.unescape(seg).lower())
            if len(w) >= 2:
                cells += 1
                n = len(w)
                doubled += n % 2 == 0 and w[: n // 2] == w[n // 2:]
    return doubled / cells if cells else 0.0


def profile(items: list[dict]) -> dict:
    """items: {text, context_type, pairs: list[(value, type)]}"""
    tok = tokens.tokenizer()
    texts = [i["text"] for i in items]
    lens = sorted(len(e.ids) for e in tok.encode_batch(texts, add_special_tokens=False))
    cap = config.get("training.kaggle.max_seq_len_tokens")
    pos = [i for i in items if i["pairs"]]
    facts = sorted(len(i["pairs"]) for i in pos)
    types = Counter(t for i in pos for _, t in i["pairs"])
    tags = Counter(m.lower() for t in texts for m in _TAG.findall(t))
    nt = sum(types.values()) or 1
    return {
        "instances": len(items), "positives": len(pos),
        "negatives_fraction": 1 - len(pos) / len(items) if items else 0.0,
        "table_share": sum(i["context_type"] == "table" for i in items) / max(len(items), 1),
        "table_share_positives": sum(i["context_type"] == "table" for i in pos) / max(len(pos), 1),
        "context_tokens_quantiles": {str(q): lens[min(int(q * len(lens)), len(lens) - 1)] for q in QUANTILES} if lens else {},
        "context_tokens_mean": statistics.fmean(lens) if lens else 0.0,
        "share_above_max_seq_len": sum(l > cap for l in lens) / max(len(lens), 1),
        "facts_per_positive_quantiles": {str(q): facts[min(int(q * len(facts)), len(facts) - 1)] for q in QUANTILES} if facts else {},
        "facts_per_positive_mean": statistics.fmean(facts) if facts else 0.0,
        "datatype_mix": {t: round(n / nt, 4) for t, n in sorted(types.items())},
        "html_tags": dict(tags.most_common()),
        "doubled_cell_share": doubled_cell_share(texts),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    rows = finni.load()
    keep = set(eval_sets.dedup_indices(rows))
    ev = [{"text": r.context, "context_type": eval_sets.context_type(r),
           "pairs": sorted(scorer.gold_set(r.answer))} for r in rows if r.idx in keep]
    tr = []
    for path in (args.files or sorted(OUT_DIR.glob("*.jsonl"))):
        for line in open(path):
            x = json.loads(line)
            pairs = sorted(scorer.pair_set({"Fact": e["value"], "Type": e["type"]} for e in x["numeric_entities"]))
            tr.append({"text": x["text"], "context_type": "table" if x["context_type"] == "table" else "text",
                       "pairs": pairs, "split": x["split"]})
    tr_filtered = [i for i in tr if i["pairs"] or fintagging_negative_filter(i["context_type"], i["text"])]
    serialize_report = json.load(open(config.REPO_ROOT / "data" / "corpus_serialize_report.json"))
    report = {
        "config_hash": config.config_hash(),
        "eval": {"set": "FinNI-eval-v2 dedup", **profile(ev)},
        "train_all": {"set": "serialized corpus after twin drop, all negatives", **profile(tr)},
        "train_fintagging_negative_filter": {"set": "serialized corpus, negatives restricted to FinTagging's filters",
                                             **profile(tr_filtered)},
        "eval_positives": {"set": "FinNI-eval-v2 dedup, positives", **profile([i for i in ev if i["pairs"]])},
        "train_positives": {"set": "serialized corpus, positives", **profile([i for i in tr if i["pairs"]])},
        "eval_negatives": {"set": "FinNI-eval-v2 dedup, negatives", **profile([i for i in ev if not i["pairs"]])},
        "train_negatives_fintagging_filter": {"set": "serialized corpus, negatives passing FinTagging's filters",
                                              **profile([i for i in tr_filtered if not i["pairs"]])},
        "twin_drop_divergence": {"twins_dropped_total": serialize_report["twins_dropped_total"],
                                 "note": "TRADEOFFS T-010: the eval keeps such twins; the training corpus does not"},
    }
    if args.files:
        report["train_source"] = args.files
    out = REPORT.with_name(f"format_parity_report_{args.tag}.json") if args.tag else REPORT
    out.write_text(json.dumps(report, indent=1))
    keys = ["instances", "positives", "negatives_fraction", "table_share", "table_share_positives",
            "context_tokens_quantiles", "context_tokens_mean", "share_above_max_seq_len",
            "facts_per_positive_quantiles", "facts_per_positive_mean", "datatype_mix", "doubled_cell_share"]
    for k in keys:
        print(f"{k}:\n  eval  {report['eval'][k]}\n  train {report['train_all'][k]}\n  trainF {report['train_fintagging_negative_filter'][k]}")
    for k in ("table_share", "context_tokens_quantiles", "context_tokens_mean", "share_above_max_seq_len"):
        print(f"{k} [positives]: eval {report['eval_positives'][k]} | train {report['train_positives'][k]}")
        print(f"{k} [negatives]: eval {report['eval_negatives'][k]} | trainF {report['train_negatives_fintagging_filter'][k]}")
    tags = lambda p: set(report[p]["html_tags"])
    print("tags only in eval:", sorted(tags("eval") - tags("train_all")), "only in train:", sorted(tags("train_all") - tags("eval")))


if __name__ == "__main__":
    main()
