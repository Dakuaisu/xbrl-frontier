"""Serialize every filing in pins.train_accessions with the benchmark serializer (PRD section 4).

Applies TRADEOFFS T-010: within one filing, an empty-labelled text that also occurs with a non-empty label
set is not written. Counts dropped twins per filing and in total. No sampling happens here: the
negatives fraction reported is the natural one after twin-dropping; sampling to
training.negatives_fraction is the corpus-build step.
Writes data/corpus/serialized/<accession>.jsonl (untracked) and data/corpus_serialize_report.json.
"""
import json
import re
from collections import Counter

from ixbrl_facts import fintagging
from xbrl_frontier import config, edgar

OUT_DIR = config.REPO_ROOT / "data" / "corpus" / "serialized"
REPORT = config.REPO_ROOT / "data" / "corpus_serialize_report.json"


def read_accessions() -> list[dict]:
    lines = [l.rstrip("\n") for l in open(config.docs_path(config.get("pins.train_accessions"))) if not l.startswith("#")]
    cols = lines[0].split("\t")
    return [dict(zip(cols, l.split("\t"))) for l in lines[1:] if l]


def fintagging_negative_filter(kind: str, text: str) -> bool:
    """The negative filters in FinTagging's sample_pos&neg_data.ipynb (commit in pins.fintagging_source)."""
    if kind == "text":
        return 100 < len(text) < 500
    return len(re.findall(r"\d", text)) >= 50


def instances(out: dict) -> tuple[list[dict], int]:
    labelled = {x["text"] for k in ("text", "table") for x in out[k]["pos"]}
    rows, twins = [], 0
    for kind in ("text", "table"):
        for x in out[kind]["pos"]:
            rows.append({"context_type": kind, "text": x["text"], "numeric_entities": x["numeric_entities"]})
        for x in out[kind]["neg"]:
            if x["text"] in labelled:
                twins += 1
                continue
            rows.append({"context_type": kind, "text": x["text"], "numeric_entities": []})
    return rows, twins


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    accs = read_accessions()
    per_filing, tot = [], Counter()
    for i, a in enumerate(accs):
        html = edgar.filing_doc(a["cik"], a["accession"], a["primary_document"]).decode("utf-8", "replace")
        rows, twins = instances(fintagging.serialize(html, unknown="raise"))
        with open(OUT_DIR / f"{a['accession']}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps({"accession": a["accession"], "split": a["split"], **r}) + "\n")
        c = Counter()
        for r in rows:
            pos = bool(r["numeric_entities"])
            c[f"{r['context_type']}_{'pos' if pos else 'neg'}"] += 1
            if not pos and fintagging_negative_filter(r["context_type"], r["text"]):
                c["neg_passing_fintagging_filter"] += 1
            c["pairs"] += len(r["numeric_entities"])
        c["twins_dropped"] = twins
        per_filing.append({"accession": a["accession"], "split": a["split"], "form": a["form"], **c})
        for k, v in c.items():
            tot[f"{a['split']}::{k}"] += v
            tot[k] += v
        if i % 25 == 0:
            print(f"serialized {i + 1}/{len(accs)}", flush=True)
    pos = tot["text_pos"] + tot["table_pos"]
    neg = tot["text_neg"] + tot["table_neg"]
    report = {
        "config_hash": config.config_hash(), "filings": len(accs),
        "instances": pos + neg, "positives": pos, "negatives": neg,
        "negatives_fraction_after_twin_drop": neg / (pos + neg),
        "twins_dropped_total": tot["twins_dropped"],
        "negatives_fraction_if_twins_kept": (neg + tot["twins_dropped"]) / (pos + neg + tot["twins_dropped"]),
        "negatives_passing_fintagging_filter": tot["neg_passing_fintagging_filter"],
        "target_train_instances": config.get("training.target_train_instances"),
        "dev_slice_instances": config.get("training.dev_slice_instances"),
        "negatives_fraction_target": config.get("training.negatives_fraction"),
        "totals": dict(sorted(tot.items())), "per_filing": per_filing,
    }
    REPORT.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("per_filing", "totals")}, indent=1))


if __name__ == "__main__":
    main()
