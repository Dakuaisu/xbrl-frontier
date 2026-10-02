"""Select the post-cutoff fresh slice once (PRD section 4; TRADEOFFS T-015) -> pins.fresh_accessions.

Universe: EDGAR monthly XBRL RSS, forms in contamination.fresh_slice_selection.forms with an inline-XBRL
primary document, filed in [contamination.fresh_slice_filing_date_min, fresh_slice_selection.filing_date_max];
CIKs outside the eval manifest, every contamination.exclude_ciks_from file and pins.train_accessions; one
filing per CIK. contamination.fresh_slice_min_filings filings, stratified like the training selection
(SEC office x filer size, eval-manifest quotas), accepted under the same serializer rule. If a cell cannot be
filled from the window, the remaining slots are filled from the same seeded order regardless of cell (the
minimum count is spec, the strata are not); such rows are marked fill=relaxed. Then builds the
fresh instances like the benchmark: every positive plus negatives passing FinTagging's filters, drawn to the
eval set's negatives fraction and negative table share (benchmark conventions: twins are not dropped).
Writes data/fresh_accessions.txt, data/fresh/instances.jsonl, data/fresh_selection_report.json.
"""
import datetime as dt
import json
import random
import urllib.error
from collections import Counter

from ixbrl_facts import fintagging
from scripts.select_train_accessions import (eval_cells, excluded_ciks, largest_remainder, office_key,
                                             quotas, size_key, universe)
from scripts.serialize_corpus import fintagging_negative_filter, read_accessions
from xbrl_frontier import config, edgar, eval_sets, finni, scorer

OUT = config.docs_path(config.get("pins.fresh_accessions"))
FRESH = config.REPO_ROOT / "data" / "fresh"
REPORT = config.REPO_ROOT / "data" / "fresh_selection_report.json"
COLS = ["accession", "cik", "form", "filing_date", "period", "sic", "sec_office", "filer_size", "fill",
        "primary_document", "company"]


def available_universe(lo: dt.date, hi: dt.date, forms: set) -> tuple[list, list]:
    out, missing = [], []
    for y, m in sorted({(d.year, d.month) for d in (lo + dt.timedelta(days=i) for i in range((hi - lo).days + 1))}):
        first = max(lo, dt.date(y, m, 1))
        last = min(hi, (dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(days=1)))
        try:
            out += universe(first, last, forms)
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
            missing.append(f"{y}-{m:02d}")
    return out, missing


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists: the fresh slice is selected once")
    sel = config.get("contamination.fresh_slice_selection")
    lo = dt.date.fromisoformat(config.get("contamination.fresh_slice_filing_date_min"))
    hi = dt.date.fromisoformat(sel["filing_date_max"])
    n = config.get("contamination.fresh_slice_min_filings")
    excl = excluded_ciks()
    train_ciks = {a["cik"] for a in read_accessions()}
    excl_all = set().union(*excl.values()) | train_ciks
    pool, missing_months = available_universe(lo, hi, set(sel["forms"]))
    eligible = [c for c in pool if c["cik"] not in excl_all]
    rng = random.Random(sel["seed"])
    rng.shuffle(eligible)
    quota = quotas(eval_cells(), largest_remainder(Counter({f: 1 for f in sel["forms"]}), n))
    taken, accepted, rejected, seen = Counter(), [], Counter(), set()
    outs = {}
    for relaxed in (False, True):
        for c in eligible:
            if len(accepted) >= n or (not relaxed and all(taken[k] >= v for k, v in quota.items())):
                break
            if c["cik"] in seen:
                continue
            sub = edgar.submissions(c["cik"])
            c["sec_office"], c["filer_size"] = office_key(sub.get("ownerOrg")), size_key(sub.get("category"))
            key = (c["form"], c["sec_office"], c["filer_size"])
            if not relaxed and taken[key] >= quota.get(key, 0):
                continue
            c["fill"] = "relaxed" if relaxed else "quota"
            seen.add(c["cik"])
            try:
                out = fintagging.serialize(edgar.filing_doc(c["cik"], c["accession"], c["primary_document"])
                                           .decode("utf-8", "replace"), unknown="raise")
            except fintagging.UnknownConcept:
                rejected["unknown_us_gaap_concept"] += 1
                continue
            if not (out["text"]["pos"] or out["table"]["pos"]):
                rejected["no_positive_instance"] += 1
                continue
            taken[key] += 1
            accepted.append(c)
            outs[c["accession"]] = out
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        f.write("# pins.fresh_accessions: selected once on 2026-10-02 by scripts/select_fresh_slice.py (TRADEOFFS T-015); never edited.\n")
        f.write("\t".join(COLS) + "\n")
        for a in sorted(accepted, key=lambda a: a["accession"]):
            f.write("\t".join(str(a[k]) for k in COLS) + "\n")

    rows = finni.load()
    keep = set(eval_sets.dedup_indices(rows))
    ev = [r for r in rows if r.idx in keep]
    ev_neg = [r for r in ev if not scorer.gold_set(r.answer)]
    nf = len(ev_neg) / len(ev)
    t_share = sum(eval_sets.context_type(r) == "table" for r in ev_neg) / len(ev_neg)
    pos, neg_t, neg_x = [], [], []
    for acc in sorted(outs):
        for kind in ("text", "table"):
            for part in ("pos", "neg"):
                for i, x in enumerate(outs[acc][kind][part]):
                    item = {"id": f"{acc}:{kind}:{part}:{i}", "accession": acc, "context_type": kind,
                            "text": x["text"], "numeric_entities": x["numeric_entities"]}
                    if part == "pos":
                        pos.append(item)
                    elif fintagging_negative_filter(kind, x["text"]):
                        (neg_t if kind == "table" else neg_x).append(item)
    n_neg = round(len(pos) * nf / (1 - nf))
    n_t = min(round(n_neg * t_share), len(neg_t))
    n_x = min(n_neg - n_t, len(neg_x))
    drng = random.Random(f"{sel['seed']}:instances")
    items = pos + drng.sample(neg_t, n_t) + drng.sample(neg_x, n_x)
    drng.shuffle(items)
    FRESH.mkdir(parents=True, exist_ok=True)
    with open(FRESH / "instances.jsonl", "w") as f:
        for x in items:
            f.write(json.dumps(x) + "\n")
    report = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "config_hash": config.config_hash(), "window": [lo.isoformat(), hi.isoformat()],
              "rss_months_missing": missing_months, "universe": len(pool), "eligible": len(eligible),
              "excluded_ciks": len(excl_all), "accepted": len(accepted), "required": n,
              "rejected": dict(rejected), "filled_relaxed": sum(a["fill"] == "relaxed" for a in accepted), "unfilled_quota": {"|".join(k): v - taken[k] for k, v in quota.items() if taken[k] < v},
              "filing_dates": [min(a["filing_date"] for a in accepted), max(a["filing_date"] for a in accepted)] if accepted else [],
              "instances": len(items), "positives": len(pos), "negatives": n_t + n_x,
              "negatives_fraction": (n_t + n_x) / max(len(items), 1), "eval_negatives_fraction_dedup": nf,
              "negatives_table": n_t, "eval_negative_table_share": t_share,
              "edgar_requests": dict(edgar.stats)}
    REPORT.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
