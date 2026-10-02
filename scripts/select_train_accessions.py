"""Select the training accessions once (PRD section 4; TRADEOFFS T-011) -> pins.train_accessions.

Universe: EDGAR monthly XBRL RSS, original 10-K/10-Q with an inline-XBRL primary document, filed in
[training.accession_selection.filing_date_min, contamination.train_filing_date_max], CIK not in any file of
contamination.exclude_ciks_from, at most max_filings_per_cik per CIK.
Size: (training.target_train_instances + training.dev_slice_instances) / planning instances-per-filing
(pins.eval_finni.rows_expected / contamination.eval_manifest_required_recovered), split by form_share.
Strata: SEC industry office (submissions `ownerOrg`) x filer size (submissions `category`), read from the same
API for eval and training CIKs; quotas follow the eval manifest's distribution (largest remainder).
Walk: one seeded shuffle of the universe; a candidate is accepted if its cell has room and it serializes
under the benchmark's rule (no unknown us-gaap concept; at least one positive instance). Dev filings are a
seeded sample of the accepted list (disjoint by construction: one filing per CIK).
Writes data/train_accessions.txt and data/train_selection_report.json; refuses to overwrite the list.
"""
import csv
import datetime as dt
import json
import math
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

from ixbrl_facts import fintagging
from xbrl_frontier import config, cost, edgar

OUT = config.docs_path(config.get("pins.train_accessions"))
REPORT = config.REPO_ROOT / "data" / "train_selection_report.json"
NS = "{https://www.sec.gov/Archives/edgar}"


def excluded_ciks() -> dict[str, set]:
    out = {}
    for rel in config.get("contamination.exclude_ciks_from"):
        path = config.docs_path(rel)
        if path.suffix == ".csv":
            ciks = {r["cik"] for r in csv.DictReader(open(path))}
        elif path.suffix == ".json":
            ciks = set(json.load(open(path)))
        else:
            ciks = {l.strip() for l in open(path) if l.strip() and not l.startswith("#")}
        out[rel] = {f"{int(c):010d}" for c in ciks}
    return out


def office_key(name: str) -> str:
    """'Office of Real Estate & Construction' and '05 Real Estate & Construction' -> 'real estate & construction'."""
    name = re.sub(r"^\s*(office of|\d+)\s+", "", (name or "").strip(), flags=re.I)
    return name.lower() or "unknown"


def size_key(category: str) -> str:
    c = (category or "").lower()
    if c.startswith("large accelerated"):
        return "large_accelerated"
    if c.startswith("accelerated"):
        return "accelerated"
    return "other"


def months(lo: dt.date, hi: dt.date):
    y, m = lo.year, lo.month
    while (y, m) <= (hi.year, hi.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def universe(lo: dt.date, hi: dt.date, forms: set) -> list[dict]:
    out = []
    for y, m in months(lo, hi):
        body = edgar.get(f"https://www.sec.gov/Archives/edgar/monthly/xbrlrss-{y}-{m:02d}.xml")
        for item in ET.fromstring(body).iter("item"):
            f = item.find(f"{NS}xbrlFiling")
            if f is None:
                continue
            form = f.findtext(f"{NS}formType")
            if form not in forms:
                continue
            fd = dt.datetime.strptime(f.findtext(f"{NS}filingDate"), "%m/%d/%Y").date()
            if not lo <= fd <= hi:
                continue
            primary = next((x for x in f.iter(f"{NS}xbrlFile")
                            if x.get(f"{NS}inlineXBRL") == "true" and x.get(f"{NS}type") == form), None)
            if primary is None:
                continue
            out.append({"accession": f.findtext(f"{NS}accessionNumber"), "cik": f.findtext(f"{NS}cikNumber"),
                        "company": f.findtext(f"{NS}companyName"), "form": form, "filing_date": fd.isoformat(),
                        "period": f.findtext(f"{NS}period") or "", "sic": f.findtext(f"{NS}assignedSic") or "",
                        "primary_document": primary.get(f"{NS}file")})
        print(f"rss {y}-{m:02d}: universe {len(out)} requests={edgar.stats['requests']}", flush=True)
    return out


def largest_remainder(weights: Counter, total: int) -> dict:
    s = sum(weights.values())
    raw = {k: total * v / s for k, v in weights.items()}
    q = {k: math.floor(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - q[k]), k))[: total - sum(q.values())]:
        q[k] += 1
    return q


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists: the training accessions are selected once and frozen")
    sel = config.get("training.accession_selection")
    lo = dt.date.fromisoformat(sel["filing_date_min"])
    hi = dt.date.fromisoformat(config.get("contamination.train_filing_date_max"))
    ipf = cost.planning_shape().instances_per_filing
    n_train = math.ceil(config.get("training.target_train_instances") / ipf)
    n_dev = math.ceil(config.get("training.dev_slice_instances") / ipf)
    form_n = largest_remainder(Counter(sel["form_share"]), n_train + n_dev)

    eval_cells = Counter()
    for m in csv.DictReader(open(config.docs_path(config.get("pins.eval_manifest")))):
        s = edgar.submissions(m["cik"])
        eval_cells[(office_key(s.get("ownerOrg")), size_key(s.get("category")))] += 1
    quota = {(form, *cell): n for form, nf in form_n.items()
             for cell, n in largest_remainder(eval_cells, nf).items()}

    excl = excluded_ciks()
    excl_all = set().union(*excl.values())
    pool = universe(lo, hi, set(sel["form_share"]))
    eligible = [c for c in pool if c["cik"] not in excl_all]
    random.Random(sel["seed"]).shuffle(eligible)

    taken: dict = defaultdict(int)
    accepted, rejected, seen_cik = [], Counter(), Counter()
    for c in eligible:
        if all(taken[k] >= v for k, v in quota.items()):
            break
        if seen_cik[c["cik"]] >= sel["max_filings_per_cik"]:
            continue
        if not any(taken[k] < v for k, v in quota.items() if k[0] == c["form"]):
            continue
        sub = edgar.submissions(c["cik"])
        c["sec_office"] = office_key(sub.get("ownerOrg"))
        c["filer_size"] = size_key(sub.get("category"))
        key = (c["form"], c["sec_office"], c["filer_size"])
        if taken[key] >= quota.get(key, 0):
            continue
        seen_cik[c["cik"]] += 1
        try:
            html = edgar.filing_doc(c["cik"], c["accession"], c["primary_document"]).decode("utf-8", "replace")
            out = fintagging.serialize(html, unknown="raise")
        except fintagging.UnknownConcept:
            rejected["unknown_us_gaap_concept"] += 1
            continue
        if not (out["text"]["pos"] or out["table"]["pos"]):
            rejected["no_positive_instance"] += 1
            continue
        taken[key] += 1
        accepted.append(c)
        if len(accepted) % 20 == 0:
            print(f"accepted {len(accepted)}/{n_train + n_dev} rejected={dict(rejected)} "
                  f"requests={edgar.stats['requests']}", flush=True)

    dev = set(a["accession"] for a in random.Random(sel["seed"] + 1).sample(accepted, min(n_dev, len(accepted))))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = ["accession", "cik", "form", "filing_date", "period", "sic", "sec_office", "filer_size", "split",
            "primary_document", "company"]
    with open(OUT, "w") as f:
        f.write("# pins.train_accessions: selected once on 2026-10-02 by scripts/select_train_accessions.py "
                "(TRADEOFFS T-011); never edited.\n")
        f.write("\t".join(cols) + "\n")
        for a in sorted(accepted, key=lambda a: a["accession"]):
            a["split"] = "dev" if a["accession"] in dev else "train"
            f.write("\t".join(str(a[k]) for k in cols) + "\n")

    def dist(rows, key):
        c = Counter(key(r) for r in rows)
        n = sum(c.values())
        return {str(k): {"n": v, "share": round(v / n, 4)} for k, v in sorted(c.items())}

    unfilled = {"|".join(k): v - taken[k] for k, v in quota.items() if taken[k] < v}
    report = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config_hash": config.config_hash(), "selection": sel,
        "window": [lo.isoformat(), hi.isoformat()],
        "planning_instances_per_filing": ipf,
        "planning_basis": "pins.eval_finni.rows_expected / contamination.eval_manifest_required_recovered",
        "filings_target": {"train": n_train, "dev": n_dev, "by_form": form_n},
        "excluded_cik_counts": {k: len(v) for k, v in excl.items()}, "excluded_ciks_total": len(excl_all),
        "universe_filings": len(pool), "eligible_after_exclusion": len(eligible),
        "accepted": len(accepted), "rejected": dict(rejected), "unfilled_quota": unfilled,
        "eval_distribution": {"sec_office": dist([k for k, n in eval_cells.items() for _ in range(n)], lambda k: k[0]),
                              "filer_size": dist([k for k, n in eval_cells.items() for _ in range(n)], lambda k: k[1])},
        "train_distribution": {"sec_office": dist(accepted, lambda a: a["sec_office"]),
                               "filer_size": dist(accepted, lambda a: a["filer_size"]),
                               "form": dist(accepted, lambda a: a["form"]),
                               "split": {"train": len(accepted) - len(dev), "dev": len(dev)}},
        "edgar_requests": edgar.stats,
    }
    REPORT.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("selection",)}, indent=1))


if __name__ == "__main__":
    main()
