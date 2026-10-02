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

--extend (TRADEOFFS T-012): replays the original walk, checks that it reproduces the frozen rows (whose
sha256 is pinned in training.accession_selection.frozen_rows), then continues the same walk from where it
stopped. The extension is sized from the measured per-form positive yields in data/corpus_serialize_report.json
to (target_train_instances + dev_slice_instances) x (1 - negatives_fraction) x (1 + extension_margin)
positives, with the same form_share and eval-distribution quotas. Extra dev filings are drawn with the
seed + 1 rule from the extension. Rows are appended only.
"""
import hashlib
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


COLS = ["accession", "cik", "form", "filing_date", "period", "sic", "sec_office", "filer_size", "split",
        "primary_document", "company"]
SERIALIZE_REPORT = config.REPO_ROOT / "data" / "corpus_serialize_report.json"


def eval_cells() -> Counter:
    cells = Counter()
    for m in csv.DictReader(open(config.docs_path(config.get("pins.eval_manifest")))):
        s = edgar.submissions(m["cik"])
        cells[(office_key(s.get("ownerOrg")), size_key(s.get("category")))] += 1
    return cells


def quotas(cells: Counter, form_n: dict) -> dict:
    return {(form, *cell): n for form, nf in form_n.items() for cell, n in largest_remainder(cells, nf).items()}


def walk(eligible: list, quota: dict, start: int, seen_cik: Counter, rejected: Counter, label: str):
    """Accept candidates from eligible[start:] until every quota is full. Returns (accepted, next index)."""
    max_per_cik = config.get("training.accession_selection.max_filings_per_cik")
    taken: dict = defaultdict(int)
    accepted = []
    i = start
    while i < len(eligible):
        if all(taken[k] >= v for k, v in quota.items()):
            break
        c = eligible[i]
        i += 1
        if seen_cik[c["cik"]] >= max_per_cik:
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
            print(f"{label}: accepted {len(accepted)}/{sum(quota.values())} rejected={dict(rejected)} "
                  f"requests={edgar.stats['requests']}", flush=True)
    unfilled = {"|".join(k): v - taken[k] for k, v in quota.items() if taken[k] < v}
    return accepted, i, unfilled


def dist(rows, key):
    c = Counter(key(r) for r in rows)
    n = sum(c.values())
    return {str(k): {"n": v, "share": round(v / n, 4)} for k, v in sorted(c.items())}


def row(a: dict) -> str:
    return "\t".join(str(a[k]) for k in COLS) + "\n"


def initial_plan():
    sel = config.get("training.accession_selection")
    ipf = cost.planning_shape().instances_per_filing
    n_train = math.ceil(config.get("training.target_train_instances") / ipf)
    n_dev = math.ceil(config.get("training.dev_slice_instances") / ipf)
    return ipf, n_train, n_dev, largest_remainder(Counter(sel["form_share"]), n_train + n_dev)


def main(extend: bool = False) -> None:
    if OUT.exists() and not extend:
        raise SystemExit(f"{OUT} exists: the training accessions are selected once and frozen (see --extend)")
    if extend and not OUT.exists():
        raise SystemExit("--extend needs the initial list")
    sel = config.get("training.accession_selection")
    lo = dt.date.fromisoformat(sel["filing_date_min"])
    hi = dt.date.fromisoformat(config.get("contamination.train_filing_date_max"))
    ipf, n_train, n_dev, form_n = initial_plan()
    cells = eval_cells()
    excl = excluded_ciks()
    excl_all = set().union(*excl.values())
    pool = universe(lo, hi, set(sel["form_share"]))
    eligible = [c for c in pool if c["cik"] not in excl_all]
    random.Random(sel["seed"]).shuffle(eligible)
    seen_cik, rejected = Counter(), Counter()

    if not extend:
        accepted, _, unfilled = walk(eligible, quotas(cells, form_n), 0, seen_cik, rejected, "initial")
        dev = set(a["accession"] for a in random.Random(sel["seed"] + 1).sample(accepted, min(n_dev, len(accepted))))
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT, "w") as f:
            f.write("# pins.train_accessions: selected once on 2026-10-02 by scripts/select_train_accessions.py "
                    "(TRADEOFFS T-011); never edited.\n")
            f.write("\t".join(COLS) + "\n")
            for a in sorted(accepted, key=lambda a: a["accession"]):
                a["split"] = "dev" if a["accession"] in dev else "train"
                f.write(row(a))
        report = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                  "config_hash": config.config_hash(), "selection": sel, "window": [lo.isoformat(), hi.isoformat()],
                  "planning_instances_per_filing": ipf,
                  "planning_basis": "pins.eval_finni.rows_expected / contamination.eval_manifest_required_recovered",
                  "filings_target": {"train": n_train, "dev": n_dev, "by_form": form_n},
                  "excluded_cik_counts": {k: len(v) for k, v in excl.items()}, "excluded_ciks_total": len(excl_all),
                  "universe_filings": len(pool), "eligible_after_exclusion": len(eligible),
                  "accepted": len(accepted), "rejected": dict(rejected), "unfilled_quota": unfilled,
                  "eval_distribution": {"sec_office": dist([k for k, n in cells.items() for _ in range(n)], lambda k: k[0]),
                                        "filer_size": dist([k for k, n in cells.items() for _ in range(n)], lambda k: k[1])},
                  "train_distribution": {"sec_office": dist(accepted, lambda a: a["sec_office"]),
                                         "filer_size": dist(accepted, lambda a: a["filer_size"]),
                                         "form": dist(accepted, lambda a: a["form"]),
                                         "split": {"train": len(accepted) - len(dev), "dev": len(dev)}},
                  "edgar_requests": dict(edgar.stats)}
        REPORT.write_text(json.dumps(report, indent=1))
        print(json.dumps({k: v for k, v in report.items() if k != "selection"}, indent=1))
        return

    # --extend
    lines = OUT.read_text().splitlines(keepends=True)
    header_comment, colrow, data = lines[0], lines[1], lines[2:]
    frozen = sel["frozen_rows"]
    if colrow.rstrip("\n").split("\t") != COLS or len(data) != frozen["count"]:
        raise SystemExit("train_accessions.txt does not have the frozen shape; refusing to extend")
    if hashlib.sha256("".join(data).encode()).hexdigest() != frozen["sha256"]:
        raise SystemExit("frozen rows do not match training.accession_selection.frozen_rows.sha256; refusing")
    initial, stop, _ = walk(eligible, quotas(cells, form_n), 0, seen_cik, rejected, "replay")
    frozen_acc = [l.split("\t")[0] for l in data]
    if sorted(a["accession"] for a in initial) != sorted(frozen_acc):
        raise SystemExit("replaying the initial walk does not reproduce the frozen rows; refusing")
    initial_rejected = dict(rejected)

    ser = json.load(open(SERIALIZE_REPORT))
    pos_by_form = defaultdict(list)
    for f in ser["per_filing"]:
        pos_by_form[f["form"]].append(f.get("text_pos", 0) + f.get("table_pos", 0))
    yields = {form: sum(v) / len(v) for form, v in pos_by_form.items()}
    share = sel["form_share"]
    mean_yield = sum(share[f] * yields[f] for f in share) / sum(share.values())
    nf = config.get("training.negatives_fraction")
    margin = sel["extension_margin"]
    need = (config.get("training.target_train_instances") + config.get("training.dev_slice_instances")) * (1 - nf)
    have = ser["positives"]
    n_ext = math.ceil(max(0.0, need * (1 + margin) - have) / mean_yield)
    ext_form_n = largest_remainder(Counter(share), n_ext)
    ext, _, ext_unfilled = walk(eligible, quotas(cells, ext_form_n), stop, seen_cik, rejected, "extension")
    dev_total = math.ceil(config.get("training.dev_slice_instances") * (1 - nf) * (1 + margin) / mean_yield)
    initial_dev = sum(1 for l in data if l.split("\t")[COLS.index("split")] == "dev")
    extra_dev = max(0, dev_total - initial_dev)
    dev = set(a["accession"] for a in random.Random(sel["seed"] + 1).sample(ext, min(extra_dev, len(ext))))
    for a in ext:
        a["split"] = "dev" if a["accession"] in dev else "train"
    new_header = ("# pins.train_accessions: selected 2026-10-02, extended once 2026-10-02 (TRADEOFFS T-012); "
                  "the first 208 rows are frozen and hash-verified (training.accession_selection.frozen_rows).\n")
    with open(OUT, "w") as f:
        f.write(new_header)
        f.write(colrow)
        f.writelines(data)
        for a in sorted(ext, key=lambda a: a["accession"]):
            f.write(row(a))
    check = OUT.read_text().splitlines(keepends=True)[2:2 + frozen["count"]]
    assert hashlib.sha256("".join(check).encode()).hexdigest() == frozen["sha256"]

    report = json.load(open(REPORT))
    report["extension"] = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config_hash": config.config_hash(),
        "replay_reproduced_frozen_rows": True, "replay_rejected": initial_rejected, "resumed_at_index": stop,
        "measured_positive_yield_per_filing": yields, "weighted_yield": mean_yield,
        "positives_need": need, "margin": margin, "positives_have": have,
        "filings_target": {"total": n_ext, "by_form": ext_form_n}, "accepted": len(ext),
        "rejected_in_extension": {k: rejected[k] - initial_rejected.get(k, 0) for k in rejected},
        "unfilled_quota": ext_unfilled,
        "dev": {"filings_total_target": dev_total, "initial": initial_dev, "added_from_extension": len(dev)},
        "sizes": {"initial_filings": len(data), "extension_filings": len(ext), "total_filings": len(data) + len(ext)},
        "extension_distribution": {"sec_office": dist(ext, lambda a: a["sec_office"]),
                                   "filer_size": dist(ext, lambda a: a["filer_size"]),
                                   "form": dist(ext, lambda a: a["form"])},
        "edgar_requests": dict(edgar.stats)}
    REPORT.write_text(json.dumps(report, indent=1))
    print(json.dumps(report["extension"], indent=1))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--extend", action="store_true")
    main(ap.parse_args().extend)
