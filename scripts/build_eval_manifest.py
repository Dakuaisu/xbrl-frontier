"""Complete the FinTagging eval manifest and build the instance-to-filing map (PRD section 4).

Resumable: every EDGAR response is cached under data/edgar/, so a rerun replays without new requests.
Outputs pins.eval_manifest and pins.eval_instance_map, plus data/manifest_work/report.json.
"""
import csv
import datetime as dt
import hashlib
import html
import json
import re
from collections import Counter, defaultdict

from xbrl_frontier import config, edgar, eval_sets, finni
from xbrl_frontier.instance_match import ContextIndex

WORK = config.REPO_ROOT / "data" / "manifest_work"
_WORD = re.compile(r"[A-Za-z0-9]+")


def log(*a):
    print(*a, flush=True)


def window() -> tuple[str, str]:
    start = dt.date.fromisoformat(config.get("contamination.train_filing_date_max")) + dt.timedelta(days=1)
    return start.isoformat(), config.get("contamination.eval_filing_window_end")


def in_window(d: str) -> bool:
    lo, hi = window()
    return lo <= d <= hi


def tenk_candidates(cik: str) -> list[dict]:
    """Every original 10-K the CIK filed inside the eval window, from the submissions API."""
    sub = edgar.submissions(cik)
    blocks = [sub["filings"]["recent"]]
    lo, hi = window()
    for f in sub["filings"].get("files", []):
        if f["filingTo"] >= lo and f["filingFrom"] <= hi:
            blocks.append(json.loads(edgar.get(f"https://data.sec.gov/submissions/{f['name']}")))
    out = []
    for b in blocks:
        for i, form in enumerate(b["form"]):
            if form == "10-K" and in_window(b["filingDate"][i]) and b["primaryDocument"][i]:
                out.append({"cik": f"{int(cik):010d}", "adsh": b["accessionNumber"][i],
                            "filename": b["primaryDocument"][i], "form": form,
                            "file_date": b["filingDate"][i], "period": b["reportDate"][i],
                            "company": sub.get("name", ""), "sic": sub.get("sic", ""),
                            "sic_description": sub.get("sicDescription", "")})
    return out


def phrase_for(context: str) -> str | None:
    """The 11-word window (edgar.fts_phrase_words) with the most alphabetic words of 4+ letters."""
    n = config.get("edgar.fts_phrase_words")
    text = html.unescape(re.sub(r"<[^>]+>", " ", context)).replace("\xa0", " ")
    words = _WORD.findall(text)
    if len(words) < n:
        return None
    score = [1 if (w.isalpha() and len(w) >= 4) else 0 for w in words]
    best, best_s, s = 0, -1, sum(score[:n])
    for i in range(len(words) - n + 1):
        if i:
            s += score[i + n - 1] - score[i - 1]
        if s > best_s:
            best, best_s = i, s
    return " ".join(words[best:best + n])


def resolve(hits: list[dict]) -> str | None:
    """One registrant iff every hit carries the same CIK group (co-registrants collapse to the first CIK)."""
    groups = {tuple(h["ciks"]) for h in hits}
    if len(groups) != 1:
        return None
    return next(iter(groups))[0]


def main(max_queries: int | None = None, prior_files: list[str] = ()) -> None:
    """prior_files: fts_queries JSON from earlier runs. Their new CIKs are seeded, their contexts are not
    re-queried, and their query count is subtracted from edgar.fts_query_budget (the budget is a total)."""
    WORK.mkdir(parents=True, exist_ok=True)
    required = config.get("contamination.eval_manifest_required_recovered")
    prior = [q for f in prior_files for q in json.loads(open(f).read())]
    budget = config.get("edgar.fts_query_budget") - len(prior)
    if max_queries is not None:
        budget = min(budget, max_queries)
    rows = finni.load()
    first_of: dict[str, str] = {}
    row_cid: list[str] = []
    for r in rows:
        row_cid.append(first_of.setdefault(r.context, str(r.idx)))
    contexts = {cid: rows[int(cid)].context for cid in set(row_cid)}
    rows_per_cid = Counter(row_cid)
    idx = ContextIndex(contexts)
    log(f"contexts: {len(contexts)} distinct over {len(rows)} rows")

    recovered = json.load(open(config.docs_path("research/recovered_eval_manifest.json")))
    ciks: dict[str, dict] = {}
    for name, cl, n in recovered["recovered_eval_registrants"]:
        ciks[cl[0]] = {"source": "recovered_manifest", "name": name}
    for q in prior:
        if q["new"] and q["resolved_cik"] not in ciks:
            ciks[q["resolved_cik"]] = {"source": "fts", "name": q.get("name", "")}
    candidates: dict[str, dict] = {}
    scanned: set[str] = set()

    def add_cik_filings(cik: str, extra: list[dict] = ()):
        for c in tenk_candidates(cik) + list(extra):
            candidates.setdefault(c["adsh"], c)
        for adsh, c in list(candidates.items()):
            if c["cik"] == cik and adsh not in scanned:
                body = edgar.filing_doc(c["cik"], adsh, c["filename"]).decode("utf-8", "replace")
                idx.scan(adsh, body)
                scanned.add(adsh)

    for i, cik in enumerate(sorted(ciks)):
        add_cik_filings(cik)
        if i % 10 == 0:
            log(f"seed {i}/{len(ciks)} cik={cik} filings scanned={len(scanned)} requests={edgar.stats['requests']}")

    min_unique = config.get("contamination.instance_match.min_unique_exact_for_filing")

    def confirmed():
        """Present filings, the assignment among them, and assigned rows per confirmed CIK."""
        ue = idx.unique_exact()
        present = {f for f, n in ue.items() if n >= min_unique}
        company_of = {f: candidates[f]["cik"] for f in scanned}
        dominant = {}
        for f in sorted(present, key=lambda f: (-ue[f], f)):
            dominant.setdefault(company_of[f], f)
        m = idx.assign(present, company_of, dominant)
        per_cik = Counter()
        for c, x in m.items():
            if x.filing:
                per_cik[company_of[x.filing]] += rows_per_cid[c]
        return m, per_cik, present, ue

    m, per_cik, present, ue = confirmed()
    log(f"after seed: present filings {len(present)}, confirmed CIKs {len(per_cik)}; "
        f"assigned rows {sum(per_cik.values())}/{len(rows)}")

    lo, hi = window()
    queries = []
    queried: set[str] = {str(q["context_row"]) for q in prior}
    order = sorted(contexts, key=lambda c: hashlib.sha256(c.encode()).hexdigest())
    while len(queries) < budget and len(per_cik) < required:
        nxt = next((c for c in order if c not in queried and m[c].method == "none"), None)
        if nxt is None:
            break
        queried.add(nxt)
        phrase = phrase_for(contexts[nxt])
        if phrase is None:
            continue
        hits = edgar.full_text_search(phrase, "10-K", lo, hi)
        who = resolve(hits)
        rec = {"context_row": int(nxt), "phrase": phrase, "hits": len(hits), "resolved_cik": who, "new": False}
        if who and who not in ciks:
            ciks[who] = {"source": "fts", "name": hits[0]["display_names"][0]}
            rec["name"] = ciks[who]["name"]
            extra = [{"cik": who, "adsh": h["adsh"], "filename": h["filename"], "form": h["form"],
                      "file_date": h["file_date"], "period": h.get("period_ending") or "",
                      "company": h["display_names"][0], "sic": (h.get("sics") or [""])[0], "sic_description": ""}
                     for h in hits if h["form"] == "10-K"]
            add_cik_filings(who, extra)
            m, per_cik, present, ue = confirmed()
            rec["new"] = True
            log(f"q{len(queries) + 1}: NEW {who} {ciks[who]['name']} -> confirmed {len(per_cik)}")
        queries.append(rec)
        if len(queries) % 50 == 0:
            log(f"q{len(queries)}: ciks={len(ciks)} confirmed={len(per_cik)} requests={edgar.stats['requests']}")
    (WORK / "fts_queries.json").write_text(json.dumps(prior + queries, indent=1))

    final = m
    manifest_path = config.docs_path(config.get("pins.eval_manifest"))
    map_path = config.docs_path(config.get("pins.eval_instance_map"))
    per_adsh = Counter()
    with open(map_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "accession", "cik", "method", "containment", "runner_up_containment"])
        for r, cid in zip(rows, row_cid):
            x = final[cid]
            cik = candidates[x.filing]["cik"] if x.filing else ""
            per_adsh[x.filing] += 1
            w.writerow([r.idx, x.filing or "", cik, x.method, f"{x.containment:.4f}", f"{x.runner_up:.4f}"])
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cik", "company", "accession", "form", "filing_date", "period_of_report", "fiscal_year",
                    "sic", "sic_description", "primary_document", "cik_source", "unique_exact_contexts",
                    "instances_assigned"])
        for adsh in sorted(present, key=lambda a: (candidates[a]["cik"], candidates[a]["file_date"])):
            c = candidates[adsh]
            w.writerow([c["cik"], c["company"], adsh, c["form"], c["file_date"], c["period"], (c["period"] or "")[:4],
                        c["sic"], c["sic_description"], c["filename"], ciks[c["cik"]]["source"], ue[adsh],
                        per_adsh[adsh]])

    keep = set(eval_sets.dedup_indices(rows))
    methods = Counter(final[c].method for c in row_cid)
    assigned_raw = sum(1 for c in row_cid if final[c].filing)
    assigned_dedup = sum(1 for r, c in zip(rows, row_cid) if r.idx in keep and final[c].filing)
    confirmed_ciks = {candidates[a]["cik"] for a in present}
    unconfirmed = sorted(set(ciks) - confirmed_ciks)
    report = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config_hash": config.config_hash(),
        "finni_revision": config.get("pins.eval_finni.hf_revision"),
        "filing_window": window(),
        "manifest_filings": len(present),
        "manifest_companies": len(confirmed_ciks),
        "filings_per_company": dict(Counter(Counter(candidates[a]["cik"] for a in present).values())),
        "required": required,
        "ciks_known": len(ciks),
        "ciks_unconfirmed": unconfirmed,
        "candidate_filings_scanned": len(scanned),
        "fts_queries_this_run": len(queries),
        "fts_queries_prior_runs": len(prior),
        "fts_query_budget_total": config.get("edgar.fts_query_budget"),
        "fts_new_ciks": sum(q["new"] for q in queries + prior),
        "assignment_rate_raw": assigned_raw / len(rows),
        "assignment_rate_dedup": assigned_dedup / len(keep),
        "instance_assignment_min_fraction": config.get("contamination.instance_assignment_min_fraction"),
        "methods_raw": dict(methods),
        "edgar_requests": edgar.stats,
        "unconfirmed_diagnostics": {
            cik: {"source": ciks[cik]["source"], "name": ciks[cik]["name"],
                  "candidates": [{"accession": a, "file_date": c["file_date"], "period": c["period"],
                                  "unique_exact": ue.get(a, 0),
                                  "rows_best_or_tied": sum(rows_per_cid[x] for x, lst in idx.best.items()
                                                           if any(f == a for _, _, f in lst))}
                                 for a, c in candidates.items() if c["cik"] == cik],
                  "fts_queries_resolving_here": sum(1 for q in queries + prior if q["resolved_cik"] == cik)}
            for cik in unconfirmed},
        "fts_resolution": dict(Counter("new" if q["new"] else "known" if q["resolved_cik"] else
                                       "ambiguous_or_none" for q in queries + prior)),
    }
    (WORK / "report.json").write_text(json.dumps(report, indent=1))
    log(json.dumps({k: v for k, v in report.items() if k != "ciks_unconfirmed"}, indent=1))
    log(f"unconfirmed CIKs ({len(unconfirmed)}): {unconfirmed}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-queries", type=int)
    ap.add_argument("--prior", nargs="*", default=[])
    a = ap.parse_args()
    main(a.max_queries, a.prior)
