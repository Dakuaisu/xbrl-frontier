"""Label audit (PRD section 3; eval.label_audit; T-029): the draw, the hand-filled sheets, and their scoring.

draw:  eval.label_audit.sample_size instances from the deduplicated FinNI set restricted to instances assigned
       to a filing (the audit checks labels against the filing). Strata = eval.label_audit.stratify_by:
       context_type x entity_type, where an instance's entity_type is its most frequent gold type ("none" for a
       negative). Proportional allocation by largest remainder, at least one per non-empty stratum. Seed
       eval.label_audit.seed. Second pass: second_pass_size of the first-pass ids, eval.label_audit.second_pass_seed,
       on a separate sheet that does not show first-pass verdicts. Writes data/label_audit/ (sample.json,
       pass1_sheet.csv, pass2_sheet.csv) once; refuses to overwrite.
score: reads the filled sheets; refuses blank verdicts. Instance error rate (raw and stratum-weighted to the
       population), fact-level counts, self-agreement on the second pass against eval.label_audit.min_self_agreement.
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict

from scripts.select_train_accessions import largest_remainder
from xbrl_frontier import config, eval_sets, finni

AUDIT = config.REPO_ROOT / "data" / "label_audit"
TYPES = ("monetaryItemType", "percentItemType", "sharesItemType", "perShareItemType", "integerItemType")
OWNER_COLS = ["verdict (correct|incorrect)", "missing_facts", "spurious_facts", "wrong_type_facts", "notes"]


def entity_type(answer: str) -> str:
    c = Counter(r["Type"] for r in json.loads(answer)["result"])
    if not c:
        return "none"
    return max(c, key=lambda t: (c[t], -TYPES.index(t) if t in TYPES else -99))


def allocate(pop: dict, n: int) -> dict:
    q = largest_remainder(Counter({k: len(v) for k, v in pop.items()}), n)
    for k in pop:
        if q.get(k, 0) == 0 and pop[k]:
            donor = max(q, key=lambda s: (q[s], s))
            q[donor] -= 1
            q[k] = 1
    return q


def draw() -> None:
    if AUDIT.exists() and any(AUDIT.iterdir()):
        raise SystemExit(f"{AUDIT} already holds a draw; the audit sample is drawn once")
    la = config.get("eval.label_audit")
    if la["stratify_by"] != ["context_type", "entity_type"]:
        raise SystemExit("only stratify_by [context_type, entity_type] is implemented")
    rows = finni.load()
    imap = eval_sets.load_instance_map()
    man = {r["accession"]: r for r in csv.DictReader(open(config.docs_path(config.get("pins.eval_manifest"))))}
    keep = set(eval_sets.dedup_indices(rows))
    pop = defaultdict(list)
    for r in rows:
        if r.idx in keep and r.idx in imap:
            pop[f"{eval_sets.context_type(r)}|{entity_type(r.answer)}"].append(r.idx)
    quota = allocate(pop, la["sample_size"])
    rng = random.Random(la["seed"])
    first = []
    for k in sorted(pop):
        first += rng.sample(sorted(pop[k]), quota[k])
    first.sort()
    second = sorted(random.Random(la["second_pass_seed"]).sample(first, la["second_pass_size"]))
    AUDIT.mkdir(parents=True, exist_ok=True)
    stratum = {i: k for k, ids in pop.items() for i in ids}
    (AUDIT / "sample.json").write_text(json.dumps({
        "population": {k: len(v) for k, v in sorted(pop.items())}, "quota": dict(sorted(quota.items())),
        "first_pass": first, "second_pass": second, "config_hash": config.config_hash(),
        "rule": "dedup FinNI instances assigned to a filing; T-029"}, indent=1))

    def sheet(path, ids):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["instance_id", "stratum", "accession", "cik", "filing_url", "context", "gold_pairs"] + OWNER_COLS)
            for i in ids:
                m = imap[i]
                d = man[m["accession"]]
                url = (f"https://www.sec.gov/Archives/edgar/data/{int(d['cik'])}/"
                       f"{d['accession'].replace('-', '')}/{d['primary_document']}")
                gold = [(x["Fact"], x["Type"]) for x in json.loads(rows[i].answer)["result"]]
                w.writerow([i, stratum[i], m["accession"], m["cik"], url, rows[i].context, json.dumps(gold)] + [""] * len(OWNER_COLS))
    sheet(AUDIT / "pass1_sheet.csv", first)
    sheet(AUDIT / "pass2_sheet.csv", second)
    print(json.dumps({"first_pass": len(first), "second_pass": len(second), "quota": quota}, indent=1))


def _read(path) -> dict:
    out = {}
    for r in csv.DictReader(open(path)):
        v = r[OWNER_COLS[0]].strip().lower()
        if v not in ("correct", "incorrect"):
            raise SystemExit(f"{path}: instance {r['instance_id']} has verdict {v!r}; every row must be correct|incorrect")
        n = lambda k: int(r[k] or 0)
        out[int(r["instance_id"])] = {"verdict": v, "stratum": r["stratum"], "missing": n("missing_facts"),
                                      "spurious": n("spurious_facts"), "wrong_type": n("wrong_type_facts"),
                                      "gold_facts": len(json.loads(r["gold_pairs"]))}
    return out


def score() -> None:
    s = json.loads((AUDIT / "sample.json").read_text())
    p1, p2 = _read(AUDIT / "pass1_sheet.csv"), _read(AUDIT / "pass2_sheet.csv")
    if sorted(p1) != s["first_pass"] or sorted(p2) != s["second_pass"]:
        raise SystemExit("sheets do not match the drawn ids; refusing")
    by = defaultdict(list)
    for v in p1.values():
        by[v["stratum"]].append(v["verdict"] == "incorrect")
    total = sum(s["population"].values())
    weighted = sum(s["population"][k] / total * (sum(x) / len(x)) for k, x in by.items())
    agree = sum(p1[i]["verdict"] == p2[i]["verdict"] for i in p2) / len(p2)
    out = {"instances": len(p1), "instance_error_rate_raw": sum(v["verdict"] == "incorrect" for v in p1.values()) / len(p1),
           "label_audit_error_rate": weighted, "weighting": "strata weighted to the assigned dedup population",
           "facts": {k: sum(v[k] for v in p1.values()) for k in ("gold_facts", "missing", "spurious", "wrong_type")},
           "self_agreement": agree, "min_self_agreement": config.get("eval.label_audit.min_self_agreement"),
           "self_agreement_met": agree > config.get("eval.label_audit.min_self_agreement"),
           "by_stratum": {k: {"n": len(x), "error_rate": sum(x) / len(x)} for k, x in sorted(by.items())}}
    (AUDIT / "audit_result.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["draw", "score"])
    {"draw": draw, "score": score}[ap.parse_args().action]()
