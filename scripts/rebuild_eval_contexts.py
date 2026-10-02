"""Re-run instance-to-filing assignment with the benchmark's own serialization (PRD section 4, T-006).

For each filing in pins.eval_manifest: serialize with ixbrl_facts.fintagging (record mode), then assign
every FinNI row whose context string equals a serialized context (exact string). Rows the serializer does
not reproduce fall back to the Phase 0 token matcher over the same filings. Also checks the serializer's
labels against the gold answers on exact matches (label parity).

Writes pins.eval_instance_map (new methods), refreshes instances_assigned in pins.eval_manifest, keeps the
Phase 0 map as data/manifest_work/eval_instance_to_filing_phase0_tokenmatch.csv, and writes
data/manifest_work/serializer_rebuild.json.
"""
import csv
import datetime as dt
import json
import shutil
from collections import Counter, defaultdict

from ixbrl_facts import fintagging
from xbrl_frontier import config, edgar, eval_sets, finni, scorer
from xbrl_frontier.instance_match import ContextIndex

WORK = config.REPO_ROOT / "data" / "manifest_work"


def main() -> None:
    rows = finni.load()
    man_path = config.docs_path(config.get("pins.eval_manifest"))
    map_path = config.docs_path(config.get("pins.eval_instance_map"))
    manifest = list(csv.DictReader(open(man_path)))
    old_map = {int(r["instance_id"]): r for r in csv.DictReader(open(map_path))}
    backup = WORK / "eval_instance_to_filing_phase0_tokenmatch.csv"
    if not backup.exists():
        shutil.copy(map_path, backup)
    company_of = {m["accession"]: m["cik"] for m in manifest}

    produced: dict[str, set] = defaultdict(set)
    labels: dict[tuple, list] = {}
    unknown: dict[str, list] = {}
    htmls = {}
    for i, m in enumerate(manifest):
        html = edgar.filing_doc(m["cik"], m["accession"], m["primary_document"]).decode("utf-8", "replace")
        htmls[m["accession"]] = html
        out = fintagging.serialize(html, unknown="record")
        if out["unknown_concepts"]:
            unknown[m["accession"]] = out["unknown_concepts"]
        for kind in ("text", "table"):
            for part in ("pos", "neg"):
                for x in out[kind][part]:
                    produced[x["text"]].add(m["accession"])
                    labels.setdefault((m["accession"], x["text"]), []).append(x["numeric_entities"])
        if i % 20 == 0:
            print(f"serialized {i + 1}/{len(manifest)}", flush=True)

    uniq = Counter()
    for ctx, accs in produced.items():
        if len(accs) == 1:
            uniq[next(iter(accs))] += 1
    dominant = {}
    for a in sorted(company_of, key=lambda a: (-uniq[a], a)):
        dominant.setdefault(company_of[a], a)

    assigned: dict[int, tuple] = {}
    for r in rows:
        accs = produced.get(r.context)
        if not accs:
            continue
        if len(accs) == 1:
            assigned[r.idx] = (next(iter(accs)), "serializer_exact")
        elif len({company_of[a] for a in accs}) == 1:
            assigned[r.idx] = (dominant[company_of[next(iter(accs))]], "serializer_exact_tie_within_company")
        else:
            assigned[r.idx] = (None, "serializer_ambiguous_cross_company")

    rest = {}
    for r in rows:
        if r.idx not in assigned:
            rest.setdefault(r.context, str(r.idx))
    idx = ContextIndex({cid: ctx for ctx, cid in rest.items()})
    for acc, html in htmls.items():
        idx.scan(acc, html)
    fb = idx.assign(set(company_of), company_of, dominant)
    for r in rows:
        if r.idx not in assigned:
            x = fb[rest[r.context]]
            assigned[r.idx] = (x.filing, f"token_{x.method}", x.containment, x.runner_up)

    with open(map_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "accession", "cik", "method", "containment", "runner_up_containment"])
        for r in rows:
            a = assigned[r.idx]
            acc, method = a[0], a[1]
            cont, run = (a[2], a[3]) if len(a) > 2 else ((1.0, 0.0) if acc else (0.0, 0.0))
            w.writerow([r.idx, acc or "", company_of.get(acc, ""), method, f"{cont:.4f}", f"{run:.4f}"])

    per_acc = Counter(a[0] for a in assigned.values() if a[0])
    for m in manifest:
        m["instances_assigned"] = per_acc[m["accession"]]
    with open(man_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)

    agree = Counter()
    for r in rows:
        acc, method = assigned[r.idx][:2]
        if method != "serializer_exact":
            continue
        gold = scorer.gold_set(r.answer)
        # One text can be serialized more than once with different labels (e.g. a div and its only child
        # p yield the same text, the div as a negative); compare against the closest candidate.
        cands = [scorer.pair_set({"Fact": e["value"], "Type": e["type"]} for e in ents)
                 for ents in labels[(acc, r.context)]]
        mine = max(cands, key=lambda c: (len(gold & c), -len(c ^ gold)))
        agree["instances"] += 1
        agree["texts_with_several_label_sets"] += len({c for c in cands}) > 1
        agree["identical_label_set"] += gold == mine
        agree["tp"] += len(gold & mine)
        agree["gold_only"] += len(gold - mine)
        agree["serializer_only"] += len(mine - gold)

    keep = set(eval_sets.dedup_indices(rows))
    n_ok = sum(1 for a in assigned.values() if a[0])
    n_ok_d = sum(1 for i, a in assigned.items() if a[0] and i in keep)
    n_ser = sum(1 for a in assigned.values() if a[0] and a[1].startswith("serializer"))
    report = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config_hash": config.config_hash(),
        "fintagging_commit": config.get("pins.fintagging_source.commit"),
        "manifest_filings": len(manifest),
        "phase0_token_matcher": {
            "assignment_rate_raw": sum(1 for r in old_map.values() if r["accession"]) / len(rows),
            "assignment_rate_dedup": sum(1 for i, r in old_map.items() if r["accession"] and i in keep) / len(keep)},
        "serializer_only_rate_raw": n_ser / len(rows),
        "assignment_rate_raw": n_ok / len(rows),
        "assignment_rate_dedup": n_ok_d / len(keep),
        "instance_assignment_min_fraction": config.get("contamination.instance_assignment_min_fraction"),
        "methods_raw": dict(Counter(a[1] for a in assigned.values())),
        "changed_filing_vs_phase0": sum(1 for i, a in assigned.items()
                                        if a[0] and old_map[i]["accession"] and a[0] != old_map[i]["accession"]),
        "newly_assigned_vs_phase0": sum(1 for i, a in assigned.items() if a[0] and not old_map[i]["accession"]),
        "lost_vs_phase0": sum(1 for i, a in assigned.items() if not a[0] and old_map[i]["accession"]),
        "label_parity_on_serializer_exact": {
            **agree, "identical_share": agree["identical_label_set"] / max(agree["instances"], 1),
            "pair_precision_vs_gold": agree["tp"] / max(agree["tp"] + agree["serializer_only"], 1),
            "pair_recall_vs_gold": agree["tp"] / max(agree["tp"] + agree["gold_only"], 1)},
        "filings_with_unknown_concepts": {a: v for a, v in sorted(unknown.items())},
    }
    (WORK / "serializer_rebuild.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "filings_with_unknown_concepts"}, indent=1))
    print("filings with unknown concepts:", len(unknown))


if __name__ == "__main__":
    main()
