"""Draw and pin the frontier sample, the probe filings and the latency ids, once (PRD section 7; T-026).

Requires data/phase3/power_check.json (written only from a real scorer output) for N and the arms; refuses
without it and refuses to overwrite a pinned sample. Filings are the eval manifest's filings with assigned
instances; table share and size come from the instance map; sector is the SEC office (submissions cache, as in
T-011). Writes data/frontier_sample.json, data/frontier_sample_accessions.txt, data/frontier_latency_ids.txt.
"""
import csv
import json
from collections import defaultdict

from frontier.sampling import draw, latency_ids, probe_filings
from scripts.power_check import KIND, OUT as POWER
from scripts.select_train_accessions import office_key
from xbrl_frontier import config, edgar, eval_sets, finni

DATA = config.REPO_ROOT / "data"
SAMPLE, ACCS, LAT = DATA / "frontier_sample.json", DATA / "frontier_sample_accessions.txt", DATA / "frontier_latency_ids.txt"


def manifest_filings() -> tuple[dict, dict]:
    rows = finni.load()
    imap = eval_sets.load_instance_map()
    per = defaultdict(lambda: {"n": 0, "tables": 0})
    for i, m in imap.items():
        per[m["accession"]]["n"] += 1
        per[m["accession"]]["tables"] += eval_sets.context_type(rows[i]) == "table"
    man = {r["accession"]: r for r in csv.DictReader(open(config.docs_path(config.get("pins.eval_manifest"))))}
    filings = {}
    for a, c in per.items():
        sub = edgar.submissions(man[a]["cik"])
        filings[a] = {"cik": man[a]["cik"], "n": c["n"], "table_share": c["tables"] / c["n"],
                      "sector": office_key(sub.get("ownerOrg"))}
    return filings, imap


def main() -> None:
    if not POWER.exists() or json.loads(POWER.read_text()).get("kind") != KIND:
        raise SystemExit("no power check from a real scorer output (data/phase3/power_check.json); refusing")
    if SAMPLE.exists() or ACCS.exists() or LAT.exists():
        raise SystemExit("the frontier sample is pinned once; refusing to overwrite")
    pc = json.loads(POWER.read_text())
    s = config.get("frontier.sampling")
    filings, imap = manifest_filings()
    d = draw(filings, pc["sample_filings"], s["seed"], s["max_redraws"])
    rows = finni.load()
    keep = set(eval_sets.dedup_indices(rows))
    in_sample = set(d["accessions"])
    cands = [i for i, m in imap.items() if m["accession"] in in_sample and i in keep]
    lat = latency_ids(cands, config.get("frontier.sync_latency_instances"), config.get("frontier.latency_sampling.seed"))
    probes = probe_filings(filings, in_sample, pc["arms"], config.get("frontier.probe_selection.seed"))
    out = {"pinned": "2026-10-02 rule; drawn by scripts/draw_frontier_sample.py (T-026)", "power_check": str(POWER),
           "arms": pc["arms"], **d, "probe_filing_per_arm": probes, "latency_ids_file": str(LAT.relative_to(config.REPO_ROOT)),
           "latency_rule": config.get("frontier.latency_sampling.rule"), "config_hash": config.config_hash()}
    SAMPLE.write_text(json.dumps(out, indent=1))
    ACCS.write_text("".join(a + "\n" for a in d["accessions"]))
    LAT.write_text("".join(f"{i}\n" for i in lat))
    print(json.dumps({k: v for k, v in out.items() if k not in ("cells",)}, indent=1))


if __name__ == "__main__":
    main()
