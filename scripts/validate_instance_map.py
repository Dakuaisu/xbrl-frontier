"""Independent check of pins.eval_instance_map: do the gold fact values occur in the assigned filing?

Uses every assigned positive instance (no sampling). Values are compared after removing , $ % and
whitespace and lowercasing; values shorter than 3 characters are skipped as uninformative.
Writes data/manifest_work/validation.json.
"""
import csv
import html
import json
import re
from collections import defaultdict

from xbrl_frontier import config, edgar, finni

_STRIP = re.compile(r"[,$%\s]")


def main() -> None:
    rows = finni.load()
    man = {r["accession"]: r for r in csv.DictReader(open(config.docs_path(config.get("pins.eval_manifest"))))}
    texts = {}
    for acc, c in man.items():
        body = edgar.filing_doc(c["cik"], acc, c["primary_document"]).decode("utf-8", "replace")
        texts[acc] = _STRIP.sub("", html.unescape(re.sub(r"<[^>]+>", " ", body))).lower()
    by_method = defaultdict(list)
    for r in csv.DictReader(open(config.docs_path(config.get("pins.eval_instance_map")))):
        if not r["accession"]:
            continue
        facts = {_STRIP.sub("", f["Fact"]).lower() for f in json.loads(rows[int(r["instance_id"])].answer)["result"]}
        facts = {f for f in facts if len(f) >= 3}
        if facts:
            t = texts[r["accession"]]
            by_method[r["method"]].append(sum(f in t for f in facts) / len(facts))
    out = {m: {"instances": len(v), "mean_share_gold_values_in_filing": sum(v) / len(v),
               "instances_below_half": sum(x < 0.5 for x in v)} for m, v in sorted(by_method.items())}
    path = config.REPO_ROOT / "data" / "manifest_work" / "validation.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
