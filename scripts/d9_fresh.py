"""D9: final 1.7B adapter on the pinned fresh slice, retargeted from the verified final-adapter dev pilot.

`notebook` rewrites the pilot notebook's dev inputs to the corpus dataset's fresh_instances.jsonl (pinned sha256 in
data/fresh_selection_report.json), the output directory to d9-fresh and the session length to SESSION_HOURS.
`score` checks one visible T4, the pinned hashes, the pilot's generation settings, completion and exact unique-id
coverage of the fresh slice, runs the oracle tests and scores raw and deduplicated results.
"""
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.d3_10k_score import check_ids
from scripts.d3_curve_dev import equivalent_config, sha
from xbrl_frontier import config

KAGGLE = Path("data/kaggle")
PILOT = KAGGLE / "d3-pilot-output/d3-pilot"
FRESH = Path("data/fresh/instances.jsonl")
DEV_SHA = "e5b1f16488cf504a05e757770d4a7b9d80a6f53656416a687e701d51be7f0746"
SESSION_HOURS = 3
OUT = KAGGLE / "d9-fresh-output/d9-fresh"


def fresh_sha() -> str:
    return json.loads(Path("data/fresh_selection_report.json").read_text())["instances_file"]["sha256"]


def retarget(notebook: dict, fresh_hash: str, fresh_count: int) -> dict:
    result = copy.deepcopy(notebook)
    if len(result["cells"]) != 1:
        raise ValueError("expected a single-cell notebook")
    source = "".join(result["cells"][0]["source"])
    changes = {
        "glob.glob('/kaggle/input/**/dev.jsonl', recursive=True)": "glob.glob('/kaggle/input/**/fresh_instances.jsonl', recursive=True)",
        DEV_SHA: fresh_hash,
        "assert len(items) == 1933 and": f"assert len(items) == {fresh_count} and",
        "'dev_sha256':": "'fresh_sha256':",
        "'dev_instances':": "'fresh_instances':",
        "'provider_timeout_seconds': 7200": f"'provider_timeout_seconds': {SESSION_HOURS * 3600}",
        "/kaggle/working/d3-pilot": "/kaggle/working/d9-fresh",
        "'--set', 'dev'": "'--set', 'fresh'",
        "'--session-hours', '2'": f"'--session-hours', '{SESSION_HOURS}'",
    }
    if any(source.count(old) != 1 for old in changes if old != DEV_SHA) or source.count(DEV_SHA) == 0:
        raise ValueError("pilot notebook does not contain every expected target exactly once")
    for old, new in changes.items():
        source = source.replace(old, new)
    result["cells"][0]["source"] = source.splitlines(keepends=True)
    return result


def notebook() -> None:
    rows = [json.loads(line) for line in FRESH.read_text().splitlines()]
    assert sha(FRESH) == fresh_sha() and len(rows) == len({row["id"] for row in rows})
    pre = json.loads((PILOT / "pilot_preflight.json").read_text())
    assert pre["dev_sha256"] == DEV_SHA
    cap = json.loads(Path("data/output_cap_check.json").read_text())
    assert cap["all_within_cap"] is True and cap["config_hash"] == config.config_hash()
    assert all(v["exceeding"] == 0 for k, v in cap["results"].items() if k.endswith("|fresh"))
    target = KAGGLE / "d9-fresh.ipynb"
    nb = retarget(json.loads((KAGGLE / "d3-supervised-pilot.ipynb").read_text()), fresh_sha(), len(rows))
    target.write_text(json.dumps(nb))
    print(json.dumps({"notebook": str(target), "fresh_sha256": fresh_sha(), "fresh_instances": len(rows),
                      "adapter_sha256": pre["adapter_sha256"], "session_hours": SESSION_HOURS}))


def score() -> None:
    pre = json.loads((OUT / "pilot_preflight.json").read_text())
    ref = json.loads((PILOT / "pilot_preflight.json").read_text())
    meta = json.loads((OUT / "predictions_fresh.meta.json").read_text())
    ref_meta = json.loads((PILOT / "predictions_dev.meta.json").read_text())
    for key in ("prompt_sha256", "runner_sha256", "adapter_sha256"):
        if pre[key] != ref[key]:
            raise ValueError(f"preflight differs from the final-adapter pilot: {key}")
    if pre["visible_device_count"] != 1 or "T4" not in pre["device"]:
        raise ValueError("run did not use one visible T4")
    if pre["fresh_sha256"] != fresh_sha() or sha(FRESH) != fresh_sha():
        raise ValueError("fresh hash mismatch")
    for key in ("model", "revision", "generation", "effective_max_model_len"):
        if meta[key] != ref_meta[key]:
            raise ValueError(f"generation provenance differs: {key}")
    if meta["set"] != "fresh" or not equivalent_config(meta["config_hash"]):
        raise ValueError("wrong set or non-equivalent config")
    if meta["complete"] is not True or meta["targets"]["final"]["complete"] is not True:
        raise ValueError("prediction run is incomplete")
    prediction = OUT / "predictions_final_fresh.jsonl"
    rows = [json.loads(line) for line in FRESH.read_text().splitlines()]
    check_ids(rows, [json.loads(line) for line in prediction.read_text().splitlines()])
    env = {**os.environ, "PYTHONPATH": ".", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    subprocess.run([sys.executable, "-m", "pytest", "tests/test_oracles.py", "-q"], env=env, check=True)
    subprocess.run([sys.executable, "-m", "scripts.score_predictions", "--set", "fresh", "--pred", str(prediction),
                    "--tag", "d9-final-fresh"], env=env, check=True)
    print("Saved data/scores/d9-final-fresh.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["notebook", "score"])
    (notebook if ap.parse_args().action == "notebook" else score)()


if __name__ == "__main__":
    main()
