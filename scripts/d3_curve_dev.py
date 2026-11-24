"""Dev-slice predictions and coverage-gated scoring for the remaining curve checkpoints (T-048).

`notebook --prefix N` retargets the proven 20k notebook to curve/prefix_N after checking the run manifest and
that the remote adapter matches the preserved local copy. `score --prefix N` checks the run's provenance against
the final-adapter pilot, scores raw and deduplicated dev results, and refuses anything incomplete.
"""
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from scripts.d3_10k_score import check_ids
from xbrl_frontier import config

KAGGLE = Path("data/kaggle")
RUN = KAGGLE / "session4-retry-output/run-Qwen3-1.7B-s1/run.json"
SOURCE = KAGGLE / "d3-curve-source-check"
DEV = Path("data/corpus/dev.jsonl")
CAP_RAISE_PARENT = "7a37c2d^"
CAP_KEY = ("training", "kaggle", "gpu_hours_cap_total")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def weight_path(prefix: int) -> str:
    return f"run-Qwen3-1.7B-s1/curve/prefix_{prefix}/adapter_model.safetensors"


def tag(prefix: int) -> str:
    return f"d3-{prefix}"


def retarget(notebook: dict, prefix: int, old_hash: str, new_hash: str) -> dict:
    result = copy.deepcopy(notebook)
    if len(result["cells"]) != 1:
        raise ValueError("expected a single-cell notebook")
    source = "".join(result["cells"][0]["source"])
    changes = {"curve/prefix_20000/adapter_model.safetensors": f"curve/prefix_{prefix}/adapter_model.safetensors",
               "d3-20k": tag(prefix), "prefix_20000=": f"prefix_{prefix}=", old_hash: new_hash}
    if any(source.count(old) == 0 for old in changes):
        raise ValueError("20k notebook does not contain every expected target")
    for old, new in changes.items():
        source = source.replace(old, new)
    result["cells"][0]["source"] = source.splitlines(keepends=True)
    return result


def drop_key(tree: dict, key: tuple[str, ...]) -> dict:
    tree = copy.deepcopy(tree)
    node = tree
    for part in key[:-1]:
        node = node[part]
    del node[key[-1]]
    return tree


def equivalent_config(hash_value: str, old_text: bytes | None = None) -> bool:
    """True for the current config, or the pre-T-047 config when it differs from today's only in the cap key."""
    if hash_value == config.config_hash():
        return True
    if old_text is None:
        old_text = subprocess.run(["git", "show", f"{CAP_RAISE_PARENT}:docs/config/thresholds.yaml"],
                                  capture_output=True, check=True, cwd=config.REPO_ROOT).stdout
    h = hashlib.sha256(old_text)
    h.update(config.docs_path(config.thresholds()["pins"]["price_sheet"]).read_bytes())
    if h.hexdigest() != hash_value:
        return False
    return drop_key(yaml.safe_load(old_text), CAP_KEY) == drop_key(config.thresholds(), CAP_KEY)


def check_reports(reports: list[dict]) -> None:
    keys = ("set", "instances_raw", "instances_dedup")
    values = [tuple(report[key] for key in keys) for report in reports]
    if not values or values[0][0] != "dev" or any(item != values[0] for item in values):
        raise ValueError("score reports differ in set or coverage")
    if not all(equivalent_config(report["config_hash"]) for report in reports):
        raise ValueError("score report config is not equivalent to the current config")


def curve_point(prefix: int) -> dict:
    run = json.loads(RUN.read_text())
    assert run["complete"] is True and run["revision"] == config.get("pins.students")[1]["hf_revision"]
    points = [p for p in run["curve_checkpoints"] if p["path"] == f"curve/prefix_{prefix}"]
    assert len(points) == 1, points
    return points[0]


def notebook(prefix: int) -> None:
    point = curve_point(prefix)
    weight_hash = sha(SOURCE / weight_path(prefix))
    assert weight_hash == sha(KAGGLE / "session3-v4" / weight_path(prefix))
    preflight = json.loads((KAGGLE / "d3-20k-output/d3-20k/pilot_preflight.json").read_text())
    rows = [json.loads(line) for line in DEV.read_text().splitlines()]
    assert len(rows) == preflight["dev_instances"] == len({row["id"] for row in rows})
    assert sha(DEV) == preflight["dev_sha256"] == json.loads(RUN.read_text())["corpus_sha256"]["dev"]
    cap = json.loads(Path("data/output_cap_check.json").read_text())
    assert cap["all_within_cap"] is True and cap["config_hash"] == config.config_hash()
    original = json.loads((KAGGLE / "d3-20k-dev.ipynb").read_text())
    target = KAGGLE / f"{tag(prefix)}-dev.ipynb"
    target.write_text(json.dumps(retarget(original, prefix, preflight["adapter_sha256"], weight_hash)))
    print(json.dumps({"notebook": str(target), "prefix": prefix, "global_step": point["global_step"],
                      "instances_seen": point["instances_seen"], "weight_sha256": weight_hash,
                      "dev_sha256": preflight["dev_sha256"], "config_hash": config.config_hash()}))


def score(prefix: int) -> None:
    curve_point(prefix)
    root = KAGGLE / f"{tag(prefix)}-output" / tag(prefix)
    final = KAGGLE / "d3-pilot-output/d3-pilot"
    pre = json.loads((root / "pilot_preflight.json").read_text())
    ref = json.loads((final / "pilot_preflight.json").read_text())
    meta = json.loads((root / "predictions_dev.meta.json").read_text())
    final_meta = json.loads((final / "predictions_dev.meta.json").read_text())
    prov = json.loads(Path("data/scores/d3-final-dev-pilot.provenance.json").read_text())
    for key in ("dev_sha256", "prompt_sha256", "runner_sha256", "dev_instances"):
        if pre[key] != ref[key]:
            raise ValueError(f"preflight differs: {key}")
    if pre["visible_device_count"] != 1 or "T4" not in pre["device"]:
        raise ValueError("run did not use one visible T4")
    if pre["adapter_sha256"] != sha(SOURCE / weight_path(prefix)):
        raise ValueError("checkpoint hash mismatch")
    if sha(DEV) != prov["dev_sha256"] or pre["dev_sha256"] != prov["dev_sha256"]:
        raise ValueError("dev hash mismatch")
    for key in ("model", "revision", "set", "generation", "effective_max_model_len"):
        if meta[key] != final_meta[key]:
            raise ValueError(f"generation provenance differs: {key}")
    if not equivalent_config(meta["config_hash"]):
        raise ValueError("generation config is not equivalent to the current config")
    if meta["complete"] is not True or meta["targets"][f"prefix_{prefix}"]["complete"] is not True:
        raise ValueError("prediction run is incomplete")
    prediction = root / f"predictions_prefix_{prefix}_dev.jsonl"
    rows = [json.loads(line) for line in DEV.read_text().splitlines()]
    check_ids(rows, [json.loads(line) for line in prediction.read_text().splitlines()])
    final_path = Path(prov["score_report"])
    if sha(final_path) != prov["score_report_sha256"]:
        raise ValueError("final score report changed")
    env = {**os.environ, "PYTHONPATH": ".", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    subprocess.run([sys.executable, "-m", "pytest", "tests/test_oracles.py", "-q"], env=env, check=True)
    out_tag = f"d3-prefix-{prefix}-dev"
    subprocess.run([sys.executable, "-m", "scripts.score_predictions", "--set", "dev", "--pred", str(prediction),
                    "--tag", out_tag], env=env, check=True)
    check_reports([json.loads(final_path.read_text()), json.loads(Path(f"data/scores/{out_tag}.json").read_text())])
    print(f"Saved data/scores/{out_tag}.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["notebook", "score"])
    ap.add_argument("--prefix", type=int, required=True)
    args = ap.parse_args()
    (notebook if args.action == "notebook" else score)(args.prefix)


if __name__ == "__main__":
    main()
