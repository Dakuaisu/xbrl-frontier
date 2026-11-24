"""Prepare the 10k evaluation from the verified final-adapter pilot notebook."""
import copy
import hashlib
import json
from pathlib import Path

from xbrl_frontier import config


def retarget(notebook: dict, old_hash: str, new_hash: str) -> dict:
    result = copy.deepcopy(notebook)
    if len(result["cells"]) != 1:
        raise ValueError("expected a single-cell verified pilot notebook")
    source = "".join(result["cells"][0]["source"])
    changes = {"final_adapter/adapter_model.safetensors": "curve/prefix_10000/adapter_model.safetensors",
               "d3-pilot": "d3-10k", "final=": "prefix_10000=", old_hash: new_hash}
    if any(source.count(old) == 0 for old in changes):
        raise ValueError("pilot notebook does not contain every expected target")
    for old, new in changes.items():
        source = source.replace(old, new)
    result["cells"][0]["source"] = source.splitlines(keepends=True)
    return result


def main() -> None:
    root = Path("data/kaggle")
    run = json.loads((root / "session4-retry-output/run-Qwen3-1.7B-s1/run.json").read_text())
    points = [point for point in run["curve_checkpoints"] if point["path"] == "curve/prefix_10000"]
    assert len(points) == 1 and points[0]["global_step"] == 625 and points[0]["instances_seen"] == 10000
    assert run["complete"] is True and run["revision"] == config.get("pins.students")[1]["hf_revision"]
    remote = root / "d3-10k-source-check/run-Qwen3-1.7B-s1/curve/prefix_10000/adapter_model.safetensors"
    local = root / "session3-v4/run-Qwen3-1.7B-s1/curve/prefix_10000/adapter_model.safetensors"
    remote_hash = hashlib.sha256(remote.read_bytes()).hexdigest()
    assert remote_hash == hashlib.sha256(local.read_bytes()).hexdigest()
    baseline = json.loads((root / "d3-pilot-output/d3-pilot/pilot_preflight.json").read_text())
    dev = Path("data/corpus/dev.jsonl")
    rows = [json.loads(line) for line in dev.read_text().splitlines()]
    assert len(rows) == baseline["dev_instances"] == len({row["id"] for row in rows})
    assert hashlib.sha256(dev.read_bytes()).hexdigest() == baseline["dev_sha256"] == run["corpus_sha256"]["dev"]
    cap = json.loads(Path("data/output_cap_check.json").read_text())
    assert cap["all_within_cap"] is True and cap["config_hash"] == config.config_hash()
    original = json.loads((root / "d3-supervised-pilot.ipynb").read_text())
    notebook = retarget(original, baseline["adapter_sha256"], remote_hash)
    path = root / "d3-10k-dev.ipynb"
    path.write_text(json.dumps(notebook))
    print(json.dumps({"notebook": str(path), "weight_sha256": remote_hash, "dev_sha256": baseline["dev_sha256"],
                      "dev_instances": len(rows), "config_hash": config.config_hash()}))


if __name__ == "__main__":
    main()
