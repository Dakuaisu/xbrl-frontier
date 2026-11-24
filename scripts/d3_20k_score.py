"""Coverage-gated dev scoring and comparison across the three measured checkpoints."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.d3_10k_score import check_ids
from xbrl_frontier import config


def check_reports(reports: list[dict]) -> None:
    keys = ("set", "config_hash", "instances_raw", "instances_dedup")
    values = [tuple(report[key] for key in keys) for report in reports]
    if not values or values[0][0] != "dev" or any(item != values[0] for item in values):
        raise ValueError("score reports differ in set, config or coverage")


def main() -> None:
    root = Path("data/kaggle/d3-20k-output/d3-20k")
    old = Path("data/kaggle/d3-10k-output/d3-10k")
    final = Path("data/kaggle/d3-pilot-output/d3-pilot")
    dev = Path("data/corpus/dev.jsonl")
    pre = json.loads((root / "pilot_preflight.json").read_text())
    ref = json.loads((old / "pilot_preflight.json").read_text())
    metadata = json.loads((root / "predictions_dev.meta.json").read_text())
    old_meta = json.loads((old / "predictions_dev.meta.json").read_text())
    final_meta = json.loads((final / "predictions_dev.meta.json").read_text())
    prov = json.loads(Path("data/scores/d3-final-dev-pilot.provenance.json").read_text())
    for key in ("dev_sha256", "prompt_sha256", "runner_sha256", "dev_instances"):
        if pre[key] != ref[key]:
            raise ValueError(f"preflight differs: {key}")
    if pre["visible_device_count"] != 1 or "T4" not in pre["device"]:
        raise ValueError("run did not use one visible T4")
    weight = Path("data/kaggle/d3-20k-source-check/run-Qwen3-1.7B-s1/curve/prefix_20000/adapter_model.safetensors")
    if pre["adapter_sha256"] != hashlib.sha256(weight.read_bytes()).hexdigest():
        raise ValueError("checkpoint hash mismatch")
    if hashlib.sha256(dev.read_bytes()).hexdigest() != prov["dev_sha256"] or pre["dev_sha256"] != prov["dev_sha256"]:
        raise ValueError("dev hash mismatch")
    for key in ("model", "revision", "set", "generation", "config_hash", "effective_max_model_len"):
        if metadata[key] != old_meta[key] or metadata[key] != final_meta[key]:
            raise ValueError(f"generation provenance differs: {key}")
    if metadata["config_hash"] != config.config_hash() or metadata["complete"] is not True or metadata["targets"]["prefix_20000"]["complete"] is not True:
        raise ValueError("prediction run is incomplete or config differs")
    rows = [json.loads(line) for line in dev.read_text().splitlines()]
    prediction = root / "predictions_prefix_20000_dev.jsonl"
    old_prediction = old / "predictions_prefix_10000_dev.jsonl"
    final_prediction = final / "predictions_final_dev.jsonl"
    for file in (prediction, old_prediction, final_prediction):
        check_ids(rows, [json.loads(line) for line in file.read_text().splitlines()])
    old_score_path = Path("data/scores/d3-prefix-10000-dev.json")
    final_score_path = Path(prov["score_report"])
    old_score = json.loads(old_score_path.read_text())
    final_score = json.loads(final_score_path.read_text())
    if hashlib.sha256(final_score_path.read_bytes()).hexdigest() != prov["score_report_sha256"]:
        raise ValueError("final score report changed")
    if old_score["predictions_sha256"] != hashlib.sha256(old_prediction.read_bytes()).hexdigest() or final_score["predictions_sha256"] != hashlib.sha256(final_prediction.read_bytes()).hexdigest():
        raise ValueError("prior score prediction hash mismatch")
    env = {**os.environ, "PYTHONPATH": ".", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    subprocess.run([sys.executable, "-m", "pytest", "tests/test_oracles.py", "-q"], env=env, check=True)
    subprocess.run([sys.executable, "-m", "scripts.score_predictions", "--set", "dev", "--pred", str(prediction),
                    "--tag", "d3-prefix-20000-dev"], env=env, check=True)
    current_path = Path("data/scores/d3-prefix-20000-dev.json")
    current = json.loads(current_path.read_text())
    check_reports([old_score, current, final_score])
    result = {"set": "dev", "dev_sha256": pre["dev_sha256"], "config_hash": config.config_hash(),
              "checkpoint_sha256": pre["adapter_sha256"], "reports": {"10000": str(old_score_path),
              "20000": str(current_path), "final": str(final_score_path)}, "metrics": {}}
    for variant in ("raw", "dedup"):
        result["metrics"][variant] = {name: {"overall": report[variant]["overall"],
                                               "context_type": report[variant]["slices"]["context_type"]}
                                      for name, report in (("10000", old_score), ("20000", current), ("final", final_score))}
    destination = Path("data/scores/d3-10k-20k-final-dev.json")
    destination.write_text(json.dumps(result, indent=2))
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
