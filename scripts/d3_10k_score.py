"""Gate full-dev checkpoint scoring and save comparison to the final adapter."""
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from xbrl_frontier import config


def check_ids(dev: list[dict], predictions: list[dict]) -> None:
    expected = Counter(row["id"] for row in dev)
    actual = Counter(row["id"] for row in predictions)
    if expected != actual or any(count != 1 for count in expected.values()):
        raise ValueError("prediction IDs do not exactly match unique dev IDs")


def main() -> None:
    root = Path("data/kaggle/d3-10k-output/d3-10k")
    prior = Path("data/kaggle/d3-pilot-output/d3-pilot")
    dev = Path("data/corpus/dev.jsonl")
    prediction = root / "predictions_prefix_10000_dev.jsonl"
    preflight = json.loads((root / "pilot_preflight.json").read_text())
    reference = json.loads((prior / "pilot_preflight.json").read_text())
    metadata = json.loads((root / "predictions_dev.meta.json").read_text())
    final_meta = json.loads((prior / "predictions_dev.meta.json").read_text())
    provenance = json.loads(Path("data/scores/d3-final-dev-pilot.provenance.json").read_text())
    final_path = Path(provenance["score_report"])
    if hashlib.sha256(final_path.read_bytes()).hexdigest() != provenance["score_report_sha256"]:
        raise ValueError("final score report changed")
    for key in ("dev_sha256", "prompt_sha256", "runner_sha256", "dev_instances"):
        if preflight[key] != reference[key]:
            raise ValueError(f"pilot preflight differs: {key}")
    if preflight["visible_device_count"] != 1 or "T4" not in preflight["device"]:
        raise ValueError("pilot did not use one visible T4")
    weight = Path("data/kaggle/d3-10k-source-check/run-Qwen3-1.7B-s1/curve/prefix_10000/adapter_model.safetensors")
    if preflight["adapter_sha256"] != hashlib.sha256(weight.read_bytes()).hexdigest():
        raise ValueError("checkpoint hash mismatch")
    if hashlib.sha256(dev.read_bytes()).hexdigest() != provenance["dev_sha256"] or preflight["dev_sha256"] != provenance["dev_sha256"]:
        raise ValueError("dev hash mismatch")
    for key in ("model", "revision", "set", "generation", "config_hash", "effective_max_model_len"):
        if metadata[key] != final_meta[key]:
            raise ValueError(f"generation provenance differs: {key}")
    if metadata["config_hash"] != config.config_hash() or metadata["complete"] is not True or metadata["targets"]["prefix_10000"]["complete"] is not True:
        raise ValueError("prediction run is incomplete or config differs")
    dev_rows = [json.loads(line) for line in dev.read_text().splitlines()]
    pred_rows = [json.loads(line) for line in prediction.read_text().splitlines()]
    check_ids(dev_rows, pred_rows)
    env = {**os.environ, "PYTHONPATH": ".", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    subprocess.run([sys.executable, "-m", "pytest", "tests/test_oracles.py", "-q"], env=env, check=True)
    subprocess.run([sys.executable, "-m", "scripts.score_predictions", "--set", "dev", "--pred", str(prediction),
                    "--tag", "d3-prefix-10000-dev"], env=env, check=True)
    current_path = Path("data/scores/d3-prefix-10000-dev.json")
    current, final = json.loads(current_path.read_text()), json.loads(final_path.read_text())
    if (current["set"], current["config_hash"], current["instances_raw"], current["instances_dedup"]) != (final["set"], final["config_hash"], final["instances_raw"], final["instances_dedup"]):
        raise ValueError("score reports cover different sets")
    comparison = {"set": "dev", "checkpoint_sha256": preflight["adapter_sha256"],
                  "dev_sha256": preflight["dev_sha256"], "config_hash": config.config_hash(),
                  "checkpoint_score": str(current_path), "final_score": str(final_path), "metrics": {}}
    for variant in ("raw", "dedup"):
        comparison["metrics"][variant] = {"checkpoint": current[variant]["overall"], "final": final[variant]["overall"],
                                          "checkpoint_context_type": current[variant]["slices"]["context_type"],
                                          "final_context_type": final[variant]["slices"]["context_type"],
                                          "f1_difference_checkpoint_minus_final": current[variant]["overall"]["finni_f1"] - final[variant]["overall"]["finni_f1"]}
    destination = Path("data/scores/d3-prefix-10000-vs-final-dev.json")
    destination.write_text(json.dumps(comparison, indent=2))
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
