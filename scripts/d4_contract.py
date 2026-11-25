"""Pinned D4 identity shared by notebook and local collection."""
import hashlib
import json
from pathlib import Path

from scripts.d4_resume_guard import Identity, Target
from xbrl_frontier import config

SHARD_SHA256 = "d13a93783e256b75c743fe4743dde1de1e9a5e80c972a137afcb783c81d73dcb"
ADAPTER_SHA256 = "e6726442327838fca975fb6a54d0154ba09f8318ff2131893f70f062b0edcfd0"
GENERATOR_FILES = ("serving/generate_predictions.py", "serving/common.py", "training/schedule.py",
                   "training/encoder_train.py", "xbrl_frontier/config.py", "xbrl_frontier/finni.py",
                   "xbrl_frontier/sft_format.py", "xbrl_frontier/output_cap.py")


def target(runner_sha256: str) -> Target:
    student = config.get("pins.students")[1]
    dataset = config.get("pins.eval_finni")
    sources = {name: hashlib.sha256((config.REPO_ROOT / name).read_bytes()).hexdigest() for name in GENERATOR_FILES}
    if sources["serving/generate_predictions.py"] != runner_sha256:
        raise ValueError("generator hash mismatch")
    payload = json.dumps({"generation": config.get("eval.generation"), "sources": sources},
                         sort_keys=True, separators=(",", ":")).encode()
    identity = Identity(student["id"], student["hf_revision"], dataset["hf_revision"], SHARD_SHA256,
                        config.config_hash(), ADAPTER_SHA256, hashlib.sha256(payload).hexdigest())
    return Target(identity, tuple(range(dataset["rows_expected"])))


def validate_local_inputs() -> str:
    audit = json.loads(Path("data/kaggle/d4-output-cap-check.json").read_text())
    if (audit["config_hash"] != config.config_hash() or not audit["all_within_cap"]
            or audit["dataset_revision"] != config.get("pins.eval_finni.hf_revision")
            or audit["split_file_sha256"] != SHARD_SHA256):
        raise ValueError("output-cap audit does not match current configuration")
    for student in config.get("pins.students"):
        row = audit["results"][student["id"] + "|finni"]
        if row["golds"] != config.get("pins.eval_finni.rows_expected") or row["exceeding"] != 0:
            raise ValueError("output-cap audit does not cover every FinNI gold")
    runner = Path("serving/generate_predictions.py")
    digest = hashlib.sha256(runner.read_bytes()).hexdigest()
    pilot = json.loads(Path("data/kaggle/d3-pilot-output/d3-pilot/pilot_preflight.json").read_text())
    if pilot["runner_sha256"] != digest:
        raise ValueError("generator differs from D3")
    pilot_meta = json.loads(Path("data/kaggle/d3-pilot-output/d3-pilot/predictions_dev.meta.json").read_text())
    if pilot_meta["effective_max_model_len"] != config.get("eval.generation.max_model_len"):
        raise ValueError("D3 effective context differs from configured context")
    adapter = Path("data/kaggle/session4-retry-output/run-Qwen3-1.7B-s1/final_adapter/adapter_model.safetensors")
    if hashlib.sha256(adapter.read_bytes()).hexdigest() != ADAPTER_SHA256:
        raise ValueError("final adapter hash differs")
    return digest
