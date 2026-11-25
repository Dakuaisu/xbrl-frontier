"""Collect a version-qualified D4 output and score only verified full coverage."""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from scripts import d4_contract, d4_receipt
from scripts.d4_resume_guard import ResumeBlocked, ResumePlan, Target
from xbrl_frontier import config


def inspect_output(directory: Path, target: Target, kernel_ref: str) -> ResumePlan:
    try:
        preflight = json.loads((directory / "preflight.json").read_text())
        envelope = json.loads((directory / "receipt.json").read_text())
        metadata_path = directory / "predictions_finni.meta.json"
        metadata = json.loads(metadata_path.read_text())
        evidence = d4_receipt.decode(json.dumps(envelope["evidence"]))
        if hashlib.sha256(metadata_path.read_bytes()).hexdigest() != envelope["metadata_sha256"]:
            raise ResumeBlocked("metadata hash mismatch")
        identity = target.identity
        expected = {"kernel_ref": kernel_ref, "visible_device_count": 1, "adapter_sha256": identity.adapter_sha256,
                    "dataset_sha256": identity.dataset_sha256, "dataset_revision": identity.dataset_revision,
                    "config_hash": identity.config_sha256, "generation_sha256": identity.generation_sha256}
        if any(preflight[key] != value for key, value in expected.items()) or "T4" not in preflight["gpu"]:
            raise ResumeBlocked("preflight mismatch")
        if (metadata["model"], metadata["revision"], metadata["set"], metadata["config_hash"]) != (
            identity.model, identity.model_revision, "finni", identity.config_sha256
        ):
            raise ResumeBlocked("generation metadata mismatch")
        if metadata["generation"] != config.get("eval.generation") or metadata["effective_max_model_len"] != config.get("eval.generation.max_model_len"):
            raise ResumeBlocked("generation settings or effective context mismatch")
        plan = d4_receipt.check(directory / "predictions_final_finni.jsonl", evidence,
                                d4_receipt.Expected(target, kernel_ref))
        actual = len(target.expected_ids) - len(plan.missing)
        final = metadata["targets"]["final"]
        if final["written"] != actual or final["complete"] is not plan.complete or metadata["complete"] is not plan.complete:
            raise ResumeBlocked("completion metadata does not match unique coverage")
        return plan
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise ResumeBlocked("D4 output is incomplete or malformed") from exc


def score_if_complete(plan: ResumePlan, destination: Path) -> bool:
    if not plan.complete:
        return False
    if plan.missing:
        raise ResumeBlocked("complete plan cannot contain missing IDs")
    env = {**os.environ, "PYTHONPATH": ".", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    report = config.REPO_ROOT / "data/scores/d4-final-finni.json"
    if report.exists():
        raise SystemExit("score already exists; refusing to overwrite")
    for command in ([sys.executable, "-m", "pytest", "tests/test_oracles.py", "-q"],
                    [sys.executable, "-m", "scripts.score_predictions", "--set", "finni", "--pred",
                     str(destination / "predictions_final_finni.jsonl"), "--tag", "d4-final-finni"]):
        run = subprocess.run(command, env=env, capture_output=True, text=True)
        if run.returncode != 0:
            raise SystemExit("oracle or full-set scoring failed; inspect locally without exposing rows")
    return True


def fetch_output(kaggle: str, kernel_ref: str, destination: Path) -> bool:
    try:
        result = subprocess.run([kaggle, "kernels", "output", kernel_ref, "-p", str(destination), "--page-size", "200",
                                 "--file-pattern", "^d4/.*"], capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True, help="owner/kernel/version")
    parser.add_argument("--out", type=Path, required=True, help="new private directory under data/kaggle")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9_-]+/[a-z0-9_-]+/[1-9][0-9]*", args.kernel):
        raise SystemExit("kernel must be version-qualified")
    destination = args.out.resolve()
    if not destination.is_relative_to((config.REPO_ROOT / "data/kaggle").resolve()) or destination.exists():
        raise SystemExit("destination must be a new directory under data/kaggle")
    runner_hash = d4_contract.validate_local_inputs()
    target = d4_contract.target(runner_hash)
    kaggle = str(config.REPO_ROOT / ".venv/bin/kaggle")
    while True:
        try:
            result = subprocess.run([kaggle, "kernels", "status", args.kernel], capture_output=True, text=True, timeout=60)
            status = re.search(r"KernelWorkerStatus\.(COMPLETE|ERROR|CANCEL_ACKNOWLEDGED)", result.stdout) if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            status = None
        if status:
            break
        print(json.dumps({"kernel": args.kernel, "status": "waiting_or_unknown"}), flush=True)
        time.sleep(60)
    if status.group(1) != "COMPLETE":
        raise SystemExit("terminal failure; no scoring or resume receipt accepted")
    with tempfile.TemporaryDirectory() as tmp:
        while not fetch_output(kaggle, args.kernel, Path(tmp)):
            print(json.dumps({"kernel": args.kernel, "status": "output_unavailable_retry"}), flush=True)
            time.sleep(60)
        source = Path(tmp) / "d4"
        plan = inspect_output(source, target, args.kernel)
        destination.mkdir(mode=0o700, parents=True)
        for name in ("preflight.json", "receipt.json", "predictions_finni.meta.json", "predictions_final_finni.jsonl"):
            shutil.copyfile(source / name, destination / name)
            os.chmod(destination / name, 0o600)
    result = {"kernel": args.kernel, "complete": plan.complete, "missing": len(plan.missing), "cutoffs": plan.cutoffs,
              "receipt_sha256": hashlib.sha256((destination / "receipt.json").read_bytes()).hexdigest(),
              "out": str(destination)}
    print(json.dumps(result), flush=True)
    if score_if_complete(plan, destination):
        print(json.dumps({"score_report": str(config.REPO_ROOT / "data/scores/d4-final-finni.json"), "status": "scored_complete"}))


if __name__ == "__main__":
    main()
