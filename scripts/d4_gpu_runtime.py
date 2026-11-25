"""Execute D4 inside the importable notebook without persisting gated inputs."""
import hashlib
import json
import logging
import os
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import d4_contract, d4_receipt, d4_security
from scripts.d4_resume_guard import ResumeBlocked
from xbrl_frontier import config

OUTPUT = Path("/kaggle/working/d4")


@dataclass(frozen=True, slots=True)
class Settings:
    kernel_ref: str
    runner_sha256: str
    previous_kernel_ref: str = ""
    previous_receipt_sha256: str = ""


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_context(llm, expected: int) -> None:
    if llm.llm_engine.model_config.max_model_len != expected:
        raise ResumeBlocked("effective context differs from the pinned generation configuration")


@contextmanager
def dataset_snapshot(target):
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    logging.disable(logging.CRITICAL)
    from kaggle_secrets import UserSecretsClient
    try:
        token = UserSecretsClient().get_secret('HF_TOKEN')
    except Exception as exc:
        print(json.dumps({"status": "secret_lookup_failed", "detail": d4_security.secret_error(exc)}))
        raise SystemExit(1) from None
    if not token:
        raise ResumeBlocked("secret lookup returned empty")
    from huggingface_hub import snapshot_download
    import pyarrow.parquet as pq
    with TemporaryDirectory(dir='/tmp') as cache:
        try:
            pin = config.get("pins.eval_finni")
            dataset = Path(snapshot_download(repo_id=pin["repo"], repo_type="dataset", revision=pin["hf_revision"],
                                             cache_dir=cache, token=token))
            shards = sorted(dataset.glob("data/" + pin["split"] + "-*.parquet"))
            if len(shards) != 1 or digest(shards[0]) != target.identity.dataset_sha256:
                raise ResumeBlocked("FinNI shard hash mismatch")
            if pq.read_metadata(shards[0]).num_rows != len(target.expected_ids):
                raise ResumeBlocked("FinNI row count mismatch")
        except Exception as exc:
            print(json.dumps({"status": "dataset_access_failed", "error_type": type(exc).__name__}))
            raise SystemExit(1) from None
        finally:
            del token
        yield dataset


def cpu_probe(runner_sha256: str) -> None:
    if os.path.exists('/dev/nvidia0'):
        raise ResumeBlocked("CPU probe requires accelerator None")
    target = d4_contract.target(runner_sha256)
    with dataset_snapshot(target):
        print(json.dumps({"status": "D4_ACCESS_OK", "revision": target.identity.dataset_revision,
                          "rows": len(target.expected_ids), "shard_sha256": target.identity.dataset_sha256}))


def run(settings: Settings) -> None:
    if not re.fullmatch(r"[a-z0-9_-]+/[a-z0-9_-]+/[1-9][0-9]*", settings.kernel_ref):
        raise ResumeBlocked("set a version-qualified kernel reference")
    if bool(settings.previous_kernel_ref) != bool(settings.previous_receipt_sha256):
        raise ResumeBlocked("both prior reference and verified receipt hash are required")
    target = d4_contract.target(settings.runner_sha256)
    output = OUTPUT
    if output.exists():
        raise ResumeBlocked("output directory already exists")
    import torch
    if torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0):
        raise ResumeBlocked("one visible T4 is required")
    import glob
    weights = [Path(p) for p in glob.glob('/kaggle/input/**/run-Qwen3-1.7B-s1/final_adapter/adapter_model.safetensors', recursive=True)]
    if len(weights) != 1 or digest(weights[0]) != target.identity.adapter_sha256:
        raise ResumeBlocked("final adapter mismatch")
    prior_dir = None
    if settings.previous_kernel_ref:
        receipts = [Path(p) for p in glob.glob('/kaggle/input/**/d4/receipt.json', recursive=True)]
        receipts = [p for p in receipts if digest(p) == settings.previous_receipt_sha256]
        if len(receipts) != 1:
            raise ResumeBlocked("prior verified receipt not found uniquely")
        envelope = json.loads(receipts[0].read_text())
        evidence = d4_receipt.decode(json.dumps(envelope["evidence"]))
        prior_dir = receipts[0].parent
        meta = prior_dir / "predictions_finni.meta.json"
        if digest(meta) != envelope["metadata_sha256"]:
            raise ResumeBlocked("prior metadata hash mismatch")
        plan = d4_receipt.check(prior_dir / "predictions_final_finni.jsonl", evidence,
                                d4_receipt.Expected(target, settings.previous_kernel_ref))
        previous = json.loads(meta.read_text())
        final = previous["targets"]["final"]
        if (previous["model"], previous["revision"], previous["set"], previous["config_hash"]) != (
            target.identity.model, target.identity.model_revision, "finni", target.identity.config_sha256
        ) or previous["generation"] != config.get("eval.generation") or previous["effective_max_model_len"] != config.get("eval.generation.max_model_len") or final["written"] != len(target.expected_ids) - len(plan.missing) or final["complete"] is not plan.complete or previous["complete"] is not plan.complete:
            raise ResumeBlocked("prior generation metadata mismatch")
        if plan.complete:
            raise ResumeBlocked("prior predictions already cover the full set")
    with dataset_snapshot(target) as dataset:
        from huggingface_hub import snapshot_download
        from xbrl_frontier import finni
        finni.download = lambda: dataset
        import serving.common
        serving.common.pinned_snapshot = lambda model: snapshot_download(repo_id=model, revision=serving.common.model_pin(model), token=False)
        import vllm
        original_llm = vllm.LLM

        def same_context(*args, **kwargs):
            llm = original_llm(*args, **kwargs)
            require_context(llm, config.get("eval.generation.max_model_len"))
            return llm

        vllm.LLM = same_context
        import serving.generate_predictions as generator
        output.mkdir(mode=0o700)
        preflight = {"kernel_ref": settings.kernel_ref, "runner_sha256": settings.runner_sha256,
                     "dataset_revision": target.identity.dataset_revision, "dataset_sha256": target.identity.dataset_sha256,
                     "adapter_sha256": target.identity.adapter_sha256, "config_hash": target.identity.config_sha256,
                     "generation_sha256": target.identity.generation_sha256, "visible_device_count": torch.cuda.device_count(),
                     "gpu": torch.cuda.get_device_name(0), "started_unix": float(os.environ["XF_SESSION_START"])}
        (output / "preflight.json").write_text(json.dumps(preflight))
        argv = ["generate_predictions", "--model", target.identity.model, "--set", "finni", "--out", str(output),
                "--adapter", "final=" + str(weights[0].parent), "--session-hours", "8"]
        if prior_dir is not None:
            argv.extend(["--resume", str(prior_dir)])
        sys.argv = argv
        generator.main()
        predictions = output / "predictions_final_finni.jsonl"
        evidence = d4_receipt.make(predictions, target, settings.kernel_ref)
        envelope = {"evidence": json.loads(d4_receipt.encode(evidence)),
                    "metadata_sha256": digest(output / "predictions_finni.meta.json"),
                    "created_unix": time.time()}
        (output / "receipt.json").write_text(json.dumps(envelope, sort_keys=True))
