"""Shared helpers for owner-run serving scripts (Kaggle T4, RTX 3050 under WSL2).

These run on the GPU machine, not on the Mac. They need the repo checkout (for docs/config) and the
software in pins.software installed at exactly the pinned versions.
"""
import importlib.metadata
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from xbrl_frontier import config

PORT = 8000


def require_pins(*names: str) -> dict:
    """Installed versions must equal pins.software.<name>; empty pins stop the run."""
    pins = config.get("pins.software")
    got = {}
    for n in names:
        want = pins.get(n, "")
        if not want:
            sys.exit(f"pins.software.{n} is empty; the owner must approve and pin it first (docs/OPEN.md)")
        have = importlib.metadata.version(n)
        if have != want:
            sys.exit(f"{n}=={have} installed but pins.software.{n}=={want}")
        got[n] = have
    return got


def env_record(card: str) -> dict:
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception as e:  # recorded, not fatal: the environment record is descriptive
            return f"unavailable: {e}"
    rec = {"card": card, "hardware": config.get(f"serving.hardware.{card}"),
           "python": sys.version, "platform": platform.platform(),
           "nvidia_smi": run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
                              "--format=csv,noheader"]),
           "config_hash": config.config_hash(), "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for pkg in ("vllm", "torch", "transformers", "llmcompressor", "compressed-tensors"):
        try:
            rec[f"version::{pkg}"] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass
    return rec


def model_pin(model_id: str) -> str:
    for s in config.get("pins.students"):
        if s["id"] == model_id:
            return s["hf_revision"]
    raise KeyError(f"{model_id} is not in pins.students")


def pinned_snapshot(model_id: str) -> str:
    """Local directory of the model at its pinned commit; served and tokenized from here."""
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=model_id, revision=model_pin(model_id))


class Server:
    """`vllm serve` subprocess; waits for /health, logs to a file, killed on exit."""

    def __init__(self, model: str, args: list[str], log_path: Path):
        self.cmd = ["vllm", "serve", model, "--port", str(PORT), *args]
        self.log_path = log_path
        self.proc = None

    def __enter__(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(self.log_path, "w")
        self.log.write(" ".join(self.cmd) + "\n")
        self.log.flush()
        self.proc = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vllm serve exited with {self.proc.returncode}; see {self.log_path}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                    if r.status == 200:
                        return self
            except OSError:
                pass
            time.sleep(2)

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
        self.log.close()


def bench(model: str, tokenizer: str, input_len: int, output_len: int, num_prompts: int, concurrency: int,
          out_json: Path, log_path: Path) -> dict:
    """One `vllm bench serve` point (random prompts of the FinNI token shape, fixed output length)."""
    slo = config.get("serving.slo")
    cmd = ["vllm", "bench", "serve", "--backend", "vllm", "--model", model, "--tokenizer", tokenizer,
           "--port", str(PORT), "--dataset-name", "random", "--random-input-len", str(input_len),
           "--random-output-len", str(output_len), "--ignore-eos", "--num-prompts", str(num_prompts),
           "--max-concurrency", str(concurrency), "--request-rate", "inf",
           "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,95",
           "--goodput", f"ttft:{slo['ttft_p95_ms']}", f"e2el:{slo['e2e_p95_ms']}",
           "--save-result", "--result-dir", str(out_json.parent), "--result-filename", out_json.name]
    with open(log_path, "a") as log:
        log.write(" ".join(cmd) + "\n")
        log.flush()
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not out_json.exists():
        raise RuntimeError(f"vllm bench serve failed (rc={rc}); see {log_path}")
    raw = json.loads(out_json.read_text())
    return {"concurrency": concurrency, "num_prompts": num_prompts,
            "completed": raw["completed"], "request_throughput": raw["request_throughput"],
            "output_throughput": raw.get("output_throughput"),
            "p50_ttft_ms": raw["median_ttft_ms"], "p95_ttft_ms": raw["p95_ttft_ms"],
            "p50_e2el_ms": raw["median_e2el_ms"], "p95_e2el_ms": raw["p95_e2el_ms"],
            "request_goodput_per_request_slo": raw.get("request_goodput"), "raw_file": out_json.name}


def archive(out_dir: Path) -> Path:
    return Path(shutil.make_archive(str(out_dir), "gztar", root_dir=out_dir.parent, base_dir=out_dir.name))
