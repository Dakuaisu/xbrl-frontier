"""Prepare the single supervised D3 timing notebook from verified local inputs."""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from scripts import d3_pilot_gate
from xbrl_frontier import config


@dataclass(frozen=True, slots=True)
class Inputs:
    template: str
    prompt_digest: str
    dev_digest: str
    weight_digest: str
    runner_digest: str
    dev_count: int


def build_notebook(inputs: Inputs) -> dict:
    source = f'''import os, sys, time, json, glob, hashlib, subprocess, runpy
from pathlib import Path
os.environ['XF_SESSION_START'] = str(time.time())
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
template = {inputs.template!r}
prompt_digest = {inputs.prompt_digest!r}
assert hashlib.sha256(template.encode()).hexdigest() == prompt_digest
sources = glob.glob('/kaggle/input/**/xbrl-frontier/serving/generate_predictions.py', recursive=True)
assert len(sources) == 1, sources
runner = Path(sources[0])
assert hashlib.sha256(runner.read_bytes()).hexdigest() == {inputs.runner_digest!r}
root = runner.parent.parent
sys.path.insert(0, str(root))
os.chdir(root)
devs = glob.glob('/kaggle/input/**/dev.jsonl', recursive=True)
devs = [Path(p) for p in devs if hashlib.sha256(Path(p).read_bytes()).hexdigest() == {inputs.dev_digest!r}]
assert len(devs) == 1, devs
items = [json.loads(line) for line in devs[0].read_text().splitlines()]
assert len(items) == {inputs.dev_count} and len({{row['id'] for row in items}}) == len(items)
weights = glob.glob('/kaggle/input/**/run-Qwen3-1.7B-s1/final_adapter/adapter_model.safetensors', recursive=True)
assert len(weights) == 1, weights
with open(weights[0], 'rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == {inputs.weight_digest!r}
from xbrl_frontier import config
assert config.get('eval.generation.temperature') == 0.0
pins = config.get('pins.software')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'vllm==' + pins['vllm'], 'pyyaml==6.0.3', 'pyarrow==25.0.1'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', '-q', 'torchaudio'], check=True)
import torch
assert torch.cuda.device_count() == 1
assert 'T4' in torch.cuda.get_device_name(0)
out = Path('/kaggle/working/d3-pilot')
out.mkdir(parents=True, exist_ok=True)
(out / 'pilot_preflight.json').write_text(json.dumps({{'visible_device_count': torch.cuda.device_count(), 'device': torch.cuda.get_device_name(0), 'prompt_sha256': prompt_digest, 'dev_sha256': {inputs.dev_digest!r}, 'adapter_sha256': {inputs.weight_digest!r}, 'runner_sha256': {inputs.runner_digest!r}, 'provider_timeout_seconds': 7200, 'dev_instances': len(items)}}))
from xbrl_frontier import sft_format
sft_format.query_template = lambda: template
import serving.common
from huggingface_hub import snapshot_download
serving.common.pinned_snapshot = lambda model: snapshot_download(repo_id=model, revision=serving.common.model_pin(model), token=False)
sys.argv = ['serving.generate_predictions', '--model', 'Qwen/Qwen3-1.7B', '--set', 'dev', '--corpus', str(devs[0].parent), '--out', str(out), '--adapter', 'final=' + str(Path(weights[0]).parent), '--session-hours', '2']
runpy.run_module('serving.generate_predictions', run_name='__main__')
'''
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source.splitlines(keepends=True)}]}


def main() -> None:
    root = config.REPO_ROOT
    final = root / "data/kaggle/session4-retry-output/run-Qwen3-1.7B-s1"
    report = json.loads((root / "data/corpus_report.json").read_text())
    dev = root / "data/corpus/dev.jsonl"
    dev_digest = hashlib.sha256(dev.read_bytes()).hexdigest()
    assert dev_digest == report["splits"]["dev"]["sha256"]
    rows = [json.loads(line) for line in dev.read_text().splitlines()]
    assert len(rows) == report["splits"]["dev"]["used"] == len({row["id"] for row in rows})
    revision = config.get("pins.students")[1]["hf_revision"]
    weight_digest = d3_pilot_gate.verify_final(final, d3_pilot_gate.Provenance(revision, dev_digest, config.config_hash()))
    assert weight_digest == "e6726442327838fca975fb6a54d0154ba09f8318ff2131893f70f062b0edcfd0"
    prompt = json.loads((root / "data/kaggle/d3-prompt.json").read_text())
    assert prompt["revision"] == config.get("pins.eval_finni.hf_revision")
    assert hashlib.sha256(prompt["template"].encode()).hexdigest() == prompt["sha256"]
    assert prompt["rows_verified"] == config.get("pins.eval_finni.rows_expected")
    runner_digest = hashlib.sha256((root / "serving/generate_predictions.py").read_bytes()).hexdigest()
    inputs = Inputs(prompt["template"], prompt["sha256"], dev_digest, weight_digest, runner_digest, len(rows))
    path = root / "data/kaggle/d3-supervised-pilot.ipynb"
    path.write_text(json.dumps(build_notebook(inputs)))
    print(json.dumps({"notebook": str(path), "prompt_sha256": inputs.prompt_digest,
                      "dev_sha256": dev_digest, "adapter_sha256": weight_digest, "runner_sha256": runner_digest,
                      "dev_instances": len(rows)}))


if __name__ == "__main__":
    main()
