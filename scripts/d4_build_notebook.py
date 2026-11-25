"""Build a D4 notebook embedding its new helpers and audited configuration."""
import base64
import hashlib
import json
from pathlib import Path

from scripts.d4_contract import GENERATOR_FILES, validate_local_inputs
from xbrl_frontier import config

MODULES = ("d4_resume_guard", "d4_receipt", "d4_security", "d4_contract", "d4_gpu_runtime")


def build() -> dict:
    runner_hash = validate_local_inputs()
    root = config.REPO_ROOT
    source = {name: base64.b64encode((root / "scripts" / (name + ".py")).read_bytes()).decode() for name in MODULES}
    thresholds = base64.b64encode(config.THRESHOLDS_PATH.read_bytes()).decode()
    prices = base64.b64encode(config.docs_path(config.get("pins.price_sheet")).read_bytes()).decode()
    source_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in GENERATOR_FILES}
    code = f'''import os, time
os.environ['XF_SESSION_START'] = str(time.time())
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
MODE = 'CPU_PROBE'
KERNEL_REF = ''
PREVIOUS_KERNEL_REF = ''
PREVIOUS_RECEIPT_SHA256 = ''
SESSION_START_UTC = ''
import base64, datetime, glob, hashlib, json, subprocess, sys, tempfile, types
from pathlib import Path
try:
    if MODE == 'GPU':
        start = datetime.datetime.fromisoformat(SESSION_START_UTC.replace('Z', '+00:00'))
        assert start.tzinfo is not None and start.timestamp() <= time.time()
        os.environ['XF_SESSION_START'] = str(start.timestamp())
    candidates = glob.glob('/kaggle/input/**/xbrl-frontier/serving/generate_predictions.py', recursive=True)
    assert len(candidates) == 1
    runner = Path(candidates[0])
    assert hashlib.sha256(runner.read_bytes()).hexdigest() == {runner_hash!r}
    root = runner.parent.parent
    for path, expected in {source_hashes!r}.items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == expected
    sys.path.insert(0, str(root))
    os.chdir(root)
    with tempfile.TemporaryDirectory(dir='/tmp') as settings_dir:
        settings = Path(settings_dir)
        (settings / 'thresholds.yaml').write_bytes(base64.b64decode({thresholds!r}))
        (settings / 'prices.yaml').write_bytes(base64.b64decode({prices!r}))
        from xbrl_frontier import config
        config.THRESHOLDS_PATH = settings / 'thresholds.yaml'
        old_docs_path = config.docs_path
        config.docs_path = lambda path: settings / 'prices.yaml' if path == config.get('pins.price_sheet') else old_docs_path(path)
        config.thresholds.cache_clear()
        config.prices.cache_clear()
        assert config.config_hash() == {config.config_hash()!r}
        if MODE == 'GPU':
            from training import schedule
            assert not schedule.session_time_up(time.time(), float(os.environ['XF_SESSION_START']), 8)
            pins = config.get('pins.software')
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'vllm==' + pins['vllm'], 'pyyaml==6.0.3', 'pyarrow==25.0.1'], check=True)
            subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', '-q', 'torchaudio'], check=True)
        elif MODE != 'CPU_PROBE':
            raise ValueError('invalid MODE')
        import scripts
        embedded = {source!r}
        for name, payload in embedded.items():
            full = 'scripts.' + name
            module = types.ModuleType(full)
            sys.modules[full] = module
            setattr(scripts, name, module)
            exec(compile(base64.b64decode(payload), '<embedded ' + name + '>', 'exec'), module.__dict__)
        from scripts.d4_gpu_runtime import Settings, cpu_probe, run
        if MODE == 'CPU_PROBE':
            cpu_probe({runner_hash!r})
        else:
            run(Settings(KERNEL_REF, {runner_hash!r}, PREVIOUS_KERNEL_REF, PREVIOUS_RECEIPT_SHA256))
except BaseException as exc:
    print(json.dumps({{'status': 'D4_STOPPED', 'error_type': type(exc).__name__}}))
    raise SystemExit(1) from None
'''
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "cells": [{"cell_type": "code", "id": "d4gpu", "metadata": {}, "execution_count": None,
                       "outputs": [], "source": code.splitlines(keepends=True)}]}


def main() -> None:
    path = Path("notebooks/d4_gpu.ipynb")
    notebook = build()
    path.write_text(json.dumps(notebook, indent=1))
    print(json.dumps({"notebook": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
