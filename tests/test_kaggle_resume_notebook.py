import json
import base64
import hashlib
import subprocess
import sys
import types
from pathlib import Path

import pytest


def test_resume_preflight_checks_the_copied_checkpoint(tmp_path):
    notebook = json.loads(Path('notebooks/phase2_kaggle_train.ipynb').read_text())
    cell = next(cell for cell in notebook['cells'] if cell.get('id') == 'resume-preflight')
    namespace = {}
    exec(''.join(cell['source']), namespace)
    source = tmp_path / 'source'
    checkpoint = source / 'checkpoints' / 'checkpoint-2903'
    checkpoint.mkdir(parents=True)
    (source / 'run.json').write_text(json.dumps({'global_step': 2903}))
    (checkpoint / 'trainer_state.json').write_text(json.dumps({'global_step': 2903, 'max_steps': 4716}))
    for name in ('adapter_config.json', 'adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'scaler.pt', 'training_args.bin'):
        (checkpoint / name).write_text('present')
    namespace['verify_resume_copy'](source, tmp_path / 'output', 2903)
    (checkpoint / 'optimizer.pt').unlink()
    with pytest.raises((AssertionError, FileNotFoundError)):
        namespace['verify_resume_copy'](source, tmp_path / 'second-output', 2903)


@pytest.fixture
def run_training():
    notebook = json.loads(Path('notebooks/phase2_kaggle_train.ipynb').read_text())
    cell = next(cell for cell in notebook['cells'] if cell.get('id') == 'training-launch')
    template = b'fixture\x00CONTEXT\x00end'
    namespace = {'FINNI_TEMPLATE_B64': base64.b64encode(template).decode(),
                 'FINNI_TEMPLATE_SHA256': hashlib.sha256(template).hexdigest()}
    exec(''.join(cell['source']), namespace)
    return namespace['run_training']


def test_offline_trainer_command_uses_embedded_template_without_loading_dataset(tmp_path):
    notebook = json.loads(Path('notebooks/phase2_kaggle_train.ipynb').read_text())
    cell = next(cell for cell in notebook['cells'] if cell.get('id') == 'training-launch')
    namespace = {}
    exec(''.join(cell['source']), namespace)
    command = namespace['offline_trainer_command']()
    package = tmp_path / 'xbrl_frontier'
    package.mkdir()
    (package / '__init__.py').write_text('')
    (package / 'sft_format.py').write_text('def query_template():\n    raise RuntimeError("dataset accessed")\n')
    training = tmp_path / 'training'
    training.mkdir()
    (training / '__init__.py').write_text('')
    (training / 'kaggle_train.py').write_text('from xbrl_frontier.sft_format import query_template\nassert query_template() == "test\\x00CONTEXT\\x00end"\n')
    import os
    raw = b'test\x00CONTEXT\x00end'
    env = {**os.environ, 'PYTHONPATH': str(tmp_path), 'FINNI_TEMPLATE_B64': base64.b64encode(raw).decode(),
           'FINNI_TEMPLATE_SHA256': hashlib.sha256(raw).hexdigest()}
    subprocess.run(command, env=env, cwd=tmp_path, check=True)
    env['FINNI_TEMPLATE_SHA256'] = '0' * 64
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(command, env=env, cwd=tmp_path, check=True, capture_output=True)


def test_training_never_requests_kaggle_secret(run_training, tmp_path, monkeypatch, capsys):
    class Secrets:
        def get_secret(self, name):
            raise AssertionError('secret lookup must not occur')

    monkeypatch.setenv('KAGGLE_KERNEL_RUN_TYPE', 'Batch')
    monkeypatch.setitem(sys.modules, 'kaggle_secrets', types.SimpleNamespace(UserSecretsClient=Secrets))
    out = tmp_path / 'run'
    out.mkdir()
    monkeypatch.setenv('HF_TOKEN', 'test-secret-value')
    program = "import os,json,sys; assert 'HF_TOKEN' not in os.environ; json.dump({'global_step':2904,'sessions':[{'resumed_from':'checkpoint-2903'}]},open(sys.argv[1],'w'))"
    run_training([sys.executable, '-c', program, str(out / 'run.json')], out, 2903)
    assert 'test-secret-value' not in capsys.readouterr().out


def test_training_subprocess_failure_is_propagated(run_training, tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        run_training([sys.executable, '-c', 'raise SystemExit(7)'], tmp_path, 2903)


def test_training_without_progress_fails(run_training, tmp_path):
    (tmp_path / 'run.json').write_text(json.dumps({'global_step': 2903, 'sessions': [{'resumed_from': 'checkpoint-2903'}]}))
    with pytest.raises(AssertionError):
        run_training([sys.executable, '-c', 'pass'], tmp_path, 2903)
