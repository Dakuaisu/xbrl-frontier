import json
import glob
import logging
import sys
import traceback
from pathlib import Path
from types import ModuleType

import pytest


def test_probe_notebook_is_cpu_and_temporary_storage_only():
    notebook = json.loads(Path("notebooks/d4_access_probe.ipynb").read_text())
    source = "".join(notebook["cells"][0]["source"])
    compile(source, "d4_access_probe", "exec")
    assert "TemporaryDirectory(dir='/tmp')" in source
    assert "UserSecretsClient().get_secret('HF_TOKEN')" in source
    assert "read_metadata" in source and "file_digest" in source
    assert "D4_ACCESS_OK" in source and "D4_ACCESS_FAILED" in source
    assert "/kaggle/working" not in source and "print(token" not in source


@pytest.mark.parametrize("message,expected", [
    ("No user secrets exist for kernel id " + "7" * 200 + " and label HF_TOKEN",
     ("No user secrets exist for kernel id " + "7" * 200 + " and label HF_TOKEN")[:120]),
    ("token=synthetic-secret row=synthetic-row", "[redacted]"),
])
def test_secret_lookup_failure_is_sanitized(monkeypatch, tmp_path, capsys, message, expected):
    config = tmp_path / "thresholds.yaml"
    config.write_text("pins:\n  eval_finni: {}\n")
    monkeypatch.setattr(glob, "glob", lambda *args, **kwargs: [str(config)])
    monkeypatch.setattr("os.path.exists", lambda path: False)
    monkeypatch.setattr(logging, "disable", lambda level: None)
    monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    secrets = ModuleType("kaggle_secrets")

    class FakeSecrets:
        def get_secret(self, label):
            raise RuntimeError(message)

    secrets.UserSecretsClient = FakeSecrets
    monkeypatch.setitem(sys.modules, "kaggle_secrets", secrets)
    notebook = json.loads(Path("notebooks/d4_access_probe.ipynb").read_text())
    source = "".join(notebook["cells"][0]["source"])
    with pytest.raises(SystemExit) as caught:
        exec(compile(source, "d4_access_probe", "exec"), {})
    result = json.loads(capsys.readouterr().out)
    assert result == {"status": "D4_ACCESS_FAILED", "error_type": "RuntimeError", "detail": expected}
    assert caught.value.__suppress_context__
    assert "synthetic-secret" not in "".join(traceback.format_exception(caught.value))
    assert "synthetic-row" not in "".join(traceback.format_exception(caught.value))
