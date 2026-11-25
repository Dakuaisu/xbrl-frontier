import base64
import hashlib
import json
from pathlib import Path

import pytest

from scripts import d4_build_notebook


@pytest.fixture
def notebook(monkeypatch):
    digest = hashlib.sha256(Path("serving/generate_predictions.py").read_bytes()).hexdigest()
    monkeypatch.setattr(d4_build_notebook, "validate_local_inputs", lambda: digest)
    return d4_build_notebook.build()


def test_notebook_embeds_new_helpers_and_has_early_clock(notebook):
    code = "".join(notebook["cells"][0]["source"])
    compile(code, "d4_gpu", "exec")
    assert code.index("XF_SESSION_START") < code.index("pip")
    assert "CUDA_VISIBLE_DEVICES" in code
    assert "PREVIOUS_RECEIPT_SHA256" in code
    assert "SESSION_START_UTC" in code and "session_time_up" in code
    for name in d4_build_notebook.MODULES:
        encoded = base64.b64encode(Path("scripts", name + ".py").read_bytes()).decode()
        assert encoded in code
    json.dumps(notebook)


def test_checked_in_notebook_matches_builder(notebook):
    saved = json.loads(Path("notebooks/d4_gpu.ipynb").read_text())
    assert saved == notebook
