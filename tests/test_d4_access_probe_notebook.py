import json
from pathlib import Path


def test_probe_notebook_is_cpu_and_temporary_storage_only():
    notebook = json.loads(Path("notebooks/d4_access_probe.ipynb").read_text())
    source = "".join(notebook["cells"][0]["source"])
    compile(source, "d4_access_probe", "exec")
    assert "TemporaryDirectory(dir='/tmp')" in source
    assert "UserSecretsClient().get_secret('HF_TOKEN')" in source
    assert "read_metadata" in source and "file_digest" in source
    assert "D4_ACCESS_OK" in source and "D4_ACCESS_FAILED" in source
    assert "/kaggle/working" not in source and "print(token" not in source
