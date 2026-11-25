import hashlib
import json
import shutil
import sys
import types

import pytest

from scripts import d4_gpu_runtime as runtime
from scripts.d4_resume_guard import Identity, ResumeBlocked, Target


def test_two_sessions_preserve_rows_and_generate_missing_only(tmp_path, monkeypatch, capsys):
    import glob
    import huggingface_hub
    import pyarrow.parquet
    import serving.common
    import serving.generate_predictions as generator
    from xbrl_frontier import finni

    weights = tmp_path / "adapter_model.safetensors"
    weights.write_bytes(b"adapter")
    dataset = tmp_path / "dataset"
    (dataset / "data").mkdir(parents=True)
    shard = dataset / "data/test-00000-of-00001.parquet"
    shard.write_bytes(b"fixture")
    identity = Identity("model", "revision", "dataset", hashlib.sha256(shard.read_bytes()).hexdigest(),
                        "config", hashlib.sha256(weights.read_bytes()).hexdigest(), "generation")
    target = Target(identity, (0, 1, 2))
    monkeypatch.setattr(runtime.d4_contract, "target", lambda _: target)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(device_count=lambda: 1,
                                                                                             get_device_name=lambda _: "Tesla T4")))
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=lambda *args, **kwargs: None))
    monkeypatch.setitem(sys.modules, "kaggle_secrets", types.SimpleNamespace(UserSecretsClient=lambda: types.SimpleNamespace(get_secret=lambda _: "fake-secret")))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **_: str(dataset))
    monkeypatch.setattr(pyarrow.parquet, "read_metadata", lambda _: types.SimpleNamespace(num_rows=3))
    monkeypatch.setattr(finni, "download", finni.download)
    monkeypatch.setattr(serving.common, "pinned_snapshot", serving.common.pinned_snapshot)
    monkeypatch.setenv("XF_SESSION_START", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    monkeypatch.setattr(runtime.logging, "disable", lambda _: None)
    monkeypatch.setattr(sys, "argv", sys.argv.copy())
    prior = tmp_path / "session1" / "d4"
    prior.parent.mkdir()
    monkeypatch.setattr(glob, "glob", lambda pattern, recursive: [str(weights)] if "final_adapter" in pattern else [str(prior / "receipt.json")])

    def generate():
        out = runtime.OUTPUT
        pred = out / "predictions_final_finni.jsonl"
        if "--resume" in sys.argv:
            source = sys.argv[sys.argv.index("--resume") + 1]
            shutil.copyfile(source + "/predictions_final_finni.jsonl", pred)
            new = (1, 2)
        else:
            new = (0,)
        with pred.open("a") as stream:
            for i in new:
                stream.write(json.dumps({"id": i, "text": "{}", "finish_reason": "length" if i == 0 else "stop"}) + "\n")
        written = len(pred.read_text().splitlines())
        (out / "predictions_finni.meta.json").write_text(json.dumps({"model": "model", "revision": "revision", "set": "finni",
            "config_hash": "config", "complete": written == 3,
            "generation": runtime.config.get("eval.generation"), "effective_max_model_len": runtime.config.get("eval.generation.max_model_len"),
            "targets": {"final": {"written": written, "complete": written == 3}}}))

    monkeypatch.setattr(generator, "main", generate)
    monkeypatch.setattr(runtime, "OUTPUT", prior)
    runtime.run(runtime.Settings("owner/session1/1", "runner"))
    first_hash = hashlib.sha256((prior / "receipt.json").read_bytes()).hexdigest()
    second = tmp_path / "session2" / "d4"
    second.parent.mkdir()
    monkeypatch.setattr(runtime, "OUTPUT", second)
    runtime.run(runtime.Settings("owner/session2/1", "runner", "owner/session1/1", first_hash))
    rows = [json.loads(line) for line in (second / "predictions_final_finni.jsonl").read_text().splitlines()]
    assert [row["id"] for row in rows] == [0, 1, 2]
    assert rows[0]["finish_reason"] == "length"
    assert "fake-secret" not in capsys.readouterr().out


def test_secret_lookup_exception_is_redacted_before_output(monkeypatch, capsys):
    class BrokenSecrets:
        def get_secret(self, name):
            raise RuntimeError("hf_fake_secret_should_never_appear")

    monkeypatch.setitem(sys.modules, "kaggle_secrets", types.SimpleNamespace(UserSecretsClient=BrokenSecrets))
    monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    monkeypatch.setattr(runtime.logging, "disable", lambda _: None)
    with pytest.raises(SystemExit):
        with runtime.dataset_snapshot(None):
            pytest.fail("lookup must fail")
    output = capsys.readouterr().out
    assert "secret_lookup_failed" in output and "hf_fake_secret_should_never_appear" not in output


def test_context_mismatch_is_rejected_before_generation():
    llm = types.SimpleNamespace(llm_engine=types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=4096)))
    with pytest.raises(ResumeBlocked):
        runtime.require_context(llm, 32768)
