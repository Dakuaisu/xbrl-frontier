import json
import sys
import types

import pytest

from serving import generate_predictions as gp


class _Out:
    def __init__(self, text):
        self.outputs = [types.SimpleNamespace(text=text, finish_reason="stop", token_ids=[1, 2])]


class _LLM:
    calls = 0

    def __init__(self, **kw):
        self.llm_engine = types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=100))

    def generate(self, prompts, params, lora_request=None):
        _LLM.calls += 1
        return [_Out("[]") for _ in prompts]


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    vllm = types.ModuleType("vllm"); vllm.LLM = _LLM; vllm.SamplingParams = lambda **kw: kw
    lora = types.ModuleType("vllm.lora.request"); lora.LoRARequest = lambda *a: a
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", types.ModuleType("vllm.lora"))
    monkeypatch.setitem(sys.modules, "vllm.lora.request", lora)
    tr = types.ModuleType("transformers")
    tr.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda p: types.SimpleNamespace(encode=lambda s, **k: [0] * len(s)))
    monkeypatch.setitem(sys.modules, "transformers", tr)
    import serving.common as sc
    monkeypatch.setattr(sc, "require_pins", lambda *a: {"vllm": "x"})
    monkeypatch.setattr(sc, "pinned_snapshot", lambda m: "/m")
    monkeypatch.setattr(sc, "model_pin", lambda m: "rev")
    monkeypatch.setattr(sc, "env_record", lambda c: {})
    items = [{"id": f"i{k}", "text": "x" * (5 if k != 3 else 200)} for k in range(7)]
    monkeypatch.setattr(gp, "load_set", lambda s, c: items)
    monkeypatch.setattr(gp.sft_format, "prompt_text", lambda tok, t: t)
    return tmp_path


def _run(monkeypatch, out, extra=()):
    monkeypatch.setattr(sys, "argv", ["gp", "--model", "Qwen/Qwen3-0.6B", "--set", "dev", "--base", "--out", str(out),
                                      "--chunk", "2", *extra])
    gp.main()
    return json.load(open(out / "predictions_dev.meta.json"))


def test_deadline_stop_then_resume_completes(stubbed, monkeypatch):
    ticks = iter([False, True])
    monkeypatch.setattr(gp.schedule, "session_time_up", lambda now, start, hours=None: next(ticks, True))
    m = _run(monkeypatch, stubbed / "s1")
    assert m["complete"] is False and m["stopped_for_session_limit"]
    first = [json.loads(l)["id"] for l in open(stubbed / "s1" / "predictions_base_dev.jsonl")]
    assert len(first) == 3 and "i3" in first  # one chunk of 2 plus the prompt over max_model_len
    monkeypatch.setattr(gp.schedule, "session_time_up", lambda now, start, hours=None: False)
    m = _run(monkeypatch, stubbed / "s2", ["--resume", str(stubbed / "s1")])
    rows = [json.loads(l) for l in open(stubbed / "s2" / "predictions_base_dev.jsonl")]
    assert m["complete"] and sorted(r["id"] for r in rows) == [f"i{k}" for k in range(7)]
    assert [r["id"] for r in rows if r["not_generated"]] == ["i3"]
    assert m["targets"]["base"]["resumed_with"] == 3
