import json
import sys
import types

import pytest

from serving import generate_predictions as gp


class _Out:
    def __init__(self, rid, text):
        self.request_id, self.finished = rid, True
        self.outputs = [types.SimpleNamespace(text=text, finish_reason="stop", token_ids=[1, 2])]


class _Engine:
    def __init__(self):
        self.pending, self.aborted, self.caps = [], [], {}

    def add_request(self, rid, prompt, params, lora_request=None):
        self.pending.append(rid)  # outputs report the external id ...
        self.caps[rid] = params["max_tokens"]
        return rid + "-9f3a"  # ... while add_request returns an internal one (vLLM 0.30, O-035)

    def step(self):  # finishes one request per step
        return [_Out(self.pending.pop(0), "[]")] if self.pending else []

    def has_unfinished_requests(self):
        return bool(self.pending)

    def abort_request(self, ids):
        ids = [i.rsplit("-", 1)[0] for i in ids]
        self.aborted += ids
        self.pending = [r for r in self.pending if r not in ids]


class _LLM:
    def __init__(self, **kw):
        self.llm_engine = _Engine()
        self.llm_engine.model_config = types.SimpleNamespace(max_model_len=100)
        _LLM.last = self


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
                                      "--window", "3", *extra])
    gp.main()
    return json.load(open(out / "predictions_dev.meta.json"))


def test_deadline_stop_then_resume_completes(stubbed, monkeypatch):
    ticks = iter([False, False, True])
    monkeypatch.setattr(gp.schedule, "session_time_up", lambda now, start, hours=None: next(ticks, True))
    m = _run(monkeypatch, stubbed / "s1")
    assert m["complete"] is False and m["stopped_for_session_limit"]
    first = [json.loads(l)["id"] for l in open(stubbed / "s1" / "predictions_base_dev.jsonl")]
    assert first == ["i3", "i0", "i1"]  # the over-long prompt, then two finished before the deadline
    assert _LLM.last.llm_engine.aborted == ["i2", "i4"]  # in flight at the deadline (window refilled): aborted, not written
    monkeypatch.setattr(gp.schedule, "session_time_up", lambda now, start, hours=None: False)
    meta1 = stubbed / "s1" / "predictions_dev.meta.json"
    legacy = json.loads(meta1.read_text())
    del legacy["sessions"]  # session 1 of D1 0.6B ran before per-session records existed
    meta1.write_text(json.dumps(legacy))
    m = _run(monkeypatch, stubbed / "s2", ["--resume", str(stubbed / "s1")])
    assert [(x["session"], x["code_path"]) for x in m["sessions"]] == [(1, "chunked"), (2, "continuous")]
    assert m["sessions"][1]["written_this_session"] == {"base": 4}
    rows = [json.loads(l) for l in open(stubbed / "s2" / "predictions_base_dev.jsonl")]
    assert m["complete"] and sorted(r["id"] for r in rows) == [f"i{k}" for k in range(7)]
    assert [r["id"] for r in rows if r["not_generated"]] == ["i3"]
    assert {r["id"]: r["session"] for r in rows} == {**{i: 1 for i in first}, **{f"i{k}": 2 for k in (2, 4, 5, 6)}}
    assert m["targets"]["base"]["resumed_with"] == 3
    assert all(r["max_tokens"] <= 100 - r["prompt_tokens"] for r in rows if not r["not_generated"])


def test_match_handles_internal_ids_and_fails_fast():
    flight = {"0": 1, "1": 1}
    assert gp._match("0", flight, {}) == "0"
    assert gp._match("1-ab12", flight, {}) == "1"
    assert gp._match("zz", flight, {"zz": "0"}) == "0"
    with pytest.raises(SystemExit, match="id mismatch"):
        gp._match("7-ab12", flight, {})
