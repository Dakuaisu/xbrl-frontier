"""Frontier harness: request shapes, parsing, and the spend guard. No network, no API keys read."""
import json
import os
import subprocess
import sys

import pytest

from frontier import guard, harness, requests as fr
from xbrl_frontier import config, sft_format

KEYS = ("OPENAI_API_KEY", "GEMINI_API_KEY")


@pytest.fixture(autouse=True)
def no_keys(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(harness, "_api_key", lambda arm: (_ for _ in ()).throw(AssertionError("key read in a test")))
    monkeypatch.setattr(harness, "_http", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network in a test")))


def test_bodies_use_verbatim_prompt_and_minimum_thinking():
    prompt = sft_format.query("<table><tr><td>Revenue</td><td>5</td></tr></table>")
    o = fr.body("gpt-6-luna", prompt)
    assert o["input"][0]["content"] == prompt and o["reasoning"]["effort"] == "none" and o["model"] == "gpt-6-luna"
    g = fr.body("gemini-3.5-flash-lite", prompt)
    assert g["contents"][0]["parts"][0]["text"] == prompt
    assert g["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "minimal"
    assert fr.sync_endpoint("gemini-3.5-flash-lite").endswith("/models/gemini-3.5-flash-lite:generateContent")


def test_batch_payload_shapes():
    lines = fr.batch_payload("gpt-6-luna", [(1, "p1"), (2, "p2")]).splitlines()
    first = json.loads(lines[0])
    assert first["custom_id"] == "1" and first["url"] == "/v1/responses" and first["method"] == "POST"
    g = fr.batch_payload("gemini-3.5-flash-lite", [(1, "p1")])
    req = g["batch"]["input_config"]["requests"]["requests"][0]
    assert req["metadata"]["key"] == "1" and req["request"]["contents"][0]["parts"][0]["text"] == "p1"


def test_parse_records_text_usage_and_returned_model():
    o = fr.parse("gpt-6-luna", {"model": "gpt-6-luna-2026-09-22", "status": "completed",
                               "output": [{"type": "reasoning"}, {"type": "message", "content": [{"type": "output_text", "text": "{\"result\": []}"}]}],
                               "usage": {"input_tokens": 900, "output_tokens": 12, "output_tokens_details": {"reasoning_tokens": 0}}})
    assert o == {"text": "{\"result\": []}", "model_returned": "gpt-6-luna-2026-09-22", "input_tokens": 900,
                 "output_tokens": 12, "reasoning_tokens": 0, "finish": "completed"}
    g = fr.parse("gemini-3.5-flash-lite", {"modelVersion": "gemini-3.5-flash-lite-001",
                                           "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "x", "thought": True}, {"text": "{}"}]}}],
                                           "usageMetadata": {"promptTokenCount": 800, "candidatesTokenCount": 10, "thoughtsTokenCount": 5}})
    assert g["text"] == "{}" and g["output_tokens"] == 15 and g["reasoning_tokens"] == 5 and g["model_returned"].startswith("gemini")


def test_dry_run_prices_from_sheet(tmp_path):
    ids = tmp_path / "ids.txt"
    ids.write_text("0\n")
    items = harness.instances("latency", None, None, ids)
    est = harness.price("gpt-6-luna", "latency", items)
    p = config.prices()["api_per_million_tokens"]["gpt-6-luna"]
    runs = config.get("frontier.sync_runs")
    assert est["runs"] == runs and est["price_mode"] == "sync" and est["margin"] == 0.0
    assert est["usd_estimate"] == pytest.approx((est["est_input_tokens"] * p["input"] + est["est_output_tokens"] * p["output"]) / 1e6)


def test_reprice_from_probe_scales_by_measured_ratios(tmp_path):
    from xbrl_frontier import finni
    from xbrl_frontier.tokens import tokenizer
    tok, rows = tokenizer(), finni.load()
    proxy_in = [len(tok.encode(sft_format.query(rows[i].context)).ids) for i in (0, 1)]
    proxy_out = [len(tok.encode(rows[i].answer).ids) for i in (0, 1)]
    probe = tmp_path / "probe"
    probe.mkdir()
    (probe / "usage.jsonl").write_text("".join(json.dumps(
        {"id": i, "run": 0, "input_tokens": 2 * proxy_in[k], "output_tokens": 3 * proxy_out[k]}) + "\n" for k, i in enumerate((0, 1))))
    est = {"est_input_tokens": 1000, "est_output_tokens": 100, "margin": 0.15, "usd_estimate": 0.0}
    assert harness.reprice_from_probe("gemini-3.5-flash-lite", est, tmp_path / "none") is None
    r = harness.reprice_from_probe("gemini-3.5-flash-lite", est, probe)
    p = config.prices()["api_per_million_tokens"]["gemini-3.5-flash-lite"]
    assert r["ratio_input_measured_over_proxy"] == pytest.approx(2) and r["ratio_output_measured_over_proxy"] == pytest.approx(3)
    assert r["usd_estimate"] == pytest.approx((2000 * p["batch_input"] + 300 * p["batch_output"]) / 1e6 * 1.15)


def test_t033_guard_allows_only_gemini_free_tier_sync(tmp_path):
    t = tmp_path / "TRADEOFFS.md"
    t.write_text("# Decisions\n\n## T-033 OWNER DECISION (Adnan): free tier\n")
    ok = guard.check("probe", "gemini-3.5-flash-lite", "T-033", "sync", t)
    assert ok["billed_usd"] == 0.0
    for arm in ("gpt-6-luna", "gpt-5.4-mini-2026-03-17", "gpt-5.4-nano-2026-03-17"):
        with pytest.raises(guard.GuardRefused, match="not in frontier.running_arms"):
            guard.check("probe", arm, "T-033", "sync", t)
    with pytest.raises(guard.GuardRefused, match="only --approval T-033"):
        guard.check("probe", "gemini-3.5-flash-lite", "T-900", "sync", t)
    with pytest.raises(guard.GuardRefused, match="only --approval T-033"):
        guard.check("probe", "gemini-3.5-flash-lite", None, "sync", t)
    with pytest.raises(guard.GuardRefused, match="Batch API could bill"):
        guard.check("batch", "gemini-3.5-flash-lite", "T-033", "batch_api", t)
    t.write_text("# Decisions\n")
    with pytest.raises(guard.GuardRefused, match="OWNER DECISION"):
        guard.check("probe", "gemini-3.5-flash-lite", "T-033", "sync", t)


def test_live_without_t033_refuses_before_any_key(tmp_path):
    ids = tmp_path / "ids.txt"
    ids.write_text("0\n")
    env = {k: v for k, v in os.environ.items() if k not in KEYS}
    env["XBRL_FRONTIER_PHASE3_DIR"] = str(tmp_path / "phase3")
    for arm, approval in (("gpt-6-luna", "T-033"), ("gemini-3.5-flash-lite", "T-024")):
        out = subprocess.run([sys.executable, "-m", "frontier.harness", "run", "--step", "latency", "--arm", arm,
                              "--ids", str(ids), "--live", "--approval", approval],
                             capture_output=True, text=True, cwd=config.REPO_ROOT, env=env)
        assert out.returncode != 0 and "GuardRefused" in out.stderr and "KeyError" not in out.stderr
    assert (tmp_path / "phase3" / "live").exists() and not (tmp_path / "phase3" / "live" / "latency_gpt-6-luna" / "usage.jsonl").exists()


def test_parse_429_and_pacer_learning(tmp_path):
    from frontier import freetier
    body = json.dumps({"error": {"code": 429, "message": "quota", "details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
            {"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
             "quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "quotaValue": "15"}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "7s"}]}}).encode()
    info = freetier.parse_429(body)
    assert info["per_minute"] and not info["per_day"] and info["quota_value"] == 15 and info["retry_s"] == 7.0
    clock = [0.0]
    slept = []
    p = freetier.Pacer(tmp_path / "state.json", clock=lambda: clock[0], sleep=lambda s: slept.append(s))
    assert p.rpm == config.get("frontier.free_tier.assumed_rpm")
    p.learn(info)
    assert p.rpm == 15 and json.loads((tmp_path / "state.json").read_text())["rpm"] == 15
    p.wait(); p.wait()
    assert slept == [pytest.approx(4.0)]


class _FakeHTTPError(Exception):
    pass


def _err(code, body=b"{}"):
    import io
    import urllib.error
    return urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(body))


def test_retries_never_score_failures_and_daily_quota_leaves_pending(tmp_path, monkeypatch):
    from frontier import freetier
    ok = json.dumps({"modelVersion": "gemini-3.5-flash-lite", "candidates": [{"content": {"parts": [{"text": "{\"result\": []}"}]},
                                                                             "finishReason": "STOP"}],
                     "usageMetadata": {"promptTokenCount": 900, "candidatesTokenCount": 5}}).encode()
    per_day = json.dumps({"error": {"code": 429, "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                          "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier", "quotaValue": "1000"}]}]}}).encode()
    script = [_err(429, b'{"error":{"details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"0s"}]}}'),
              _err(503), ok, ok, _err(429, per_day)]
    def fake_http(*a, **k):
        x = script.pop(0)
        if isinstance(x, Exception):
            raise x
        return x
    monkeypatch.setattr(harness, "_http", fake_http)
    monkeypatch.setattr(freetier.time, "sleep", lambda s: None)
    pacer = freetier.Pacer(tmp_path / "state.json", clock=lambda: 0.0, sleep=lambda s: None)
    items = [{"id": i, "context": "Revenue was 5 million.", "answer": "{}"} for i in range(3)]
    with pytest.raises(freetier.DailyQuotaExhausted):
        harness.run_free_tier("gemini-3.5-flash-lite", "probe", items, "k", tmp_path, pacer)
    preds = [json.loads(l) for l in open(tmp_path / "predictions.jsonl")]
    usage = [json.loads(l) for l in open(tmp_path / "usage.jsonl")]
    assert [p["id"] for p in preds] == [0, 1] and all(p["text"] == '{"result": []}' for p in preds)
    assert usage[0]["retries_429"] == 1 and usage[0]["retries_5xx"] == 1 and usage[0]["tier"] == "free"
    assert json.loads((tmp_path / "state.json").read_text())["rpd"] == 1000
    script.extend([ok])
    res = harness.run_free_tier("gemini-3.5-flash-lite", "probe", items, "k", tmp_path, pacer)
    assert res["new_calls"] == 1 and [json.loads(l)["id"] for l in open(tmp_path / "predictions.jsonl")] == [0, 1, 2]

