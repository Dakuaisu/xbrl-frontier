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


def _tradeoffs(tmp_path, body):
    p = tmp_path / "TRADEOFFS.md"
    p.write_text("# Decisions log\n\n## T-900 owner pre-approval\n\n" + body + "\n\n## T-901 other\n\nPRE-APPROVED SPEND: step=batch arm=gpt-6-luna usd_max=9\n")
    return p


def test_guard_refusals_and_pass(tmp_path):
    t = _tradeoffs(tmp_path, "PRE-APPROVED SPEND: step=probe arm=gpt-6-luna usd_max=0.10")
    ledger = tmp_path / "ledger.jsonl"
    with pytest.raises(guard.GuardRefused, match="no --approval"):
        guard.check("probe", "gpt-6-luna", 0.05, None, t, ledger)
    with pytest.raises(guard.GuardRefused, match="has no"):
        guard.check("probe", "gemini-3.5-flash-lite", 0.05, "T-900", t, ledger)
    with pytest.raises(guard.GuardRefused, match="has no"):
        guard.check("batch", "gpt-6-luna", 0.05, "T-900", t, ledger)
    with pytest.raises(guard.GuardRefused, match="exceeds the approved"):
        guard.check("probe", "gpt-6-luna", 0.11, "T-900", t, ledger)
    limit = config.get("budget.usd_cap_frontier_api") - config.get("frontier.usd_reserve_min")
    ledger.write_text(json.dumps({"step": "x", "arm": "y", "usd": limit - 0.01}) + "\n")
    with pytest.raises(guard.GuardRefused, match="usd_reserve_min"):
        guard.check("probe", "gpt-6-luna", 0.05, "T-900", t, ledger)
    ledger.unlink()
    assert guard.check("probe", "gpt-6-luna", 0.05, "T-900", t, ledger)["usd_max"] == 0.10


def test_live_without_approval_refuses_before_any_key(tmp_path):
    ids = tmp_path / "ids.txt"
    ids.write_text("0\n1\n")
    env = {k: v for k, v in os.environ.items() if k not in KEYS}
    out = subprocess.run([sys.executable, "-m", "frontier.harness", "run", "--step", "latency", "--arm", "gpt-6-luna",
                          "--ids", str(ids), "--live"], capture_output=True, text=True, cwd=config.REPO_ROOT, env=env)
    assert out.returncode != 0
    assert "GuardRefused" in out.stderr and "no --approval" in out.stderr and "KeyError" not in out.stderr


def test_dry_run_prices_from_sheet(tmp_path):
    ids = tmp_path / "ids.txt"
    ids.write_text("0\n")
    items = harness.instances("latency", None, None, ids)
    est = harness.price("gpt-6-luna", "latency", items)
    p = config.prices()["api_per_million_tokens"]["gpt-6-luna"]
    runs = config.get("frontier.sync_runs")
    assert est["runs"] == runs and est["price_mode"] == "sync" and est["margin"] == 0.0
    assert est["usd_estimate"] == pytest.approx((est["est_input_tokens"] * p["input"] + est["est_output_tokens"] * p["output"]) / 1e6)
