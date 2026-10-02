"""Request builders and response parsers for the pinned frontier arms (pure; no network, no keys).

Prompt: the benchmark's verbatim FinNI query (xbrl_frontier.sft_format.query), zero-shot, one user message.
Thinking: frontier.thinking_per_arm (the minimum each API allows). No temperature or output cap is set, so
each arm runs at its API defaults; the settings sent are recorded with every response.
"""
import json

from xbrl_frontier import config

OPENAI = "https://api.openai.com/v1"
GEMINI = "https://generativelanguage.googleapis.com/v1beta"


def arm_spec(arm: str) -> dict:
    spec = config.get("frontier.thinking_per_arm").get(arm)
    if spec is None:
        raise KeyError(f"{arm} has no frontier.thinking_per_arm entry")
    return spec


def body(arm: str, prompt: str) -> dict:
    s = arm_spec(arm)
    if s["api"] == "openai_responses":
        return {"model": arm, "input": [{"role": "user", "content": prompt}],
                "reasoning": {"effort": s["reasoning_effort"]}, "store": False}
    if s["api"] == "gemini_generate_content":
        return {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"thinkingConfig": {"thinkingLevel": s["thinking_level"]}}}
    raise ValueError(s["api"])


def sync_endpoint(arm: str) -> str:
    api = arm_spec(arm)["api"]
    return f"{OPENAI}/responses" if api == "openai_responses" else f"{GEMINI}/models/{arm}:generateContent"


def batch_payload(arm: str, items: list[tuple[str, str]]) -> str | dict:
    """OpenAI: JSONL text for the Batch API (/v1/responses). Gemini: the batchGenerateContent body (inline)."""
    api = arm_spec(arm)["api"]
    if api == "openai_responses":
        return "".join(json.dumps({"custom_id": str(cid), "method": "POST", "url": "/v1/responses",
                                   "body": body(arm, p)}) + "\n" for cid, p in items)
    return {"batch": {"display_name": f"xbrl-frontier-{arm}",
                      "input_config": {"requests": {"requests": [
                          {"request": body(arm, p), "metadata": {"key": str(cid)}} for cid, p in items]}}}}


def parse(arm: str, resp: dict) -> dict:
    """{text, model_returned, input_tokens, output_tokens, reasoning_tokens, finish} from one response body."""
    api = arm_spec(arm)["api"]
    if api == "openai_responses":
        text = "".join(c.get("text", "") for o in resp.get("output", []) if o.get("type") == "message"
                       for c in o.get("content", []) if c.get("type") == "output_text")
        u = resp.get("usage") or {}
        return {"text": text, "model_returned": resp.get("model"), "input_tokens": u.get("input_tokens"),
                "output_tokens": u.get("output_tokens"),
                "reasoning_tokens": (u.get("output_tokens_details") or {}).get("reasoning_tokens"),
                "finish": resp.get("status")}
    cand = (resp.get("candidates") or [{}])[0]
    text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []) if not p.get("thought"))
    u = resp.get("usageMetadata") or {}
    out = (u.get("candidatesTokenCount") or 0) + (u.get("thoughtsTokenCount") or 0)
    return {"text": text, "model_returned": resp.get("modelVersion"), "input_tokens": u.get("promptTokenCount"),
            "output_tokens": out, "reasoning_tokens": u.get("thoughtsTokenCount"), "finish": cand.get("finishReason")}
