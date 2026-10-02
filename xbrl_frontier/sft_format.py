"""The one training/inference format for students (Mac pilot, Kaggle training, scoring).

Prompt: the benchmark's verbatim FinNI query (template taken from the pinned dataset) as the single user
turn, rendered with the student's chat template, add_generation_prompt=True, enable_thinking=False. For
Qwen3 that ends in an empty think block, so inference and training see the same prefix.
Completion: the gold answer in the benchmark's own schema, json.dumps({"result": [{"Fact", "Type"}]}) with
default separators (byte-identical to FinNI-eval-v2 answers), followed by the template's turn end.
Loss: completion tokens only (everything after the rendered prompt).
"""
import json
from functools import lru_cache

from . import finni

_SENTINEL = "\x00CONTEXT\x00"


@lru_cache(maxsize=None)
def query_template() -> str:
    r = finni.load()[0]
    if r.query.count(r.context) != 1:
        raise ValueError("FinNI query does not contain its context exactly once")
    return r.query.replace(r.context, _SENTINEL)


def query(context: str) -> str:
    return query_template().replace(_SENTINEL, context)


def answer(numeric_entities: list[dict]) -> str:
    return json.dumps({"result": [{"Fact": e["value"], "Type": e["type"]} for e in numeric_entities]})


def prompt_text(tokenizer, context: str) -> str:
    return tokenizer.apply_chat_template([{"role": "user", "content": query(context)}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def full_text(tokenizer, context: str, numeric_entities: list[dict]) -> tuple[str, str]:
    """(prompt, full) where full = prompt + completion, checked against the template's own rendering."""
    p = prompt_text(tokenizer, context)
    a = answer(numeric_entities)
    full = tokenizer.apply_chat_template([{"role": "user", "content": query(context)},
                                          {"role": "assistant", "content": a}], tokenize=False)
    if not full.startswith(p) or a not in full[len(p):]:
        raise ValueError("chat template does not render prompt + answer as a prefix-consistent pair")
    return p, full


def encode(tokenizer, context: str, numeric_entities: list[dict]) -> tuple[list[int], int]:
    """(token ids of the full sequence, number of prompt tokens = loss offset)."""
    p, full = full_text(tokenizer, context, numeric_entities)
    p_ids = tokenizer.encode(p, add_special_tokens=False)
    f_ids = tokenizer.encode(full, add_special_tokens=False)
    if f_ids[:len(p_ids)] != p_ids:
        raise ValueError("prompt tokens are not a prefix of the full sequence")
    return f_ids, len(p_ids)
