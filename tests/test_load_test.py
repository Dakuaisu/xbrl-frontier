import json

import pytest

from serving import load_test
from xbrl_frontier import config


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    from serving.common import pinned_snapshot
    return AutoTokenizer.from_pretrained(pinned_snapshot("Qwen/Qwen3-1.7B"))


def test_refuses_until_latency_ids_are_pinned(tmp_path, monkeypatch, tok):
    monkeypatch.setattr(config, "docs_path", lambda rel: tmp_path / "missing.txt")
    with pytest.raises(SystemExit, match="not pinned yet"):
        load_test.prompts_file(tok, tmp_path)


def test_prompts_are_rendered_once_per_pinned_id_with_the_output_cap(tmp_path, monkeypatch, tok):
    ids = tmp_path / "ids.txt"
    ids.write_text("0\n5\n")
    monkeypatch.setattr(config, "docs_path", lambda rel: ids)
    path, longest, n = load_test.prompts_file(tok, tmp_path)
    rows = [json.loads(l) for l in open(path)]
    assert n == 2 and len(rows) == 2 and longest > 0
    assert all(r["output_tokens"] == config.get("serving.load_test.max_output_tokens") for r in rows)
    assert rows[0]["prompt"].startswith("<|im_start|>user\n") and rows[0]["prompt"].endswith("<think>\n\n</think>\n\n")
