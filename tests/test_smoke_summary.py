import json

import pytest

from scripts.smoke_summary import summarize


def test_failed_variants_removed_untested_listed(tmp_path):
    t4 = tmp_path / "t4.json"
    t4.write_text(json.dumps({"card": "kaggle_t4", "versions": {"vllm": "x"}, "results": [
        {"variant": "0.6B-fp16", "ok": True}, {"variant": "1.7B-awq-int4", "ok": False, "error": "RuntimeError: OOM"}]}))
    out = summarize({"kaggle_t4": t4, "rtx3050_wsl2": tmp_path / "missing.json"})
    c = out["cards"]["kaggle_t4"]
    assert c["kept"] == ["0.6B-fp16"] and c["removed"] == [{"variant": "1.7B-awq-int4", "error": "RuntimeError: OOM"}]
    assert "1.7B-fp16" in c["not_tested"]
    assert out["cards"]["rtx3050_wsl2"]["source"] is None and out["cards"]["rtx3050_wsl2"]["kept"] == []
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"card": "rtx3050_wsl2"}))
    with pytest.raises(SystemExit, match="not a serving/smoke_test.py output"):
        summarize({"kaggle_t4": bad})
