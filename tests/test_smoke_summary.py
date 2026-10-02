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


def test_missing_checkpoint_is_not_tested_and_rerun_overrides(tmp_path):
    first = tmp_path / "a.json"
    first.write_text(json.dumps({"card": "kaggle_t4", "results": [
        {"variant": "0.6B-fp16", "ok": True},
        {"variant": "0.6B-awq-int4", "ok": False, "error": "FileNotFoundError: missing quantized checkpoint /x (run it)"},
        {"variant": "1.7B-awq-int4", "ok": False, "error": "FileNotFoundError: missing quantized checkpoint /y (run it)"}]}))
    c = summarize({"kaggle_t4": [first]})["cards"]["kaggle_t4"]
    assert c["removed"] == [] and {"0.6B-awq-int4", "1.7B-awq-int4"} <= set(c["not_tested"])
    assert set(c["not_tested_reasons"]) == {"0.6B-awq-int4", "1.7B-awq-int4"}
    rerun = tmp_path / "b.json"
    rerun.write_text(json.dumps({"card": "kaggle_t4", "results": [
        {"variant": "0.6B-awq-int4", "ok": True}, {"variant": "1.7B-awq-int4", "ok": False, "error": "ValueError: bad kernel"}]}))
    c = summarize({"kaggle_t4": [first, rerun]})["cards"]["kaggle_t4"]
    assert "0.6B-awq-int4" in c["kept"] and c["removed"] == [{"variant": "1.7B-awq-int4", "error": "ValueError: bad kernel"}]
    assert c["not_tested_reasons"] == {}
