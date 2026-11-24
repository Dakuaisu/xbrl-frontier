from scripts import d3_pilot_summary


def test_partial_timing_reports_cutoffs_and_no_accuracy():
    rows = [{"id": "a", "finish_reason": "length", "not_generated": False},
            {"id": "b", "not_generated": True}]
    result = d3_pilot_summary.summarize(rows, {"a", "b", "c"}, 5.0)
    assert result["status"] == "partial"
    assert result["rows"] == 2 and result["cut_off"] == 1 and result["not_generated"] == 1
    assert result["seconds"] == 5.0 and "accuracy" not in result
