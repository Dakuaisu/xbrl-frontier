from scripts.generation_stats import stats


def test_stats_split_by_context_type_and_count_cap_hits():
    rows = [{"id": 1, "finish_reason": "length", "completion_tokens": 88, "max_tokens": 88, "session": 1},
            {"id": 2, "finish_reason": "stop", "completion_tokens": 10, "max_tokens": 88, "session": 2},
            {"id": 3, "not_generated": True, "session": 2}]
    s = stats(rows, {1: "table", 2: "text", 3: "table"})
    assert s["all"]["rows"] == 3 and s["all"]["not_generated"] == 1 and s["all"]["cap_hits"] == 1
    assert s["table"]["finish_reason"] == {"length": 1} and s["text"]["finish_reason"] == {"stop": 1}
    assert s["all"]["sessions"] == {1: 1, 2: 2}
