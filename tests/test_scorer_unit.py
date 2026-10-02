from xbrl_frontier import scorer, oracles
from xbrl_frontier.scorer import Parsed, parse_prediction, normalize_value, gold_set


def test_normalization_mirrors_prompt_rules():
    assert normalize_value("$1,234.5") == "1234.5"
    assert normalize_value("12.5%") == "12.5"
    assert normalize_value("USD 300") == "300"
    assert normalize_value("Two") == "two"
    assert normalize_value("0\xa0million") == "0 million"
    assert normalize_value(" — ") == "—"


def test_parse_valid_fenced_and_think():
    out = parse_prediction('<think>x</think>\n```json\n{"result":[{"Fact":"1,000","Type":"monetaryItemType"}]}\n```')
    assert not out.parse_failed and out.pairs == {("1000", "monetaryItemType")}


def test_parse_numeric_fact_and_trailing_prose():
    out = parse_prediction('Here: {"result":[{"Fact":58166,"Type":"monetaryItemType"}]} done')
    assert out.pairs == {("58166", "monetaryItemType")}


def test_parse_failure_is_empty_and_counted():
    for bad in [None, "", "no json", '{"result": "x"}', '[{"Fact":"1"}]', '{"result":[{"Fact":"1"']:
        out = parse_prediction(bad)
        assert out.parse_failed and out.pairs == frozenset()


def test_malformed_items_counted_not_scored():
    out = parse_prediction('{"result":[{"Fact":"1","Type":"monetaryItemType"},{"Fact":"2"},"x",{"Fact":true,"Type":"t"}]}')
    assert out.pairs == {("1", "monetaryItemType")} and out.malformed_items == 3 and not out.parse_failed


def test_set_semantics_and_global_pooling():
    g = [gold_set('{"result":[{"Fact":"1","Type":"monetaryItemType"},{"Fact":"1","Type":"monetaryItemType"}]}'),
         gold_set('{"result":[{"Fact":"2","Type":"percentItemType"},{"Fact":"3","Type":"percentItemType"}]}')]
    assert len(g[0]) == 1
    p = [Parsed(frozenset({("1", "monetaryItemType")}), False, 0),
         Parsed(frozenset({("2", "percentItemType"), ("9", "percentItemType")}), False, 0)]
    s = scorer.score(g, p)
    assert (s.overall["tp"], s.overall["fp"], s.overall["fn"]) == (2, 1, 1)
    assert abs(s.overall["finni_f1"] - 2 / 3) < 1e-12
    assert s.slices["entity_type"]["percentItemType"]["fn"] == 1


def test_negative_instance_per_instance_convention():
    s = scorer.score([frozenset()], [Parsed(frozenset(), False, 0)])
    assert s.overall["per_instance_mean"]["f1"] == 1.0 and s.overall["finni_f1"] == 0.0


def test_parse_failure_rate_reported_per_slice():
    g = [frozenset({("1", "monetaryItemType")})] * 2
    p = [parse_prediction("garbage"), Parsed(g[1], False, 0)]
    s = scorer.score(g, p, {"context_type": ["text", "table"]})
    assert s.overall["parse_failure_rate"] == 0.5
    assert s.slices["context_type"]["text"]["parse_failure_rate"] == 1.0


def test_types_swapped_never_hits_gold():
    g = [frozenset({("5", "monetaryItemType"), ("5", "perShareItemType"), ("7", "percentItemType")})]
    sw = oracles.types_swapped(g)[0].pairs
    assert not (sw & g[0]) and {v for v, _ in sw} == {"5", "7"}
