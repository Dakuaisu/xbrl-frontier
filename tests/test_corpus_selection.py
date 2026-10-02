from collections import Counter

from scripts.select_train_accessions import excluded_ciks, largest_remainder, office_key, size_key
from scripts.serialize_corpus import fintagging_negative_filter, instances
from xbrl_frontier import config


def test_office_and_size_keys_agree_across_sources():
    assert office_key("Office of Real Estate & Construction") == office_key("05 Real Estate & Construction")
    assert office_key("") == "unknown"
    assert size_key("Large accelerated filer") == "large_accelerated"
    assert size_key("Accelerated filer<br>Smaller reporting company") == "accelerated"
    assert size_key("Non-accelerated filer<br>Smaller reporting company") == "other"


def test_largest_remainder_sums_and_follows_weights():
    q = largest_remainder(Counter({"a": 125, "b": 7, "c": 7}), 104)
    assert sum(q.values()) == 104 and q["a"] > q["b"] >= 5


def test_every_exclusion_file_is_read():
    ex = excluded_ciks()
    assert list(ex) == config.get("contamination.exclude_ciks_from")
    assert all(ex.values()) and all(len(c) == 10 for s in ex.values() for c in s)


def test_negative_twin_dropped_only_when_same_text_is_labelled():
    ent = [{"value": "5", "type": "monetaryItemType", "concept": "us-gaap:Revenues"}]
    out = {"text": {"pos": [{"text": "T labelled", "numeric_entities": ent}],
                    "neg": [{"text": "T labelled", "numeric_entities": []}, {"text": "T plain", "numeric_entities": []}]},
           "table": {"pos": [], "neg": [{"text": "<table>x</table>", "numeric_entities": []}]}}
    rows, twins = instances(out)
    assert twins == 1
    assert [(r["text"], bool(r["numeric_entities"])) for r in rows] == [
        ("T labelled", True), ("T plain", False), ("<table>x</table>", False)]


def test_fintagging_negative_filters():
    assert fintagging_negative_filter("text", "x" * 101) and not fintagging_negative_filter("text", "x" * 100)
    assert fintagging_negative_filter("table", "1" * 50) and not fintagging_negative_filter("table", "1" * 49)
