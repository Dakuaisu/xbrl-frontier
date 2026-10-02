from xbrl_frontier import config
from xbrl_frontier.instance_match import ContextIndex, tokens_from_html
from scripts.build_eval_manifest import phrase_for, resolve

LONG_A = "Omega owns ninety one skilled nursing facilities leased to third party operators across forty two states"
LONG_B = "Revenue from rental income increased due to new acquisitions completed during the fourth quarter of the year"
BOILER = "The information required by this item is incorporated herein by reference to our proxy statement"


def _filing(*paras):
    return "<html><ix:header>hidden facts here</ix:header><body>" + "".join(f"<p>{p}</p>" for p in paras) + "</body></html>"


def test_tokens_strip_tags_entities_and_ix_header():
    assert tokens_from_html("<td>Net&#160;income $1,234</td><ix:header>x</ix:header>") == ["net", "income", "1", "234"]


def test_doubled_segments_collapse_symmetrically():
    assert tokens_from_html("<td>Date Declared Date Declared</td><td>1 1</td><td>a b a</td>") == \
        ["date", "declared", "1", "a", "b", "a"]
    ctx = "<table><tr><td>Average Price Paid per Share Average Price Paid per Share</td><td>12.50</td></tr></table>"
    fil = "<table><tr><td><span>Average Price Paid per Share</span></td><td>$12.50</td></tr></table>"
    assert tokens_from_html(ctx) == tokens_from_html(fil)


def test_unique_exact_presence_and_assignment():
    idx = ContextIndex({"a": f"<p>{LONG_A}</p>", "b": f"<table><tr><td>{LONG_B}</td></tr></table>",
                        "boiler": BOILER, "short": "2024 Total"})
    idx.scan("f1", _filing(LONG_A, BOILER))
    idx.scan("f2", _filing(LONG_B, BOILER))
    idx.scan("g1", _filing(BOILER))
    assert idx.unique_exact() == {"f1": 1, "f2": 1}
    company = {"f1": "c1", "f2": "c1", "g1": "c2"}
    m = idx.assign(company_of=company, dominant={"c1": "f1"})
    assert m["a"].filing == "f1" and m["a"].method == "exact"
    assert m["b"].filing == "f2"
    assert m["boiler"].filing is None and m["boiler"].method == "ambiguous_cross_company"
    assert m["short"].method == "too_short"
    m2 = idx.assign(allowed={"f1", "f2"}, company_of=company, dominant={"c1": "f2"})
    assert m2["boiler"].filing == "f2" and m2["boiler"].method == "exact_tie_within_company"


def test_fuzzy_threshold_from_config():
    thr = config.get("contamination.instance_match.fuzzy_min_containment")
    words = LONG_A.split()
    idx = ContextIndex({"a": LONG_A + " " + LONG_B})
    idx.scan("f1", _filing(LONG_A + " " + LONG_B.replace("year", "period")))
    m = idx.assign()["a"]
    assert m.method == "fuzzy" and m.containment >= thr and len(words) > 8


def test_phrase_and_resolution():
    n = config.get("edgar.fts_phrase_words")
    p = phrase_for(f"<p>{LONG_A}</p>")
    assert len(p.split()) == n
    assert phrase_for("<td>too short</td>") is None
    assert resolve([{"ciks": ["1"]}, {"ciks": ["1"]}]) == "1"
    assert resolve([{"ciks": ["1", "2"]}, {"ciks": ["1", "2"]}]) == "1"
    assert resolve([{"ciks": ["1"]}, {"ciks": ["2"]}]) is None
    assert resolve([]) is None
