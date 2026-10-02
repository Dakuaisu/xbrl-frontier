import pytest

from ixbrl_facts import dei, dom, fintagging


def _nf(name, value):
    return f'<ix:nonFraction name="us-gaap:{name}" contextRef="c1" unitRef="usd">{value}</ix:nonFraction>'


def test_dom_names_text_and_navigation():
    root = dom.parse('<html><body><div id="a"><P>Net&#160;income <span>is</span> <ix:nonFraction name="x">1,234</ix:nonFraction>'
                     '</P><script>var x = 1;</script><!-- note --></div></body></html>')
    div = root.find("div")
    assert div.get_text(" ", strip=True) == "Net\xa0income is 1,234"
    assert [t.name for t in div.find_all(["p"], recursive=False)] == ["p"]
    nf = root.find("ix:nonfraction")
    assert nf.text == "1,234" and nf.find_parent(["div"]) is div and nf.find_parent(["td"]) is None


def test_dom_p_autoclose_and_stray_end_tag():
    root = dom.parse("<body><p>intro text<div>block</div></p></span><p>after</p></body>")
    body = root.find("body")
    assert [t.name for t in body.find_all(["p", "div"], recursive=False)] == ["p", "div", "p"]


def test_text_blocks_follow_notebook_rules():
    html = ("<body>"
            f"<div><p>Revenue for the year was {_nf('Revenues', '1,500')} million dollars.</p></div>"
            f"<p>Nested facts do not count here <span>{_nf('Revenues', '9')} inside a span element</span></p>"
            "<p>short</p>"
            "<p>A long negative paragraph without facts.</p><p>A long negative paragraph without facts.</p>"
            "</body>")
    out = fintagging.extract_text(dom.parse(html))
    pos = {x["text"]: x["numeric_entities"] for x in out["pos"]}
    assert pos == {"Revenue for the year was 1,500 million dollars.":
                   [{"value": "1500", "type": "monetaryItemType", "concept": "us-gaap:Revenues"}],
                   "9 inside a span element": [{"value": "9", "type": "monetaryItemType", "concept": "us-gaap:Revenues"}]}
    neg = [x["text"] for x in out["neg"]]
    assert neg.count("Revenue for the year was 1,500 million dollars.") == 1
    assert neg.count("A long negative paragraph without facts.") == 2
    assert "short" not in neg


def test_table_cells_repeat_nested_text_and_take_all_facts():
    html = ("<table><tr><td><div><span>Date Declared</span></div></td><td>"
            f"<p>{_nf('EarningsPerShareBasic', '1.25')}</p></td></tr>"
            "<tbody><tr><td>skipped row inside tbody</td></tr></tbody></table>")
    out = fintagging.extract_table(dom.parse(html))
    assert out["pos"][0]["text"] == "<table><tr><td>Date Declared Date Declared</td><td>1.25</td></tr></table>"
    assert out["pos"][0]["numeric_entities"] == [
        {"value": "1.25", "type": "perShareItemType", "concept": "us-gaap:EarningsPerShareBasic"}]


def test_unknown_concept_raises_like_the_notebook_or_is_recorded():
    html = f"<p>Something that is long enough {_nf('NotARealConceptAtAll', '5')}</p>"
    with pytest.raises(fintagging.UnknownConcept):
        fintagging.serialize(html)
    out = fintagging.serialize(html, unknown="record")
    assert out["unknown_concepts"] == ["NotARealConceptAtAll"] and not out["text"]["pos"]


def test_taxonomy_is_the_2023_2024_union():
    tax = fintagging.taxonomy()
    assert len(tax) == 17688
    assert tax["StockRepurchaseProgramAuthorizedAmount1"] == "monetaryItemType"


def test_dei_harvest_includes_hidden():
    root = dom.parse('<div style="display:none"><ix:hidden><ix:nonNumeric name="dei:DocumentFiscalYearFocus">2024'
                     '</ix:nonNumeric></ix:hidden></div>')
    d = dei.harvest_dei(root)
    assert dei.fiscal_year(d) == 2024
