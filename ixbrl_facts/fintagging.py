"""FinTagging's own context serialization and labelling, ported from `parse_xbrl_report.ipynb`.

Source: https://github.com/The-FinAI/FinTagging at commit 653650cbbab525227d20228c94a65600c1209c01
(functions `extract_text` and `extract_table`). The control flow, filters and string building are kept
line for line, including the quirks that shape the benchmark's contexts:
- text blocks: every p/div/span outside a table cell, text = get_text(" ", strip=True), facts only from
  DIRECT-child leaf ix:nonFraction elements; positives dedup on text, negatives do not add to the seen set;
- tables: direct-child tr, direct-child th/td; a cell's text joins get_text of EVERY nested p/div/span, so
  nested elements repeat their text (the doubled cells seen in FinNI); facts from all nested
  ix:nonFraction (no leaf check);
- values: element text stripped, commas removed; only `us-gaap:` concepts; dedup on (value, concept);
  type looked up in the taxonomy and kept if one of the five numeric item types. The notebook's
  `us-gaap-2024_taxonomy.json` is not in the repo; the 2023+2024 xsd union reproduces the paper's 17,688
  concepts and covers eval filings that use 2023-only elements (pins.fintagging_source).
The notebook indexes the taxonomy directly, so an unknown concept raises KeyError and the notebook's
caller skips the whole filing. `unknown="raise"` reproduces that; `unknown="record"` lists the concept
and continues (used to rebuild contexts, where the type does not matter).
"""
import hashlib
import urllib.request
import xml.etree.ElementTree as ET
from functools import lru_cache

from xbrl_frontier import config

from . import dom

ENTITY_TYPES = ("integerItemType", "monetaryItemType", "perShareItemType", "percentItemType", "sharesItemType")
TAXONOMY_DIR = config.REPO_ROOT / "data" / "fintagging"
_XSD = "{http://www.w3.org/2001/XMLSchema}element"


def _xsd_bytes(url: str, sha256: str) -> bytes:
    path = TAXONOMY_DIR / url.rsplit("/", 1)[-1]
    if not path.exists():
        TAXONOMY_DIR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as r:
            path.write_bytes(r.read())
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != sha256:
        raise RuntimeError(f"{path} does not match its pinned sha256")
    return body


@lru_cache(maxsize=None)
def taxonomy() -> dict[str, str]:
    """Concept local name -> item type over the pinned us-gaap xsds (parse_taxonomy.ipynb rule: the type
    attribute's local part); later files override earlier ones."""
    out: dict[str, str] = {}
    for x in config.get("pins.fintagging_source.taxonomy_xsd"):
        for e in ET.fromstring(_xsd_bytes(x["url"], x["sha256"])).iter(_XSD):
            name, typ = e.get("name"), e.get("type")
            if name and typ:
                out[name] = typ.split(":")[-1]
    return out


class UnknownConcept(KeyError):
    pass


def _type_of(concept: str, unknown: str, missing: set):
    local = concept.split(":")[-1].strip()
    t = taxonomy().get(local)
    if t is None:
        if unknown == "raise":
            raise UnknownConcept(local)
        missing.add(local)
    return t


def extract_text(soup: dom.Tag, unknown: str = "raise", missing: set | None = None) -> dict:
    missing = set() if missing is None else missing
    positive_data, negative_data, seen_texts = [], [], set()
    for element in soup.find_all(["p", "div", "span"]):
        text = element.get_text(separator=" ", strip=True)
        if element.find_parent(["td", "th"]) is not None:
            continue
        numeric_entities, seen = [], {}
        for non_fraction in element.find_all("ix:nonfraction", recursive=False):
            if not non_fraction.find("ix:nonfraction"):
                concept = non_fraction.get("name", "None")
                value = non_fraction.text.strip().replace(",", "")
                if concept.startswith("us-gaap:") and value != "" and (value, concept) not in seen:
                    seen[(value, concept)] = True
                    entity_type = _type_of(concept, unknown, missing)
                    if entity_type in ENTITY_TYPES:
                        numeric_entities.append({"value": value, "type": entity_type, "concept": concept})
        if numeric_entities:
            if len(text) > 20 and text not in seen_texts:
                seen_texts.add(text)
                positive_data.append({"text": text, "numeric_entities": numeric_entities})
        elif len(text) > 20 and text not in seen_texts:
            negative_data.append({"text": text, "numeric_entities": []})
    return {"pos": positive_data, "neg": negative_data}


def extract_table(soup: dom.Tag, unknown: str = "raise", missing: set | None = None) -> dict:
    missing = set() if missing is None else missing
    positive_data, negative_data, seen_texts = [], [], set()
    for table in soup.find_all("table"):
        table_rows, numeric_entities, seen = [], [], {}
        for table_row in table.find_all("tr", recursive=False):
            extracted_cells = []
            for cell in table_row.find_all(["th", "td"], recursive=False):
                extracted_text_parts = []
                for sub_element in cell.find_all(["p", "div", "span"], recursive=True):
                    extracted_text_parts.append(sub_element.get_text(separator=" ", strip=True))
                    for non_fraction in sub_element.find_all("ix:nonfraction", recursive=True):
                        concept = non_fraction.get("name", "None")
                        value = non_fraction.text.strip().replace(",", "")
                        if concept.startswith("us-gaap:") and value and (value, concept) not in seen:
                            seen[(value, concept)] = True
                            entity_type = _type_of(concept, unknown, missing)
                            if entity_type in ENTITY_TYPES:
                                numeric_entities.append({"value": value, "type": entity_type, "concept": concept})
                extracted_text = (" ".join(extracted_text_parts) if extracted_text_parts
                                  else cell.get_text(separator=" ", strip=True))
                extracted_cells.append(f"<{cell.name}>{extracted_text}</{cell.name}>")
            table_rows.append(f"<tr>{''.join(extracted_cells)}</tr>")
        table_text = "".join(table_rows)
        if numeric_entities:
            if table_text and table_text not in seen_texts:
                seen_texts.add(table_text)
                positive_data.append({"text": f"<table>{table_text}</table>", "numeric_entities": numeric_entities})
        elif table_text and table_text not in seen_texts:
            negative_data.append({"text": f"<table>{table_text}</table>", "numeric_entities": []})
    return {"pos": positive_data, "neg": negative_data}


def serialize(html_text: str, unknown: str = "raise") -> dict:
    """Both extractors over one filing; `unknown_concepts` lists taxonomy misses in record mode."""
    soup = dom.parse(html_text)
    missing: set = set()
    out = {"text": extract_text(soup, unknown, missing), "table": extract_table(soup, unknown, missing)}
    out["unknown_concepts"] = sorted(missing)
    return out
