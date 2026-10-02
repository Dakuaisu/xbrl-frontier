"""dei:* labels (fiscal year and period focus, registrant name) for a filing.

Forked from FilingQA commit 895bf6d7d1b4c94254c64b378d02c349d1e7399b, `api/parse/ixbrl.py::harvest_dei`
(TRADEOFFS T-009). Changes: walks the stdlib `dom` tree instead of an lxml tree (ix elements are matched
by their lowercased prefixed name, as the HTML parser sees them, rather than by namespace URI).
Semantics kept: dei facts are collected from the whole document, including `ix:hidden` inside
`display:none` blocks; the first value per concept wins; text is the element's full text, stripped.
"""
from . import dom

_FACT_TAGS = ("ix:nonnumeric", "ix:nonfraction")


def harvest_dei(root: dom.Tag) -> dict[str, str]:
    found: dict[str, str] = {}
    for fact in root.find_all(_FACT_TAGS):
        name = fact.get("name")
        if name and name.startswith("dei:"):
            found.setdefault(name, fact.text.strip())
    return found


def fiscal_year(dei: dict[str, str]) -> int | None:
    raw = dei.get("dei:DocumentFiscalYearFocus")
    return int(raw) if raw and raw.isdigit() else None
