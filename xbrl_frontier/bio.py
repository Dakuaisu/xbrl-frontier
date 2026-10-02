"""Word-level BIO labels over the benchmark's concept label space, and decoding back to FinNI pairs.

Words: the context with HTML tags removed (entities decoded), split into numbers (thousands separators and
decimals kept inside one word, so "58,166" is one word), alphabetic words, and single other symbols.
Alignment: the serializer's facts are in document order; each fact's value (commas removed, tokenized
the same way) is matched to the first word sequence at or after the previous fact's position whose words,
with commas removed, equal it; failing that, the first unlabelled match anywhere. Facts whose concept is
outside the label space, or whose value is not found, stay O and are counted.
Decoding: a B-X followed by I-X words is one span of concept X; its value is the span's words joined by
spaces with commas removed; its type is the taxonomy type of X (FinNI pair = (value, type)).
"""
import html
import json
import re
from functools import lru_cache

from . import config

_TAG = re.compile(r"<[^>]+>")
_WORD = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?|\.\d+|[A-Za-z]+(?:['’][A-Za-z]+)*|[^\sA-Za-z\d]")


def words(context: str) -> list[str]:
    return _WORD.findall(html.unescape(_TAG.sub(" ", context)))


def _norm(w: str) -> str:
    return w.replace(",", "")


@lru_cache(maxsize=None)
def label_space() -> dict[str, int]:
    import hashlib
    import urllib.request
    spec = config.get("training.encoder.label_space")
    path = config.REPO_ROOT / "data" / "fintagging" / "label2id.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(spec["url"], timeout=120) as r:
            path.write_bytes(r.read())
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != spec["sha256"]:
        raise RuntimeError("label2id.json does not match training.encoder.label_space.sha256")
    return json.loads(body)


def label_words(ws: list[str], entities: list[dict], space: dict[str, int] | None = None) -> tuple[list[str], dict]:
    space = label_space() if space is None else space
    labels = ["O"] * len(ws)
    norm = [_norm(w) for w in ws]
    stats = {"facts": 0, "aligned": 0, "not_found": 0, "outside_label_space": 0}
    cursor = 0
    for e in entities:
        stats["facts"] += 1
        concept = e["concept"]
        if f"B-{concept}" not in space:
            stats["outside_label_space"] += 1
            continue
        target = [_norm(w) for w in words(e["value"])]
        n = len(target)
        if not n:
            stats["not_found"] += 1
            continue

        def free_match(i):
            return norm[i:i + n] == target and all(l == "O" for l in labels[i:i + n])

        pos = next((i for i in range(cursor, len(ws) - n + 1) if free_match(i)), None)
        if pos is None:
            pos = next((i for i in range(0, len(ws) - n + 1) if free_match(i)), None)
        if pos is None:
            stats["not_found"] += 1
            continue
        labels[pos] = f"B-{concept}"
        for j in range(pos + 1, pos + n):
            labels[j] = f"I-{concept}"
        cursor = pos + n
        stats["aligned"] += 1
    return labels, stats


def decode(ws: list[str], labels: list[str]) -> list[tuple[str, str]]:
    """(value, concept) spans in order."""
    spans, i = [], 0
    while i < len(ws):
        lab = labels[i]
        if lab.startswith("B-"):
            concept = lab[2:]
            j = i + 1
            while j < len(ws) and labels[j] == f"I-{concept}":
                j += 1
            spans.append((_norm(" ".join(ws[i:j])), concept))
            i = j
        else:
            i += 1
    return spans


def pairs_from_spans(spans: list[tuple[str, str]], taxonomy: dict[str, str]) -> list[dict]:
    """FinNI items {"Fact", "Type"} for spans whose concept type is one of the five numeric types."""
    from ixbrl_facts.fintagging import ENTITY_TYPES
    out = []
    for value, concept in spans:
        t = taxonomy.get(concept.split(":")[-1])
        if t in ENTITY_TYPES:
            out.append({"Fact": value, "Type": t})
    return out


def chunks(subtoken_counts: list[int], window: int) -> list[tuple[int, int]]:
    """Consecutive word ranges whose subtokens fit window - 2 ([CLS], [SEP]); a single word longer than that
    is cut at the subtoken level by the caller (counted)."""
    cap = window - 2
    out, start, used = [], 0, 0
    for i, c in enumerate(subtoken_counts):
        if used and used + c > cap:
            out.append((start, i))
            start, used = i, 0
        used += c
    if start < len(subtoken_counts):
        out.append((start, len(subtoken_counts)))
    return out
