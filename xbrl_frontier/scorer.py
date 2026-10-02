"""FinNI scorer: set-based (normalized value, type) pair P/R/F1 with exact match.

Parse failures count as empty predictions and are reported as their own rate.
"""
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field

from . import config

TYPES = ("integerItemType", "monetaryItemType", "perShareItemType", "sharesItemType", "percentItemType")

_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_CURRENCY = re.compile(r"\$|\bUSD\b|%|,")
_SPACE = re.compile(r"\s+")


def normalize_value(v: str) -> str:
    rule = config.get("eval.value_normalization")
    if rule != "fintagging_prompt_rules":
        raise ValueError(f"unknown eval.value_normalization {rule!r}")
    # Prompt rule (4): drop currency symbols, % and commas. Rule (3): no magnitude expansion, so word
    # numbers stay words and are only lowercased. Whitespace (incl. NBSP) is collapsed.
    v = _CURRENCY.sub("", v)
    return _SPACE.sub(" ", v).strip().lower()


def pair_set(items) -> frozenset:
    return frozenset((normalize_value(str(it["Fact"])), it["Type"].strip()) for it in items)


@dataclass(frozen=True)
class Parsed:
    pairs: frozenset
    parse_failed: bool
    malformed_items: int


def parse_prediction(text) -> Parsed:
    """Model output -> pair set. Only the answer JSON is read; nothing is retried or repaired."""
    if text is None:
        return Parsed(frozenset(), True, 0)
    s = _THINK.sub("", text).strip()
    m = _FENCE.search(s)
    if m:
        s = m.group(1).strip()
    obj = None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        i, j = s.find("{"), s.rfind("}")
        if 0 <= i < j:
            try:
                obj = json.loads(s[i:j + 1])
            except json.JSONDecodeError:
                obj = None
    if not isinstance(obj, dict) or not isinstance(obj.get("result"), list):
        return Parsed(frozenset(), True, 0)
    good, bad = [], 0
    for it in obj["result"]:
        if (isinstance(it, dict) and isinstance(it.get("Type"), str)
                and isinstance(it.get("Fact"), (str, int, float)) and not isinstance(it.get("Fact"), bool)):
            good.append(it)
        else:
            bad += 1
    return Parsed(pair_set(good), False, bad)


def gold_set(answer: str) -> frozenset:
    return pair_set(json.loads(answer)["result"])


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def instance_prf(gold: frozenset, pred: frozenset) -> tuple[float, float, float]:
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    tp = len(gold & pred)
    return prf(tp, len(pred - gold), len(gold - pred))


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    n: int = 0
    parse_failures: int = 0
    malformed_items: int = 0
    inst_p: float = 0.0
    inst_r: float = 0.0
    inst_f: float = 0.0

    def report(self) -> dict:
        p, r, f = prf(self.tp, self.fp, self.fn)
        n = self.n or 1
        return {
            "finni_precision": p, "finni_recall": r, "finni_f1": f,
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "instances": self.n,
            "parse_failure_rate": self.parse_failures / n, "parse_failures": self.parse_failures,
            "malformed_items": self.malformed_items,
            "per_instance_mean": {"precision": self.inst_p / n, "recall": self.inst_r / n, "f1": self.inst_f / n},
        }


@dataclass
class Score:
    overall: dict
    slices: dict = field(default_factory=dict)
    per_instance: list = field(default_factory=list)


def score(golds: list[frozenset], preds: list[Parsed], slice_keys: dict | None = None) -> Score:
    """golds[i], preds[i] align. slice_keys maps slice name -> list of per-instance keys.

    The entity_type slice is computed from the pairs themselves: FP by predicted type, FN by gold type.
    """
    if config.get("eval.aggregation") != "global_pairs":
        raise ValueError("only eval.aggregation=global_pairs is implemented")
    if len(golds) != len(preds):
        raise ValueError("golds and preds differ in length")
    overall = Counts()
    by_slice: dict = defaultdict(lambda: defaultdict(Counts))
    by_type: dict = defaultdict(Counts)
    per_instance = []
    for i, (g, p) in enumerate(zip(golds, preds)):
        tp, fp, fn = len(g & p.pairs), len(p.pairs - g), len(g - p.pairs)
        ip, ir, iff = instance_prf(g, p.pairs)
        per_instance.append({"i": i, "tp": tp, "fp": fp, "fn": fn, "f1": iff, "parse_failed": p.parse_failed})
        targets = [overall] + [by_slice[name][keys[i]] for name, keys in (slice_keys or {}).items()]
        for c in targets:
            c.tp += tp; c.fp += fp; c.fn += fn; c.n += 1
            c.parse_failures += p.parse_failed; c.malformed_items += p.malformed_items
            c.inst_p += ip; c.inst_r += ir; c.inst_f += iff
        for v, t in g & p.pairs:
            by_type[t].tp += 1
        for v, t in p.pairs - g:
            by_type[t].fp += 1
        for v, t in g - p.pairs:
            by_type[t].fn += 1
    slices = {name: {str(k): c.report() for k, c in sorted(d.items(), key=lambda kv: str(kv[0]))}
              for name, d in by_slice.items()}
    slices["entity_type"] = {t: {k: v for k, v in c.report().items()
                                 if k in ("finni_precision", "finni_recall", "finni_f1", "tp", "fp", "fn")}
                             for t, c in sorted(by_type.items())}
    return Score(overall.report(), slices, per_instance)
