"""The calibration predictors in eval.oracles, built from gold pair sets."""
from .scorer import TYPES, Parsed


def gold_copy(golds):
    return [Parsed(g, False, 0) for g in golds]


def empty(golds):
    return [Parsed(frozenset(), False, 0) for _ in golds]


def types_swapped(golds):
    """Every gold pair re-typed to the next type (cyclic) that is not itself gold for that value."""
    out = []
    for g in golds:
        pairs = set()
        for v, t in g:
            k = TYPES.index(t) if t in TYPES else -1
            for step in range(1, len(TYPES) + 1):
                cand = TYPES[(k + step) % len(TYPES)]
                if (v, cand) not in g:
                    pairs.add((v, cand))
                    break
        out.append(Parsed(frozenset(pairs), False, 0))
    return out


def half_dropped(golds):
    """Drops every second gold pair in a global deterministic order."""
    out, k = [], 0
    for g in golds:
        kept = set()
        for pair in sorted(g):
            if k % 2 == 0:
                kept.add(pair)
            k += 1
        out.append(Parsed(frozenset(kept), False, 0))
    return out
