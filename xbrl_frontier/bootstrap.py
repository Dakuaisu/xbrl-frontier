"""Company-cluster bootstrap and the gate decision rule (PRD sections 3 and 8; T-007, T-028).

Resampling unit: the company (CIK). Each resample draws companies with replacement and pools tp/fp/fn over
their instances; F1 is computed per arm on the same resample, so differences and ratios are paired.
eval.bootstrap_resamples resamples, percentile interval at eval.ci_level, seed eval.bootstrap_seed.
"""
import random
from collections import defaultdict

from . import config


def _f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else 0.0


def clustered_counts(per_instance: dict, company_of: dict) -> dict:
    """{id: (tp, fp, fn)} -> {cik: [tp, fp, fn]} over instances with a company."""
    out = defaultdict(lambda: [0, 0, 0])
    for i, (tp, fp, fn) in per_instance.items():
        c = company_of.get(i)
        if c is not None:
            o = out[c]
            o[0] += tp; o[1] += fp; o[2] += fn
    return dict(out)


def paired(arms: dict, company_of: dict, companies: list | None = None) -> dict:
    """arms: name -> {id: (tp, fp, fn)} on the same instances. Returns point F1 per arm and resampled F1 lists."""
    clusters = {a: clustered_counts(v, company_of) for a, v in arms.items()}
    comps = sorted(companies or set.intersection(*(set(c) for c in clusters.values())))
    if len(comps) < 2:
        raise ValueError("fewer than two companies to resample")
    rng = random.Random(config.get("eval.bootstrap_seed"))
    n = config.get("eval.bootstrap_resamples")
    point = {a: _f1(*map(sum, zip(*(c[k] for k in comps)))) for a, c in clusters.items()}
    samples = {a: [] for a in arms}
    for _ in range(n):
        draw = [rng.choice(comps) for _ in comps]
        for a, c in clusters.items():
            samples[a].append(_f1(*map(sum, zip(*(c[k] for k in draw)))))
    return {"point": point, "samples": samples, "companies": len(comps), "resamples": n,
            "unit": "company (CIK)"}


def interval(xs: list[float]) -> tuple[float, float]:
    level = config.get("eval.ci_level")
    s = sorted(xs)
    lo = s[int((1 - level) / 2 * (len(s) - 1))]
    hi = s[int((1 + level) / 2 * (len(s) - 1) + 0.5)]
    return lo, hi


def decide(point: float, ci: tuple[float, float], threshold: float, higher_is_better: bool = True) -> str:
    """gates.decision_rule: pass if the point and the whole CI meet the threshold, fail if the point and the whole
    CI miss it, inconclusive if the CI straddles it."""
    meets = (lambda v: v >= threshold) if higher_is_better else (lambda v: v <= threshold)
    if meets(point) and meets(ci[0]) and meets(ci[1]):
        return "pass"
    if not meets(point) and not meets(ci[0]) and not meets(ci[1]):
        return "fail"
    return "inconclusive"
