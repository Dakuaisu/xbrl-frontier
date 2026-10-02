"""Power check and frontier-sample draw (PRD section 7; T-007, T-026). Pure functions; the CLIs feed them.

Power check: company-level F1 (pooled tp/fp/fn per CIK over the primary student's dedup full-set
predictions), SD across companies, expected half-width by the expression in frontier.power_check, compared
with gates.headline_margin_f1; if exceeded, frontier.power_check.if_exceeded is applied as written (drop
pins.frontier_arms[2]; N = largest N whose priced plan leaves frontier.usd_reserve_min). Otherwise
N = min(frontier.sample_filings, largest N for all arms) (the v2.1 reserve rule, through the priced plan).

Sample: filings of the eval manifest with assigned instances (the unassigned stratum is excluded), strata
table-share tercile x instances-per-filing tercile over those filings, cell targets floor/ceil of N / 9
(extra slots to cells in seeded order). The sampling unit is the company: a company sits in the cell of its
largest filing and all its filings move together. Sector balance: a draw in which a sector whose
proportional share of the sample is at least one filing is empty is redrawn; redraws are counted and capped.
"""
import math
import random
import statistics
from collections import defaultdict


def company_f1_sd(per_instance: list, company_of: dict) -> dict:
    """per_instance rows [id, tp, fp, fn, ...] from a scorer output; company_of: id -> cik (assigned only)."""
    agg = defaultdict(lambda: [0, 0, 0])
    for row in per_instance:
        cik = company_of.get(row[0])
        if cik is None:
            continue
        a = agg[cik]
        a[0] += row[1]; a[1] += row[2]; a[2] += row[3]
    f1 = {c: (2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0) for c, (tp, fp, fn) in agg.items()}
    if len(f1) < 2:
        raise ValueError("fewer than two companies with assigned instances")
    return {"companies": len(f1), "sd": statistics.stdev(f1.values()), "company_f1": f1}


def half_width(expression: str, sd: float, n: int) -> float:
    """Evaluate frontier.power_check.expected_half_width (e.g. '1.96 * sd / sqrt(sample_filings)')."""
    return eval(expression, {"__builtins__": {}}, {"sd": sd, "sample_filings": n, "sqrt": math.sqrt})


def terciles(values: list[float]) -> list[float]:
    s = sorted(values)
    return [s[(k * len(s)) // 3] for k in (1, 2)]


def tercile_of(v: float, edges: list[float]) -> int:
    return 0 if v < edges[0] else (1 if v < edges[1] else 2)


def strata(filings: dict) -> dict:
    """filings: acc -> {"cik", "n", "table_share", "sector"}; returns acc -> (table tercile, size tercile)."""
    te = terciles([f["table_share"] for f in filings.values()])
    se = terciles([f["n"] for f in filings.values()])
    return {a: (tercile_of(f["table_share"], te), tercile_of(f["n"], se)) for a, f in filings.items()}


def draw(filings: dict, n_target: int, seed: int, max_redraws: int) -> dict:
    cell = strata(filings)
    by_company = defaultdict(list)
    for a, f in filings.items():
        by_company[f["cik"]].append(a)
    company_cell = {c: cell[max(accs, key=lambda a: (filings[a]["n"], a))] for c, accs in by_company.items()}
    cells = sorted({v for v in cell.values()} | {(i, j) for i in range(3) for j in range(3)})
    rng = random.Random(seed)
    base, extra = divmod(n_target, len(cells))
    order = cells[:]
    rng.shuffle(order)
    target = {c: base + (1 if k < extra else 0) for k, c in enumerate(order)}
    sector_share = defaultdict(int)
    for f in filings.values():
        sector_share[f["sector"]] += 1
    redraws = 0
    while True:
        chosen, shortfall = [], {}
        for c in cells:
            comps = sorted(k for k, v in company_cell.items() if v == c)
            rng.shuffle(comps)
            got = 0
            for comp in comps:
                if got >= target[c]:
                    break
                size = len(by_company[comp])
                if got + size > target[c] and any(len(by_company[o]) == 1 for o in comps if o not in chosen and o != comp):
                    continue
                chosen.append(comp)
                got += size
            if got < target[c]:
                shortfall["|".join(map(str, c))] = target[c] - got
        accs = sorted(a for comp in chosen for a in by_company[comp])
        drawn_sectors = {filings[a]["sector"] for a in accs}
        empty = [s for s, k in sector_share.items() if k / len(filings) * len(accs) >= 1 and s not in drawn_sectors]
        if not empty:
            break
        redraws += 1
        if redraws > max_redraws:
            raise RuntimeError(f"sector balance not met after {max_redraws} redraws (empty: {empty})")
    return {"accessions": accs, "companies": sorted(chosen), "filings": len(accs), "n_target": n_target,
            "cell_targets": {"|".join(map(str, c)): t for c, t in target.items()}, "cell_shortfall": shortfall,
            "cells": {a: "|".join(map(str, cell[a])) for a in accs}, "redraws": redraws,
            "sector_counts_sample": dict(sorted((s, sum(filings[a]["sector"] == s for a in accs)) for s in drawn_sectors)),
            "sector_counts_manifest": dict(sorted(sector_share.items()))}


def latency_ids(candidate_ids: list[int], k: int, seed: int) -> list[int]:
    if len(candidate_ids) < k:
        raise ValueError(f"only {len(candidate_ids)} candidate instances for {k} latency ids")
    return sorted(random.Random(seed).sample(sorted(candidate_ids), k))


def probe_filings(filings: dict, sample: set, arms: list[str], seed: int) -> dict:
    """One probe filing per arm from non-sample filings in the middle instances-per-filing tercile."""
    se = terciles([f["n"] for f in filings.values()])
    pool = sorted(a for a, f in filings.items() if a not in sample and tercile_of(f["n"], se) == 1)
    rng = random.Random(seed)
    return {arm: rng.choice(pool) for arm in arms}
