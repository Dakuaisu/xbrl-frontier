"""Power check and sampler: synthetic inputs for the logic, real-file refusals for the CLIs."""
import json
import subprocess
import sys
from collections import Counter

import pytest

from frontier import sampling
from scripts import draw_frontier_sample, power_check
from xbrl_frontier import config


def _sh(*args):
    return subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True, cwd=config.REPO_ROOT)


def test_power_check_refuses_anything_but_a_finni_scorer_output(tmp_path):
    fake = tmp_path / "fake.json"
    fake.write_text(json.dumps({"set": "finni", "raw": {}, "dedup": {"per_instance": [[0, 1, 0, 0, False, False]]}}))
    out = _sh("scripts.power_check", "--score", str(fake))
    assert out.returncode != 0 and "not a scripts/score_predictions.py output" in out.stdout + out.stderr
    assert not power_check.OUT.exists()


def test_sampler_refuses_without_power_check(monkeypatch, tmp_path):
    monkeypatch.setattr(draw_frontier_sample, "POWER", tmp_path / "missing.json")
    with pytest.raises(SystemExit, match="no power check"):
        draw_frontier_sample.main()


def test_company_sd_and_half_width_expression():
    per = [[0, 8, 2, 0], [1, 2, 0, 0], [2, 0, 5, 5], [3, 1, 1, 1], [9, 50, 0, 0]]
    sd = sampling.company_f1_sd(per, {0: "A", 1: "A", 2: "B", 3: "C"})
    assert sd["companies"] == 3
    expr = config.get("frontier.power_check.expected_half_width")
    assert sampling.half_width(expr, 0.1, 25) == pytest.approx(1.96 * 0.1 / 5)


def _filings(n=60):
    f = {}
    for i in range(n):
        f[f"acc{i:03d}"] = {"cik": f"c{i // 2 if i < 20 else i}", "n": 50 + 7 * i % 300,
                            "table_share": (i * 37 % 100) / 100, "sector": "rare" if i == 59 else f"s{i % 4}"}
    return f


def test_draw_keeps_companies_whole_and_fills_cells():
    f = _filings()
    d = sampling.draw(f, 23, seed=1, max_redraws=100)
    by_company = Counter(f[a]["cik"] for a in d["accessions"])
    for cik in by_company:
        assert by_company[cik] == sum(1 for v in f.values() if v["cik"] == cik)
    assert sorted(d["cell_targets"].values()) == [2] * 4 + [3] * 5
    assert abs(d["filings"] - 23) <= 2 and d["redraws"] >= 0
    assert sampling.draw(f, 23, seed=1, max_redraws=100) == d


def test_sector_balance_redraw_counted_and_capped():
    f = {f"a{i:03d}": {"cik": f"c{i}", "n": 20 + i, "table_share": (i * 37 % 100) / 100,
                       "sector": "B" if i % 20 == 0 else "A"} for i in range(100)}
    redrawn, refused = [], 0
    for seed in range(40):
        d = sampling.draw(f, 23, seed=seed, max_redraws=1000)
        assert d["sector_counts_sample"].get("B", 0) >= 1
        redrawn.append(d["redraws"])
        try:
            sampling.draw(f, 23, seed=seed, max_redraws=0)
        except RuntimeError as e:
            assert "sector balance" in str(e)
            refused += 1
    assert any(r > 0 for r in redrawn) and refused == sum(r > 0 for r in redrawn)
    rare = {a: dict(v, sector="R" if a == "a000" else "A") for a, v in f.items()}
    assert sampling.draw(rare, 23, seed=0, max_redraws=0)["redraws"] == 0


def test_latency_ids_seeded_subset():
    ids = sampling.latency_ids(list(range(500)), 100, seed=7)
    assert ids == sampling.latency_ids(list(range(500))[::-1], 100, seed=7) and len(set(ids)) == 100
    with pytest.raises(ValueError):
        sampling.latency_ids([1, 2], 3, seed=7)
