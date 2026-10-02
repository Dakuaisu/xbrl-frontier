"""Bootstrap, decision rule, cost model and charts on synthetic inputs; refusals on non-result files."""
import json
import subprocess
import sys

import pytest

from scripts.score_predictions import SCORES, gold_set
from serving.load_test import KIND
from xbrl_frontier import bootstrap, config, eval_sets


def _sh(*args):
    return subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True, cwd=config.REPO_ROOT)


def test_decision_rule_three_outcomes():
    assert bootstrap.decide(0.95, (0.91, 0.99), 0.90) == "pass"
    assert bootstrap.decide(0.80, (0.70, 0.85), 0.90) == "fail"
    assert bootstrap.decide(0.92, (0.88, 0.96), 0.90) == "inconclusive"
    assert bootstrap.decide(0.88, (0.85, 0.91), 0.90) == "inconclusive"
    assert bootstrap.decide(0.01, (0.0, 0.015), 0.02, higher_is_better=False) == "pass"


def test_paired_bootstrap_is_seeded_and_company_level():
    a = {i: (5, 1, 1) for i in range(30)}
    b = {i: ((2, 3, 3) if i % 3 else (6, 0, 0)) for i in range(30)}
    comp = {i: f"c{i % 10}" for i in range(30)}
    r1, r2 = bootstrap.paired({"a": a, "b": b}, comp), bootstrap.paired({"a": a, "b": b}, comp)
    assert r1["samples"] == r2["samples"] and r1["companies"] == 10 and r1["unit"] == "company (CIK)"
    assert r1["point"]["a"] == pytest.approx(10 / 12)
    lo, hi = bootstrap.interval(r1["samples"]["b"])
    assert lo <= r1["point"]["b"] <= hi


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    d = tmp_path_factory.mktemp("p4")
    imap = eval_sets.load_instance_map()
    ciks = sorted({m["cik"] for m in imap.values()})[:3]
    ids = sorted(i for i, m in imap.items() if m["cik"] in ciks)
    (d / "ids.txt").write_text("".join(f"{i}\n" for i in ids))
    items, _ = gold_set("finni", None)
    want = set(ids)
    def score(tag, empty):
        p = d / f"{tag}.jsonl"
        p.write_text("".join(json.dumps({"id": it.id, "pairs": []} if empty else {"id": it.id, "text": it.answer}) + "\n"
                             for it in items if it.id in want))
        assert _sh("scripts.score_predictions", "--pred", str(p), "--set", "finni", "--ids", str(d / "ids.txt"), "--tag", tag).returncode == 0
        return SCORES / f"{tag}.json"
    files = {"student": score("_t4_student", False), "luna": score("_t4_luna", True), "gem": score("_t4_gem", False)}
    for arm in ("luna", "gem"):
        for kind in ("batch", "sync"):
            (d / f"{arm}_{kind}.jsonl").write_text("".join(json.dumps({"id": i, "run": 0, "input_tokens": 1000, "output_tokens": 150}) + "\n" for i in ids))
    pts = [{"regime": "closed_loop", "concurrency": c, "request_rate": None, "num_prompts": 200, "completed": 200,
            "request_throughput": rps, "p95_ttft_ms": t, "p95_e2el_ms": e}
           for c, rps, t, e in ((1, 1.0, 100, 900), (4, 3.0, 300, 1800), (16, 6.0, 900, 4000))]
    (d / "load.json").write_text(json.dumps({"kind": KIND, "card": "kaggle_t4", "variant": "fp16", "points": pts,
                                             "checkpoint_hashes": {}}))
    yield d, files
    for f in files.values():
        f.unlink(missing_ok=True)


@pytest.fixture
def clean_outputs():
    yield
    (config.REPO_ROOT / "data" / "phase4" / "cost_model.json").unlink(missing_ok=True)
    for p in (config.REPO_ROOT / "data" / "charts").glob("pareto_*.vl.json"):
        p.unlink()


def test_cost_model_end_to_end_and_comparator(sample, clean_outputs):
    d, files = sample
    args = ["scripts.cost_model", "--load", f"fp16={d / 'load.json'}", "--student-score", str(files["student"]),
            "--arm", f"gpt-6-luna={files['luna']},{d / 'luna_batch.jsonl'},{d / 'luna_sync.jsonl'}",
            "--arm", f"gemini-3.5-flash-lite={files['gem']},{d / 'gem_batch.jsonl'},{d / 'gem_sync.jsonl'}"]
    out = _sh(*args)
    assert out.returncode == 0, out.stderr
    cm = json.loads((config.REPO_ROOT / "data" / "phase4" / "cost_model.json").read_text())
    assert cm["comparator"]["arm"] == "gemini-3.5-flash-lite" and cm["comparator"]["rule_applied"] == "primary"
    g = cm["gate"][f"fp16|{config.get('gates.cost_gate_regime')}"]
    assert g["gated"] and g["outcome"] in ("pass", "fail") and g["cost_ratio"] > 0
    assert cm["student"]["bootstrap_unit"] == "company (CIK)"
    full = _sh("scripts.cost_model", "--load", f"fp16={d / 'load.json'}", "--student-score", str(files["luna"]).replace("_t4_luna", "_nope"),
               "--arm", "x=a,b,c")
    assert full.returncode != 0
    bad = d / "bad_load.json"
    bad.write_text(json.dumps({"card": "kaggle_t4", "points": [{}]}))
    out = _sh("scripts.cost_model", "--load", f"fp16={bad}", "--student-score", str(files["student"]),
              "--arm", f"gpt-6-luna={files['luna']},{d / 'luna_batch.jsonl'},{d / 'luna_sync.jsonl'}")
    assert out.returncode != 0 and "not a kaggle_t4 load-test output" in out.stdout + out.stderr
    pareto = _sh("scripts.charts", "pareto", "--cost", str(config.REPO_ROOT / "data" / "phase4" / "cost_model.json"))
    assert pareto.returncode == 0, pareto.stderr
    spec = json.loads((config.REPO_ROOT / "data" / "charts" / "pareto_offline.vl.json").read_text())
    assert spec["usermeta"]["missing"] == ["fp16"] and {v["name"] for v in spec["data"]["values"]} == {"gpt-6-luna", "gemini-3.5-flash-lite"}


def test_charts_refuse_non_result_files(tmp_path):
    f = tmp_path / "x.json"
    f.write_text(json.dumps({"rows": []}))
    for args in (("pareto", "--cost", str(f)), ("curve", "--curve-file", str(f)), ("latency", "--load", str(f))):
        out = _sh("scripts.charts", *args)
        assert out.returncode != 0 and "refusing" in out.stdout + out.stderr
