import json
import subprocess
import sys

from scripts.score_predictions import SCORES, gold_set
from training import schedule
from xbrl_frontier import config


def _sh(*args):
    return subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True, cwd=config.REPO_ROOT)


def test_rows_only_from_scorer_outputs_and_missing_stays_missing(tmp_path):
    n = 38788
    cps = []
    for step in range(1, schedule.steps_per_epoch(n) + 1):
        for h in schedule.curve_hits(step - 1, step, n):
            cps.append({**h, "global_step": step, "path": f"curve/prefix_{h['prefix']}"})
    run = tmp_path / "run.json"
    run.write_text(json.dumps({"model": "Qwen/Qwen3-1.7B", "seed": 1, "trained_instances": n,
                               "skipped_over_max_seq_len": 0, "curve_checkpoints": cps}))
    items, _ = gold_set("dev", None)
    pred = tmp_path / "p.jsonl"
    pred.write_text("".join(json.dumps({"id": it.id, "text": it.answer}) + "\n" for it in items))
    assert _sh("scripts.score_predictions", "--pred", str(pred), "--set", "dev", "--tag", "_test_curve").returncode == 0
    score = SCORES / "_test_curve.json"
    fake = tmp_path / "fake.json"
    fake.write_text(json.dumps({"raw": {"overall": {"finni_f1": 0.9}}, "set": "dev"}))
    try:
        bad = _sh("scripts.curve_table", "--run", str(run), "--point", f"2500={fake}", "--tag", "_test_curve_bad")
        assert bad.returncode != 0 and "not a scripts/score_predictions.py output" in bad.stdout + bad.stderr
        ok = _sh("scripts.curve_table", "--run", str(run), "--point", f"2500={score}", "--point", f"{n}={score}",
                 "--tag", "_test_curve_ok")
        assert ok.returncode == 0, ok.stderr
        out = json.loads((config.REPO_ROOT / "data" / "curve" / "_test_curve_ok.json").read_text())
        assert [r["prefix"] for r in out["rows"]] == [2500, n]
        assert out["rows"][1]["label"] == "38,788 (cap 40,000 not reached after decontamination, O-016)"
        assert out["rows"][0]["instances_seen"] == next(c["instances_seen"] for c in cps if c["prefix"] == 2500)
        assert [m["prefix"] for m in out["missing"]] == [5000, 10000, 20000]
        assert len(out["series"]["dedup"]) == 2
    finally:
        score.unlink(missing_ok=True)
        (config.REPO_ROOT / "data" / "curve" / "_test_curve_ok.json").unlink(missing_ok=True)
