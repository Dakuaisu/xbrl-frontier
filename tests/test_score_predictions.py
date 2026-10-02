"""The predictions-file scorer, calibrated with the eval.oracles predictors through real files."""
import json
import subprocess
import sys

import pytest

from scripts.score_predictions import SCORES, gold_set
from xbrl_frontier import config


def _run(tmp_path, set_name, rows, tag):
    p = tmp_path / f"{tag}.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = subprocess.run([sys.executable, "-m", "scripts.score_predictions", "--pred", str(p), "--set", set_name,
                          "--tag", tag], capture_output=True, text=True, cwd=config.REPO_ROOT)
    return out, (SCORES / f"{tag}.json")


@pytest.mark.parametrize("set_name", ["dev", "finni"])
def test_gold_copy_and_empty_oracles_through_files(tmp_path, set_name):
    tol = config.get("eval.oracles.tolerance")
    items, _ = gold_set(set_name, None)
    out, path = _run(tmp_path, set_name, [{"id": it.id, "text": it.answer} for it in items], f"_test_gold_{set_name}")
    assert out.returncode == 0, out.stderr
    rep = json.loads(path.read_text())
    for k in ("raw", "dedup"):
        o = rep[k]["overall"]
        assert abs(o["finni_f1"] - config.get("eval.oracles.gold_copy_f1")) <= tol
        assert o["parse_failure_rate"] == 0 and o["not_generated_rate"] == 0
        assert "context_type" in rep[k]["slices"] and "context_token_bucket" in rep[k]["slices"]
    out, path = _run(tmp_path, set_name, [{"id": it.id, "pairs": []} for it in items], f"_test_empty_{set_name}")
    assert abs(json.loads(path.read_text())["raw"]["overall"]["finni_f1"] - config.get("eval.oracles.empty_prediction_f1")) <= tol
    path.unlink()
    (SCORES / f"_test_gold_{set_name}.json").unlink()


def test_parse_failures_not_generated_and_missing(tmp_path):
    items, _ = gold_set("dev", None)
    rows = [{"id": it.id, "text": it.answer} for it in items]
    rows[0] = {"id": items[0].id, "text": "I cannot comply"}
    rows[1] = {"id": items[1].id, "text": None, "not_generated": True}
    out, path = _run(tmp_path, "dev", rows, "_test_mixed")
    o = json.loads(path.read_text())["raw"]["overall"]
    assert o["parse_failures"] == 1 and o["not_generated"] == 1
    path.unlink()
    out, _ = _run(tmp_path, "dev", rows[:-1], "_test_missing")
    assert out.returncode != 0 and "no prediction" in out.stdout + out.stderr
