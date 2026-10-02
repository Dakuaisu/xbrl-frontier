import json
import subprocess
import sys

from scripts.curve_validation import compare
from scripts.score_predictions import SCORES, gold_set
from xbrl_frontier import config


def _score(tmp_path, tag, rows):
    p = tmp_path / f"{tag}.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = subprocess.run([sys.executable, "-m", "scripts.score_predictions", "--pred", str(p), "--set", "dev", "--tag", tag],
                         capture_output=True, text=True, cwd=config.REPO_ROOT)
    assert out.returncode == 0, out.stderr
    return SCORES / f"{tag}.json"


def test_gap_and_caveat(tmp_path):
    items, _ = gold_set("dev", None)
    gold = [{"id": it.id, "text": it.answer} for it in items]
    worse = [{"id": it.id, "pairs": []} if k % 2 else {"id": it.id, "text": it.answer} for k, it in enumerate(items)]
    files = [_score(tmp_path, "_cv_a", gold), _score(tmp_path, "_cv_b", gold), _score(tmp_path, "_cv_c", worse)]
    try:
        same = compare(files[0], files[1], "dev")
        assert same["gap"] == 0 and not same["caveat"] and same["bootstrap_unit"] == "company (CIK)"
        diff = compare(files[0], files[2], "dev")
        assert diff["gap"] > 0 and diff["caveat"] and diff["gap_ci"][0] > 0
    finally:
        for f in files:
            f.unlink(missing_ok=True)
