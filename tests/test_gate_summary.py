import json
import subprocess
import sys

from scripts.gate_summary import OUT
from scripts.score_predictions import SCORES, gold_set
from xbrl_frontier import config


def _sh(*a):
    return subprocess.run([sys.executable, "-m", *a], capture_output=True, text=True, cwd=config.REPO_ROOT)


def test_gain_gate_slot_and_full_set_check(tmp_path):
    items, _ = gold_set("finni", None)
    files = {}
    for tag, rows in (("_g_stu", [{"id": it.id, "text": it.answer} for it in items]),
                      ("_g_zs", [{"id": it.id, "pairs": []} for it in items])):
        p = tmp_path / f"{tag}.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        assert _sh("scripts.score_predictions", "--pred", str(p), "--set", "finni", "--tag", tag).returncode == 0
        files[tag] = SCORES / f"{tag}.json"
    ids = tmp_path / "ids.txt"
    ids.write_text("0\n1\n2\n")
    p = tmp_path / "sub.jsonl"
    p.write_text("".join(json.dumps({"id": i, "pairs": []}) + "\n" for i in (0, 1, 2)))
    assert _sh("scripts.score_predictions", "--pred", str(p), "--set", "finni", "--ids", str(ids), "--tag", "_g_sub").returncode == 0
    try:
        out = _sh("scripts.gate_summary", "--student-full", str(files["_g_stu"]), "--zero-shot-full", str(files["_g_zs"]))
        assert out.returncode == 0, out.stderr
        g = {r["gate"]: r for r in json.loads(OUT.read_text())["gates"]}
        gain = g["finni_f1_gain_over_zero_shot_min"]
        assert gain["status"] == "pass" and gain["point"] == 1.0 and gain["threshold_key"] == "gates.finni_f1_gain_over_zero_shot_min"
        assert g["cost_ratio_vs_comparator_min"]["status"] == "not measured"
        bad = _sh("scripts.gate_summary", "--student-full", str(SCORES / "_g_sub.json"), "--zero-shot-full", str(files["_g_zs"]))
        assert bad.returncode != 0 and "full set is required" in bad.stdout + bad.stderr
    finally:
        for t in ("_g_stu", "_g_zs", "_g_sub"):
            (SCORES / f"{t}.json").unlink(missing_ok=True)
        OUT.unlink(missing_ok=True)
