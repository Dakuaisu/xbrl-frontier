import json
import subprocess
import sys

from scripts.score_predictions import SCORES, gold_set
from serving.export_quantized import calibration_texts_from
from xbrl_frontier import config


def test_calibration_sample_is_fixed_and_rendered_in_training_format():
    from transformers import AutoTokenizer
    from serving.common import pinned_snapshot
    tok = AutoTokenizer.from_pretrained(pinned_snapshot("Qwen/Qwen3-1.7B"))
    spec = config.get("serving.export_quantization.awq_calibration")
    path = config.REPO_ROOT / spec["source"]
    t1, m1 = calibration_texts_from(path, tok, spec)
    t2, m2 = calibration_texts_from(path, tok, spec)
    assert len(t1) == spec["samples"] and t1 == t2 and m1["texts_sha256"] == m2["texts_sha256"]
    assert all(t.startswith("<|im_start|>user\n") and t.rstrip().endswith("<|im_end|>") for t in t1)
    assert all(i.split(":")[0] for i in m1["instance_ids"])


def _flip(tmp_path, a_rows, b_rows):
    pa, pb = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    pa.write_text("".join(json.dumps(r) + "\n" for r in a_rows))
    pb.write_text("".join(json.dumps(r) + "\n" for r in b_rows))
    out = subprocess.run([sys.executable, "-m", "scripts.flip_rate", "--fp16", str(pa), "--quantized", str(pb),
                          "--set", "dev", "--tag", "_test"], capture_output=True, text=True, cwd=config.REPO_ROOT)
    assert out.returncode == 0, out.stderr
    rep = json.loads((SCORES / "flip__test.json").read_text())
    (SCORES / "flip__test.json").unlink()
    return rep


def test_flip_rate_identical_zero_and_one_change_counted(tmp_path):
    items, _ = gold_set("dev", None)
    gold = [{"id": it.id, "text": it.answer} for it in items]
    rep = _flip(tmp_path, gold, gold)
    assert rep["raw"]["overall"]["flip_rate"] == 0 and rep["raw"]["overall"]["f1_drop"] == 0
    k = next(i for i, it in enumerate(items) if json.loads(it.answer)["result"])
    n_facts = len({(f["Fact"], f["Type"]) for f in json.loads(items[k].answer)["result"]})
    changed = list(gold)
    changed[k] = {"id": items[k].id, "text": "not json"}
    o = _flip(tmp_path, gold, changed)["raw"]["overall"]
    assert o["flipped_facts"] == n_facts and o["correct_to_wrong"] == n_facts and o["wrong_to_correct"] == 0
    assert o["instance_flip_rate"] == 1 / len(items) and o["parse_failure_rate_quantized"] == 1 / len(items)
    assert o["f1_drop"] > 0
