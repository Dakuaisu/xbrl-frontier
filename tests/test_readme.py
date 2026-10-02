import json

from scripts.readme import build
from xbrl_frontier import config


def test_readme_skeleton_blanks_keys_and_recorded_disclosures():
    text = build()
    pilot = json.loads((config.REPO_ROOT / "data" / "phase2_mac_pilot_Qwen3-0.6B.json").read_text())["format_check_on_val"]
    assert f"{pilot['finni_f1']:.3f}" not in text and f"{pilot['finni_f1']:.4f}" not in text
    assert "__% of the best frontier arm's FinNI F1" in text and text.count("__") > 20
    for key in ("finni_f1_fraction_of_best_frontier_min", "finni_f1_gain_over_zero_shot_min", "fresh_slice_f1_gap_max"):
        assert f"`{key}`" in text
    ser = json.loads((config.REPO_ROOT / "data" / "manifest_work" / "serializer_rebuild.json").read_text())
    assert f"{ser['assignment_rate_raw']:.2%} raw / {ser['assignment_rate_dedup']:.2%} dedup" in text
    assert f"{ser['phase0_token_matcher']['assignment_rate_raw']:.2%}" in text
    assert "(cap 40,000 not reached after decontamination, O-016)" in text and "0 flagged in the final corpus" in text
    assert str(config.get("gates.finni_f1_fraction_of_best_frontier_min")) not in text.split("## Gates")[1].split("## Results")[0]
