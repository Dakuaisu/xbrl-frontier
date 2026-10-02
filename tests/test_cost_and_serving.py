import pytest

from xbrl_frontier import config, cost, throughput
from serving.smoke_test import planned, server_flags


def test_planned_variants_cover_config():
    t4 = planned("kaggle_t4")
    assert len(t4) == len(config.get("serving.hardware.kaggle_t4.variants")) * len(config.get("pins.students"))
    rtx = planned("rtx3050_wsl2")
    assert [v for v, _, _ in rtx] == config.get("serving.hardware.rtx3050_wsl2.variants")
    assert {m for _, m, _ in rtx} <= {s["id"] for s in config.get("pins.students")}


def test_rtx_flags_from_config():
    f = server_flags("rtx3050_wsl2", "Qwen/Qwen3-0.6B", "bf16")
    vf = config.get("serving.hardware.rtx3050_wsl2.vllm_flags")
    assert "--enforce-eager" in f and f[f.index("--max-num-seqs") + 1] == str(vf["max_num_seqs"])
    assert f[f.index("--dtype") + 1] == "bfloat16"
    assert server_flags("kaggle_t4", "Qwen/Qwen3-1.7B", "fp16")[3] == "half"


def test_break_even_identities():
    shape = cost.planning_shape()
    arm = config.get("pins.frontier_arms")[0]
    rate = next(v for v in cost.t4_rates().values() if v is not None)
    for regime in ("offline", "online"):
        util = config.get(f"gates.cost_regimes.{regime}.utilization")
        usd = cost.arm_usd_per_filing(arm, regime, shape)
        be = cost.break_even_filings_per_hour(rate, usd, util)
        assert cost.self_host_usd_per_filing(rate, be, util) == pytest.approx(usd)
        assert cost.gate_pass_filings_per_hour(be) == pytest.approx(be * config.get("gates.cost_ratio_vs_comparator_min"))
    assert cost.arm_usd_per_filing(arm, "online", shape) > cost.arm_usd_per_filing(arm, "offline", shape)


def test_blank_t4_source_is_none_not_guessed():
    rates = cost.t4_rates()
    assert list(rates) == config.get("serving.hardware.kaggle_t4.market_rate_keys")
    assert all(v is None or v > 0 for v in rates.values())


def _pt(c, rps, ttft, e2e, done=200):
    return {"concurrency": c, "request_throughput": rps, "p95_ttft_ms": ttft, "p95_e2el_ms": e2e,
            "completed": done, "num_prompts": 200}


def test_goodput_is_best_point_meeting_slo():
    slo = config.get("serving.slo")
    shape = cost.planning_shape()
    pts = [_pt(1, 1.0, slo["ttft_p95_ms"] / 2, slo["e2e_p95_ms"] / 2),
           _pt(4, 3.0, slo["ttft_p95_ms"], slo["e2e_p95_ms"]),
           _pt(16, 6.0, slo["ttft_p95_ms"] * 3, slo["e2e_p95_ms"] * 3),
           _pt(32, 9.0, 1, 1, done=150)]
    s = throughput.summarize(pts, shape)
    assert s["max_throughput_rps"] == 6.0 and s["goodput_rps_at_slo"] == 3.0
    assert s["incomplete_points"] == 1
    assert s["max_throughput_filings_per_hour"] == pytest.approx(6.0 * 3600 / shape.instances_per_filing)


def test_summarize_rejects_missing_fields():
    with pytest.raises(KeyError):
        throughput.summarize([{"concurrency": 1}], cost.planning_shape())
