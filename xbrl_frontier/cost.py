"""Cost model (PRD section 7): frontier $/filing, self-host $/filing, break-even and gate-pass goodput."""
from dataclasses import dataclass

from . import config

REGIME_PRICE_FIELDS = {"batch": ("batch_input", "batch_output"), "sync": ("input", "output")}


@dataclass(frozen=True)
class TokenShape:
    input_tokens_per_instance: float
    output_tokens_per_instance: float
    instances_per_filing: float
    source: str


def planning_shape() -> TokenShape:
    """FinNI token shape from config: per-instance lengths of the Phase 0 benchmark, instances per filing
    from the pinned row count over the benchmark's filing count. Replaced by probe measurements in Phase 3."""
    b = config.get("serving.phase0_throughput_benchmark")
    ipf = config.get("pins.eval_finni.rows_expected") / config.get("contamination.eval_manifest_required_recovered")
    return TokenShape(b["input_tokens"], b["output_tokens"], ipf,
                      "serving.phase0_throughput_benchmark.{input,output}_tokens; "
                      "pins.eval_finni.rows_expected / contamination.eval_manifest_required_recovered")


def arm_usd_per_filing(arm: str, regime: str, shape: TokenShape) -> float:
    price = config.prices()["api_per_million_tokens"][arm]
    fin, fout = REGIME_PRICE_FIELDS[config.get(f"gates.cost_regimes.{regime}.frontier_price")]
    per_instance = (shape.input_tokens_per_instance * price[fin] + shape.output_tokens_per_instance * price[fout]) / 1e6
    return per_instance * shape.instances_per_filing


def t4_rates() -> dict[str, float | None]:
    """Every key in serving.hardware.kaggle_t4.market_rate_keys -> $/hour (None if blank in the sheet)."""
    out = {}
    for key in config.get("serving.hardware.kaggle_t4.market_rate_keys"):
        node = config.prices()
        for part in key.split("."):
            node = node[part]
        out[key] = node
    return out


def break_even_filings_per_hour(t4_usd_per_hour: float, arm_usd: float, utilization: float) -> float:
    return t4_usd_per_hour / (arm_usd * utilization)


def gate_pass_filings_per_hour(break_even: float) -> float:
    return config.get("gates.cost_ratio_vs_comparator_min") * break_even


def self_host_usd_per_filing(t4_usd_per_hour: float, filings_per_hour: float, utilization: float) -> float:
    return t4_usd_per_hour / filings_per_hour / utilization
