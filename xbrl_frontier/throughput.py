"""Turn load-test points into filings/hour: maximum throughput and goodput at serving.slo."""
from . import config
from .cost import TokenShape

REQUIRED = ("concurrency", "request_throughput", "p95_ttft_ms", "p95_e2el_ms", "completed", "num_prompts")


def meets_slo(point: dict) -> bool:
    slo = config.get("serving.slo")
    return point["p95_ttft_ms"] <= slo["ttft_p95_ms"] and point["p95_e2el_ms"] <= slo["e2e_p95_ms"]


def summarize(points: list[dict], shape: TokenShape) -> dict:
    """Points with failed requests are kept in the record but excluded from both maxima.

    goodput = the highest request throughput among complete points whose p95 TTFT and p95 E2E both meet
    serving.slo (0 if none does).
    """
    for p in points:
        missing = [k for k in REQUIRED if k not in p]
        if missing:
            raise KeyError(f"load-test point lacks {missing}")
    complete = [p for p in points if p["completed"] == p["num_prompts"]]
    max_rps = max((p["request_throughput"] for p in complete), default=0.0)
    good_rps = max((p["request_throughput"] for p in complete if meets_slo(p)), default=0.0)
    to_fph = 3600.0 / shape.instances_per_filing
    return {
        "max_throughput_rps": max_rps,
        "goodput_rps_at_slo": good_rps,
        "max_throughput_filings_per_hour": max_rps * to_fph,
        "goodput_filings_per_hour_at_slo": good_rps * to_fph,
        "points": [{**{k: p[k] for k in REQUIRED}, "meets_slo": meets_slo(p),
                    "complete": p["completed"] == p["num_prompts"]} for p in points],
        "incomplete_points": len(points) - len(complete),
    }
