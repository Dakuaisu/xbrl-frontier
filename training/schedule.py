"""Pure scheduling rules for student training (no torch; unit-tested locally).

Curve points count DISTINCT instances: nested prefixes of the one fixed corpus order, consumed in epoch 1
(TRADEOFFS T-016). They are never optimizer examples summed across epochs, so a point larger than the
corpus is not "reached in epoch 2"; it is realised as "full corpus" at the corpus size.
"""
import math
import os
import time

from xbrl_frontier import config


def curve_targets(corpus_size: int) -> list[dict]:
    """Each configured curve point -> the prefix it is realised at, in increasing order."""
    out = []
    for p in config.get("training.curve_points"):
        if p < corpus_size:
            out.append({"curve_point": p, "prefix": p, "full_corpus": False})
        else:
            out.append({"curve_point": p, "prefix": corpus_size, "full_corpus": True})
    seen, uniq = set(), []
    for t in out:
        if t["prefix"] not in seen:
            seen.add(t["prefix"])
            uniq.append(t)
        else:
            uniq[-1].setdefault("also_covers", []).append(t["curve_point"])
    return uniq


def instances_seen(global_step: int, corpus_size: int) -> int:
    """Distinct instances consumed after global_step optimizer steps of epoch 1 (sequential order)."""
    return min(global_step * config.get("training.global_batch_instances"), corpus_size)


def steps_per_epoch(corpus_size: int) -> int:
    return math.ceil(corpus_size / config.get("training.global_batch_instances"))


def curve_hits(prev_step: int, step: int, corpus_size: int) -> list[dict]:
    """Curve targets first covered by the optimizer step that moved prev_step -> step (epoch 1 only).

    Saved at the first batch boundary at or past the prefix; the instances actually seen are recorded."""
    if prev_step >= steps_per_epoch(corpus_size):
        return []
    a, b = instances_seen(prev_step, corpus_size), instances_seen(step, corpus_size)
    return [dict(t, instances_seen=b) for t in curve_targets(corpus_size) if a < t["prefix"] <= b]


def due_for_resume_checkpoint(now_s: float, last_save_s: float) -> bool:
    return now_s - last_save_s >= config.get("training.kaggle.checkpoint_every_minutes") * 60


def session_hours(requested: float | None) -> float:
    """The planned session length: requested (sized to the remaining weekly quota, O-033) or session_hours_max."""
    k = config.get("training.kaggle")
    h = k["session_hours_max"] if requested is None else requested
    if not 0 < h <= k["session_hours_max"]:
        raise SystemExit(f"session hours {h} must be in (0, training.kaggle.session_hours_max={k['session_hours_max']}]")
    if h * 60 <= k["session_stop_margin_minutes"]:
        raise SystemExit(f"session hours {h} leave no time after the {k['session_stop_margin_minutes']}-minute stop margin")
    return h


def session_time_up(now_s: float, session_start_s: float, hours: float | None = None) -> bool:
    """True once the session should checkpoint and exit so Kaggle keeps its output (T-038)."""
    k = config.get("training.kaggle")
    return now_s - session_start_s >= session_hours(hours) * 3600 - k["session_stop_margin_minutes"] * 60


def session_start() -> float:
    """Kaggle's timeout runs from session start, not from this process: the notebook's first cell exports XF_SESSION_START
    (O-034); outside a notebook, now."""
    return float(os.environ.get("XF_SESSION_START", time.time()))
