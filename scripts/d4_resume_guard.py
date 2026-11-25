"""Validate D4 prediction continuity before any resumed generation or scoring."""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


class ResumeBlocked(Exception):
    """Prior predictions cannot be safely reused."""


@dataclass(frozen=True, slots=True)
class Identity:
    model: str
    model_revision: str
    dataset_revision: str
    dataset_sha256: str
    config_sha256: str
    adapter_sha256: str
    generation_sha256: str


@dataclass(frozen=True, slots=True)
class Receipt:
    identity: Identity
    predictions_sha256: str


@dataclass(frozen=True, slots=True)
class Target:
    identity: Identity
    expected_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ResumePlan:
    missing: tuple[int, ...]
    cutoffs: int
    complete: bool


def validate(path: Path, receipt: Receipt, target: Target) -> ResumePlan:
    if receipt.identity != target.identity or not all(vars_value for vars_value in (
        target.identity.model, target.identity.model_revision, target.identity.dataset_revision,
        target.identity.dataset_sha256, target.identity.config_sha256, target.identity.adapter_sha256,
        target.identity.generation_sha256,
    )):
        raise ResumeBlocked("generation identity mismatch or missing")
    expected = set(target.expected_ids)
    if len(expected) != len(target.expected_ids) or not expected:
        raise ResumeBlocked("expected IDs are not unique and nonempty")
    try:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != receipt.predictions_sha256:
            raise ResumeBlocked("prediction hash mismatch")
        rows = [json.loads(line) for line in raw.splitlines()]
        ids = [row["id"] for row in rows]
        seen = set(ids)
        if any(type(i) is not int for i in ids) or len(ids) != len(seen) or not seen <= expected:
            raise ResumeBlocked("prediction IDs are duplicate, invalid or outside the pinned set")
        if any("text" not in row for row in rows):
            raise ResumeBlocked("prediction row is malformed")
        missing = tuple(i for i in target.expected_ids if i not in seen)
        cutoffs = sum(row.get("finish_reason") == "length" for row in rows)
        return ResumePlan(missing, cutoffs, not missing)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ResumeBlocked("prior prediction file is unavailable or malformed") from exc
