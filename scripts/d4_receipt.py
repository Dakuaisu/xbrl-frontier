"""Bind private D4 prediction files to an immutable run identity and version."""
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from scripts.d4_resume_guard import Identity, Receipt, ResumeBlocked, ResumePlan, Target, validate


@dataclass(frozen=True, slots=True)
class Evidence:
    kernel_ref: str
    receipt: Receipt


@dataclass(frozen=True, slots=True)
class Expected:
    target: Target
    kernel_ref: str


def make(predictions: Path, target: Target, kernel_ref: str) -> Evidence:
    if not re.fullmatch(r"[a-z0-9_-]+/[a-z0-9_-]+/[1-9][0-9]*", kernel_ref):
        raise ResumeBlocked("kernel reference must include an exact version")
    digest = hashlib.sha256(predictions.read_bytes()).hexdigest()
    record = Receipt(target.identity, digest)
    validate(predictions, record, target)
    return Evidence(kernel_ref, record)


def check(predictions: Path, evidence: Evidence, expected: Expected) -> ResumePlan:
    if evidence.kernel_ref != expected.kernel_ref:
        raise ResumeBlocked("kernel version mismatch")
    return validate(predictions, evidence.receipt, expected.target)


def encode(evidence: Evidence) -> str:
    return json.dumps(asdict(evidence), sort_keys=True)


def decode(raw: str) -> Evidence:
    try:
        data = json.loads(raw)
        record = data["receipt"]
        return Evidence(data["kernel_ref"], Receipt(Identity(**record["identity"]), record["predictions_sha256"]))
    except (ValueError, KeyError, TypeError) as exc:
        raise ResumeBlocked("receipt is malformed") from exc
