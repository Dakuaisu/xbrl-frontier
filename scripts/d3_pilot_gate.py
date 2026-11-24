"""Fail-closed D3 timing-pilot admission and dispatch state."""
import hashlib
import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path


class PilotBlocked(Exception):
    """Admission evidence is missing or invalid."""


@dataclass(frozen=True, slots=True)
class Provenance:
    revision: str
    dev_sha256: str
    config_hash: str


@dataclass(frozen=True, slots=True)
class Budget:
    planned_hours: float
    weekly_committed: float
    cumulative_committed: float
    weekly_cap: float
    cumulative_cap: float
    reservations_known: bool


def training_ready(version: int, status: str) -> bool:
    return version > 2 and status == "COMPLETE"


def budget_allows(budget: Budget) -> bool:
    values = (budget.planned_hours, budget.weekly_committed, budget.cumulative_committed,
              budget.weekly_cap, budget.cumulative_cap)
    return (all(math.isfinite(x) and x >= 0 for x in values)
            and budget.reservations_known and 0 < budget.planned_hours <= 2
            and budget.weekly_committed + budget.planned_hours <= budget.weekly_cap
            and budget.cumulative_committed + budget.planned_hours <= budget.cumulative_cap)


def verify_final(root: Path, expected: Provenance) -> str:
    try:
        run = json.loads((root / "run.json").read_text())
        if not (run["complete"] is True and run["global_step"] == 4716
                and run["final_adapter"] == "final_adapter" and run["model"] == "Qwen/Qwen3-1.7B"
                and run["revision"] == expected.revision and run["seed"] == 1
                and run["sessions"][-1]["resumed_from"] == "checkpoint-4180"
                and run["corpus_sha256"]["dev"] == expected.dev_sha256
                and run["config_hash"] == expected.config_hash):
            raise PilotBlocked("final run provenance or completion mismatch")
        adapter = root / "final_adapter"
        settings = json.loads((adapter / "adapter_config.json").read_text())
        base = settings["base_model_name_or_path"]
        snapshot_suffix = f"/models--Qwen--Qwen3-1.7B/snapshots/{expected.revision}"
        if base != "Qwen/Qwen3-1.7B" and not base.endswith(snapshot_suffix):
            raise PilotBlocked("adapter base model mismatch")
        weight = adapter / "adapter_model.safetensors"
        with weight.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            if not 0 < header_size < min(weight.stat().st_size - 8, 100_000_000):
                raise PilotBlocked("invalid safetensors header size")
            header = json.loads(stream.read(header_size))
            size = weight.stat().st_size - header_size - 8
            tensors = [value for name, value in header.items() if name != "__metadata__"]
            widths = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8}
            if not tensors or any(
                t["dtype"] not in widths or not all(type(n) is int and n > 0 for n in t["shape"])
                or not (0 <= t["data_offsets"][0] < t["data_offsets"][1] <= size)
                or math.prod(t["shape"]) * widths[t["dtype"]] != t["data_offsets"][1] - t["data_offsets"][0]
                for t in tensors
            ):
                raise PilotBlocked("invalid safetensors tensor offsets")
            offsets = sorted(t["data_offsets"] for t in tensors)
            if offsets[0][0] != 0 or offsets[-1][1] != size or any(
                left[1] != right[0] for left, right in zip(offsets, offsets[1:])
            ):
                raise PilotBlocked("safetensors tensor storage is not contiguous")
            stream.seek(0)
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except (OSError, KeyError, IndexError, TypeError, ValueError, struct.error) as exc:
        raise PilotBlocked("missing or malformed final artifact") from exc


def claim_once(path: Path, version: int, weight_sha256: str) -> bool:
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as stream:
        json.dump({"state": "push_outcome_unknown", "training_version": version,
                   "weights_sha256": weight_sha256}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return True
