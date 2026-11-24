"""Provider-independent, fail-closed D3 dispatch state machine."""
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Protocol

from scripts import d3_pilot_gate as gate


@dataclass(frozen=True, slots=True)
class TrainingState:
    version: int
    status: str
    output_dir: Path


@dataclass(frozen=True, slots=True)
class PilotOutput:
    rows: int
    complete: bool
    seconds: float


@dataclass(frozen=True, slots=True)
class Options:
    attempt_path: Path
    provenance: gate.Provenance


@dataclass(frozen=True, slots=True)
class Result:
    state: str
    rows: int = 0
    seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class Polling:
    sleep: Callable[[float], None]
    report: Callable[[Result], None]
    interval: float
    max_polls: int


class Provider(Protocol):
    def training(self) -> TrainingState: ...
    def budget(self) -> gate.Budget: ...
    def submit(self, version: int, digest: str, hours: float) -> None: ...
    def pilot(self) -> str: ...
    def collect(self) -> PilotOutput: ...


def step(provider: Provider, options: Options) -> Result:
    if options.attempt_path.exists():
        state = provider.pilot()
        if state in {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"}:
            output = provider.collect()
            return Result("complete_timing_only" if output.complete else "partial_timing_only",
                          output.rows, output.seconds)
        return Result("pilot_outcome_unknown")
    if provider.pilot() != "ABSENT":
        return Result("pilot_outcome_unknown")
    training = provider.training()
    if not gate.training_ready(training.version, training.status):
        return Result("waiting_for_new_completed_training")
    try:
        digest = gate.verify_final(training.output_dir, options.provenance)
    except gate.PilotBlocked:
        return Result("final_artifact_blocked")
    budget = provider.budget()
    if not gate.budget_allows(budget):
        return Result("budget_blocked")
    if not gate.claim_once(options.attempt_path, training.version, digest):
        return Result("pilot_outcome_unknown")
    provider.submit(training.version, digest, budget.planned_hours)
    return Result("pilot_outcome_unknown")


def watch(provider: Provider, options: Options, polling: Polling) -> None:
    if polling.interval <= 0 or polling.max_polls <= 0:
        raise gate.PilotBlocked("invalid polling bounds")
    for index in range(polling.max_polls):
        result = step(provider, options)
        polling.report(result)
        if result.state in {"partial_timing_only", "complete_timing_only"}:
            return
        if index + 1 < polling.max_polls:
            polling.sleep(polling.interval)


def write_status(path: Path, result: Result) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(asdict(result), stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
