import json
import struct

from scripts import d3_pilot_dispatch as dispatch
from scripts import d3_pilot_gate as gate


class Provider:
    def __init__(self, final_run, status="ERROR", version=2):
        self.final_run = final_run
        self.version = version
        self.status = status
        self.pushes = 0
        self.pilot_state = "ABSENT"

    def training(self):
        return dispatch.TrainingState(self.version, self.status, self.final_run)

    def budget(self):
        return gate.Budget(1.5, 20, 60, 30, 70, True)

    def submit(self, version, digest, hours):
        self.pushes += 1
        self.pilot_state = "UNKNOWN"

    def pilot(self):
        return self.pilot_state

    def collect(self):
        return dispatch.PilotOutput(12, False, 20.0)


def test_old_error_is_waiting_without_push(tmp_path):
    provider = Provider(tmp_path)
    result = dispatch.step(provider, dispatch.Options(tmp_path / "attempt.json", gate.Provenance("pinned", "devhash", "confighash")))
    assert (result.state, provider.pushes) == ("waiting_for_new_completed_training", 0)


def test_unknown_pilot_never_pushes_again(tmp_path):
    marker = tmp_path / "attempt.json"
    marker.write_text('{"state":"push_outcome_unknown"}')
    provider = Provider(tmp_path, "COMPLETE", 3)
    provider.pilot_state = "UNKNOWN"
    result = dispatch.step(provider, dispatch.Options(marker, gate.Provenance("pinned", "devhash", "confighash")))
    assert (result.state, provider.pushes) == ("pilot_outcome_unknown", 0)


def test_terminal_partial_output_is_not_score(tmp_path):
    marker = tmp_path / "attempt.json"
    marker.write_text('{"state":"push_outcome_unknown"}')
    provider = Provider(tmp_path)
    provider.pilot_state = "COMPLETE"
    result = dispatch.step(provider, dispatch.Options(marker, gate.Provenance("pinned", "devhash", "confighash")))
    assert result.state == "partial_timing_only"
    assert result.rows == 12 and result.seconds == 20.0 and provider.pushes == 0


def test_completed_training_dispatches_only_once(tmp_path):
    adapter = tmp_path / "final_adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"base_model_name_or_path":"Qwen/Qwen3-1.7B"}')
    header = json.dumps({"layer": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (adapter / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"1234")
    (tmp_path / "run.json").write_text(json.dumps({
        "complete": True, "global_step": 4716, "final_adapter": "final_adapter", "seed": 1,
        "model": "Qwen/Qwen3-1.7B", "revision": "pinned", "config_hash": "confighash",
        "sessions": [{"resumed_from": "checkpoint-4180"}], "corpus_sha256": {"dev": "devhash"},
    }))
    provider = Provider(tmp_path, "COMPLETE", 3)
    options = dispatch.Options(tmp_path / "attempt.json", gate.Provenance("pinned", "devhash", "confighash"))
    assert dispatch.step(provider, options).state == "pilot_outcome_unknown"
    assert dispatch.step(provider, options).state == "pilot_outcome_unknown"
    assert provider.pushes == 1


def test_remote_pilot_unknown_does_not_submit(tmp_path):
    provider = Provider(tmp_path, "COMPLETE", 3)
    provider.pilot_state = "UNKNOWN"
    result = dispatch.step(provider, dispatch.Options(tmp_path / "attempt.json", gate.Provenance("pinned", "devhash", "confighash")))
    assert result.state == "pilot_outcome_unknown" and provider.pushes == 0


def test_watcher_sleeps_between_waiting_polls(tmp_path):
    provider = Provider(tmp_path)
    sleeps = []
    statuses = []
    options = dispatch.Options(tmp_path / "attempt.json", gate.Provenance("pinned", "devhash", "confighash"))
    dispatch.watch(provider, options, dispatch.Polling(sleeps.append, statuses.append, 5, 2))
    assert sleeps == [5] and [item.state for item in statuses] == ["waiting_for_new_completed_training"] * 2


def test_status_file_preserves_partial_timing(tmp_path):
    path = tmp_path / "status.json"
    dispatch.write_status(path, dispatch.Result("partial_timing_only", 12, 20.0))
    assert json.loads(path.read_text()) == {"state": "partial_timing_only", "rows": 12, "seconds": 20.0}
