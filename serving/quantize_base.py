"""Quantize the pinned base students for the Phase 0 smoke test (serving.smoke_test.smoke_quantization).

Owner-run on Kaggle (T4). These checkpoints exist only to prove each planned variant loads and runs in
vLLM on each card; they are never scored. Writes OUT/<model>-int8-w8a8 and OUT/<model>-awq-int4.

INT8 needs llmcompressor (pins.software.llmcompressor); AWQ needs autoawq (pins.software.autoawq), which
may need its own environment because of its transformers bound: run with --only int8 or --only awq.
The quantization steps themselves are shared with the trained-student export (serving/export_quantized.py).
"""
import argparse
import json
from pathlib import Path

from xbrl_frontier import config
from serving.common import env_record, model_pin, pinned_snapshot, require_pins


def int8_from_dir(src: str, out: Path) -> None:
    """INT8 W8A8 with llm-compressor (data-free; dynamic per-token activations)."""
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec = config.get("serving.smoke_test.smoke_quantization.int8_w8a8")
    model = AutoModelForCausalLM.from_pretrained(src, torch_dtype="auto")
    tok = AutoTokenizer.from_pretrained(src)
    oneshot(model=model, recipe=QuantizationModifier(targets="Linear", scheme=spec["scheme"], ignore=["lm_head"]))
    model.save_pretrained(out, save_compressed=True)
    tok.save_pretrained(out)


def awq_from_dir(src: str, out: Path, calib_data, samples: int, seq_len: int) -> None:
    """AWQ-INT4 (AutoAWQ GEMM) with the given calibration data (a dataset name or a list of texts)."""
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer
    spec = config.get("serving.smoke_test.smoke_quantization.awq_int4")
    model = AutoAWQForCausalLM.from_pretrained(src)
    tok = AutoTokenizer.from_pretrained(src)
    model.quantize(tok, quant_config={"w_bit": spec["w_bit"], "q_group_size": spec["q_group_size"],
                                      "zero_point": spec["zero_point"], "version": spec["version"]},
                   calib_data=calib_data, n_parallel_calib_samples=None,
                   max_calib_samples=samples, max_calib_seq_len=seq_len)
    model.save_quantized(str(out))
    tok.save_pretrained(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/smoke_checkpoints"))
    ap.add_argument("--only", choices=["int8", "awq"], required=True)
    args = ap.parse_args()
    versions = require_pins("llmcompressor" if args.only == "int8" else "autoawq")
    spec = config.get("serving.smoke_test.smoke_quantization.awq_int4")
    args.out.mkdir(parents=True, exist_ok=True)
    log = []
    for s in config.get("pins.students"):
        tag = s["id"].split("/")[-1]
        dest = args.out / f"{tag}-{'int8-w8a8' if args.only == 'int8' else 'awq-int4'}"
        entry = {"model": s["id"], "revision": model_pin(s["id"]), "variant": dest.name, "versions": versions}
        try:
            src = pinned_snapshot(s["id"])
            if args.only == "int8":
                int8_from_dir(src, dest)
            else:
                awq_from_dir(src, dest, spec["calib_data"], spec["calib_samples"], spec["calib_max_seq_len"])
            entry["ok"] = True
        except Exception as e:  # recorded per serving.smoke_test.failure_action
            entry.update(ok=False, error=f"{type(e).__name__}: {e}")
        log.append(entry)
        print(entry, flush=True)
    (args.out / f"quantize_{args.only}.json").write_text(json.dumps({"env": env_record("kaggle_t4"), "runs": log}, indent=1))


if __name__ == "__main__":
    main()
