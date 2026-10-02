"""Export a trained student: merge the adapter into the pinned base (fp16), then INT8 W8A8 and AWQ-INT4
(PRD sections 5 and 9; TRADEOFFS T-023). Owner-run on Kaggle (OPEN O-019, export steps).

Stages, each in the environment its tool needs:
  merge  transformers + peft at their pins: base at pins.students revision in fp16 + LoRA -> OUT/merged-fp16
  int8   llmcompressor at its pin: OUT/merged-fp16 -> OUT/int8-w8a8 (same scheme as the smoke test)
  awq    autoawq at its pin (own venv): OUT/merged-fp16 -> OUT/awq-int4, calibrated on a fixed seeded sample of
         training instances rendered as prompt + answer (serving.export_quantization.awq_calibration)
Each stage appends to OUT/export.json: versions, env, inputs, output file hashes, calibration hash.
Every checkpoint is scored on the full set before any load test (gates.quantized_*; scripts/flip_rate.py).
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from xbrl_frontier import config, sft_format


def calibration_texts_from(path: Path, tokenizer, spec: dict) -> tuple[list[str], dict]:
    """The fixed calibration sample: random.Random(seed).sample over the training file's lines, rendered with
    the training format (prompt + answer)."""
    rows = [json.loads(l) for l in open(path)]
    picked = random.Random(spec["seed"]).sample(range(len(rows)), spec["samples"])
    texts = [sft_format.full_text(tokenizer, rows[i]["text"], rows[i]["numeric_entities"])[1] for i in picked]
    meta = {"source": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "seed": spec["seed"], "samples": spec["samples"], "max_seq_len": spec["max_seq_len"],
            "instance_ids": [rows[i]["id"] for i in picked],
            "texts_sha256": hashlib.sha256("\x1e".join(texts).encode()).hexdigest()}
    return texts, meta


def dir_hashes(d: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.glob("*.safetensors"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["merge", "int8", "awq"])
    ap.add_argument("--model", required=True, choices=[s["id"] for s in config.get("pins.students")])
    ap.add_argument("--adapter", type=Path, help="merge: the run's final_adapter directory")
    ap.add_argument("--corpus", type=Path, help="awq: directory holding train.jsonl (defaults to the config source)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from serving.common import env_record, model_pin, pinned_snapshot, require_pins
    from serving.quantize_base import awq_from_dir, int8_from_dir

    args.out.mkdir(parents=True, exist_ok=True)
    merged = args.out / "merged-fp16"
    rec = {"stage": args.stage, "model": args.model, "revision": model_pin(args.model),
           "config_hash": config.config_hash(), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if args.stage == "merge":
        rec["versions"] = require_pins("transformers", "peft")
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        local = pinned_snapshot(args.model)
        base = AutoModelForCausalLM.from_pretrained(local, torch_dtype=torch.float16)
        model = PeftModel.from_pretrained(base, str(args.adapter)).merge_and_unload()
        model.save_pretrained(merged, safe_serialization=True)
        AutoTokenizer.from_pretrained(local).save_pretrained(merged)
        rec.update(adapter=str(args.adapter), adapter_hashes=dir_hashes(args.adapter), output=str(merged),
                   output_hashes=dir_hashes(merged))
    elif args.stage == "int8":
        rec["versions"] = require_pins("llmcompressor")
        dest = args.out / "int8-w8a8"
        int8_from_dir(str(merged), dest)
        rec.update(input_hashes=dir_hashes(merged), output=str(dest), output_hashes=dir_hashes(dest))
    else:
        rec["versions"] = require_pins("autoawq")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(merged))
        spec = config.get("serving.export_quantization.awq_calibration")
        src = (args.corpus / "train.jsonl") if args.corpus else config.REPO_ROOT / spec["source"]
        texts, cal = calibration_texts_from(src, tok, spec)
        dest = args.out / "awq-int4"
        awq_from_dir(str(merged), dest, texts, spec["samples"], spec["max_seq_len"])
        rec.update(calibration=cal, input_hashes=dir_hashes(merged), output=str(dest), output_hashes=dir_hashes(dest))
    rec["env"] = env_record("kaggle_t4")
    log_path = args.out / "export.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else []
    log.append(rec)
    log_path.write_text(json.dumps(log, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k not in ("env",)}, indent=1)[:3000])


if __name__ == "__main__":
    main()
