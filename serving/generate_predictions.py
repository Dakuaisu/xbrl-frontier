"""Greedy student predictions with vLLM for scoring (base model and/or LoRA adapters; T-021).

Owner-run on Kaggle (OPEN O-019). Same prompt as training (xbrl_frontier.sft_format.prompt_text,
enable_thinking=False), temperature eval.generation.temperature, max_model_len eval.generation.max_model_len.
A prompt that does not fit is not truncated: it is written as not_generated and scored as an empty
prediction, counted separately. One output file per target: OUT/predictions_<target>_<set>.jsonl.
"""
import argparse
import json
import time
from pathlib import Path

from training.encoder_train import load_set
from xbrl_frontier import config, sft_format


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=[s["id"] for s in config.get("pins.students")])
    ap.add_argument("--adapter", action="append", default=[], help="NAME=DIR, repeatable")
    ap.add_argument("--base", action="store_true", help="also predict with the base model (zero-shot)")
    ap.add_argument("--set", choices=["finni", "dev", "fresh"], required=True)
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from serving.common import env_record, model_pin, pinned_snapshot, require_pins
    versions = require_pins("vllm")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    g = config.get("eval.generation")
    local = pinned_snapshot(args.model)
    tok = AutoTokenizer.from_pretrained(local)
    items = load_set(args.set, args.corpus)
    prompts, fits = [], []
    for x in items:
        p = sft_format.prompt_text(tok, x["text"])
        n = len(tok.encode(p, add_special_tokens=False))
        prompts.append((p, n))
        fits.append(n < g["max_model_len"])
    targets = ([("base", None)] if args.base else []) + [tuple(a.split("=", 1)) for a in args.adapter]
    if not targets:
        raise SystemExit("nothing to predict: pass --base and/or --adapter NAME=DIR")
    llm = LLM(model=local, tokenizer=local, dtype="half", max_model_len=g["max_model_len"],
              enable_lora=any(d for _, d in targets), max_lora_rank=config.get("training.lora_rank"),
              seed=config.get("pins.seeds")[0])
    args.out.mkdir(parents=True, exist_ok=True)
    run_prompts = [p for (p, n), ok in zip(prompts, fits) if ok]
    params = [SamplingParams(temperature=g["temperature"], max_tokens=g["max_model_len"] - n)
              for (p, n), ok in zip(prompts, fits) if ok]
    meta = {"model": args.model, "revision": model_pin(args.model), "set": args.set, "versions": versions,
            "env": env_record("kaggle_t4"), "config_hash": config.config_hash(), "generation": g,
            "instances": len(items), "not_generated_over_max_model_len": fits.count(False), "targets": {}}
    for k, (name, path) in enumerate(targets):
        lora = LoRARequest(name, k + 1, path) if path else None
        t0 = time.time()
        outs = iter(llm.generate(run_prompts, params, lora_request=lora))
        out_path = args.out / f"predictions_{name}_{args.set}.jsonl"
        with open(out_path, "w") as f:
            for x, (p, n), ok in zip(items, prompts, fits):
                if not ok:
                    f.write(json.dumps({"id": x["id"], "text": None, "not_generated": True, "prompt_tokens": n}) + "\n")
                    continue
                o = next(outs).outputs[0]
                f.write(json.dumps({"id": x["id"], "text": o.text, "finish_reason": o.finish_reason,
                                    "prompt_tokens": n, "completion_tokens": len(o.token_ids),
                                    "not_generated": False}) + "\n")
        meta["targets"][name] = {"adapter": path, "file": out_path.name, "seconds": time.time() - t0}
    (args.out / f"predictions_{args.set}.meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "env"}, indent=1))


if __name__ == "__main__":
    main()
