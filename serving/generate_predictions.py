"""Greedy student predictions with vLLM for scoring (base model and/or LoRA adapters; T-021).

Owner-run on Kaggle (OPEN O-019). Same prompt as training (xbrl_frontier.sft_format.prompt_text,
enable_thinking=False), temperature eval.generation.temperature, max_model_len eval.generation.max_model_len,
max_tokens from the input per xbrl_frontier.output_cap (T-039), bounded by the remaining context.
A prompt that does not fit is not truncated: it is written as not_generated and scored as an empty
prediction, counted separately. If vLLM refuses the configured max_model_len (e.g. not enough KV cache on a
16 GB T4), each refusal is recorded and the next halving is tried; the accepted (effective) value is
recorded in the meta file and decides which prompts fit. Nothing is lowered silently. One output file per target: OUT/predictions_<target>_<set>.jsonl.
Requests are fed to the engine continuously (a window in flight) and each output is appended as it finishes, so one long
generation never idles the batch. Before each engine step the run checks the Kaggle session
deadline (training.kaggle.session_hours_max less session_stop_margin_minutes, from process start; T-038) and stops cleanly,
with meta "complete": false; --resume DIR (a previous session's OUT) keeps the ids already written and continues.
"""
import argparse
import shutil
import json
import time
from pathlib import Path

from training.encoder_train import load_set
from training import schedule
from xbrl_frontier import config, output_cap, sft_format


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=[s["id"] for s in config.get("pins.students")])
    ap.add_argument("--adapter", action="append", default=[], help="NAME=DIR, repeatable")
    ap.add_argument("--base", action="store_true", help="also predict with the base model (zero-shot)")
    ap.add_argument("--checkpoint", help="NAME=DIR: an exported checkpoint (merged fp16, INT8, AWQ) served directly")
    ap.add_argument("--set", choices=["finni", "dev", "fresh"], required=True)
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", type=Path, help="previous session's OUT directory; its written predictions are kept")
    ap.add_argument("--window", type=int, default=2048,
                    help="requests kept in the engine at once; each output is written as it finishes (continuous batching)")
    ap.add_argument("--session-hours", type=float, help="planned session length (<= training.kaggle.session_hours_max)")
    args = ap.parse_args()
    started = time.time()
    planned_hours = schedule.session_hours(args.session_hours)

    from serving.common import env_record, model_pin, pinned_snapshot, require_pins
    versions = require_pins("vllm")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    g = config.get("eval.generation")
    if args.checkpoint and (args.base or args.adapter):
        raise SystemExit("--checkpoint is exclusive with --base/--adapter (one served model per run)")
    local = args.checkpoint.split("=", 1)[1] if args.checkpoint else pinned_snapshot(args.model)
    tok = AutoTokenizer.from_pretrained(local)
    items = load_set(args.set, args.corpus)
    prompts = []
    for x in items:
        p = sft_format.prompt_text(tok, x["text"])
        prompts.append((p, len(tok.encode(p, add_special_tokens=False))))
    targets = ([("base", None)] if args.base else []) + [tuple(a.split("=", 1)) for a in args.adapter]
    if args.checkpoint:
        targets = [(args.checkpoint.split("=", 1)[0], None)]
    if not targets:
        raise SystemExit("nothing to predict: pass --base, --adapter NAME=DIR or --checkpoint NAME=DIR")
    llm, refusals, limit = None, [], g["max_model_len"]
    floor = config.get("training.kaggle.max_seq_len_tokens")
    while llm is None:
        try:
            llm = LLM(model=local, tokenizer=local, dtype="half", max_model_len=limit,
                      enable_lora=any(d for _, d in targets), max_lora_rank=config.get("training.lora_rank"),
                      seed=config.get("pins.seeds")[0])
        except (ValueError, RuntimeError) as e:
            refusals.append({"max_model_len": limit, "error": f"{type(e).__name__}: {str(e)[:500]}"})
            print("vLLM refused max_model_len", limit, "->", refusals[-1]["error"], flush=True)
            if limit // 2 < floor:
                raise SystemExit(f"vLLM refused every max_model_len down to {limit}; see the error above")
            limit //= 2
    effective = llm.llm_engine.model_config.max_model_len
    fits = [n < effective for _, n in prompts]
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"model": args.model, "revision": model_pin(args.model), "set": args.set, "versions": versions,
            "env": env_record("kaggle_t4"), "config_hash": config.config_hash(), "generation": g,
            "served_from": local, "configured_max_model_len": g["max_model_len"], "effective_max_model_len": effective,
            "max_model_len_refusals": refusals,
            "instances": len(items), "not_generated_over_max_model_len": fits.count(False), "targets": {}}
    meta["complete"] = True
    for k, (name, path) in enumerate(targets):
        lora = LoRARequest(name, k + 1, path) if path else None
        t0 = time.time()
        out_path = args.out / f"predictions_{name}_{args.set}.jsonl"
        if args.resume and (args.resume / out_path.name).exists() and not out_path.exists():
            shutil.copy(args.resume / out_path.name, out_path)
        done = {json.loads(l)["id"] for l in open(out_path)} if out_path.exists() else set()
        todo = [(x, p, n, ok) for x, (p, n), ok in zip(items, prompts, fits) if x["id"] not in done]
        stopped = False
        with open(out_path, "a") as f:
            for x, p, n, ok in todo:
                if not ok:
                    f.write(json.dumps({"id": x["id"], "text": None, "not_generated": True, "prompt_tokens": n}) + "\n")
            queue = [(x, p, n, output_cap.cap(x["text"], effective - n)) for x, p, n, ok in todo if ok]
            queue.reverse()
            engine, flight = llm.llm_engine, {}
            while queue or flight:
                if not stopped and schedule.session_time_up(time.time(), started, planned_hours):
                    stopped = True
                    queue.clear()
                    if flight:
                        engine.abort_request(list(flight))
                    break
                while queue and len(flight) < args.window:
                    x, pr, n, c = queue.pop()
                    rid = engine.add_request(str(x["id"]), pr, SamplingParams(temperature=g["temperature"], max_tokens=c),
                                             lora_request=lora)
                    flight[rid] = (x, n, c)
                for o in engine.step():
                    if not o.finished or o.request_id not in flight:
                        continue
                    x, n, c = flight.pop(o.request_id)
                    o = o.outputs[0]
                    f.write(json.dumps({"id": x["id"], "text": o.text, "finish_reason": o.finish_reason,
                                        "prompt_tokens": n, "completion_tokens": len(o.token_ids), "max_tokens": c,
                                        "not_generated": False}) + "\n")
                    f.flush()
                if flight and not engine.has_unfinished_requests():
                    raise SystemExit(f"engine idle with {len(flight)} unmatched request ids (e.g. {next(iter(flight))}); vLLM id mismatch")
        written = sum(1 for _ in open(out_path))
        meta["targets"][name] = {"adapter": path, "file": out_path.name, "seconds_this_session": time.time() - t0,
                                 "resumed_with": len(done), "written": written, "complete": written == len(items)}
        meta["complete"] = meta["complete"] and written == len(items)
        if stopped:
            meta["stopped_for_session_limit"] = True
            break
    (args.out / f"predictions_{args.set}.meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "env"}, indent=1))


if __name__ == "__main__":
    main()
