"""Phase 4 load test of one serving variant on one card (PRD section 6; TRADEOFFS T-027). Owner-run.

Prompts: the pinned latency ids (serving.load_test.prompts, identical for every arm and student), rendered with
the training/inference format, sent pre-rendered with a per-prompt output cap of serving.load_test.max_output_tokens
(generation stops at EOS first). Warm-up of serving.warmup_requests (discarded), then serving.requests_per_point
requests per point: closed loop at each serving.concurrency_levels.<card> (single-request regime = concurrency 1)
and Poisson arrivals at each serving.request_rate_sweep_rps (batched regime). Server flags: the card's flags
(serving.hardware.<card>.vllm_flags on the 3050); on the T4 max_model_len covers the longest pinned prompt plus the
output cap. Writes OUT/load_<card>_<variant>.json (points + env + flags + checkpoint hashes) and OUT.tar.gz.
"""
import argparse
import hashlib
import json
from pathlib import Path

from serving.common import Server, archive, bench_custom, env_record, pinned_snapshot, require_pins
from serving.smoke_test import server_flags
from xbrl_frontier import config, finni, sft_format


def prompts_file(tok, out: Path) -> tuple[Path, int, int]:
    ids_path = config.docs_path(config.get("serving.load_test.prompts"))
    if not ids_path.exists():
        raise SystemExit(f"{ids_path} is not pinned yet (scripts/draw_frontier_sample.py); refusing")
    rows = finni.load()
    cap = config.get("serving.load_test.max_output_tokens")
    path, longest, n = out / "prompts.jsonl", 0, 0
    with open(path, "w") as f:
        for i in (int(l) for l in open(ids_path) if l.strip()):
            p = sft_format.prompt_text(tok, rows[i].context)
            longest = max(longest, len(tok.encode(p, add_special_tokens=False)))
            f.write(json.dumps({"prompt": p, "output_tokens": cap}) + "\n")
            n += 1
    return path, longest, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", choices=list(config.get("serving.hardware")), required=True)
    ap.add_argument("--model", required=True, choices=[s["id"] for s in config.get("pins.students")])
    ap.add_argument("--variant", required=True, help="fp16 | bf16 | int8-w8a8 | awq-int4")
    ap.add_argument("--checkpoint", type=Path, help="exported checkpoint dir; omitted = pinned base (bf16/fp16 only)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    versions = require_pins("vllm")
    from transformers import AutoTokenizer
    path = str(args.checkpoint) if args.checkpoint else pinned_snapshot(args.model)
    tok = AutoTokenizer.from_pretrained(path)
    args.out.mkdir(parents=True, exist_ok=True)
    pfile, longest, n_prompts = prompts_file(tok, args.out)
    flags = server_flags(args.card, args.model, args.variant)
    if "max_model_len" not in config.get(f"serving.hardware.{args.card}").get("vllm_flags", {}):
        i = flags.index("--max-model-len")
        flags[i + 1] = str(longest + config.get("serving.load_test.max_output_tokens"))
    tag = f"{args.card}_{args.model.split('/')[-1]}_{args.variant}"
    log = args.out / f"server_{tag}.log"
    rec = {"card": args.card, "model": args.model, "variant": args.variant, "checkpoint": path, "versions": versions,
           "checkpoint_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(path).glob("*.safetensors"))},
           "server_flags": flags, "prompts": n_prompts, "longest_prompt_tokens": longest,
           "env": env_record(args.card), "config_hash": config.config_hash(), "points": [], "errors": []}
    per = config.get("serving.requests_per_point")
    try:
        with Server(path, flags, log):
            bench_custom(args.model, path, pfile, config.get("serving.warmup_requests"),
                         args.out / f"warmup_{tag}.json", args.out / f"bench_{tag}.log", concurrency=1)
            for c in config.get(f"serving.concurrency_levels.{args.card}"):
                try:
                    rec["points"].append({"regime": "closed_loop", **bench_custom(
                        args.model, path, pfile, per, args.out / f"raw_{tag}_c{c}.json", args.out / f"bench_{tag}.log", concurrency=c)})
                except RuntimeError as e:
                    rec["errors"].append({"concurrency": c, "error": str(e)})
            for r in config.get("serving.request_rate_sweep_rps"):
                try:
                    rec["points"].append({"regime": "poisson_rate", **bench_custom(
                        args.model, path, pfile, per, args.out / f"raw_{tag}_r{r}.json", args.out / f"bench_{tag}.log", request_rate=r)})
                except RuntimeError as e:
                    rec["errors"].append({"request_rate": r, "error": str(e)})
    except RuntimeError as e:
        rec["errors"].append({"stage": "server", "error": str(e)})
    (args.out / f"load_{tag}.json").write_text(json.dumps(rec, indent=1))
    print(tag, "points:", len(rec["points"]), "errors:", rec["errors"], flush=True)
    print("archive:", archive(args.out))


if __name__ == "__main__":
    main()
