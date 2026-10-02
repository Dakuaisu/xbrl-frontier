"""vLLM smoke test of every planned serving variant on one card (serving.smoke_test).

Owner-run: --card kaggle_t4 on Kaggle, --card rtx3050_wsl2 on the laptop under WSL2. Per variant: start
vLLM, one generation on a pinned FinNI prompt, then a short throughput check of
serving.smoke_test.requests_per_variant requests. Writes OUT/smoke_<card>.json and OUT.tar.gz.
"""
import argparse
import json
import urllib.request
from pathlib import Path

from xbrl_frontier import config
from serving.common import PORT, Server, archive, bench, env_record, pinned_snapshot, require_pins

SIZE_TO_MODEL = {s["id"].split("-")[-1]: s["id"] for s in config.get("pins.students")}


def planned(card: str) -> list[tuple[str, str, str]]:
    """(variant label, model id, precision) for every planned variant on the card."""
    variants = config.get(f"serving.hardware.{card}.variants")
    out = []
    for v in variants:
        if "-" in v and v.split("-")[0] in SIZE_TO_MODEL:
            size, precision = v.split("-", 1)
            out.append((v, SIZE_TO_MODEL[size], precision))
        else:
            out += [(f"{m.split('-')[-1]}-{v}", m, v) for m in SIZE_TO_MODEL.values()]
    return out


def checkpoint(model_id: str, precision: str, ckpt_dir: Path) -> str:
    if precision in ("fp16", "bf16"):
        return pinned_snapshot(model_id)
    path = ckpt_dir / f"{model_id.split('/')[-1]}-{precision}"
    if not path.exists():
        raise FileNotFoundError(f"missing quantized checkpoint {path} (run serving.quantize_base first)")
    return str(path)


def server_flags(card: str, model_id: str, precision: str) -> list[str]:
    hw = config.get(f"serving.hardware.{card}")
    dtype = "bfloat16" if precision == "bf16" else "half"
    flags = ["--served-model-name", model_id, "--dtype", dtype, "--seed", str(config.get("pins.seeds")[0])]
    vf = hw.get("vllm_flags", {})
    if vf.get("enforce_eager"):
        flags.append("--enforce-eager")
    for key, flag in (("max_model_len", "--max-model-len"), ("gpu_memory_utilization", "--gpu-memory-utilization"),
                      ("max_num_seqs", "--max-num-seqs")):
        if key in vf:
            flags += [flag, str(vf[key])]
    if "max_model_len" not in vf:
        flags += ["--max-model-len", str(config.get("training.kaggle.max_seq_len_tokens"))]
    return flags


def one_generation(model_id: str, prompt: str) -> dict:
    body = json.dumps({"model": model_id, "prompt": prompt, "temperature": 0,
                       "max_tokens": config.get("serving.phase0_throughput_benchmark.output_tokens")}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    return {"text": out["choices"][0]["text"], "usage": out.get("usage")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", choices=config.get("serving.smoke_test.cards"), required=True)
    ap.add_argument("--checkpoints", type=Path, required=True, help="directory from serving.quantize_base")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--precision", action="append", help="test only these planned precisions (a rerun); repeatable")
    args = ap.parse_args()
    versions = require_pins("vllm")
    from xbrl_frontier import finni
    prompt = finni.load()[0].query
    spec = config.get("serving.smoke_test")
    conc = max(config.get(f"serving.concurrency_levels.{args.card}"))
    shape = config.get("serving.phase0_throughput_benchmark")
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    todo = [x for x in planned(args.card) if not args.precision or x[2] in args.precision]
    if not todo:
        raise SystemExit(f"no planned {args.card} variant has precision {args.precision}")
    for label, model_id, precision in todo:
        r = {"variant": label, "model": model_id, "precision": precision, "ok": False}
        log = args.out / f"server_{label}.log"
        try:
            path = checkpoint(model_id, precision, args.checkpoints)
            r["checkpoint"] = path
            r["server_flags"] = server_flags(args.card, model_id, precision)
            with Server(path, r["server_flags"], log):
                r["generation"] = one_generation(model_id, prompt)
                r["throughput_check"] = bench(model_id, path, shape["input_tokens"], shape["output_tokens"],
                                              spec["requests_per_variant"], conc,
                                              args.out / f"raw_{label}.json", args.out / f"bench_{label}.log")
            r["ok"] = (bool(r["generation"]["text"].strip())
                       and r["throughput_check"]["completed"] == spec["requests_per_variant"])
        except Exception as e:  # every failure is recorded; failure_action applies to the variant
            r["error"] = f"{type(e).__name__}: {e}"
            if log.exists():
                r["server_log_tail"] = log.read_text()[-4000:]
        results.append(r)
        print(label, "OK" if r["ok"] else f"FAIL {r.get('error', '')}", flush=True)
    (args.out / f"smoke_{args.card}.json").write_text(json.dumps(
        {"card": args.card, "versions": versions, "env": env_record(args.card),
         "failure_action": spec["failure_action"], "precision_filter": args.precision, "results": results}, indent=1))
    print("archive:", archive(args.out))


if __name__ == "__main__":
    main()
