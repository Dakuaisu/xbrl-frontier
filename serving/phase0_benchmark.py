"""Phase 0 base-model throughput benchmark on a Kaggle T4 (serving.phase0_throughput_benchmark).

Owner-run on Kaggle. Writes OUT/benchmark_<model>.json per model plus logs and env, then OUT.tar.gz.
Analyse locally with: .venv/bin/python -m scripts.phase0_cost --results data/phase0/kaggle_t4
"""
import argparse
import json
from pathlib import Path

from xbrl_frontier import config
from serving.common import Server, archive, bench, env_record, model_pin, pinned_snapshot, require_pins


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/phase0_kaggle_t4"))
    args = ap.parse_args()
    versions = require_pins("vllm")
    spec = config.get("serving.phase0_throughput_benchmark")
    hw = config.get("serving.hardware.kaggle_t4")
    max_len = config.get("training.kaggle.max_seq_len_tokens")
    warmup = config.get("serving.warmup_requests")
    args.out.mkdir(parents=True, exist_ok=True)
    for model in spec["models"]:
        rev = model_pin(model)
        local = pinned_snapshot(model)
        tag = model.split("/")[-1]
        log = args.out / f"server_{tag}.log"
        flags = ["--served-model-name", model, "--dtype", "half" if hw["dtype"] == "fp16" else hw["dtype"],
                 "--max-model-len", str(max_len), "--seed", str(config.get("pins.seeds")[0])]
        rec = {"model": model, "revision": rev, "versions": versions, "server_flags": flags,
               "env": env_record("kaggle_t4"), "spec": spec, "points": [], "errors": []}
        try:
            with Server(local, flags, log):
                bench(model, local, spec["input_tokens"], spec["output_tokens"], warmup, 1,
                      args.out / f"warmup_{tag}.json", args.out / f"bench_{tag}.log")
                for c in spec["concurrency_levels"]:
                    try:
                        rec["points"].append(bench(model, local, spec["input_tokens"], spec["output_tokens"],
                                                   spec["requests_per_point"], c,
                                                   args.out / f"raw_{tag}_c{c}.json", args.out / f"bench_{tag}.log"))
                    except RuntimeError as e:
                        rec["errors"].append({"concurrency": c, "error": str(e)})
        except RuntimeError as e:
            rec["errors"].append({"stage": "server", "error": str(e)})
        (args.out / f"benchmark_{tag}.json").write_text(json.dumps(rec, indent=1))
        print(tag, "points:", len(rec["points"]), "errors:", rec["errors"], flush=True)
    print("archive:", archive(args.out))


if __name__ == "__main__":
    main()
