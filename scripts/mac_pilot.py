"""Phase 2 Mac pilot (PRD section 5; TRADEOFFS T-017): QLoRA on a 4-bit base with mlx-lm.

Settles the prompt format and loss masking (xbrl_frontier.sft_format, shared with Kaggle training), runs one
epoch over training.mac.pilot_instances at training.mac.max_seq_len_tokens and the configured LoRA/LR, and
measures tokens per second against training.mac.min_tokens_per_sec_to_train_0p6b_locally.

Pilot data: the first pilot_instances of the fixed corpus order (data/corpus/train.jsonl) whose full
sequence fits max_seq_len (mlx-lm would otherwise truncate the answer); val = the first pilot_val_instances
dev instances that fit. Tokens/sec is reported two ways: processed tokens (whole sequences, the headline,
comparable to the planning rates; the pilot is exactly one epoch over full micro-batches, so it is the sum of
all pilot sequence lengths) and loss tokens (mlx-lm's own figure, completion tokens only).
After training, the adapter generates greedily on the val instances and is scored with the FinNI scorer:
a format check, not a result.
Writes data/phase2/mac_pilot_<model>/ (adapter, untracked) and data/phase2_mac_pilot_<model>.json.
"""
import argparse
import json
import platform
import time
from pathlib import Path
from types import SimpleNamespace

from xbrl_frontier import config, scorer, sft_format

PHASE2 = config.REPO_ROOT / "data" / "phase2"


class _Item(list):
    """Token ids whose len() is the sequence length (mlx-lm buckets by len of the raw item)."""

    def __init__(self, ids, offset):
        super().__init__(ids)
        self.offset = offset


class PilotDataset:
    """(token ids, loss offset) items in the shape mlx-lm's CacheDataset expects."""

    def __init__(self, items: list[tuple[list[int], int]]):
        self.items = [_Item(ids, off) for ids, off in items]

    def process(self, item):
        return list(item), item.offset

    def __getitem__(self, i):
        return self.items[i]

    def __len__(self):
        return len(self.items)


def first_fitting(path: Path, n: int, tok, max_len: int) -> tuple[list, list, int]:
    """First n instances of a corpus file whose encoded sequence fits max_len."""
    raw, enc, skipped = [], [], 0
    for line in open(path):
        if len(enc) >= n:
            break
        x = json.loads(line)
        ids, off = sft_format.encode(tok, x["text"], x["numeric_entities"])
        if len(ids) > max_len:
            skipped += 1
            continue
        raw.append(x)
        enc.append((ids, off))
    return raw, enc, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    args = ap.parse_args()
    import mlx.core as mx
    import mlx_lm
    from mlx_lm import convert, generate, load
    from mlx_lm.lora import CONFIG_DEFAULTS, train_model
    from mlx_lm.tuner.callbacks import TrainingCallback

    from serving.common import model_pin, pinned_snapshot

    mac = config.get("training.mac")
    if mlx_lm.__version__ != config.get("pins.software.mlx_lm"):
        raise SystemExit(f"mlx_lm {mlx_lm.__version__} != pins.software.mlx_lm")
    tag = args.model.split("/")[-1]
    qpath = PHASE2 / f"{tag}-mlx-{mac['quant_base']}"
    if not qpath.exists():
        if mac["quant_base"] != "4bit":
            raise SystemExit("only training.mac.quant_base=4bit is implemented")
        convert(hf_path=pinned_snapshot(args.model), mlx_path=str(qpath), quantize=True, q_bits=4,
                q_group_size=mac["quant_group_size"])
    model, tok = load(str(qpath))
    hf_tok = tok._tokenizer if hasattr(tok, "_tokenizer") else tok

    max_len = mac["max_seq_len_tokens"]
    corpus = config.REPO_ROOT / "data" / "corpus"
    _, train_items, train_skipped = first_fitting(corpus / "train.jsonl", mac["pilot_instances"], hf_tok, max_len)
    val_raw, val_items, val_skipped = first_fitting(corpus / "dev.jsonl", mac["pilot_val_instances"], hf_tok, max_len)
    train_set, val_set = PilotDataset(train_items), PilotDataset(val_items)

    micro = mac["micro_batch"]
    accum = config.get("training.global_batch_instances") // micro
    iters = len(train_items) // micro
    rank, alpha = config.get("training.lora_rank"), config.get("training.lora_alpha")
    keys = [("self_attn." if m in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp.") + m
            for m in config.get("training.lora_target_modules")]
    out_dir = PHASE2 / f"mac_pilot_{tag}"
    a = SimpleNamespace(**{**CONFIG_DEFAULTS,
                           "model": str(qpath), "data": "data/corpus (pilot prefix)", "train": True, "fine_tune_type": "lora", "optimizer": "adam",
                           "seed": config.get("pins.seeds")[0], "num_layers": -1, "batch_size": micro,
                           "iters": iters, "val_batches": -1, "learning_rate": config.get("training.learning_rate"),
                           "steps_per_report": 10, "steps_per_eval": iters, "adapter_path": str(out_dir),
                           "save_every": iters, "max_seq_length": max_len, "grad_accumulation_steps": accum,
                           "grad_checkpoint": mac["grad_checkpoint"],
                           "lora_parameters": {"rank": rank, "scale": alpha / rank,
                                               "dropout": config.get("training.lora_dropout"), "keys": keys}})

    reports, vals = [], []

    class CB(TrainingCallback):
        def on_train_loss_report(self, info):
            reports.append(dict(info))
            print(json.dumps(info), flush=True)

        def on_val_loss_report(self, info):
            vals.append(dict(info))
            print("val", json.dumps(info), flush=True)

    mx.reset_peak_memory()
    t0 = time.perf_counter()
    train_model(a, model, train_set, val_set, CB())
    wall = time.perf_counter() - t0
    train_time = sum(10 / r["iterations_per_second"] for r in reports)
    loss_tokens = reports[-1]["trained_tokens"] if reports else 0
    # One epoch over iters full micro-batches visits each of the first iters * micro items exactly once.
    processed = sum(len(ids) for ids, _ in sorted(train_items, key=lambda t: len(t[0]))[: iters * micro]) \
        if len(train_items) == iters * micro else None

    model, tok = load(str(qpath), adapter_path=str(out_dir))
    golds, preds, gen_tokens, t1 = [], [], 0, time.perf_counter()
    for x in val_raw:
        prompt = sft_format.prompt_text(hf_tok, x["text"])
        n_prompt = len(hf_tok.encode(prompt, add_special_tokens=False))
        text = generate(model, tok, prompt, max_tokens=max(max_len - n_prompt, 1))
        gen_tokens += len(hf_tok.encode(text, add_special_tokens=False))
        golds.append(scorer.pair_set({"Fact": e["value"], "Type": e["type"]} for e in x["numeric_entities"]))
        preds.append(scorer.parse_prediction(text))
    gen_time = time.perf_counter() - t1
    s = scorer.score(golds, preds).overall

    threshold = mac["min_tokens_per_sec_to_train_0p6b_locally"]
    rep = {
        "model": args.model, "revision": model_pin(args.model), "base": f"mlx {mac['quant_base']} group {mac['quant_group_size']}",
        "mlx_lm": mlx_lm.__version__, "mlx": mx.__version__, "machine": platform.machine(), "platform": platform.platform(),
        "config_hash": config.config_hash(), "seed": a.seed,
        "train_instances": len(train_items), "train_skipped_over_max_len": train_skipped,
        "val_instances": len(val_items), "val_skipped_over_max_len": val_skipped,
        "iters": iters, "micro_batch": micro, "grad_accumulation": accum, "grad_checkpoint": mac["grad_checkpoint"], "optimizer_steps": iters // accum,
        "lora": a.lora_parameters, "learning_rate": a.learning_rate, "max_seq_len": max_len,
        "train_time_s": train_time, "wall_time_s": wall, "peak_memory_gb": max((r["peak_memory"] for r in reports), default=None),
        "processed_tokens": processed, "processed_tokens_per_sec": processed / train_time if train_time else None,
        "loss_tokens": loss_tokens, "loss_tokens_per_sec": loss_tokens / train_time if train_time else None,
        "min_tokens_per_sec_to_train_0p6b_locally": threshold,
        "train_0p6b_locally": (args.model.endswith("0.6B") and train_time > 0 and processed / train_time > threshold),
        "train_loss_first_last": [reports[0]["train_loss"], reports[-1]["train_loss"]] if reports else None,
        "val_loss": vals,
        "format_check_on_val": {"instances": len(val_raw), **{k: s[k] for k in ("finni_f1", "finni_precision",
                                "finni_recall", "parse_failure_rate", "malformed_items")},
                                "generation_tokens_per_sec": gen_tokens / gen_time if gen_time else None,
                                "note": "pilot format check on dev instances; not a reported result"},
        "reports": reports}
    out = config.REPO_ROOT / "data" / f"phase2_mac_pilot_{tag}.json"
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "reports"}, indent=1))


if __name__ == "__main__":
    main()
