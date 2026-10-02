"""QLoRA training of a student on a Kaggle T4, resumable across sessions (PRD section 5; TRADEOFFS T-018).

Owner-run on Kaggle (OPEN O-019). Uses the shared format (xbrl_frontier.sft_format: benchmark prompt,
benchmark answer schema, completion-only loss), the fixed corpus order (no trainer shuffling), base
quantized per training.kaggle.quant_base in training.kaggle.dtype, LoRA per training.lora_*, constant LR
(training.curve_lr_schedule), training.epochs epochs, global batch training.global_batch_instances.

Curve checkpoints (--curve): adapters saved when epoch 1 first covers each curve point, counted in DISTINCT
instances of the fixed order (training.schedule; T-016). Resume checkpoints: full trainer state every
training.kaggle.checkpoint_every_minutes into OUT/checkpoints; --resume DIR continues from the latest one
found there (a previous session's output attached as a Kaggle dataset).
Instances whose full sequence exceeds training.kaggle.max_seq_len_tokens are skipped, never truncated, and
counted (T-018).
--validation (PRD section 5, T-030): an independent run on exactly the first training.curve_validation_run TRAINED
instances of the fixed order (the same prefix the nested checkpoint covers), with the full recipe (seed, epochs,
LR, batch); compared with the nested checkpoint by scripts/curve_validation.py.
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from training import schedule
from xbrl_frontier import config, sft_format


def corpus_check(corpus_dir: Path) -> dict:
    rep = json.load(open(config.REPO_ROOT / "data" / "corpus_report.json"))
    out = {}
    for split in ("train", "dev"):
        h = hashlib.sha256((corpus_dir / f"{split}.jsonl").read_bytes()).hexdigest()
        if h != rep["splits"][split]["sha256"]:
            raise SystemExit(f"{split}.jsonl sha256 {h} does not match data/corpus_report.json")
        out[split] = h
    return out


def latest_checkpoint(d: Path) -> Path | None:
    cks = sorted(d.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])) if d.exists() else []
    return cks[-1] if cks else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=[s["id"] for s in config.get("pins.students")])
    ap.add_argument("--seed", type=int, required=True, choices=config.get("pins.seeds"))
    ap.add_argument("--corpus", type=Path, required=True, help="directory holding train.jsonl and dev.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", type=Path, help="previous session's OUT directory")
    ap.add_argument("--curve", action="store_true", help="save curve checkpoints (primary student, seed 1)")
    ap.add_argument("--validation", action="store_true",
                    help="independent curve-validation run on the first training.curve_validation_run trained instances")
    args = ap.parse_args()
    session_start = time.time()

    from serving.common import env_record, model_pin, pinned_snapshot, require_pins
    versions = require_pins("transformers", "peft", "bitsandbytes", "accelerate")
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import SequentialSampler
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
                              TrainerCallback, TrainingArguments)

    k = config.get("training.kaggle")
    if k["dtype"] != "fp16" or k["quant_base"] != "nf4":
        raise SystemExit("only training.kaggle.dtype=fp16 with quant_base=nf4 is implemented")
    hashes = corpus_check(args.corpus)
    args.out.mkdir(parents=True, exist_ok=True)
    ck_dir = args.out / "checkpoints"
    if args.resume:
        for sub in ("checkpoints", "curve"):
            if (args.resume / sub).exists():
                shutil.copytree(args.resume / sub, args.out / sub, dirs_exist_ok=True)
        if (args.resume / "run.json").exists() and not (args.out / "run.json").exists():
            shutil.copy(args.resume / "run.json", args.out / "run.json")
    run_path = args.out / "run.json"
    run = json.load(open(run_path)) if run_path.exists() else {"sessions": [], "curve_checkpoints": []}

    local = pinned_snapshot(args.model)
    tok = AutoTokenizer.from_pretrained(local)
    rows = [json.loads(l) for l in open(args.corpus / "train.jsonl")]
    max_len = k["max_seq_len_tokens"]
    data, skipped = [], 0
    for x in rows:
        ids, off = sft_format.encode(tok, x["text"], x["numeric_entities"])
        if len(ids) > max_len:
            skipped += 1
            continue
        data.append({"input_ids": ids, "offset": off, "id": x["id"]})
    if args.validation:
        if args.curve:
            raise SystemExit("--validation and --curve are exclusive")
        k = config.get("training.curve_validation_run")
        if len(data) < k:
            raise SystemExit(f"only {len(data)} trained instances, fewer than training.curve_validation_run={k}")
        data = data[:k]
    n = len(data)

    def collate(batch):
        L = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id
        ids = torch.full((len(batch), L), pad, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            t = torch.tensor(b["input_ids"])
            ids[i, :len(t)] = t
            att[i, :len(t)] = 1
            lab[i, b["offset"]:len(t)] = t[b["offset"]:]
        return {"input_ids": ids, "attention_mask": att, "labels": lab}

    class FixedOrderTrainer(Trainer):
        def _get_train_sampler(self, *a, **kw):
            return SequentialSampler(self.train_dataset)

    model = AutoModelForCausalLM.from_pretrained(
        local, torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.float16))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=config.get("training.lora_rank"), lora_alpha=config.get("training.lora_alpha"),
        lora_dropout=config.get("training.lora_dropout"), target_modules=config.get("training.lora_target_modules"),
        task_type="CAUSAL_LM"))

    micro = k["micro_batch"]
    targs = TrainingArguments(
        output_dir=str(ck_dir), per_device_train_batch_size=micro,
        gradient_accumulation_steps=config.get("training.global_batch_instances") // micro,
        num_train_epochs=config.get("training.epochs"), learning_rate=config.get("training.learning_rate"),
        lr_scheduler_type=config.get("training.curve_lr_schedule"), warmup_steps=0, fp16=True,
        logging_steps=10, save_strategy="no", save_total_limit=2, seed=args.seed, data_seed=args.seed,
        report_to=[], remove_unused_columns=False, dataloader_drop_last=False)

    session = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "resumed_from": None,
               "processed_tokens": 0, "train_seconds": 0.0}
    tok_len = [len(d["input_ids"]) for d in data]

    class Saver(TrainerCallback):
        def __init__(self):
            self.last = time.time()
            self.t0 = time.time()
            self.prev = None

        def on_step_begin(self, a, state, control, **kw):
            if self.prev is None:
                self.prev = state.global_step

        def on_step_end(self, a, state, control, **kw):
            g = state.global_step
            step_in_epoch = g % schedule.steps_per_epoch(n) or schedule.steps_per_epoch(n)
            lo = (step_in_epoch - 1) * config.get("training.global_batch_instances")
            hi = min(step_in_epoch * config.get("training.global_batch_instances"), n)
            session["processed_tokens"] += sum(tok_len[lo:hi])
            if args.curve:
                for hit in schedule.curve_hits(g - 1, g, n):
                    d = args.out / "curve" / f"prefix_{hit['prefix']}"
                    kw["model"].save_pretrained(d)
                    run["curve_checkpoints"].append({**hit, "global_step": g, "path": str(d.relative_to(args.out))})
                    run_path.write_text(json.dumps(run, indent=1))
            if schedule.due_for_resume_checkpoint(time.time(), self.last):
                control.should_save = True
                self.last = time.time()
            if schedule.session_time_up(time.time(), session_start):
                control.should_save = True
                control.should_training_stop = True
                session["stopped_for_session_limit"] = True

        def on_train_end(self, a, state, control, **kw):
            session["train_seconds"] = time.time() - self.t0

    trainer = FixedOrderTrainer(model=model, args=targs, train_dataset=data, data_collator=collate,
                                callbacks=[Saver()])
    resume_from = latest_checkpoint(ck_dir)
    session["resumed_from"] = resume_from.name if resume_from else None
    trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    complete = trainer.state.global_step >= trainer.state.max_steps
    final = args.out / "final_adapter"
    if complete:
        trainer.model.save_pretrained(final)
    session["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    session["processed_tokens_per_sec"] = (session["processed_tokens"] / session["train_seconds"]
                                           if session["train_seconds"] else None)
    run.update({"model": args.model, "revision": model_pin(args.model), "seed": args.seed,
                "versions": versions, "env": env_record("kaggle_t4"), "config_hash": config.config_hash(),
                "corpus_sha256": hashes, "corpus_size": len(rows), "trained_instances": n,
                "skipped_over_max_seq_len": skipped, "max_seq_len": max_len,
                "validation_run": {"first_n_trained_instances": n, "first_id": data[0]["id"], "last_id": data[-1]["id"]}
                if args.validation else None,
                "steps_per_epoch": schedule.steps_per_epoch(n), "global_step": trainer.state.global_step,
                "curve_targets": schedule.curve_targets(n) if args.curve else None,
                "complete": complete, "final_adapter": str(final.relative_to(args.out)) if complete else None})
    run["sessions"].append(session)
    run_path.write_text(json.dumps(run, indent=1))
    print(json.dumps({k2: v for k2, v in run.items() if k2 != "env"}, indent=1))


if __name__ == "__main__":
    main()
