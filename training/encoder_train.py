"""SEC-BERT encoder baseline: token classification over the benchmark's BIO label space (PRD section 5; T-020).

Owner-run on Kaggle (OPEN O-019, run 4).
train:   pins.encoder_baseline at its pinned revision, trained on the same corpus, in the same fixed order
         (sequential sampler), with the same instance set as the students: instances whose student sequence
         exceeds training.kaggle.max_seq_len_tokens are skipped (never truncated). Contexts longer than
         training.encoder.window_subtokens are chunked into consecutive word windows, never truncated.
         Labels: xbrl_frontier.bio (first subtoken of each word carries the word's label, as in FinTagging's
         BERT/train_ner_model.py). Resume checkpoints every training.kaggle.checkpoint_every_minutes.
predict: chunk, argmax per word, decode B/I spans to (value, concept), map concepts to FinNI types; writes
         {"id", "pairs"} jsonl for scripts/score_predictions.py. No length limit: every instance is predicted.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

from training import schedule
from xbrl_frontier import bio, config, sft_format


def fresh_instances_path(corpus: Path | None) -> Path:
    """data/fresh/instances.jsonl, or the copy in the attached corpus dataset (Kaggle), checked against the sha256
    pinned in data/fresh_selection_report.json."""
    import hashlib
    local = config.REPO_ROOT / "data" / "fresh" / "instances.jsonl"
    path = local if local.exists() else (corpus or Path()) / "fresh_instances.jsonl"
    want = json.loads((config.REPO_ROOT / "data" / "fresh_selection_report.json").read_text())["instances_file"]["sha256"]
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != want:
        raise SystemExit(f"fresh instances not found or not matching the pinned sha256 ({path}); refusing")
    return path


def load_set(name: str, corpus: Path | None) -> list[dict]:
    if name == "finni":
        from xbrl_frontier import finni
        return [{"id": r.idx, "text": r.context} for r in finni.load()]
    path = (corpus or Path()) / "dev.jsonl" if name == "dev" else fresh_instances_path(corpus)
    return [{"id": x["id"], "text": x["text"]} for x in map(json.loads, open(path))]


def encode_chunks(tok, ws: list[str], label_ids: list[int] | None, window: int) -> tuple[list[dict], int]:
    """Model inputs per chunk; returns (chunks, words cut because one word exceeded the window)."""
    enc = tok(ws, is_split_into_words=True, add_special_tokens=False)
    per_word: list[list[int]] = [[] for _ in ws]
    for t, w in zip(enc["input_ids"], enc.word_ids()):
        per_word[w].append(t)
    out, cut = [], 0
    for a, b in bio.chunks([len(p) for p in per_word], window):
        ids, labels, first = [tok.cls_token_id], [-100], []
        for w in range(a, b):
            first.append(len(ids))
            ids.extend(per_word[w])
            labels.append(label_ids[w] if label_ids is not None else 0)
            labels.extend([-100] * (len(per_word[w]) - 1))
        if len(ids) > window - 1:
            cut += 1
            ids, labels = ids[:window - 1], labels[:window - 1]
            first = [f for f in first if f < window - 1]
        ids.append(tok.sep_token_id)
        labels.append(-100)
        out.append({"input_ids": ids, "labels": labels, "first_subtoken": first, "words": (a, a + len(first))})
    return out, cut


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "predict"])
    ap.add_argument("--corpus", type=Path, help="directory holding train.jsonl and dev.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--model-dir", type=Path, help="predict: trained model directory")
    ap.add_argument("--set", choices=["finni", "dev", "fresh"], default="finni")
    args = ap.parse_args()

    from serving.common import env_record, pinned_snapshot, require_pins
    versions = require_pins("transformers", "accelerate")
    import torch
    from torch.utils.data import SequentialSampler
    from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                              Trainer, TrainerCallback, TrainingArguments)

    from huggingface_hub import snapshot_download
    enc_pin = config.get("pins.encoder_baseline")
    enc_dir = snapshot_download(enc_pin["id"], revision=enc_pin["hf_revision"])
    tok = AutoTokenizer.from_pretrained(enc_dir)
    space = bio.label_space()
    id2label = {v: k for k, v in space.items()}
    e = config.get("training.encoder")
    window = e["window_subtokens"]
    args.out.mkdir(parents=True, exist_ok=True)

    if args.mode == "predict":
        from ixbrl_facts.fintagging import taxonomy
        model = AutoModelForTokenClassification.from_pretrained(args.model_dir).cuda().half().eval()
        tax = taxonomy()
        out_path = args.out / f"encoder_predictions_{args.set}.jsonl"
        n_cut = 0
        with open(out_path, "w") as f, torch.no_grad():
            for x in load_set(args.set, args.corpus):
                ws = bio.words(x["text"])
                labels = ["O"] * len(ws)
                chunks, cut = encode_chunks(tok, ws, None, window)
                n_cut += cut
                for c in chunks:
                    logits = model(input_ids=torch.tensor([c["input_ids"]]).cuda()).logits[0]
                    pred = logits.argmax(-1).tolist()
                    for k, pos in enumerate(c["first_subtoken"]):
                        labels[c["words"][0] + k] = id2label[pred[pos]]
                pairs = bio.pairs_from_spans(bio.decode(ws, labels), tax)
                f.write(json.dumps({"id": x["id"], "pairs": pairs, "chunks": len(chunks)}) + "\n")
        (args.out / f"encoder_predictions_{args.set}.meta.json").write_text(json.dumps(
            {"set": args.set, "model_dir": str(args.model_dir), "words_cut_over_window": n_cut,
             "versions": versions, "env": env_record("kaggle_t4"), "config_hash": config.config_hash()}, indent=1))
        return

    from training.kaggle_train import corpus_check, latest_checkpoint
    hashes = corpus_check(args.corpus)
    qtok = AutoTokenizer.from_pretrained(pinned_snapshot(config.get("pins.students")[-1]["id"]))
    max_len = config.get("training.kaggle.max_seq_len_tokens")
    feats, skipped, cut, align = [], 0, 0, {"facts": 0, "aligned": 0, "not_found": 0, "outside_label_space": 0}
    for x in map(json.loads, open(args.corpus / "train.jsonl")):
        ids, _ = sft_format.encode(qtok, x["text"], x["numeric_entities"])
        if len(ids) > max_len:
            skipped += 1
            continue
        ws = bio.words(x["text"])
        labels, st = bio.label_words(ws, x["numeric_entities"], space)
        for k2 in align:
            align[k2] += st[k2]
        chunks, c = encode_chunks(tok, ws, [space[l] for l in labels], window)
        cut += c
        feats += [{"input_ids": ch["input_ids"], "labels": ch["labels"]} for ch in chunks]

    ck_dir = args.out / "checkpoints"
    if args.resume and (args.resume / "checkpoints").exists():
        shutil.copytree(args.resume / "checkpoints", ck_dir, dirs_exist_ok=True)
    model = AutoModelForTokenClassification.from_pretrained(
        enc_dir, num_labels=len(space), id2label=id2label, label2id=space, ignore_mismatched_sizes=True)

    class FixedOrderTrainer(Trainer):
        def _get_train_sampler(self, *a, **kw):
            return SequentialSampler(self.train_dataset)

    class Saver(TrainerCallback):
        last = time.time()

        def on_step_end(self, a, state, control, **kw):
            if schedule.due_for_resume_checkpoint(time.time(), Saver.last):
                control.should_save = True
                Saver.last = time.time()

    t0 = time.time()
    trainer = FixedOrderTrainer(
        model=model, train_dataset=feats, data_collator=DataCollatorForTokenClassification(tok),
        callbacks=[Saver()],
        args=TrainingArguments(output_dir=str(ck_dir), per_device_train_batch_size=e["micro_batch"],
                               num_train_epochs=e["epochs"], learning_rate=e["learning_rate"], fp16=True,
                               logging_steps=50, save_strategy="no", save_total_limit=2,
                               seed=config.get("pins.seeds")[0], report_to=[], dataloader_drop_last=False))
    ck = latest_checkpoint(ck_dir)
    trainer.train(resume_from_checkpoint=str(ck) if ck else None)
    final = args.out / "final_model"
    trainer.save_model(final)
    tok.save_pretrained(final)
    run = {"model": enc_pin["id"], "revision": enc_pin["hf_revision"], "versions": versions,
           "env": env_record("kaggle_t4"), "config_hash": config.config_hash(), "corpus_sha256": hashes,
           "skipped_over_max_seq_len": skipped, "train_chunks": len(feats), "words_cut_over_window": cut,
           "alignment": align, "label_space_size": len(space), "resumed_from": ck.name if ck else None,
           "train_seconds_this_session": time.time() - t0, "final_model": str(final.relative_to(args.out))}
    (args.out / "run.json").write_text(json.dumps(run, indent=1))
    print(json.dumps({k: v for k, v in run.items() if k != "env"}, indent=1))


if __name__ == "__main__":
    main()
