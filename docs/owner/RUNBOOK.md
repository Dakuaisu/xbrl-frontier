# Owner runbook: every OWNER-BLOCKED item, in dependency order

`docs/OPEN.md` is the source of truth for each item (status, evidence, full step text). This runbook orders the items
and says, for each one, what to run, what comes back, where it goes, and which builder script runs on the return.
**Update 2026-10-02 (owner decision T-033):** sections B, C1, D and E5 are now run by the builder with the
`kaggle` CLI, and E1-E4 are run by the builder for `gemini-3.5-flash-lite` only (free tier, `--approval T-033`,
synchronous, $0 billed; the other arms are not run). Both are blocked first on **O-025: phone-verify the Kaggle
account** (https://www.kaggle.com/settings). Still owner-only: that verification, A1-A4 (preprint, GitHub issue after
the owner approves the text, both label-audit passes) and the RTX 3050 rows (C2, E6).

Paths are relative to the repo root. "Builder" means I run it locally once the files are in place. It spends nothing.

## A. Anytime (no dependencies)

| # | Item | You run | Comes back / where | Builder runs on return |
|---|---|---|---|---|
| A1 | O-009 preprint | Open DOI 10.13140/RG.2.2.19518.86080 in a logged-in browser (or email the authors); save the PDF **outside** the repo | Tell me the PDF path | Summary against PRD section 1; repositioning if needed |
| A2 | O-010 (a) GitHub issue | Post `docs/owner/fintagging_issue_draft.md` at github.com/The-FinAI/FinTagging/issues (add the repo link where marked) | The issue URL; any reply with the 142 filings | On a reply: manifest rebuild (`scripts/build_eval_manifest.py` from their list), `scripts/rebuild_eval_contexts.py`, report old and new numbers |
| A3 | O-024 label audit, pass 1 | Fill `data/label_audit/pass1_sheet.csv` (verdict, missing/spurious/wrong-type counts, notes) against each `filing_url`; commit | The committed sheet | (after A4) `scripts/label_audit.py score` |
| A4 | O-024 label audit, pass 2 | At least one week after A3: fill `data/label_audit/pass2_sheet.csv` blind; commit | The committed sheet | `scripts/label_audit.py score`, then `scripts/contamination_statement.py` and `scripts/readme.py` |

## B. Kaggle setup (once, then again whenever the source changes)

| # | Item | You run | Notes |
|---|---|---|---|
| B1 | Source dataset (O-005 step 1, O-019 step 1) | `git archive --format=tar.gz -o /tmp/xbrl-frontier-src.tar.gz HEAD`, uploaded as a new version of the Kaggle dataset `xbrl-frontier-src` | Rebuild after any commit the run needs (e.g. after the frontier sample is pinned, for D1/E1) |
| B2 | Corpus dataset (O-019 step 2) | Upload `data/corpus/train.jsonl`, `data/corpus/dev.jsonl` and `data/fresh/instances.jsonl` **renamed** `fresh_instances.jsonl` as the private dataset `xbrl-frontier-corpus` | Scripts check the hashes in `data/corpus_report.json` and `data/fresh_selection_report.json` |

## C. Phase 0 hardware checks (need B1)

| # | Item | You run | Comes back / where | Builder runs on return |
|---|---|---|---|---|
| C1 | O-005 T4 benchmark + smoke quantization + T4 smoke test | `notebooks/phase0_kaggle_t4.ipynb` (GPU T4, Internet on, attach `xbrl-frontier-src`) | `phase0_kaggle_t4.tar.gz`, `smoke_kaggle_t4.tar.gz`, `smoke_checkpoints.tar.gz` -> `data/phase0/` | `scripts/phase0_cost.py --results data/phase0/phase0_kaggle_t4`; `serving.smoke_test.failure_action` applied to failed variants |
| C2 | O-006 RTX 3050 smoke test (needs C1's `smoke_checkpoints`) | WSL2: `serving/run_rtx3050_smoke.sh /path/to/smoke_checkpoints` | `data/phase0/rtx3050_wsl2.tar.gz` | failure_action applied; serving table updated |

## D. Phase 2 training and scoring (need B1, B2; O-019)

| # | Item | You run | Comes back / where | Builder runs on return |
|---|---|---|---|---|
| D1 | Zero-shot base, 1.7B and 0.6B | `notebooks/kaggle_generate_predictions.ipynb`: `MODEL` each, `SET='finni'`, `BASE=True`, `ADAPTERS={}` | `predictions_base_finni.jsonl` + meta -> `data/phase2/kaggle/predictions/zeroshot-1p7b/`, `.../zeroshot-0p6b/` | `scripts/score_predictions.py --set finni`; later `--ids data/frontier_sample_ids.txt` |
| D2 | Run 1: 1.7B seed 1, curve | `notebooks/phase2_kaggle_train.ipynb`: `MODEL='Qwen/Qwen3-1.7B'`, `SEED=1`, `CURVE=True`; resume across sessions | `run.json`, `curve/`, `final_adapter/` -> `data/phase2/kaggle/run-Qwen3-1.7B-s1/` | none yet (D3, D4) |
| D3 | Curve checkpoints on dev | `kaggle_generate_predictions.ipynb`: `SET='dev'`, `ADAPTERS` = run 1's `curve/prefix_*` + `final_adapter` | `predictions_<name>_dev.jsonl` -> `data/phase2/kaggle/predictions/` | `scripts/score_predictions.py --set dev` per file, then `scripts/curve_table.py`, then `scripts/charts.py curve` |
| D4 | Run 1 full set | `kaggle_generate_predictions.ipynb`: `SET='finni'`, `ADAPTERS={'1p7b-s1': final_adapter}` | predictions -> `data/phase2/kaggle/predictions/` | `scripts/score_predictions.py --set finni`, then `scripts/power_check.py`, then `scripts/draw_frontier_sample.py` (pins the sample, latency ids, probe filings). **Commit, then rebuild B1.** |
| D5 | Run 5: curve validation (cut-order item 4) | `phase2_kaggle_train.ipynb`: 1.7B, `SEED=1`, `CURVE=False`, `VALIDATION=True` | `run-Qwen3-1.7B-s1-validation/` | D3-style dev predictions for its `final_adapter` and run 1's `prefix_10000`, then `scripts/curve_validation.py` |
| D6 | Run 2: 1.7B seed 2 (first in the cut order) | `phase2_kaggle_train.ipynb`: `SEED=2`, `CURVE=False` | `run-Qwen3-1.7B-s2/` | full-set predictions as D4, `scripts/score_predictions.py` |
| D7 | Run 3: 0.6B seed 1 (required, O-021) | `phase2_kaggle_train.ipynb`: `MODEL='Qwen/Qwen3-0.6B'`, `SEED=1`, `CURVE=False` | `run-Qwen3-0.6B-s1/` | full-set predictions as D4, `scripts/score_predictions.py` |
| D8 | Run 4: encoder | `notebooks/phase2_kaggle_encoder.ipynb`: `MODE='train'`, then `MODE='predict'` with `SET` = `finni`, `dev`, `fresh` | `run.json`, `encoder_predictions_<set>.jsonl` -> `data/phase2/kaggle/encoder/` | `scripts/score_predictions.py` per set |
| D9 | Fresh-slice student predictions | `kaggle_generate_predictions.ipynb`: `SET='fresh'`, the final adapters | predictions -> `data/phase2/kaggle/predictions/` | `scripts/score_predictions.py --set fresh` |
| D10 | Export: 1.7B s1 (and 0.6B for the 3050) | `notebooks/phase2_kaggle_export.ipynb`: `MODEL`, `ADAPTER`; merge -> int8 -> awq | `export.json`, `merged-fp16/`, `int8-w8a8/`, `awq-int4/` as a dataset | (D11) |
| D11 | Exported variants on the full set | `kaggle_generate_predictions.ipynb`: `SET='finni'`, `CHECKPOINT='<variant>=<dir>'`, one run per variant | predictions -> `data/phase2/kaggle/predictions/` | `scripts/score_predictions.py`, `scripts/flip_rate.py` (fp16 vs each); variants failing `gates.quantized_*` are dropped before E3 |

## E. Phase 3-4 (need D4's pinned sample; O-022, O-023)

| # | Item | You run (on the Mac) | Comes back / where | Builder runs on return |
|---|---|---|---|---|
| E1 | Probe, each arm (`gpt-6-luna`, `gemini-3.5-flash-lite`, `gpt-5.4-mini-2026-03-17`, or the arms in `data/phase3/power_check.json`) | Dry run `.venv/bin/python -m frontier.harness run --step probe --arm <arm> --accession <data/frontier_sample.json probe_filing_per_arm[arm]>`; add `## T-NNN` with `PRE-APPROVED SPEND: step=probe arm=<arm> usd_max=<x>` to TRADEOFFS and commit; export the key; rerun with `--live --approval T-NNN` | `data/phase3/live/probe_<arm>/` (usage, model field), ledger row | Re-plan N from measured tokens (`scripts/frontier_plan.py`) |
| E2 | Batch, each arm (after its probe) | Same pattern: `--step batch --accessions data/frontier_sample_accessions.txt`, approval line `step=batch`; then `collect` until done | `data/phase3/live/batch_<arm>/predictions.jsonl`, usage | `scripts/score_predictions.py --set finni --ids data/frontier_sample_ids.txt` per arm |
| E3 | Synchronous latency sample, each arm | `--step latency --ids data/frontier_latency_ids.txt`, approval line `step=latency` | `data/phase3/live/latency_<arm>/usage.jsonl` (latency, model field) | latency table, frontier online $/filing (`scripts/cost_model.py`) |
| E4 | Fresh-slice control (`gpt-6-luna`, after its probe) | `--step fresh_control`, approval line `step=fresh_control`; then `collect` | `data/phase3/live/fresh_control_gpt-6-luna/predictions.jsonl` | `scripts/score_predictions.py --set fresh` |
| E5 | T4 load tests (variants that passed D11) | `notebooks/phase4_kaggle_load.ipynb` (source from a commit containing `data/frontier_latency_ids.txt`) | `load-kaggle_t4-<model>.tar.gz` -> `data/phase4/kaggle_t4/` | `scripts/cost_model.py`, `scripts/charts.py pareto` and `latency` |
| E6 | RTX 3050 load tests | WSL2: `serving/run_rtx3050_load.sh <export 0.6B> <export 1.7B>` | `data/phase4/rtx3050_wsl2/` | `scripts/charts.py latency` (no dollars) |

## F. Closing (builder, after everything above)

`scripts/gate_summary.py` (every gate), `scripts/contamination_statement.py`, `scripts/readme.py`. Every blank in the
README fills only from these result files.
