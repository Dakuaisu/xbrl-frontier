# Work log

Dated implementation notes: what was built, how it was verified, what is next.

## 2026-10-02 Phase 0, unit 1: environment

- Built `.venv/` (Python 3.12.14) with the owner-approved set in T-001; `requirements.txt`
  (direct pins) and `requirements.lock` (pip freeze).
- Verified: `pip freeze` shows the four pins. SEC endpoints return 200 with the new User-Agent (O-001).
- Next: pin the HF revisions and verify the FinNI row count.

## 2026-10-02 Phase 0, unit 2: HF pins

- Filled `hf_revision` for `pins.eval_finni`, both `pins.students` and `pins.encoder_baseline`
  with each repo's live `main` commit SHA from `https://huggingface.co/api/<repo>` (queried
  2026-10-01 and re-queried 2026-10-02, same SHAs). These are frozen from now on.
- Built `xbrl_frontier/config.py` (runtime config access and config hash) and `xbrl_frontier/finni.py`
  (snapshot_download at the pinned revision into git-ignored `data/hf/`; raises if the row count
  differs from `rows_expected`).
- Verified: `.venv/bin/python -m scripts.verify_finni` loads the pinned split and the row count
  matches `rows_expected`.
- Data audit at the pin (inputs to the scorer design): one prompt template; 12,310 negative rows;
  261,457 gold pairs; 3,625 instances repeat a (Fact, Type) pair; 41 distinct non-numeric gold
  values (word numbers, dashes, `nil`, `0\xa0million`); 12,758 table contexts; 25,765 unique contexts
  and 25,969 unique (context, answer) rows; `context_index` is a 0..N-1 row id.

## 2026-10-02 Phase 0, unit 3: FinNI scorer, oracles, dedup, slices

- Built `xbrl_frontier/scorer.py` (normalization, parser, global pooled P/R/F1, per-instance mean,
  parse-failure rate, malformed-item count, slice and entity_type breakdowns),
  `xbrl_frontier/oracles.py` (the four `eval.oracles` predictors) and `xbrl_frontier/eval_sets.py`
  (`eval.dedup_policy`; slice keys for context_type, company, fiscal_year, context_token_bucket).
  `train_size_point` is a run attribute and `entity_type` is computed per pair. Decisions are in T-002.
- Verified: `.venv/bin/python -m pytest -q tests` gives 17 passed. Oracles on the pinned set
  (raw 28,787 / dedup 25,969 instances): gold copy F1 1.000000; empty F1 0.000000; types swapped
  F1 0.000000; half dropped R 0.500002 with P 1.000000. All are within `eval.oracles.tolerance`.
- Enforcement: `.githooks/pre-commit` (installed into `.git/hooks/`) and `.github/workflows/oracles.yml`.
- Open: O-003 (token-bucket edges), O-004 (company/fiscal-year need the instance map).

## 2026-10-02 Phase 0, unit 5: owner-run serving tooling and break-even script

- Built `xbrl_frontier/cost.py` (arm $/filing per regime, break-even and gate-pass goodput,
  self-host $/filing), `xbrl_frontier/throughput.py` (max throughput and SLO goodput in filings/hour),
  `scripts/phase0_cost.py` (writes `data/phase0/break_even.{json,md}`; with `--results` it adds measured
  base-model throughput and a beatable flag per arm, regime and T4 source),
  `serving/{common,phase0_benchmark,quantize_base,smoke_test}.py`, `serving/run_rtx3050_smoke.sh`
  and `notebooks/phase0_kaggle_t4.ipynb`. Decisions are in T-004.
- Verified: `tests/test_cost_and_serving.py` (6 tests) checks the variant expansion against config, the
  3050 vLLM flags, the break-even/self-host identity, the blank-source handling and the goodput
  selection. `.venv/bin/python -m scripts.phase0_cost` runs on the price sheet alone and reproduces the
  PRD section 7 break-even and gate-pass figures from config keys.
- Owner-blocked: O-005 (Kaggle), O-006 (RTX 3050), both waiting on O-007 (software pins).

- 2026-10-02: observed the AWS g4dn.xlarge T4 rate at pin time (O-008 RESOLVED). `scripts/phase0_cost.py` now uses all three `market_rate_keys` sources.

## 2026-10-02 Phase 0, unit 6: Schema-Aware XBRL preprint

- Tried DOI resolution, ResearchGate, Crossref, OpenAlex, Semantic Scholar, arXiv and Exa. Only metadata is reachable; logged as O-009.

## 2026-10-02 Phase 0, unit 4: eval manifest and instance-to-filing map

- Built `xbrl_frontier/edgar.py` (declared User-Agent, `edgar.max_requests_per_second` throttle,
  URL-keyed disk cache under `data/edgar/`, retry on 429/5xx, hard stop on 403),
  `xbrl_frontier/instance_match.py` (token normalization with doubled-segment collapse, shingle index,
  exact/fuzzy assignment, unique-exact filing presence), `scripts/build_eval_manifest.py` (seed ->
  submissions -> download -> match -> FTS discovery loop -> outputs) and
  `scripts/validate_instance_map.py`. Decisions are in T-003.
- Run 1 (2026-10-02 04:37-05:11 UTC, 1,962 FTS queries, 1,864 SEC requests): 129 companies, 139
  filings, assignment 90.10% raw / 91.30% dedup.
- Run 2 (doubled-segment normalization, `--prior` run 1, 4 new queries): 129 companies, 139 filings,
  assignment 94.29% raw / 94.43% dedup. These are the shipped files: `data/eval_manifest.csv` and
  `data/eval_instance_to_filing.csv`.
- Verified: `pytest -q tests` (29 passed, including manifest-tool tests and the shipped-map slice test).
  Full-set validation is in O-011.
- Findings: O-010 (129/142), O-011 (rate below threshold), O-012 (multi-filing companies).
  O-004 is resolved.

## 2026-10-02 Phase 0 exit: owner decisions applied

- `pins.software.{vllm,llmcompressor,autoawq}` filled (O-007 RESOLVED; T-004 items 5 and 7).
- `tokenizers==0.23.2` installed and pinned. `xbrl_frontier/tokens.py` counts context tokens.
  `scripts/derive_token_bucket_edges.py` wrote `eval.context_token_bucket_edges` once (O-003 RESOLVED,
  T-001a). Verified that bucket sizes are near-equal thirds of the dedup set and that the two students'
  tokenizer files are byte-identical. The longest context is 20,575 tokens, past
  `training.kaggle.max_seq_len_tokens`; that is noted for the Phase 2 sequence-cap handling.
- Manifest shortfall accepted and disclosed (T-005, O-010 NOT MET at 129/142). Issue draft in
  `docs/owner/fintagging_issue_draft.md`. Assignment rate disclosed (T-006). Company-level bootstrap
  (T-007, O-012 RESOLVED). Unconfirmed CIKs excluded from training (T-008).
- TRADEOFFS reordered (T-003 before T-004). T-002 now quotes prompt rules (3) and (4) verbatim and
  states how negatives enter the per-instance mean.
- Kaggle source tarball (O-005 step 1): `/tmp/xbrl-frontier-src.tar.gz` built with `git archive` from
  commit 1c30e28 (pins filled, notebook accepts an auto-extracted dataset); sha256
  517747cd3b60cf3e408535a70458bb13df6f6c68b8fd8a8a150f0ce7ec411a06.

## 2026-10-02 Phase 1, unit 1: parser fork, benchmark serializer, assignment re-run

- Built `ixbrl_facts/` (no FilingQA import): `dom.py` (stdlib HTML tree with the BeautifulSoup calls the
  benchmark uses), `fintagging.py` (port of FinTagging `parse_xbrl_report.ipynb` at 653650c, with the
  2023 ∪ 2024 us-gaap taxonomy from sha256-pinned xsds), `dei.py` (forked from FilingQA 895bf6d
  `harvest_dei`). Provenance pins: `pins.fintagging_source` and `pins.filingqa_fork`. Decisions in T-009.
- Built `scripts/rebuild_eval_contexts.py` and re-ran assignment over the 139 manifest filings (cached
  EDGAR documents, no new SEC requests). Assignment is 95.25% raw / 95.39% dedup (Phase 0 token matcher:
  94.29% / 94.43%), so O-011 is RESOLVED. Label parity is 25,768/25,770. Finding O-013.
- A first parity run compared each context with only the first serialized label set and reported 180
  mismatched instances. The cause was the twin texts in O-013, not the port. Fixed by comparing against
  every candidate label set, then re-ran from the restored Phase 0 map so the before/after numbers are
  correct.
- Verified: `pytest -q tests` (36 passed, including 7 in `tests/test_ixbrl_facts.py`), plus
  `scripts/validate_instance_map.py` on the new map.
- Next: accession selection for the training corpus (filing date <= `train_filing_date_max`, excluding
  every CIK under `contamination.exclude_ciks_from`, now including the unconfirmed list).

## 2026-10-02 Phase 1, unit 2: training accessions, serialization pass, first parity comparison

- Issue draft updated to the Phase 1 figures, with O-013 (and the taxonomy question) added as
  observations. O-011 now says the token_fuzzy validation is not comparable across phases. O-013 is
  RESOLVED by T-010 (drop negative twins in training).
- Built `scripts/select_train_accessions.py` (T-011), `scripts/serialize_corpus.py` (T-010 applied) and
  `scripts/format_parity.py`. New config block `training.accession_selection`.
- Selection (2026-10-02 06:27 UTC, cached and throttled, User-Agent from `xbrl_frontier.edgar.USER_AGENT`):
  universe 46,482, eligible 45,385, accepted 208 (104 10-K / 104 10-Q; 198 train / 10 dev), 1 rejected
  (unknown concept), all quotas filled. `data/train_accessions.txt` is frozen. Office and size distribution
  against the eval manifest: `data/train_selection_report.json`.
- Serialization: 321,080 instances = 17,094 positives + 303,986 negatives. 4,511 twins dropped (median 17
  per filing, max 123). Negatives fraction after twin drop 0.9468 (0.9475 if kept), before sampling.
  Positives fall short of the target (O-014).
- First parity comparison: O-015.
- Verified: `pytest -q tests` (41 passed, including `tests/test_corpus_selection.py`).

## 2026-10-02 Phase 1, unit 3: accession extension (T-012)

- `--extend` verified the frozen rows (hash and replay), resumed at index 2,490, and added 130 filings
  (65 10-K / 65 10-Q; 7 new dev, 17 dev in total). `git diff` on `data/train_accessions.txt`: the header line
  plus 131 appended lines; the 208 frozen rows are unchanged (sha256 re-verified).
- Re-serialized all 338: 524,439 instances, 27,234 positives, 7,617 twins dropped, negatives fraction 0.9481
  before sampling.

## 2026-10-02 Phase 1, unit 4: sampled corpus, contamination audit, fresh slice

- Built `scripts/build_corpus.py` (T-013), `scripts/contamination_audit.py` (T-014) and
  `scripts/select_fresh_slice.py` (T-015). `scripts/format_parity.py` gained `--files`/`--tag`. New keys:
  `training.corpus_sampling.seed`, `contamination.minhash_shingle_words`, `contamination.minhash_seed`,
  `contamination.fresh_slice_selection`. `contamination.fresh_slice_filing_date_min` filled (2026-05-19).
- Contamination audit (pool, 215,284 instances): 18,184 flagged (ngram only 12,867, both 5,317, minhash only
  0; 2,801 positives / 15,383 negatives; 1,617 eval contexts matched). Unaudited sample (v0): 4,527 of 42,000
  flagged. Final corpus re-audit: 40,721 audited, 0 flagged.
- Final corpus (`data/corpus/{train,dev}.jsonl`, untracked; sha256 in `data/corpus_report.json`): train
  38,788 (23,273 pos, negatives fraction 0.40000, negative table share 0.2042), dev 1,933 (1,160 pos, 0.3999).
  Both below cap (O-016). No surplus.
- Parity on the sampled corpus: negatives at parity (O-015 resolved for negatives). Decontamination skews
  positives shorter and less table-heavy (O-017).
- Fresh slice: universe 560 10-Ks, 542 eligible, 30 accepted (24 by quota, 6 relaxed fill), filed 2026-05-19
  to 2026-09-29, 0 rejections. 3,434 instances (2,140 pos, negatives fraction 0.3768, 264 table negatives).
- Verified: `pytest -q tests` (see commit), final audit 0 flagged.

## 2026-10-02 Phase 1 close: O-016 decision

- T-016: train on 38,788 / dev 1,933 with config caps unchanged; curve points count distinct instances of the
  fixed order. O-016 is RESOLVED. O-017 stays OPEN (reported next to the table slice).
- `data/fresh_selection_report.json` now carries the fresh instance file's sha256
  (18b5f52c12e3bdd2da6744f9391caac4e27f0fac5f28ac717cdde21148b5ab25), with 3,434 instances, 2,140 pos and
  1,294 neg, computed from the existing file. `scripts/select_fresh_slice.py` writes that block itself.
- O-018 carries the Phase 5 contamination-statement checklist (fresh-slice relaxed fill and unfillable
  cells, audit counts, size disclosures).
- Phase 1 complete: parser fork, serializer, assignment re-run, accession selection and extension, serialization,
  format parity, sampled corpus, contamination audit and fresh slice are all built and verified.

## 2026-10-02 Phase 2, unit 1: Mac pilot, Kaggle training tooling

- Pinned `mlx_lm` 0.32.0 (T-017). Built `xbrl_frontier/sft_format.py` (one format, completion-only loss; the
  answer serializer reproduces all 28,787 FinNI gold answers byte for byte), `scripts/mac_pilot.py`,
  `training/schedule.py`, `training/kaggle_train.py` and `notebooks/phase2_kaggle_train.ipynb` (T-018).
- Pilot attempt 1 (micro-batch 4) ran out of Metal memory within 10 iterations. Attempt 2 (micro-batch 1,
  grad checkpointing) completed: 560.6 processed tokens/s against the 1,500 threshold, so the 0.6B trains on
  Kaggle (O-021). Val loss 0.879 -> 0.044, parse failures 0/200.
- Owner-blocked: O-019 (Kaggle training sessions), waiting on O-020 (training-stack pins).
- Verified: `pytest -q tests` (48 passed, including `tests/test_training_schedule.py`).
- Next: encoder baseline script (SEC-BERT token classifier on the same corpus) and the curve-checkpoint
  scoring script, both runnable once Kaggle access is approved.

## 2026-10-02 Pilot log check: one process, one uninterrupted run

During the pilot, successive reads of `data/phase2/mac_pilot_0.6B.log` seemed to go backwards (iteration
2,170, then 1,960), and I suspected a second pilot process writing the same file. Findings:
- Processes: launch attempts before 12:54 IST were (a) attempt 1, which ran out of memory and whose log was
  renamed `mac_pilot_0.6B_attempt1_oom.log` (last write 12:46:59); (b) one `nohup` launch and two foreground
  launches, each killed when its shell call timed out, before or at the first validation. Each truncated
  the log on start. PID 8675 started at 12:54:06 and truncated the log again. At 13:46:52, `ps` listed exactly
  one `scripts.mac_pilot` process, PID 8675.
- The log after 12:54 has a single writer. It holds one run header, one step-0 validation line, no NUL bytes,
  and 300 training reports at iterations 10, 20, ..., 3,000, strictly increasing. A second process writing
  after 8675's truncation would have added its own header and step-0 validation, or left NUL holes from a
  separate file offset. None are present.
- `data/phase2_mac_pilot_Qwen3-0.6B.json` was written by PID 8675. Its 300 in-memory reports (from the
  training callback, independent of the log) are contiguous, and `trained_tokens` increases monotonically.
  `adapter_config.json` is stamped 12:54:17 (8675's train start) and the final adapter 14:07:16, which is
  12:54:17 + 4,380 s wall. The JSON and the log's last write are both 14:09:04, after generation. No other
  process reached the end of a run.
- One uninterrupted training: train_time 4,196 s + two validations (91.1 s + 89.8 s) = 4,377 s, against
  wall 4,380 s. The it/s range 0.44-1.11 (median 0.73) has no step change suggesting a second GPU job.
- The backwards reading was the read side. Several polling calls were dispatched together and ran in the
  same second (13:56:41-13:56:44), so their outputs did not come back in chronological order. The file
  itself never went backwards. The pilot does not need a rerun, and 560.6 tokens/s stands.

## 2026-10-02 Phase 2: training-stack pins (T-019)

- Filled `pins.software.{transformers 5.18.0, peft 0.21.2, bitsandbytes 0.50.2, accelerate 1.15.0}`. The
  Kaggle notebook asserts installed and imported versions against the pins. O-020 RESOLVED; O-019 has only
  the sessions left (owner).

## 2026-10-02 Phase 2, unit 2: encoder baseline and checkpoint scoring tooling

- Built `xbrl_frontier/bio.py` (words, BIO alignment over the benchmark's 34,777-label space, decoding to FinNI
  pairs, chunking), `training/encoder_train.py` (Kaggle train/predict, T-020), `serving/generate_predictions.py`
  (vLLM greedy predictions for base and adapters, T-021), `scripts/score_predictions.py` (raw and dedup, per
  slice, parse-failure and not-generated rates) and two notebooks (`phase2_kaggle_encoder.ipynb`,
  `kaggle_generate_predictions.ipynb`). New keys: `training.encoder.*`, `eval.generation`.
- Corpus alignment check: 288,037 of 289,088 facts aligned (99.64%), 441 not found, 610 outside the label
  space.
- Verified: `tests/test_bio.py` (6, including chunking with the real SEC-BERT tokenizer) and
  `tests/test_score_predictions.py` (3: gold-copy/empty oracles through files on dev and the full FinNI set,
  parse failures, not-generated rows, refusal on missing predictions). No local torch install: the encoder
  trains on Kaggle per PRD section 5.
- Owner-blocked: O-019 run 4 (encoder) and the prediction runs, each with its own step list.

## 2026-10-02 Phase 2, unit 3: curve aggregation; effective max_model_len

- Built `scripts/curve_table.py` (T-022). Scorer outputs now carry `kind` and `predictions_sha256`. The vLLM runner
  records the effective `max_model_len` and refusals (T-021 addendum).
- Verified: `tests/test_curve_table.py` (non-scorer file refused; rows only for scored prefixes; exact O-016
  label; recorded instances_seen; missing listed) and `tests/test_score_predictions.py` (still calibrated).
