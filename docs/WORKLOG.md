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
