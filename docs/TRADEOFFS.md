# Decisions log

Each entry: decision, alternatives, why. Decisions taken without the owner go under a heading containing "AUTONOMOUS DECISION - owner to review".

## T-001 Phase 0 dependency set (owner decision, 2026-10-02)

Decision: `.venv/` on Python 3.12.14 (Homebrew), with exactly these direct dependencies, pinned in
`requirements.txt` (full transitive freeze in `requirements.lock`):

| Package | Version | Reason |
|---|---|---|
| huggingface_hub | 2.1.1 | download FinNI-eval-v2 and model metadata at pinned commit SHAs |
| pyarrow | 25.0.1 | read the FinNI parquet |
| pyyaml | 6.0.3 | read `docs/config/thresholds.yaml` and the price sheet at runtime |
| pytest | 9.1.1 | oracle tests and unit tests |

Alternatives: `datasets` (heavier and pulls in pandas/multiprocess for a single parquet read); the
system Python 3.9.6 (too old for every pinned package). vLLM and GuideLLM are not installed locally.
They are installed only inside the owner-run Kaggle and WSL2 scripts, at the versions in
`pins.software`.

Approved by the owner on 2026-10-02.

## T-002 AUTONOMOUS DECISION - owner to review: FinNI scorer details the spec leaves open (2026-10-02)

FinTagging Appendix E defines a match as entity string AND type exactly equal, with set-based P/R/F1. It
leaves the remaining details unspecified (evidence_brief section 3), so they were decided as follows:

1. **Set, not multiset.** The (value, type) pairs of an instance form a set. Duplicate gold pairs
   (3,625 instances at the pin) and duplicate predictions each count once. Alternative: multiset counting,
   which would reward repeating a fact. The PRD says "set-based".
2. **Normalization (`eval.value_normalization = fintagging_prompt_rules`)**, applied identically to
   gold and predictions: remove `$`, the token `USD`, `%` and `,` (prompt rule 4); no magnitude
   expansion and no word-to-digit conversion (rule 3; the gold keeps `two`, `nil`, `—`); lowercase;
   collapse whitespace, including NBSP, and strip. Type strings are compared exactly, after stripping only.
   Alternative: also converting words to digits. Rejected because the gold does not do it, so it would
   create mismatches against correct answers.
3. **Parsing.** Remove `<think>…</think>`, take the first ```` ```json ```` fence if there is one, then
   `json.loads`. Failing that, parse the outermost `{…}` substring. Anything without a `result` list is a
   parse failure: an empty prediction, counted in `parse_failure_rate`. Items missing a string `Type`
   or a str/number `Fact` are dropped and counted as `malformed_items`, reported next to the parse-failure
   rate. Numeric `Fact`s are `str()`-ed. Nothing is retried or repaired. Alternative: strict
   whole-string JSON only. Rejected because it would score fence/prose wrapping as total failure and make
   the parse-failure rate measure formatting habit rather than extraction. Both the raw text and the
   parse outcome are stored, so a strict re-score is possible.
4. **Per-instance mean** (stored next to the global pooled score): an instance with empty gold and an
   empty prediction scores P=R=F1=1. Otherwise P or R is 0 when its denominator is 0.
5. **Oracle constructions.** "Types swapped" re-types every gold pair to the next type in a fixed cyclic
   order that is not itself gold for that value, so two gold types on one value never collide into a
   true positive. "Half dropped" keeps every other pair in one global sorted order, giving recall
   0.500002 on 238,929 raw pairs, within `eval.oracles.tolerance`.
6. **Dedup (`exact_context_and_answer`).** The first occurrence by row order is kept. Rows with the same
   context but a different answer (204 groups per the brief) are all kept, as the policy name implies.
7. **Oracle enforcement.** A versioned hook in `.githooks/pre-commit` (copied into `.git/hooks/`, with no
   change to git config) runs `tests/test_scorer_unit.py` and `tests/test_oracles.py` whenever scorer,
   config or test files are staged. `.github/workflows/oracles.yml` runs the same tests on push, for when a
   remote exists.

## T-004 AUTONOMOUS DECISION - owner to review: Phase 0 serving tooling (2026-10-02)

1. **Load tool for Phase 0: `vllm bench serve`**, with random prompts at the FinNI token shape
   (`serving.phase0_throughput_benchmark.input_tokens` / `output_tokens`, `--ignore-eos` so the output length
   is fixed), `--request-rate inf` at each `concurrency_levels` value, and `requests_per_point` prompts. A
   warm-up of `serving.warmup_requests` runs first and is discarded. Alternative: GuideLLM. It is kept for
   the Phase 4 sweeps (`serving.request_rate_sweep_rps`). Here it would add a second pinned package for no
   extra information.
2. **Goodput definition**: the highest request throughput among complete points whose p95 TTFT and p95
   E2E both meet `serving.slo`. `serving.slo` is stated at p95, so per-request SLO goodput (vLLM's
   `--goodput`) is recorded only as a secondary field. Points with failed requests are excluded from both
   maxima and counted. Filings/hour = req/s x 3600 / instances-per-filing.
3. **Instances per filing for the token shape** = `pins.eval_finni.rows_expected` /
   `contamination.eval_manifest_required_recovered`, which is the PRD section 7 basis. Note that O-005 shows
   some companies contribute more than one filing. The Phase 3 probe replaces this shape with measured
   per-filing tokens in any case.
4. **Pinned weights are served from a local `snapshot_download` at the pinned SHA**
   (`--served-model-name` keeps the public id), so the revision is enforced by path and not by a server
   flag. Scripts refuse to run unless the installed version equals `pins.software.<pkg>`, and they refuse
   if that pin is empty.
5. **Smoke-test quantization** (new keys `serving.smoke_test.smoke_quantization`): INT8 W8A8 uses
   llm-compressor `QuantizationModifier` (data-free, dynamic per-token activations). AWQ-INT4 uses AutoAWQ
   GEMM, as PRD section 5 names it, with AutoAWQ's own default calibration (pile-val), because the
   training corpus does not exist yet. These checkpoints only prove that each variant loads and runs.
   They are never scored, and Phase 2 re-quantizes the fine-tuned students with training-context
   calibration. AutoAWQ's last release (0.2.9) is 2025-05-11 and it is unmaintained, so it runs in its own
   virtualenv. Whether its GEMM format still runs on vLLM 0.30 on Turing is exactly what the smoke test
   answers. See DECISIONS NEEDED (AutoAWQ vs llm-compressor AWQ).
6. **Break-even table** (`scripts/phase0_cost.py`): every pinned arm plus the swap candidate, both
   regimes, every non-blank T4 source in `serving.hardware.kaggle_t4.market_rate_keys`, at the regime's
   utilization and at every `serving.utilization_scenarios` value. The blank AWS source is dropped and
   listed as dropped (see O-008).

## T-003 AUTONOMOUS DECISION - owner to review: eval manifest recovery and instance-to-filing map (2026-10-02)

1. **New config keys** (all numbers stay in `thresholds.yaml`): `contamination.eval_filing_window_end`
   (FinTagging v5's stated window end; the start is `train_filing_date_max` + 1 day),
   `contamination.instance_match.*` and `edgar.*` (fair-access rate below SEC's 10 req/s ceiling, the
   PRD's ~2,000-query budget, the 11-word phrase length from the PRD recipe, and retry backoff for
   429/5xx only; a 403 stops the run). They were set before any matching result was seen.
2. **The data contradicts "one 10-K per company" (data wins).** Omega Healthcare's FY2023 and FY2024
   10-Ks both supply FinNI contexts that occur in only that year's document: for example "As of December 31,
   2023, our portfolio ... consisted of 891 healthcare facilities" and "the 2024 Annual Meeting" occur only
   in the FY2023 filing. Six of the 111 seed companies show two present fiscal years. So the manifest is
   **filing-level**: a filing is in the eval set when at least
   `instance_match.min_unique_exact_for_filing` contexts match it exactly and match no other scanned
   filing. A company may have several filings. `eval_manifest_required_recovered` is read as
   **registrants** (the recovered manifest and the 142 figure both count companies).
   Alternative: force one filing per company. Rejected because it mis-assigns the other year's contexts.
3. **Matching.** Contexts and filings are tokenized identically: tags removed, `ix:header`, script and
   style removed, HTML unescaped, NBSP collapsed, lowercased, tokens `[a-z0-9]+`. **Exact** = the
   context's token string is a contiguous substring of the filing's token string. **Fuzzy** = at least
   `fuzzy_min_containment` of up to `max_shingles_per_context` evenly spaced word `shingle_words`-grams
   found. Exact beats fuzzy, then higher containment wins. A top tie between filings of one company goes
   to that company's dominant filing (most unique-exact contexts), labelled `*_tie_within_company`. A top
   tie across companies is left unassigned as `ambiguous_cross_company` (generic boilerplate and generic
   table headers). Contexts shorter than one shingle are `too_short`. Unassigned instances form the
   unassigned stratum, which is excluded from the frontier sample per PRD section 4.
   This is token-level matching, not a rebuild of the benchmark's serializer. The serializer is Phase 1
   work (forked parser). FinNI contexts contain duplicated label text (for example "Net investment income
   Net investment income"), which is why table contexts mostly land as fuzzy rather than exact.
4. **Discovery.** Seed = the 111 CIKs in `research/recovered_eval_manifest.json`. All their original 10-Ks
   in the window come from the submissions API (older pages included). After every match pass, the next
   FTS query is built from a context still unmatched (`none`), in sha256 order of its row id. The phrase is
   the 11-word window with the most 4+-letter alphabetic words (forms=10-K, window as above). A query
   counts only if every hit carries the same CIK group, and co-registrants collapse to the first CIK. A new
   CIK's in-window 10-Ks are downloaded and scanned, and matching is recomputed. Stop when the confirmed
   companies reach `eval_manifest_required_recovered`, the budget is spent, or no unmatched context is
   left. A CIK found by FTS but without a present filing is reported as unconfirmed and is not in the
   manifest.
5. **fiscal_year** = the year of the filing's `period_of_report` (the slice key).
6. **Doubled-segment normalization (added after run 1; thresholds unchanged).** Run 1 left 90.1% of rows
   assigned. The unmatched rows were 82% tables with best containment mostly 0.1-0.5 against filings
   already scanned. The cause is a FinNI serializer artifact: 59,750 of 406,686 multi-token segments
   (14.7%, in 5,927 of 25,765 contexts) are exact doubles ("Date Declared Date Declared"). A segment
   (text between tags) whose tokens are two identical halves now collapses to one half, on **both**
   sides (context and filing). This normalizes the comparison and does not change any metric or threshold.
   Run 1 and run 2 numbers are both recorded (WORKLOG). Alternative: re-implement the benchmark
   serializer now. Deferred to the Phase 1 parser fork.
7. **The FTS budget is a total across runs.** Run 2 reused run 1's queries and CIKs (`--prior`) and could
   only spend the remainder. No unqueried unmatched context was left, so it issued 4.
8. **Shipped provenance** (whitelisted in `.gitignore`): `data/manifest_work/report.json` (rates, methods,
   unconfirmed-CIK diagnostics), `fts_queries.json` (every query phrase and resolution) and
   `validation.json` (gold values found in the assigned filing). These are small and make the
   assignment-rate claim checkable. The EDGAR cache (about 1.3 GB) stays untracked and is rebuilt by
   rerunning the script.
