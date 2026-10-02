# Decisions log

Each entry: decision, alternatives, why. Decisions taken without the owner go under a heading containing "AUTONOMOUS DECISION - owner to review".

## T-001 Phase 0 dependency set (approved on the owner's behalf in the resume message, 2026-10-02; T-033 item 7)

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

Approved on the owner's behalf in the resume message, 2026-10-02 (T-033 item 7).

### T-001a AUTONOMOUS DECISION - owner to review: `tokenizers` and the context-token buckets (2026-10-02)

The supervisor approved (supervisor decision, owner to review) `tokenizers==0.23.2` (pinned in `requirements.txt` / `requirements.lock`) for the
`context_token_bucket` slice. `transformers` is not installed locally. Counts use the primary student's
`tokenizer.json` (`Qwen/Qwen3-1.7B` at its pinned `hf_revision`). Both Qwen3 students ship byte-identical
`tokenizer.json`, `tokenizer_config.json`, `vocab.json` and `merges.txt` at their pinned revisions (sha256
prefixes `aeb13307a71acd8f`, `d5d09f07b48c3086`, `ca10d7e9fb3ed185`, `8831e4f1a0444713`), so one count serves
both. Counts are the context string alone, with no prompt template and no special tokens.
`eval.context_token_bucket_edges` are the tercile boundaries over the deduplicated pinned set, written once by
`scripts/derive_token_bucket_edges.py` (the derivation is in the key's comment; the script refuses to
overwrite). Alternatives: fixed round-number edges (arbitrary), or counting with the full prompt
(adds a constant and blurs the slice). Why: terciles give equal-mass slices that do not depend on a guess.

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
   The benchmark prompt's rules, verbatim from every FinNI-eval-v2 `query` at the pinned revision:
   rule (3): "If a number is followed by a magnitude term (e.g., Hundred, Thousand, Million, Billion), do not
   expand it into the full numerical value. * "Two hundred" → Extract only "two", not "200". * "10.6
   million" → Extract only "10.6", not "10,600,000"." Rule (4): "Standardize numerical formatting by
   removing currency symbols (e.g., "USD"), percentage signs ("%"), and commas (",") while preserving the
   numeric value. These elements must be removed to ensure consistency." Removing `USD` follows rule (4)'s
   own example, and removing `$` follows its "currency symbols".
3. **Parsing.** Remove `<think>…</think>`, take the first ```` ```json ```` fence if there is one, then
   `json.loads`. Failing that, parse the outermost `{…}` substring. Anything without a `result` list is a
   parse failure: an empty prediction, counted in `parse_failure_rate`. Items missing a string `Type`
   or a str/number `Fact` are dropped and counted as `malformed_items`, reported next to the parse-failure
   rate. Numeric `Fact`s are `str()`-ed. Nothing is retried or repaired. Alternative: strict
   whole-string JSON only. Rejected because it would score fence/prose wrapping as total failure and make
   the parse-failure rate measure formatting habit rather than extraction. Both the raw text and the
   parse outcome are stored, so a strict re-score is possible.
4. **Per-instance mean** (stored next to the global pooled score): an instance with empty gold and an
   empty prediction scores P=R=F1=1. Otherwise P or R is 0 when its denominator is 0. The per-instance
   mean therefore **includes the negative instances**, each scored 1 when correctly left empty (and 0
   when anything is predicted). The global pooled score is unaffected by correct negatives.
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

Confirmed as written on 2026-10-02 (supervisor decision, owner to review). It stays AUTONOMOUS for owner review.

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
   `contamination.eval_manifest_required_recovered`, which is the PRD section 7 basis. Note that O-012 shows
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
   answers. Supervisor decision (owner to review), 2026-10-02: keep AutoAWQ, as PRD section 5 names it, with no switch before the
   T4 smoke test exists. If AutoAWQ GEMM fails there, the failure is recorded per
   `serving.smoke_test.failure_action` and the llm-compressor AWQ question is raised then, as a data-wins
   entry.
   Status 2026-10-02 (C1): AWQ int4 is NOT TESTED, not removed. The AWQ virtualenv failed at ensurepip, so no checkpoint was
   built and vLLM never served it. The rerun with virtualenv is O-029.
6. **Break-even table** (`scripts/phase0_cost.py`): every pinned arm plus the swap candidate, both
   regimes, every non-blank T4 source in `serving.hardware.kaggle_t4.market_rate_keys`, at the regime's
   utilization and at every `serving.utilization_scenarios` value. The blank AWS source is dropped and
   listed as dropped (see O-008).
7. **Software pins (supervisor decision (owner to review), 2026-10-02, O-007).** `pins.software.vllm = 0.30.0`,
   `llmcompressor = 0.14.0`, `autoawq = 0.2.9`. Only these three are filled now. Every other
   `pins.software` field is filled when the phase that first installs it arrives, and is never edited
   after. `transformers`, `torch` and `compressed-tensors` are **not pinned**: they are left to vLLM's and
   llm-compressor's resolvers, and the installed versions are captured in every run's `env` record
   (`serving.common.env_record`). Alternative: pin them too. Rejected because a hand-picked combination
   risks a resolver conflict with vLLM's own pins, and the `env` record keeps every run reproducible.

## T-005 AUTONOMOUS DECISION - owner to review: manifest shortfall accepted at 129 of 142 (2026-10-02)

Supervisor decision (owner to review): option (c), accept 129 recovered registrants and disclose them.
`contamination.eval_manifest_required_recovered` is **NOT MET** (129/142) and the threshold is unchanged.
Option (a), the public GitHub issue to the FinTagging authors, is OWNER-BLOCKED. The draft is
`docs/owner/fintagging_issue_draft.md`. Option (b), a second FTS budget with a new phrase strategy, is
**rejected**. Why: `edgar.fts_query_budget` was set before any result and is not raised after seeing
results. The diagnostics also show the remaining `none` rows are mostly tables, which phrase search cannot
resolve, and no unqueried unmatched context was left. The README and resume bullet will say 129 of 142.

## T-006 Instance assignment below threshold: disclosed, not retuned (supervisor decision (owner to review), 2026-10-02; SUPERSEDED by O-011)

Superseded by O-011 (T-033 item 7): with the benchmark's own serializer the rate is 95.25% raw / 95.39% dedup,
which meets `contamination.instance_assignment_min_fraction`. The text below is the original record.

The assignment rate stays reported as below `contamination.instance_assignment_min_fraction`, disclosed,
with the threshold unchanged and `instance_match.*` values untouched. PRD section 4 prescribes rebuilding
contexts "with the benchmark's own serialization" before matching, and the Phase 0 token matcher is a
stand-in for that. The in-spec next step is to re-run assignment with the Phase 1 serializer and record
both rates.

## T-007 AUTONOMOUS DECISION - owner to review: bootstrap resampling unit is the company (CIK) (2026-10-02)

Supervisor decision (owner to review): the resampling cluster is the company (CIK), for the full set and for the shared frontier
sample. `between_filing_f1_sd` and `frontier.power_check` are computed **per company**, and that is stated
next to the number wherever it appears. Phase 3 draws the frontier sample so that a company is never split
across in/out of the sample (mechanics decided in Phase 3). Alternatives: resample filings, as PRD
sections 3 and 7 word it. Why: data wins over "one 10-K per company" (O-012). Ten companies contribute
two fiscal years, their filings are correlated, and 1,183 rows tie between two years of the same company,
so attributing those ties to a filing is arbitrary. The company cluster is the conservative choice.

## T-008 AUTONOMOUS DECISION - owner to review: unconfirmed FTS CIKs are excluded from training (2026-10-02)

Supervisor decision (owner to review): the 15 CIKs that FTS resolved but that have no present filing (O-010) are written to
`data/eval_cik_candidates_unconfirmed.txt` (shipped, whitelisted in `.gitignore`), and that file is added to
`contamination.exclude_ciks_from`. Alternative: exclude only the 129 confirmed registrants. Why: an FTS
phrase from an evaluation context resolved to each of them, so some may be among the 13 missing
registrants. Excluding them is the conservative side of contamination control.

## T-009 AUTONOMOUS DECISION - owner to review: parser fork and benchmark serializer (2026-10-02)

1. **FilingQA fork.** Source: `/Users/apple/FillingQA` at commit `895bf6d7d1b4c94254c64b378d02c349d1e7399b`
   (`pins.filingqa_fork`), read with `git show <commit>:<path>`. FilingQA's working tree has uncommitted
   changes, and they were not used. Nothing in FilingQA was modified, and nothing imports it. Forked:
   `api/parse/ixbrl.py::harvest_dei` -> `ixbrl_facts/dei.py` (dei fiscal labels from the whole document,
   including `ix:hidden`), adapted to the stdlib tree. Not forked: FilingQA's offset-preserving text walker,
   contexts/units and number parsing. They serve FilingQA's QA grounding, while FinTagging's labels are raw
   value strings with no period, unit or offset, so carrying them would be unused code.
2. **Serializer = FinTagging's own code, ported.** `ixbrl_facts/fintagging.py` ports `extract_text` /
   `extract_table` from `parse_xbrl_report.ipynb` at The-FinAI/FinTagging commit
   `653650cbbab525227d20228c94a65600c1209c01` (`pins.fintagging_source`), keeping the control flow line for
   line. That includes the quirks: cells join `get_text` of every nested p/div/span, which is the doubled
   text; text facts come only from direct-child leaf `ix:nonFraction`; negatives do not enter the seen-text
   set; an unknown concept raises (`unknown="raise"`, the notebook's behaviour) or is listed
   (`unknown="record"`). PRD section 4 asks for "the benchmark's own serialization", and this is it.
3. **No new dependency.** The notebook uses BeautifulSoup over lxml's HTML parser. `ixbrl_facts/dom.py`
   reproduces the parts of that tree and API the serializer uses, on stdlib `html.parser`: lowercased
   prefixed names, decoded entities, comments and script/style text excluded, void elements, and libxml2's
   p-auto-close. Alternative: ask for `beautifulsoup4` + `lxml`. Not needed: the port reproduces 93.1% of
   all FinNI rows as exact strings, and label parity on those is 25,768/25,770 (O-011, O-013). If Phase 1
   parity checks show a parser-driven divergence, the bs4+lxml question is raised then with the evidence.
4. **Taxonomy = the 2023 ∪ 2024 us-gaap xsd element types** (`pins.fintagging_source.taxonomy_xsd`, sha256
   verified on load; the later year wins on the one non-numeric type conflict). The notebook loads
   `us-gaap-2024_taxonomy.json`, which the repo does not ship. The repo's 2024 xsd alone (17,388 elements,
   identical to `research/bm25.jsonl`) lacks `StockRepurchaseProgramAuthorizedAmount1`, which an
   evaluation filing (Omega FY2023) uses. Under the notebook's KeyError rule, that filing could not have
   been in the benchmark. The union has exactly 17,688 elements, the paper's own count. With it, 0 of the
   139 eval filings hit an unknown concept. Alternative: 2024 only. Rejected because of that
   contradiction.
5. **Assignment with the serializer** (`scripts/rebuild_eval_contexts.py`): exact string equality between
   a FinNI context and a serialized context of a manifest filing comes first. Ties within a company go to
   the filing with the most serializer-unique contexts, and ties across companies are ambiguous. Remaining
   rows fall back to the unchanged Phase 0 token matcher over the same 139 filings. `instance_match.*` is
   untouched. The Phase 0 map is kept as `data/manifest_work/eval_instance_to_filing_phase0_tokenmatch.csv`,
   and both rates are recorded.

## T-010 AUTONOMOUS DECISION - owner to review: drop the negative twin in training (2026-10-02)

Supervisor decision (owner to review), O-013: within one filing, when the serializer emits a text that also appears with a
non-empty label set, the empty-labelled copy is **not** written to the training corpus. Dropped twins are
counted per filing and in total, and the corpus negatives fraction is computed after dropping.
Alternative: keep both copies, faithful to the benchmark's conventions. Why rejected: the negative label on
that text is demonstrably wrong. The same text's facts are tagged, and the empty label is an artifact of the
`div` wrapper, not an absence of facts. Training on a known-false label is noise, not fidelity. The
benchmark's conventions stay in the serializer (`ixbrl_facts/fintagging.py`, unchanged). The rule is
applied only when the corpus is written, the eval set is untouched, and the format-parity script reports
the divergence.

## T-011 AUTONOMOUS DECISION - owner to review: training accession selection rule and seed (2026-10-02)

Selected once by `scripts/select_train_accessions.py` into `data/train_accessions.txt`
(`pins.train_accessions`). The script refuses to overwrite it. The rule:

1. **Universe**: EDGAR monthly XBRL RSS (`/Archives/edgar/monthly/xbrlrss-YYYY-MM.xml`), original `10-K` and
   `10-Q` only (no amendments) whose primary document is inline XBRL, filed in
   [`training.accession_selection.filing_date_min`, `contamination.train_filing_date_max`]. That window is
   two years ending at the cutoff, matching the length of the eval window. 46,482 filings.
2. **Exclusions**: every CIK in the three `contamination.exclude_ciks_from` files (129 + 11 + 15 = 151 CIKs)
   leaves 45,385 eligible. At most `max_filings_per_cik` (1) filing per CIK, so the dev filings are disjoint
   from the train filings by construction.
3. **Size**: `ceil(target_train_instances / ipf)` train + `ceil(dev_slice_instances / ipf)` dev filings =
   198 + 10, where ipf is the **planning** instances-per-filing `pins.eval_finni.rows_expected /
   contamination.eval_manifest_required_recovered` (the eval set's figure, as the owner directed), split by
   `form_share` (50/50; the PRD names both forms and gives no ratio).
4. **Strata**: SEC industry office (`ownerOrg`) x filer size (`category`, collapsed to large accelerated /
   accelerated / other), both read from the submissions API for eval and training CIKs alike. Quotas follow
   the eval manifest's 139-filing distribution, allocated by largest remainder per form. The RSS carries its
   own office field, but it uses filing-time office names (no "Industrial Applications and Services" in
   2022). Mixing the two sources would have left an eval cell unfillable, so both sides use the same
   source. Caveat: `category` and `ownerOrg` are the current values, not values as of the filing date.
5. **Walk**: one shuffle of the eligible list with `random.Random(training.accession_selection.seed)`.
   A candidate is accepted when its (form, office, size) cell has room and it serializes under the
   benchmark's rule: no unknown us-gaap concept (the notebook's KeyError skip) and at least one positive
   instance. Rejections are counted (1 unknown-concept rejection). Every quota was filled.
6. **Dev filings**: `random.Random(seed + 1).sample(accepted, n_dev)`.
Alternatives: SIC divisions as the sector key (not what the eval's SEC-office data offers); a 10-K-only
corpus (the PRD names both forms); seeding from `pins.seeds` (those are training seeds). Report:
`data/train_selection_report.json`.

## T-012 AUTONOMOUS DECISION - owner to review: extend the training accession list once (2026-10-02)

Supervisor decision (owner to review), O-014 option a: continue the **same** seeded walk under the same rule, quotas and 50/50
`form_share`, appending only. `scripts/select_train_accessions.py --extend` (1) checks the column header,
the row count and the sha256 of the 208 frozen rows (`training.accession_selection.frozen_rows`) and refuses
on any mismatch; (2) replays the original walk and refuses unless it reproduces exactly those 208
accessions; (3) continues from the stopping point (eligible index 2,490). Candidates the first walk skipped
because their cell was full are not revisited. The extension is sized from the **measured** per-form
positive yields of the first 208 (10-K 108.37, 10-Q 56.00 positives per filing; 50/50-weighted 82.18) to
`(target_train_instances + dev_slice_instances) x (1 - negatives_fraction) x (1 + extension_margin)` =
25,200 x 1.10 = 27,720 positives. The first 208 had 17,094, so 130 more filings (65 + 65), with quotas
allocated over the eval distribution exactly as before. Dev: `dev_slice_instances x (1 - negatives_fraction) x
(1 + margin) / weighted yield` = 17 dev filings in total. The frozen rows keep their 10, and the 7 extra
are drawn with the `seed + 1` rule (`random.Random(seed + 1).sample`) **over the extension rows**. That
departs from "over the extended list", because redrawing over all 338 would change the `split` column of
frozen rows, which must stay byte-identical. One filing per CIK keeps dev disjoint from train. Training
uses the first `target_train_instances` of one seeded order, and the surplus is listed, not trained on
(T-013). The header now reads "selected 2026-10-02, extended once 2026-10-02 (T-012); the first 208 rows
are frozen and hash-verified". Result: 130 accepted, 1 rejected (no positive instance), all quotas filled;
the 338 filings serialize to 27,234 positives. Both sizes and the yields are in
`data/train_selection_report.json` (`extension`).
Alternatives: (b) keep 208 and shrink `target_train_instances` / `curve_points`. Rejected because those are
spec and the shortfall is a planning-figure error (rows/142 counts the eval's sampled negatives), so the
measured yield wins. (c) A higher 10-K share for the extension. Rejected because it puts two rules in one
corpus.

## T-013 AUTONOMOUS DECISION - owner to review: sampled training corpus (2026-10-02)

`scripts/build_corpus.py`, per split (train / dev from `pins.train_accessions`):
1. Every positive. Negatives are drawn only from those passing FinTagging's own negative filters
   (`sample_pos&neg_data.ipynb` at `pins.fintagging_source.commit`: text 100 < len < 500 characters, tables
   with at least 50 digits). No further matching knobs (owner instruction).
2. `n_neg = round(n_pos x negatives_fraction / (1 - negatives_fraction))`, split by context type so the
   negatives' table share equals the eval set's, measured at run time on the FinNI dedup negatives (0.2042).
   If the table pool is short, every pool table is taken and the shortfall is reported.
3. Positives and drawn negatives go into one order shuffled with `random.Random(f"{training.corpus_sampling.seed}:{split}")`.
   Train uses the first `target_train_instances`, dev the first `dev_slice_instances`, and the rest is listed
   in `data/corpus/surplus_ids.txt`, not trained on. This order is also the fixed order for the nested
   learning-curve prefixes.
4. `--exclude` removes instance ids before the draw. The final corpus excludes every contamination flag
   (T-014). An unaudited build (`--tag v0_unaudited`) is kept only to report "before removal".
Alternatives: negatives drawn without FinTagging's filters (diverges from how the eval negatives were made);
FinTagging's per-filing cap of negatives at the number of positives (it does not reach the configured
fraction).

## T-014 AUTONOMOUS DECISION - owner to review: contamination audit implementation (2026-10-02)

`scripts/contamination_audit.py`, stdlib only (no new dependency):
1. **Normalization** (lm-eval-harness decontamination style): HTML tags replaced by spaces (otherwise
   `<td>`-only grams would flag every table), entities decoded, lowercased, ASCII punctuation removed,
   whitespace split.
2. **n-gram**: a training instance is flagged if any `contamination.ngram_n`-gram occurs in any FinNI eval
   context (all 25,765 distinct contexts, not just dedup). Texts shorter than n count as one gram of
   their whole length, on both sides, as lm-eval does.
3. **MinHash**: `minhash_num_perm` universal-hash permutations (mod 2^61-1, parameters from
   `random.Random(contamination.minhash_seed)`) over word `minhash_shingle_words`-grams (new key: 5, the
   Lee et al. dedup shingle). LSH bands x rows = 9 x 13 from datasketch's optimal-parameter rule
   (equal FP/FN weights) at `minhash_jaccard_threshold`. A candidate is flagged if its estimated Jaccard is
   at least the threshold. Alternative: the `datasketch` package (a new dependency, not needed).
4. **Unit**: a training *instance* (PRD section 4 "every training instance"). Audited set: every
   serialized instance that can enter the corpus (positives plus filter-passing negatives). Flagged
   instances are excluded before sampling, and the final corpus is re-audited to show 0.
Alternative: flag whole filings when any instance is flagged. Rejected because boilerplate 13-grams would
remove most filings.

## T-015 AUTONOMOUS DECISION - owner to review: fresh slice (2026-10-02)

1. **Cutoffs** (vendor documentation, checked 2026-10-02): `gpt-6-luna` "May 18, 2026 knowledge cutoff"
   (developers.openai.com/api/docs/models/gpt-6-luna); `gpt-5.4-mini-2026-03-17` "Aug 31, 2025"
   (developers.openai.com/api/docs/models/gpt-5.4-mini); `gemini-3.5-flash-lite` "March 2026"
   (deepmind.google model card, which adds that some domains stop at January 2025). The latest is
   2026-05-18, so `contamination.fresh_slice_filing_date_min` = 2026-05-19. Month-only "March 2026" is
   earlier than May in any reading.
2. **Selection** (`scripts/select_fresh_slice.py`, new key `contamination.fresh_slice_selection`): 10-K only,
   like the eval. Window [2026-05-19, selection date 2026-10-02]. CIKs outside the eval manifest, all
   exclusion files and every training CIK (so the gap is not confounded by company overlap with
   training). One filing per CIK, the same strata and serializer acceptance rule as training, seeded walk.
   Two large-accelerated cells (real estate & construction 5, energy & transportation 1) cannot be filled,
   because those filers have December year-ends and file no 10-K in the window. The 6 slots are filled from the same
   seeded order regardless of cell and marked `fill=relaxed`: the minimum count is spec, my strata are not.
   A first run without that fallback produced 24 rows. It was my own uncommitted output and was replaced.
3. **Instances** mirror the benchmark (`data/fresh/instances.jsonl`, untracked): every positive plus
   negatives passing FinTagging's filters, drawn to the eval set's dedup negatives fraction (0.3768) and
   negative table share (0.2042). **Twins are not dropped**, because this is an evaluation set built to the
   benchmark's conventions (T-010 applies to training only).

## T-016 AUTONOMOUS DECISION - owner to review: train on the decontaminated corpus below cap (2026-10-02)

Supervisor decision (owner to review), O-016 option b: train on the corpus as built, with 38,788 train and 1,933 dev
instances. `training.target_train_instances`, `dev_slice_instances` and `curve_points` stay as written in
`thresholds.yaml` and are **not** edited to the realised sizes. The top curve point is "full corpus",
realised at 38,788 distinct instances. Every chart, table and the README say "38,788 (cap 40,000 not
reached after decontamination, O-016)", and dev says "1,933 (cap 2,000 not reached after decontamination,
O-016)". **Curve points count distinct instances**: nested prefixes of the one fixed shuffled order
(T-013), seen in epoch 1. They are not optimizer examples across `training.epochs`, so the 40,000 point is
not "reached in epoch 2". The training script states the same.
Alternatives: (a) a second extension. Rejected because it breaks T-012's "extended once", re-opens the
pinned list a second time, and needs a third sizing rule and a re-audit, all to close a 3% gap. (c) defer
to the Phase 2 pilot. Rejected because a Phase 2 result would decide a Phase 1 corpus question, which is the
wrong direction.

## T-017 AUTONOMOUS DECISION - owner to review: mlx_lm pin and Mac pilot design (2026-10-02)

1. **Pin**: `pins.software.mlx_lm = "0.32.0"`, the version installed at the moment Phase 2 first needed it
   (PyPI release 2026-10-01), never edited afterwards. Installed into `.venv/` with its dependencies:
   mlx 0.32.3, mlx-metal 0.32.3, transformers 5.18.0, numpy 2.5.3, sentencepiece 0.2.2, protobuf 7.36.2,
   safetensors 0.8.0, jinja2 3.1.6 and their small helpers (full freeze in `requirements.lock`).
   `transformers` is therefore present locally as an mlx-lm dependency. PyTorch is not installed.
2. **One format everywhere** (`xbrl_frontier/sft_format.py`). The user turn is the benchmark's verbatim
   query (template from the pinned dataset). The chat template is applied with `add_generation_prompt=True,
   enable_thinking=False`, which ends in Qwen3's empty think block. The completion is the answer in the
   benchmark schema; `json.dumps` with default separators reproduces all 28,787 FinNI gold answers byte for
   byte. Loss covers completion tokens only. mlx-lm's own `--mask-prompt` masks up to `assistant\n` and would
   put the empty think block into the loss, so the pilot feeds mlx-lm (tokens, offset) pairs from
   `sft_format` instead, and the Kaggle trainer uses the same function.
3. **Pilot data**: the first `training.mac.pilot_instances` of the fixed corpus order whose full sequence fits
   `training.mac.max_seq_len_tokens`. mlx-lm truncates longer sequences, which would cut answers. Val is the
   first `training.mac.pilot_val_instances` dev instances that fit.
4. **Settings** (new keys): `training.global_batch_instances` 16 on every machine. `training.mac.micro_batch`
   was set to 4 before any run; the first attempt ran out of Metal memory within 10 iterations
   (`data/phase2/mac_pilot_0.6B_attempt1_oom.log`), so it is 1, with `training.mac.grad_checkpoint` true.
   This is the mlx attention-backward limitation the PRD cites. LoRA on every layer, all seven projections
   (`training.lora_target_modules`), `lora_dropout` 0.0, mlx `scale = lora_alpha / lora_rank` (the PEFT
   equivalence the PRD's converter uses). Optimizer Adam with no weight decay, matching HF's default AdamW
   with weight_decay 0. Constant LR (`training.curve_lr_schedule`). 4-bit affine base with group size
   `training.mac.quant_group_size` (64, the `mlx_lm.convert` default), converted from the pinned snapshot.
5. **Tokens/sec**: headline = processed tokens (whole sequences) per second of train time. That is the unit
   of `training.planning_tokens_per_sec` and of the "about 1,000 tokens per instance" arithmetic in PRD
   section 5, and it is the number compared with `min_tokens_per_sec_to_train_0p6b_locally`. mlx-lm's printed
   tok/s counts loss tokens only and is recorded beside it.
6. **Format check**: after training, greedy generation on the val instances, scored with the FinNI scorer
   (F1 and parse-failure rate). It is labelled as a pilot check and never used as a result.

## T-018 AUTONOMOUS DECISION - owner to review: Kaggle training script (2026-10-02)

`training/kaggle_train.py`, driven by `notebooks/phase2_kaggle_train.ipynb`:
1. Transformers Trainer + PEFT + bitsandbytes NF4 with fp16 compute (`training.kaggle.quant_base`, `dtype`),
   gradient checkpointing, LoRA as in T-017, constant LR, `training.epochs`, global batch 16 as micro-batch
   `training.kaggle.micro_batch` (1) x 16 accumulation. Unsloth and TRL are not used: plain Trainer keeps
   the fixed order and completion-only masks explicit, with fewer pins. `pins.software.unsloth`/`trl`
   stay empty.
2. **Fixed order**: a `SequentialSampler` over `data/corpus/train.jsonl` in file order (the T-013 order).
   The corpus files are checked against their sha256 in `data/corpus_report.json` before training.
3. **Curve checkpoints** (`--curve`, primary student seed 1): the adapter is saved at the first optimizer-step
   boundary where epoch 1 has covered each prefix. Prefixes count distinct trained instances
   (`training/schedule.py`, T-016). The configured point above the corpus size is realised as "full corpus".
   The actual instances seen (the batch boundary) are recorded per checkpoint.
4. **Resume**: full trainer state is saved every `training.kaggle.checkpoint_every_minutes` (callback,
   `save_total_limit` 2). `--resume DIR` copies the previous session's `checkpoints/`, `curve/` and `run.json`
   and continues from the latest checkpoint. The sequential sampler makes the resumed data order identical.
5. **Long instances**: those over `training.kaggle.max_seq_len_tokens` are skipped, never truncated (a
   truncated context could cut off facts its answer lists), and counted in `run.json`. Curve prefixes then
   count trained instances. The parked 20,575-token eval context concerns scoring, which is not done here.
6. `run.json` records versions, env, config hash, corpus hashes, per-session processed tokens/sec, the curve
   checkpoints and the final adapter. The script refuses to run while `pins.software.{transformers, peft,
   bitsandbytes, accelerate}` are empty. Two new empty fields (`bitsandbytes`, `accelerate`) were added to
   `pins.software` for this.

## T-019 AUTONOMOUS DECISION - owner to review: Kaggle training-stack pins (2026-10-02)

Supervisor decision (owner to review), approving O-020: `pins.software.transformers = "5.18.0"`, `peft = "0.21.2"`,
`bitsandbytes = "0.50.2"`, `accelerate = "1.15.0"`, filled once and never edited after. Why these:
`transformers` 5.18.0 is the version mlx-lm installed locally (T-017), so the Mac pilot and Kaggle training
tokenize and render the chat template identically. The others are the current PyPI releases, checked
2026-10-02. `trl` and `unsloth` stay empty because they are unused (T-018). Kaggle's preinstalled `torch` is
**not** pinned; its version and CUDA build are recorded in every run's `env` and printed by the notebook.
Enforcement: `notebooks/phase2_kaggle_train.ipynb` installs exactly these versions and asserts both the
installed distribution versions and the imported module versions equal the pins (a kernel that imported an
older preinstalled transformers must restart). `training/kaggle_train.py` refuses to start on a mismatch
or an empty pin (`serving.common.require_pins`), as the serving scripts do.
Alternatives: let Kaggle's preinstalled transformers/peft stand. Rejected because the tokenizer and chat
template could differ from the Mac and from scoring.

## T-020 AUTONOMOUS DECISION - owner to review: SEC-BERT encoder baseline (2026-10-02)

1. **Where it runs**: Kaggle, as PRD section 5 assigns ("Kaggle: ... the encoder"), as run 4 in O-019.
   torch is therefore **not** installed locally. Everything that does not need torch (word splitting, BIO
   alignment, chunking, decoding) lives in `xbrl_frontier/bio.py` and `training/encoder_train.py::encode_chunks`
   and is tested on the Mac with the pinned SEC-BERT fast tokenizer (`tests/test_bio.py`).
2. **Label space** = the benchmark's own BIO space: FinTagging `BERT/data/label2id.json` at
   `pins.fintagging_source.commit` (34,777 labels: B-/I- over the 17,388 us-gaap 2024 concepts, plus `O`),
   sha256-pinned in `training.encoder.label_space`. Facts with 2023-only concepts are outside that space
   and stay `O` (610 of 289,088 corpus facts). Alternative: a 5-type BIO space. Rejected because the PRD says
   "the benchmark's BIO label space". Predicted concept spans become FinNI pairs through the taxonomy type,
   so the encoder is scored by the same FinNI scorer as everything else.
3. **Words and alignment**: FinTagging's `.bio` construction code is not in the repo (the notebook only
   splits data). Their files show punctuation-split words with each value as one word ("51", "3.1", "two").
   Ours: HTML tags removed, numbers kept whole with thousands separators ("58,166"), alphabetic words, and
   single other symbols. Facts are matched in document order to the first free word sequence equal to the
   comma-stripped value. 99.64% of the corpus's 289,088 facts align, 441 values are not found, and 610 are
   outside the label space. All three are counted in `run.json`.
4. **Chunking, not truncation**: the PRD says the benchmark's BERT baselines chunked long contexts, while
   their `train_ner_model.py` truncates. The PRD governs. Consecutive non-overlapping word windows fit
   `training.encoder.window_subtokens` (512, the model's position limit). A single word longer than a window
   is cut and counted. First-subtoken labelling follows their `tokenize_and_align_labels`.
5. **Training**: same corpus, same fixed order (sequential sampler over chunks in corpus order), same
   instance set as the students (instances skipped by the students' `max_seq_len_tokens` rule are skipped
   here too). `training.encoder` epochs 3 (PRD section 5), learning rate 5e-5 and micro-batch 4 (their
   script's defaults), fp16, timed resume checkpoints.

## T-021 AUTONOMOUS DECISION - owner to review: student prediction and scoring pipeline (2026-10-02)

1. **Generation** (`serving/generate_predictions.py`, Kaggle, vLLM `pins.software.vllm`): base model at its pin,
   LoRA adapters as `LoRARequest`s, the training prompt (`sft_format.prompt_text`, `enable_thinking=False`),
   greedy (`eval.generation.temperature` 0.0), `eval.generation.max_model_len` 32,768 (Qwen3's native context).
   That covers the parked 20,575-token FinNI context; students fine-tuned at 4,096 tokens being scored on
   longer contexts is part of what the context-length slice measures. Prompts that still do not fit are
   written `not_generated`. They are scored as empty predictions and counted in `not_generated_rate`, never
   truncated. The output budget is the remaining context, so a cut-off answer shows up as
   `finish_reason=length` and is usually a parse failure, which counts.
2. **Scoring** (`scripts/score_predictions.py`, Mac): any predictions file on `finni`, `dev` or `fresh`. Raw and
   dedup (`eval.dedup_policy` applied to each set), overall and per slice (context type, entity type,
   company, fiscal year, context-token bucket), with `parse_failure_rate`, `malformed_items` and
   `not_generated_rate` beside F1. Per-instance tp/fp/fn are kept for the company-level bootstrap (T-007).
   It refuses if any instance lacks a prediction. A parse failure is an empty prediction. There are no
   retries, no repairs and no judge. Calibrated through real files: gold copy gives F1 1.0 and empty gives
   0, raw and dedup, on dev and on the full FinNI set (`tests/test_score_predictions.py`).
3. **Curve checkpoints**: scored on the dev slice now. The frontier sample is added once Phase 3 pins its ids.
   The pilot's F1 (O-021) is a format check and is never charted or tabled.
4. **Addendum (supervisor note, owner to review, 2026-10-02): effective `max_model_len`.** A 1.7B in fp16 on a 16 GB T4 may not leave
   vLLM enough KV cache for 32,768 tokens. `serving/generate_predictions.py` therefore records every startup
   refusal (`max_model_len_refusals`) and retries at the next halving, never below
   `training.kaggle.max_seq_len_tokens`. The accepted value is recorded as `effective_max_model_len` and
   printed by the notebook. Prompts that do not fit the effective value are `not_generated`. The limit is
   never lowered silently, and the configured value stays as written.

## T-022 Curve aggregation (supervisor decision (owner to review), 2026-10-02)

`scripts/curve_table.py` builds the learning-curve table and chart series from a run's `run.json` and one
scorer output per checkpoint (`--point PREFIX=SCORE_JSON`). It refuses any file that does not carry the
scorer's `kind` marker (`xbrl_frontier.score_predictions/v1`, added to every scorer output together with the
predictions file's sha256), the expected set, and the raw/dedup metrics with the context-type slice. A
checkpoint without a score file gets no row and is listed under `missing`. Rows carry the recorded
`instances_seen` and `global_step`. The full-corpus point is labelled
"38,788 (cap 40,000 not reached after decontamination, O-016)" (built from the run's own numbers), with a
skip note if any instance was skipped (T-018). Series are given overall and by table vs text, raw and dedup.

## T-023 AUTONOMOUS DECISION - owner to review: quantization export and flip rate (2026-10-02)

1. **Export** (`serving/export_quantized.py`, `notebooks/phase2_kaggle_export.ipynb`, Kaggle): the merge stage loads
   the base at its pin in fp16, applies the trained LoRA with PEFT, `merge_and_unload`, and saves
   `merged-fp16`, which is also the fp16 serving variant. INT8 W8A8 and AWQ-INT4 are produced from the merged
   checkpoint with exactly the tools, schemes and pins of `serving.smoke_test.smoke_quantization` (llm-compressor
   W8A8 data-free; AutoAWQ GEMM w4 g128 zero-point, in its own venv). The INT8/AWQ code is now one shared
   implementation (`serving/quantize_base.py::int8_from_dir`, `awq_from_dir`) used by both the smoke test
   and the export.
2. **AWQ calibration on training contexts** (as T-004 promised): `random.Random(seed).sample` of
   `serving.export_quantization.awq_calibration.samples` (128) instances from `data/corpus/train.jsonl`,
   seed 20261006, each rendered in the training format (prompt + answer, `sft_format.full_text`), at
   `max_seq_len` 512 per AutoAWQ calibration block. The sample's instance ids, the source file's sha256 and
   the sha256 of the joined texts are recorded in `export.json`. 128 / 512 are AutoAWQ's own defaults, so only
   the data source changes from the smoke test. Alternative: calibrate on eval contexts. Rejected because it
   leaks the eval set into a checkpoint.
3. **Flip rate** (`scripts/flip_rate.py`): two prediction files on the same set. Per-fact flip rate = pooled
   `sum |A xor B| / sum |A union B|`, where a parse failure or not_generated row is the empty set, as in scoring.
   Also reported: instance flip rate, correct-to-wrong and wrong-to-correct counts against gold, F1 for
   both and the F1 drop, raw and dedup, overall and by table/text, with `gates.quantized_flip_rate_max` and
   `gates.quantized_f1_drop_max` beside them. Alternative denominators: fp16 facts only (|A|). Rejected
   because a variant that only adds facts would then read as more stable. Point values only; bootstrap
   decisions belong to the scoring phase (`gates.decision_rule`).

## T-024 AUTONOMOUS DECISION - owner to review: frontier-arm harness and spend guard (2026-10-02)

1. **Requests** (`frontier/requests.py`): one user message holding the benchmark's verbatim FinNI query, zero-shot.
   OpenAI arms use the Responses API (`store: false`); Gemini uses `generateContent`. Thinking per
   `frontier.thinking_per_arm` (new key), the minimum each API allows per its documentation (checked
   2026-10-02): `gpt-6-luna` reasoning effort `none` ("GPT-6 Sol and GPT-6 Luna support `none`"),
   `gpt-5.4-mini-2026-03-17` `none` ("Reasoning.effort supports: none (default)"), `gemini-3.5-flash-lite`
   thinking level `minimal` (its default; the Cloud docs name minimal as the low end). The swap candidate
   `gpt-5.4-nano-2026-03-17` is set to `none` by analogy with gpt-5.4-mini. This is **unverified** and must be
   checked before the swap is ever used. No temperature or output cap is set, so arms run at API defaults.
   Every response's text, returned model field, input/output/reasoning tokens and finish reason are logged.
2. **Batch formats**: OpenAI Batch API JSONL against `/v1/responses` (file upload, then a batch with a 24 h
   window); Gemini `batchGenerateContent` with inline requests keyed by instance id, results read from
   `dest.inlinedResponses` (both per the vendor docs fetched 2026-10-02). Outputs are matched by
   `custom_id` / metadata key, not by order.
3. **Dry run is the default**: it renders the exact request payload and prices it from the dated price sheet.
   Input is estimated with the Qwen3 tokenizer as a proxy and output with the gold answer's tokens, labelled
   as estimates. Batch steps add `frontier.reasoning_token_margin`, synchronous steps do not (PRD section 7).
   `scripts/frontier_plan.py` prices every live step, using the planning shape until the probe measures
   real tokens, and the actual fresh instances for the control.
4. **Spend guard** (`frontier/guard.py`): `--live` requires `--approval T-NNN`, and that TRADEOFFS section must
   contain `PRE-APPROVED SPEND: step=<step> arm=<arm> usd_max=<dollars>`, written by the owner (this paid-call mechanism is
   superseded by T-034 under T-033). The estimate must
   be at most `usd_max`, and the ledger (`data/phase3/spend_ledger.jsonl`, priced from returned usage) plus the
   estimate must leave `frontier.usd_reserve_min` under `budget.usd_cap_frontier_api`. The guard runs **before**
   the API key is read. Tests strip the key variables, replace key reading and HTTP with failing stubs, and
   exercise every refusal path. The builder makes no live call; every live step is OWNER-BLOCKED (O-022).

## T-025 AUTONOMOUS DECISION - owner to review: guard refuses unverified arms and unmeasured batch pricing (2026-10-02)

1. **Verified thinking setting required.** Every `frontier.thinking_per_arm` entry carries a `source` (URL, check
   date, quoted text). The guard refuses a live call for an arm whose `source` is empty. `gpt-5.4-nano-2026-03-17`
   (the swap candidate, `none` by analogy) has an empty source, so the swap cannot run until someone checks its
   documentation and records the source line.
2. **Batch-priced steps are priced from the probe.** For `batch` and `fresh_control`, the harness re-prices the
   proxy estimate with that arm's measured/proxy token ratios on its own probe instances
   (`reprice_from_probe`: run 0 of `data/phase3/live/probe_<arm>/usage.jsonl`; measured output includes reasoning
   tokens; batch prices; reasoning margin kept). The guard compares the re-priced figure, not the proxy, with
   `usd_max` and the ledger. If the arm has no probe usage, the guard refuses. Check order: approval given,
   approval line matches, estimate at most `usd_max`, verified source, probe re-pricing for batch-priced steps,
   ledger plus estimate within cap minus reserve. All of it runs before the key is read.
Alternatives: re-price from the probe's mean tokens per instance. Rejected because the ratio method is
robust to the probe filing's instance mix differing from the sample's.

## T-026 AUTONOMOUS DECISION - owner to review: power check and frontier-sample draw (2026-10-02)

1. **Power check** (`scripts/power_check.py`, `frontier/sampling.py`): input is the primary student's full-set
   scorer output only (the T-022 `kind` check plus set `finni` plus dedup per-instance counts). No placeholder or
   planning SD exists. SD unit per T-007: pooled F1 per company (CIK) over dedup assigned instances, sample SD
   across companies, labelled "company (CIK)". The half-width evaluates the configured expression
   `frontier.power_check.expected_half_width` verbatim. N for all arms is `min(frontier.sample_filings,
   largest N whose priced plan leaves frontier.usd_reserve_min)`: the v2.1 "reduce the sample to keep the
   reserve" rule, applied through the priced plan (`scripts/frontier_plan.py::plan/largest_n`). At today's
   prices it does not bind: all arms fit up to N = 24 > 23. If the half-width exceeds `gates.headline_margin_f1`,
   `if_exceeded` is applied as written: drop `pins.frontier_arms[2]`, N = the largest N for the remaining arms
   (73 at today's planning prices), and the headline becomes "fraction of the best remaining arm".
2. **Sampler** (`scripts/draw_frontier_sample.py`): filings = the 139 manifest filings with assigned instances
   (the unassigned stratum is excluded). Table share and size come from the instance map, with terciles over
   filings (edge k = sorted[floor(k*n/3)], as in T-001a). Cell targets are floor/ceil of N/9, with extra
   slots to cells in seeded order (2-3 per cell at N = 23). **The sampling unit is the company**: a company sits
   in the cell of its largest filing and all its filings move together. A two-filing company is skipped while
   a single-filing company still fits the cell. Filings and companies drawn, and any cell shortfall, are
   recorded. Sector = SEC office (as in T-011). A draw is redrawn when a sector whose proportional share of the
   sample is at least one filing is empty. Redraws are counted and capped by `frontier.sampling.max_redraws`.
   Seed `frontier.sampling.seed`.
3. **Latency ids**: `frontier.sync_latency_instances` ids, `random.Random(frontier.latency_sampling.seed).sample`
   over the sorted dedup assigned instance ids of the sample filings (`frontier.latency_sampling.rule`), pinned
   to `data/frontier_latency_ids.txt`, identical for every arm and student.
4. **Probe filings**: one per arm, uniform over non-sample filings in the middle instances-per-filing tercile
   (`frontier.probe_selection`), so the probe neither duplicates sample filings nor sits at a size extreme.
5. Outputs are written only from a real power check and never overwritten. The tests use synthetic filings
   and assert every refusal. A no-output check on the real tables found 139 filings and 129 companies, all 9
   cells populated (7-22 filings each), and a trial draw of 23 filings / 22 companies with 0 redraws.

## T-027 AUTONOMOUS DECISION - owner to review: Phase 4 load tests on `vllm bench serve` (2026-10-02)

1. **Tool**: `vllm bench serve`, not GuideLLM, for the sweeps. PRD section 6 allows either. GuideLLM's current
   CLI (`guidellm run --profile kind=...`, v0.7+ syntax) writes a JSON report whose field names are not documented
   outside its own Python loader, so a parser could not be verified offline before spending Kaggle hours.
   `pins.software.guidellm` is also empty. `vllm bench serve` is already pinned, and its JSON fields
   (`request_throughput`, `median/p95_ttft_ms`, `median/p95_e2el_ms`, `completed`) are parsed by the Phase 0 code.
   Supervisor decision (owner to review), 2026-10-02: keep `vllm bench serve`. `pins.software.guidellm` stays empty: GuideLLM was not
   installed because it is not used.
2. **Prompts**: the pinned latency ids (`serving.load_test.prompts` -> `data/frontier_latency_ids.txt`), rendered with
   the inference format, sent pre-rendered (`--dataset-name custom --custom-skip-chat-template`), oversampled to
   `serving.requests_per_point`. Per-request output cap `serving.load_test.max_output_tokens` (new key, 4,096);
   generation stops at EOS first, so latencies reflect real answer lengths.
3. **Points**: warm-up of `serving.warmup_requests` (discarded). Closed loop at each `serving.concurrency_levels.<card>`
   (concurrency 1 = the single-request row) and Poisson arrivals at each `serving.request_rate_sweep_rps` (the
   batched rows), `--goodput` at `serving.slo` recorded beside the p95-based goodput of T-004.
4. **Server**: the card's flags (`serving.smoke_test.server_flags`). On the 3050 the configured `vllm_flags`
   (including `max_model_len`) are used as written, and prompts that do not fit fail and are counted as
   incomplete points. On the T4, `max_model_len` = the longest pinned prompt + the output cap, so every pinned
   prompt fits. Every run records flags, env (GPU, driver, versions) and checkpoint sha256s.

## T-028 AUTONOMOUS DECISION - owner to review: bootstrap, cost model (both regimes) and charts (2026-10-02)

1. **Bootstrap** (`xbrl_frontier/bootstrap.py`): company (CIK) clusters (T-007), resampled with replacement
   `eval.bootstrap_resamples` times with `eval.bootstrap_seed` (new key). tp/fp/fn are pooled per resample. Every
   arm is evaluated on the **same** resample (paired), and the percentile interval is taken at `eval.ci_level`.
   `decide()` implements `gates.decision_rule` word for word: pass needs the point and the whole CI to meet the
   threshold, fail needs all three to miss it, and anything else (the CI straddles) is inconclusive.
2. **Shared-sample scoring**: `scripts/score_predictions.py --ids FILE` scores only those instances, refuses ids
   outside the set, and records `ids_file`/`ids_sha256`. The sampler writes `data/frontier_sample_ids.txt`. The power
   check refuses subset scores, and the cost model refuses full-set student scores.
3. **Cost model** (`scripts/cost_model.py`): inputs are real result files only (load-test `kind`, scorer `kind`, harness
   usage rows). Self-host $/filing = T4 $/h / filings per hour / utilization, offline from max throughput and online
   from p95-SLO goodput, for every T4 source and every utilization scenario. Filings per hour use the **measured**
   instances per sample filing, not the planning shape. Frontier $/filing comes from measured usage per instance
   x instances per filing, at batch prices offline and sync prices online. Comparator per
   `gates.cost_comparator_rule`, with arms ranked by cost in the gated regime; the arm F1s are paired-bootstrap
   point estimates against the student's CI. The gate uses the headline rate key, `gates.cost_gate_regime` and
   that regime's utilization; the other regime is "reported, not gated". The cost ratio itself is not a bootstrap
   statistic, so its outcome is a plain comparison, while the comparator choice carries the uncertainty. 3050
   rows never get dollars. Idle cost is excluded and stated next to the formula.
4. **Charts** (`scripts/charts.py`): Vega-Lite JSON specs (no plotting dependency) for the Pareto chart (each
   regime), the learning curve (overall, table and text) and latency, built only from the cost-model, curve-table and
   load-test outputs (refusing anything else). Points without a measured value are omitted and listed under
   `missing`.
5. Tests use synthetic inputs and clean up every output they write, so no synthetic file is left where real
   results go.

## T-029 AUTONOMOUS DECISION - owner to review: label-audit draw and contamination-statement draft (2026-10-02)

1. **Label-audit population**: dedup FinNI instances **assigned to a filing** (an audit is a check against the filing).
   The unassigned stratum cannot be audited, and the reported error rate is weighted to the assigned dedup
   population. Strata per `eval.label_audit.stratify_by`: context type x entity type, where an instance's entity
   type is its most frequent gold type and negatives are "none". Proportional largest-remainder allocation of
   `sample_size`, at least one per non-empty stratum (12 strata, 300 instances). Seeds `eval.label_audit.seed` and
   `second_pass_seed` (new keys). The second pass draws `second_pass_size` from the first-pass ids, on its own
   sheet with no first-pass verdicts.
2. **Sheets** (`data/label_audit/pass1_sheet.csv`, `pass2_sheet.csv`, shipped): instance id, stratum, accession,
   CIK, the filing's EDGAR URL, the context and the gold pairs, plus owner columns (verdict correct/incorrect;
   counts of missing, spurious and wrong-type facts; notes). `scripts/label_audit.py score` refuses blank or
   unknown verdicts and sheets that do not match the drawn ids. It reports the raw and stratum-weighted
   instance error rates (`label_audit_error_rate`), fact counts, and self-agreement against
   `min_self_agreement`. The draw was run now: it is spend-free and its inputs exist.
3. **Contamination statement** (`scripts/contamination_statement.py` -> `docs/drafts/contamination_statement_draft.md`):
   every figure is read from a recorded file or config at generation time. Unmeasured figures print as
   `__ (not yet measured: ...)`, and gate values are named by key, not restated. It covers the O-018 checklist:
   fresh-slice relaxed rows and unfillable cells, audit counts before and after, size disclosures, and the fresh
   instance hash. The arms' knowledge cutoffs are recorded once in config with sources
   (`contamination.frontier_arm_cutoffs`), so the statement and `fresh_slice_filing_date_min` share one source.

## T-030 AUTONOMOUS DECISION - owner to review: curve validation run (2026-10-02)

`training/kaggle_train.py --validation` trains an independent run on exactly the first
`training.curve_validation_run` **trained** instances of the fixed order (the prefix the nested checkpoint covers;
the first and last ids are recorded), seed 1, with the **full recipe**: `training.epochs` epochs, constant LR, the
same batch and LoRA. One epoch over the prefix would recompute the nested checkpoint, so it would test nothing.
The full recipe checks whether a curve point read off one long epoch matches a model trained to completion at
that size, which is the PRD's "matches a from-scratch run at that size". `scripts/curve_validation.py` compares the
two on the dev slice and (when scored) the frontier sample with the company-cluster paired bootstrap. It reports
the gap with its paired interval and raises the curve caveat when |gap| exceeds the half-width of the nested
checkpoint's own interval. In the cut order it is item 4: planned, not cut, and scheduled after run 1 and before
run 2 (O-019 run 5).

## T-031 AUTONOMOUS DECISION - owner to review: gate summary and the zero-shot comparison (2026-10-02)

`scripts/gate_summary.py` evaluates every key under `gates` from result files only. A gate without its inputs is
"not measured", with blank values and `ships: false`. Bootstrap statistics use the company-cluster paired
bootstrap (T-007, T-028) and `gates.decision_rule`:
- **gain over zero-shot** = student - zero-shot base, paired on the **full set** (both must be full-set scorer
  outputs; subsets are refused). The same comparison on the shared sample (`--ids` scores of the same predictions,
  so no extra run) is reported beside it, not gated. The zero-shot runs use the same prompt, template and greedy
  decoding as the students (`enable_thinking=False`).
- **fraction of best frontier arm** = student / best arm (highest point F1 on the shared sample), paired.
- **quantized F1 drop** = fp16 merge - variant, paired, lower is better. The **flip rate** comes from
  `scripts/flip_rate.py` and is compared at its point value (not a bootstrap statistic).
- **cost ratio**: the gated-regime entries of `scripts/cost_model.py`, at their point value.
- **fresh-slice gap** = F1(full set) - F1(fresh): different instance sets, so each is resampled on its own
  companies and the interval is taken over the differences of the resamples. The control arm's gap and the
  student-minus-control difference are reported in the note (`fresh_slice_control_gap_reported`).
- **encoder / learning curve / Pareto chart**: presence of their result files.

## T-032 AUTONOMOUS DECISION - owner to review: README skeleton generator (2026-10-02)

`scripts/readme.py` writes the root `README.md`. Every result number is `__` (headline, results, resume bullets per
PRD section 11). Gates are listed by key, with status, point and interval read from `data/gates/summary.json`
when it exists ("not measured" otherwise), and threshold values are never printed. The disclosure lines (129/142
registrants; 95.25% / 95.39% serializer assignment beside the Phase 0 94.29% / 94.43%; 38,788 and 1,933 with the
O-016 cap note; 18,184 flagged and removed, 0 after removal) and the contamination statement are read from the
recorded files at generation time. The Mac pilot's format-check F1 is absent (O-021), which a test asserts. The
README is regenerated after each result: nothing in it is typed by hand.

## T-033 OWNER DECISION (Adnan, stated directly, 2026-10-02): $0 budget, Gemini free tier only, Kaggle automated

This entry comes directly from the owner, unlike T-001 to T-032. Their "owner decision" and "owner
approved" labels mostly record supervisor replies; the owner reviews those separately (see the end of
this entry).

1. **Money.** Billed spend is $0. The owner has no funds and no OpenAI key. The only frontier key is a
   free-tier Google AI Studio key, which lists `models/gemini-3.5-flash-lite`. That was checked
   2026-10-02 with a free model-list call, generating nothing. `budget.*` values stay as written. The
   $0 billed rule sits on top of them.
2. **Frontier arms.** Only `gemini-3.5-flash-lite` runs. `gpt-6-luna` and `gpt-5.4-mini-2026-03-17`
   are not run, for lack of a key and funds. They stay in `pins.frontier_arms` as the planned design,
   and their absence is disclosed in the README. The headline becomes "fraction of
   gemini-3.5-flash-lite's F1", and the README names the arm. This goes further than the PRD's cut
   order, which keeps the cheapest and mid-tier arms. The cause is money, not results.
3. **Fresh-slice control arm.** `gpt-6-luna` can't run, so the control is `gemini-3.5-flash-lite`
   (config change by the builder, citing this entry). `fresh_slice_filing_date_min` is unchanged:
   2026-05-19 is still after the Gemini arm's cutoff (2026-03).
4. **Free-tier collection.** Calls run within the free tier's rate limits, possibly synchronous only.
   429 and 5xx responses are retried with backoff and are never scored as empty predictions. The
   README states that tokens, the model field and latency come from the free tier. It also states that
   dollars per filing are measured tokens priced at the dated sheet's batch and sync rates, and that
   free-tier latency may include throttling. Google may use free-tier inputs; they are public SEC text.
5. **Kaggle.** The builder runs every Kaggle step itself through the `kaggle` CLI with the owner's API
   token, which is in the environment and never in the repo. It stays inside the `training.kaggle`
   quota and cap keys. Kaggle is no longer an owner-only item.
6. **Still owner-only:** the two label-audit passes (never model-filled), the RTX 3050 runs, reading the
   preprint, and posting the GitHub issue (after the owner approves the text).
7. **Review of earlier labels.** Entries T-001a, T-004 (pins and AutoAWQ), T-005, T-006, T-007, T-010,
   T-019, T-022 and T-027, and findings O-003, O-007, O-012, O-013 and O-020, say "owner" where the
   decision came from the autonomous supervisor. The builder relabels them "supervisor decision (owner
   to review)" without changing their substance. T-001's four dependencies were approved on the
   owner's behalf in the resume message. T-006 is superseded by O-011 (95.25%, which passes).

### T-033a Builder addendum: records relabelled; `kaggle` pinned (2026-10-02)

- Relabelled per item 7, substance unchanged: T-001 (approved on the owner's behalf in the resume message), T-001a,
  T-004 items 5 and 7, T-005, T-006, T-007, T-010, T-019, T-022 and T-027 now read "supervisor decision (owner to
  review)", as do O-003, O-007, O-012, O-013, O-020 and the O-010 reference to T-005. T-006 is marked SUPERSEDED by
  O-011.
- `kaggle==2.2.4` (installed by the owner in `.venv/`) is pinned in `requirements.txt`. `requirements.lock` is
  re-frozen, which adds kaggle 2.2.4, kagglesdk 0.1.37 and their dependencies (bleach, jupytext, python-dotenv,
  python-slugify, requests, urllib3 and others). The token comes from `KAGGLE_API_TOKEN` in the environment and is
  never printed or written.

## T-034 AUTONOMOUS DECISION - owner to review: free-tier Gemini collection and the T-033 guard (2026-10-02)

1. **Config (citing T-033)**: `frontier.fresh_slice_control_arm` = `gemini-3.5-flash-lite` (item 3).
   `frontier.running_arms` = [`gemini-3.5-flash-lite`] records which arms run and why. `pins.frontier_arms` is
   unchanged as the planned design. The power check, plan, sampler and README read `running_arms`. With one
   arm, `frontier.power_check.if_exceeded` cannot drop an arm, so only its second half applies (N = the largest N
   whose notional priced plan leaves the reserve), and the headline names the arm.
2. **Free-tier limits, as found** (2026-10-02): the Gemini rate-limits page publishes the rules (RPM, input TPM
   and RPD per project; RPD resets at midnight Pacific; "can be viewed in Google AI Studio") but no per-model
   free-tier numbers. Its Batch API quota table lists Tiers 1-3 only. `models/gemini-3.5-flash-lite` (version
   `3.5-flash-lite-07-2026`) lists `batchGenerateContent`, but batch has no free-tier quota, so it is treated as
   potentially billable and refused. One minimal `generateContent` call (9 tokens) returned no rate-limit
   headers, only `x-gemini-service-tier: standard`. Hence `frontier.free_tier`: start at `assumed_rpm` (10), learn
   the real per-minute and per-day limits from 429 `QuotaFailure`/`RetryInfo` details, persist them in
   `data/phase3/gemini_quota_state.json`, and pace by them (`frontier/freetier.py`).
3. **Retries, never empty**: a per-minute 429 waits the server's `retryDelay` (or the backoff list) and retries
   the same request, as do 5xx and network errors (`backoff_seconds`). A per-day 429 stops the run with
   `resume_after` = the next midnight in `daily_reset_tz`, spreading the work across days. If retries are
   exhausted the run stops. In every case the instance stays pending and no prediction is written, so the
   scorer refuses an incomplete set. Collection is resumable: completed (id, run) pairs are skipped.
4. **Guard** (`frontier/guard.py`, replacing the T-024/T-025 spend approvals, which assumed paid calls): a live call
   needs `--approval T-033` and that owner entry in TRADEOFFS. The arm must be in `running_arms`, use the
   Gemini API with `GEMINI_API_KEY`, and have a verified thinking source. Only synchronous `generateContent` is
   allowed. Everything else is refused before the key is read: other arms, other providers (the HTTP layer
   only speaks to Gemini), any other approval, and Batch-API mode. The OpenAI batch submit/collect code is
   removed. The ledger records `billed_usd` 0 and the notional `priced_usd`.
5. **Pricing and latency**: every step is collected synchronously. Accuracy steps (`batch`, `fresh_control`) are
   priced notionally at the sheet's batch rates, and `probe`/`latency` at sync rates (T-033 item 4). Usage rows
   keep the request time (`latency_s`) separate from the client-side throttle wait, and the README states that
   free-tier latency may include throttling. The plan is about 8,600 free-tier requests: probe 202, accuracy
   4,662, latency 300, fresh control 3,434.

## T-035 AUTONOMOUS DECISION - owner to review: start the fresh-slice control (E4) before D4 (2026-10-02)

Decision (supervisor decision, owner to review): collect E4 (`--step fresh_control --arm gemini-3.5-flash-lite`) now, on
the free tier under T-033. Why: Phase 2 is blocked solely on O-025. The fresh slice is pinned (3,434 instances,
sha256 in `data/fresh_selection_report.json`) and depends on neither the frontier sample nor any student output. The
free-tier daily quota makes this a multi-day collection that would otherwise sit idle. Alternative: wait for D4,
the original order, which costs calendar days for no information gain.
Conditions: dry run first, then `--live --approval T-033`. Every call's returned model field and UTC timestamp are
recorded in `usage.jsonl`. When E2 runs, the model fields are compared, and a difference is a finding in OPEN.md
reported next to the control gap, not merged silently. An instance that exhausts its retries stays pending and is
listed in the step's `status.json`. The run is not done until the pending list is empty or the remainder is logged
as a finding. Billed spend stays $0, and any path that would need billing stops the run and is logged. The
collection date range is recorded here when it finishes.
Also in this commit: RUNBOOK E1-E4 rewritten for T-033. T-008, T-012, T-016, T-002 (confirmation), the T-021 addendum
and O-002 relabelled "supervisor decision (owner to review)". The T-024 paid-approval mechanism is marked superseded by T-034.

## T-036 AUTONOMOUS DECISION - owner to review: collect the free tier at 500 a day until done (2026-10-02)

Supervisor decision (owner to review), O-027 option (a): collect E4, and later E1-E3, at the free tier's 500 requests per
day until each step's `status.json` is `complete` with an empty pending list, about 18 days for the one-arm plan.
Rejected: (b) trimming E3, the latency sample. Its size (`frontier.sync_latency_instances` x `frontier.sync_runs`) is a
measurement design choice, not a schedule variable, and nothing waits on the frontier steps while Kaggle is blocked
(O-025). The days-per-filing figure in O-027 is scheduling information only: after D4, N is set by the power check and
`gates.decision_rule`, never by the day count.
Also: `scripts/smoke_summary.py` (new) consumes the T4 and 3050 smoke outputs under `serving.smoke_test.failure_action`, and
every OWNER-BLOCKED entry in OPEN.md now names the action, the artefact path and the consuming script, matching the RUNBOOK.

## T-037 AUTONOMOUS DECISION - owner to review: no free-tier Batch API in this project (2026-10-02)

Supervisor decision (owner to review), resolving O-028: the Batch API is never used in this project, on the free tier or
otherwise. Collection stays synchronous (`frontier.free_tier.batch_api: false`; `frontier/guard.py` refuses any mode but
`sync`). Why: Google's own pages disagree on whether free-tier Batch exists (the pricing page has a free-tier Batch row, while
the rate-limits page gives Batch quotas for paid tiers only), and T-033 says to stop any path that could bill. Staying
synchronous costs only calendar days, which O-027 already budgets.
Rejected: probing free-tier Batch with one request. A single mis-tiered request would break the $0-billed rule, and the
ledger records only notional prices from returned usage, so it could not prove that nothing was billed.
No code change.

## T-038 AUTONOMOUS DECISION - owner to review: training stops itself before the Kaggle session limit (2026-10-02)

Kaggle kills a session at `training.kaggle.session_hours_max` and keeps no output from a killed session, so the 30-minute resume
checkpoints would be lost with it. `training/kaggle_train.py` now saves a checkpoint and stops cleanly once
`training.kaggle.session_stop_margin_minutes` remain; `run.json` records `complete` and writes `final_adapter` only when
every step has run. The next session resumes with `RESUME` pointing at this session's output (`--kernel-source`).
Rejected: relying on the timer alone. A killed session discards its checkpoints, so up to a full session of GPU-hours would be lost.
Cost: up to the margin of GPU time per session goes unused.
