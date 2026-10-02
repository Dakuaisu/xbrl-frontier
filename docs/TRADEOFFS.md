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

### T-001a AUTONOMOUS DECISION - owner to review: `tokenizers` and the context-token buckets (2026-10-02)

The owner approved `tokenizers==0.23.2` (pinned in `requirements.txt` / `requirements.lock`) for the
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

Confirmed as written by the owner on 2026-10-02. It stays AUTONOMOUS for owner review.

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
   answers. Owner decision 2026-10-02: keep AutoAWQ, as PRD section 5 names it, with no switch before the
   T4 smoke test exists. If AutoAWQ GEMM fails there, the failure is recorded per
   `serving.smoke_test.failure_action` and the llm-compressor AWQ question is raised then, as a data-wins
   entry.
6. **Break-even table** (`scripts/phase0_cost.py`): every pinned arm plus the swap candidate, both
   regimes, every non-blank T4 source in `serving.hardware.kaggle_t4.market_rate_keys`, at the regime's
   utilization and at every `serving.utilization_scenarios` value. The blank AWS source is dropped and
   listed as dropped (see O-008).
7. **Software pins (owner approval 2026-10-02, O-007).** `pins.software.vllm = 0.30.0`,
   `llmcompressor = 0.14.0`, `autoawq = 0.2.9`. Only these three are filled now. Every other
   `pins.software` field is filled when the phase that first installs it arrives, and is never edited
   after. `transformers`, `torch` and `compressed-tensors` are **not pinned**: they are left to vLLM's and
   llm-compressor's resolvers, and the installed versions are captured in every run's `env` record
   (`serving.common.env_record`). Alternative: pin them too. Rejected because a hand-picked combination
   risks a resolver conflict with vLLM's own pins, and the `env` record keeps every run reproducible.

## T-005 AUTONOMOUS DECISION - owner to review: manifest shortfall accepted at 129 of 142 (2026-10-02)

Decision (owner): option (c), accept 129 recovered registrants and disclose them.
`contamination.eval_manifest_required_recovered` is **NOT MET** (129/142) and the threshold is unchanged.
Option (a), the public GitHub issue to the FinTagging authors, is OWNER-BLOCKED. The draft is
`docs/owner/fintagging_issue_draft.md`. Option (b), a second FTS budget with a new phrase strategy, is
**rejected**. Why: `edgar.fts_query_budget` was set before any result and is not raised after seeing
results. The diagnostics also show the remaining `none` rows are mostly tables, which phrase search cannot
resolve, and no unqueried unmatched context was left. The README and resume bullet will say 129 of 142.

## T-006 Instance assignment below threshold: disclosed, not retuned (owner decision, 2026-10-02)

The assignment rate stays reported as below `contamination.instance_assignment_min_fraction`, disclosed,
with the threshold unchanged and `instance_match.*` values untouched. PRD section 4 prescribes rebuilding
contexts "with the benchmark's own serialization" before matching, and the Phase 0 token matcher is a
stand-in for that. The in-spec next step is to re-run assignment with the Phase 1 serializer and record
both rates.

## T-007 AUTONOMOUS DECISION - owner to review: bootstrap resampling unit is the company (CIK) (2026-10-02)

Decision (owner): the resampling cluster is the company (CIK), for the full set and for the shared frontier
sample. `between_filing_f1_sd` and `frontier.power_check` are computed **per company**, and that is stated
next to the number wherever it appears. Phase 3 draws the frontier sample so that a company is never split
across in/out of the sample (mechanics decided in Phase 3). Alternatives: resample filings, as PRD
sections 3 and 7 word it. Why: data wins over "one 10-K per company" (O-012). Ten companies contribute
two fiscal years, their filings are correlated, and 1,183 rows tie between two years of the same company,
so attributing those ties to a filing is arbitrary. The company cluster is the conservative choice.

## T-008 AUTONOMOUS DECISION - owner to review: unconfirmed FTS CIKs are excluded from training (2026-10-02)

Decision (owner): the 15 CIKs that FTS resolved but that have no present filing (O-010) are written to
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

Decision (owner, O-013): within one filing, when the serializer emits a text that also appears with a
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

Decision (owner, O-014 option a): continue the **same** seeded walk under the same rule, quotas and 50/50
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
