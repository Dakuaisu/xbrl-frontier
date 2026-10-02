# Findings register

Owner-run items in dependency order: `docs/owner/RUNBOOK.md` (it points back here).

Each entry: id, status (OPEN | OWNER-BLOCKED | RESOLVED), what was found, evidence, next action.

## O-001 SEC rejected the original declared User-Agent (RESOLVED)

- Status: RESOLVED (2026-10-02)
- Found: on 2026-10-01, requests to `efts.sec.gov/LATEST/search-index` and
  `data.sec.gov/submissions/` carrying the User-Agent
  `xbrl-frontier Dakuaisu 88734719+Dakuaisu@users.noreply.github.com` returned HTTP 403 from
  the Akamai edge, with the page titled "Your Request Originates from an Undeclared Automated Tool".
- Evidence: `curl -D -` response headers (`server: AkamaiGHost`, `HTTP/2 403`).
- Resolution: the owner replaced the declared User-Agent with `Adnan Rashid adnanrashid230@gmail.com`
  (CLAUDE.md commit 247238c). Re-checked on 2026-10-02: efts.sec.gov and data.sec.gov both
  return 200.
- Next action: none. The EDGAR client reads the User-Agent from one constant and enforces the
  fair-access throttle in code.

## O-002 Scorer details are not specified upstream (OPEN)

- Status: OPEN (the scorer choices are confirmed by a supervisor decision, owner to review, T-002; a cross-check waits on the FinTagging
  authors' reply to the O-010 issue)
- Found: no public FinNI scorer exists. Set vs multiset, normalization, aggregation details and
  parse handling are unspecified in FinTagging.
- Evidence: evidence_brief section 3. Our choices are in TRADEOFFS T-002.
- Next action: the owner reviews T-002. If the FinTagging authors publish or share their scorer,
  re-score with it as a cross-check and report any difference (do not swap metrics silently).

## O-003 `context_token_bucket` slice has no bucket edges in config (RESOLVED)

- Status: RESOLVED (2026-10-02): the supervisor approved (supervisor decision, owner to review) `tokenizers==0.23.2` and tercile edges.
  `eval.context_token_bucket_edges` was written once by `scripts/derive_token_bucket_edges.py` over the
  deduplicated set with the `Qwen/Qwen3-1.7B` tokenizer at its pin (identical in both students).
  Buckets hold 8,627 / 8,677 / 8,665 instances. See TRADEOFFS T-001a.
- Found: `eval.slices` lists `context_token_bucket`, but `thresholds.yaml` defines no bucket edges or
  tokenizer, and the config is frozen. `xbrl_frontier/eval_sets.py` raises rather than invent edges.
- Evidence: `grep context_token docs/config/thresholds.yaml`.
- Next action: none.

## O-004 `company` and `fiscal_year` slices wait on the instance map (RESOLVED)

- Status: RESOLVED (2026-10-02): `eval_sets.load_instance_map()` joins the shipped map and manifest;
  tested in `tests/test_oracles.py::test_company_and_fiscal_year_slices_from_shipped_map`.
- Found: both slice keys come from `pins.eval_instance_map`. `eval_sets.slice_keys` reads them from that
  map and labels unmapped instances `unassigned`.

## O-005 Kaggle T4 Phase 0 runs: throughput benchmark, smoke quantization, T4 smoke test (OWNER-BLOCKED via O-025)

- Status: OWNER-BLOCKED via O-025. Under T-033 the builder runs this with the `kaggle` CLI; the only owner action
  is the phone verification in O-025. Pins are filled (O-007).
- Ready: `notebooks/phase0_kaggle_t4.ipynb`, `serving/phase0_benchmark.py`, `serving/quantize_base.py`,
  `serving/smoke_test.py`, `serving/common.py` (tested in `tests/test_cost_and_serving.py`, no GPU needed).
- Builder steps (RUNBOOK B1, C1): `git archive --format=tar.gz -o /tmp/xbrl-frontier-src.tar.gz HEAD`, then
  `kaggle datasets create`/`version` for `xbrl-frontier-src`, then push the notebook as a kernel with
  `--accelerator NvidiaTeslaT4` and internet on, then `kaggle kernels status` and `kaggle kernels output`.
- Artefacts: `data/phase0/phase0_kaggle_t4/benchmark_*.json`, `data/phase0/smoke_kaggle_t4/smoke_kaggle_t4.json`,
  `data/phase0/smoke_checkpoints/` (also needed by O-006).
- Consumers: `scripts/phase0_cost.py --results data/phase0/phase0_kaggle_t4` (break-even and gate-pass goodput);
  `scripts/smoke_summary.py` (`serving.smoke_test.failure_action`: writes `data/phase0/serving_variants.json` with the
  failed variants removed and their errors recorded).

## O-006 RTX 3050 (WSL2) smoke test (OWNER-BLOCKED)

- Status: OWNER-BLOCKED (the laptop is owner-only; it needs O-005's `smoke_checkpoints`, so O-025 first).
- Ready: `serving/run_rtx3050_smoke.sh`, which runs `serving.smoke_test --card rtx3050_wsl2` with
  `serving.hardware.rtx3050_wsl2.vllm_flags`.
- Owner action (WSL2 with the NVIDIA WSL CUDA driver; RUNBOOK C2): copy the repo at the commit I name and the
  unpacked `smoke_checkpoints/` to the laptop, then from the repo root run
  `serving/run_rtx3050_smoke.sh /path/to/smoke_checkpoints`.
- Artefact: `data/phase0/rtx3050_wsl2/smoke_rtx3050_wsl2.json` (and `data/phase0/rtx3050_wsl2.tar.gz`, printed by the script);
  copy it back into the same path on the Mac.
- Consumer: `scripts/smoke_summary.py` (applies `serving.smoke_test.failure_action`; PRD risk: the 1.7B may not fit, in
  which case the 3050 row is 0.6B only and says so).

## O-007 Kaggle/WSL2 software pins need approval (RESOLVED)

- Status: RESOLVED (2026-10-02): the supervisor approved the three pins (supervisor decision, owner to review) and they are filled in
  `pins.software` (TRADEOFFS T-004 item 7).
- Found: `pins.software` is empty. Phase 0 remote runs need vllm, llmcompressor and autoawq. Proposed
  versions (current PyPI releases, checked 2026-10-02): vllm 0.30.0 (2026-09-22), llmcompressor 0.14.0
  (2026-09-22), autoawq 0.2.9 (2025-05-11, unmaintained). Transformers, torch and compressed-tensors
  are left to vllm/llmcompressor's resolver, and the installed versions are recorded in every run's `env`.
- Next action: none. The other `pins.software` fields are filled when their phase first installs them.

## O-008 AWS g4dn.xlarge T4 price is blank in the sheet (RESOLVED)

- Status: RESOLVED (2026-10-02)
- Found: `gpu_per_hour.aws.g4dn_xlarge_T4` was null and could only be filled from a dated observation
  with a source URL.
- Resolution: observed on 2026-10-02T04:45:03Z from AWS's public on-demand price file for us-east-1 Linux
  (the file the AWS EC2 pricing page loads; publication date 2026-09-25). The value, record, rate code,
  URL and file hash are in the price sheet under `gpu_per_hour.aws_g4dn_xlarge_T4_observation`. The
  sheet's top-level `observed` date stays 2026-10-01, and this source carries its own date. All three T4
  sources now appear in `scripts/phase0_cost.py`.

## O-009 "A Schema-Aware Large Language Model for XBRL Tagging and Cross-Statement Consistency Checking" could not be read (OWNER-BLOCKED)

- Status: OWNER-BLOCKED (PRD section 10 risk: possible scoop; needs a logged-in browser or the authors; RUNBOOK A1)
- Found (metadata only): Zhaoyue Peng, Shuncheng Liu, Xuhao Meng, Zhe Su, Ningjing Sang; posted
  2026-07-07; DOI 10.13140/RG.2.2.19518.86080 (a ResearchGate-minted DOI); OpenAlex W7167572144 lists it
  as an article with no PDF location, no abstract and 0 citations.
- Attempts on 2026-10-02: the DOI resolves to `researchgate.net/doi/10.13140/RG.2.2.19518.86080`, which
  returns HTTP 403 ("ResearchGate - Temporarily Unavailable"). ResearchGate search also returns 403.
  Crossref has no record. Semantic Scholar does not know the DOI (its title search was rate-limited, 429).
  The arXiv title search returned nothing. An Exa library page has the metadata only.
- Retry on 2026-10-02 14:05 UTC: the Semantic Scholar title search is still rate-limited (429) and its DOI lookup says "not
  found"; OpenAlex is unchanged (no abstract, no PDF location, 0 citations); Unpaywall returns 404 for the DOI; the Crossref title
  query finds no match; a web search finds only the same metadata page (all five authors with h-index 0) and unrelated papers.
  No full text or abstract is reachable from here.
- Evidence of scope: none beyond the title. The title suggests concept tagging plus cross-statement
  consistency checks, not a serving cost/latency frontier or a small-model learning curve. That is
  unverified until the text is read.
- Owner action (RUNBOOK A1): open DOI 10.13140/RG.2.2.19518.86080 in a logged-in browser (ResearchGate often needs
  an account) or email the authors; save the PDF **outside** the repo.
- Artefact: the PDF's local path (tell me).
- Consumer: no script. I write a summary against PRD section 1 into this entry and reposition the README if needed.
  No further builder retries are useful: every open index has been checked twice.

## O-010 Eval manifest recovered 129 of 142 registrants: `eval_manifest_required_recovered` NOT MET (OWNER-BLOCKED)

- Status: OWNER-BLOCKED for option (a). `contamination.eval_manifest_required_recovered` is **NOT MET**
  at 129/142, and the threshold is unchanged. Supervisor decision (owner to review; TRADEOFFS T-005): (c) accept and disclose
  now, (a) owner-blocked, (b) rejected.
- Found: `data/eval_manifest.csv` lists 139 filings from 129 companies. 117 filings come from the 111
  seed CIKs and 22 from 18 CIKs found by FTS. Filing dates run 2024-02-06 to 2025-02-13 (fiscal years
  2023 and 2024 only). The `edgar.fts_query_budget` total is spent (1,966 queries across two runs: 33
  single-CIK resolutions to new CIKs, 304 to known CIKs, 1,629 ambiguous or no hits). 15 FTS-found CIKs
  are unconfirmed: their in-window 10-Ks hold no (or fewer than
  `instance_match.min_unique_exact_for_filing`) uniquely exact-matching contexts, so the phrase hit was
  shared wording (diagnostics in `data/manifest_work/report.json`).
- Evidence: `data/manifest_work/report.json`, `data/manifest_work/fts_queries.json`.
- The 15 unconfirmed CIKs are listed in `data/eval_cik_candidates_unconfirmed.txt` and excluded from
  training through `contamination.exclude_ciks_from` (T-008).
- Owner action for (a) (RUNBOOK A2): first approve or edit the text in `docs/owner/fintagging_issue_draft.md`
  (T-033 item 6: posting comes after the owner approves the text), then post it at
  `https://github.com/The-FinAI/FinTagging/issues`, adding the repository link where marked.
- Artefact: the issue URL (tell me; I record it here). If the authors reply with filings, save their list as
  `data/manifest_work/authors_filing_list.*` (any format they send).
- Consumers: `scripts/build_eval_manifest.py` (manifest from their list), `scripts/rebuild_eval_contexts.py`
  (assignment), `scripts/validate_instance_map.py`; both the old and the new numbers are reported.
- What I need back: the issue URL. If the authors reply with the filing list, I rebuild the manifest from
  it, re-run assignment, and report both the old and new numbers. The README and resume bullet say
  "129 of 142" until then.

## O-011 Instance assignment rate: 94.29% with the Phase 0 token matcher, 95.25% with the benchmark serializer (RESOLVED)

- Status: RESOLVED (2026-10-02). With the PRD section 4 method (the benchmark's own serialization, then
  the fuzzy fallback), the rate is **95.25% raw / 95.39% dedup**, which meets
  `contamination.instance_assignment_min_fraction` with the threshold and `instance_match.*` unchanged. Both
  rates are recorded: the Phase 0 token matcher alone gave 94.29% / 94.43% (run 1 before
  de-doubling: 90.10% / 91.30%). Serializer exact strings alone cover 93.09% of rows. Source:
  `data/manifest_work/serializer_rebuild.json`.
- Phase 1 map methods (rows): serializer_exact 25,770; serializer_exact_tie_within_company 1,029;
  serializer_ambiguous_cross_company 243; token fallback: exact 107, fuzzy 514, none 1,115, too_short 6,
  ambiguous 3. Versus Phase 0: 278 rows newly assigned, 0 lost, and 717 moved to another filing, all
  within the same company (year ties).
- Validation of the Phase 1 map (`scripts/validate_instance_map.py`, positive instances): gold values found
  in the assigned filing are 1.000 for serializer_exact (13,185), serializer ties (306) and token_exact (32),
  and 0.819 for token_fuzzy (239; 23 below half). **This is not a regression of the matcher.** In
  Phase 0 the fuzzy method scored 0.959 over 1,062 instances, most of them easy table contexts that
  failed exact token matching only because of doubled cells. The serializer now takes all of those as
  exact strings, so the fuzzy fallback sees only the hard residue the serializer cannot reproduce. The
  two figures cover different populations and are not comparable. The Phase 0 matcher and thresholds are
  unchanged.
- History (Phase 0):
- Found: 27,142 of 28,787 rows are assigned. Unassigned: 1,201 `none`, 372 `ambiguous_cross_company`
  (text shared verbatim by filings of different companies, for example parent and subsidiary), and
  72 `too_short`. Per PRD section 4 these form the unassigned stratum, excluded from the frontier sample,
  and the rate is disclosed. Run 1 (before the doubled-segment normalization, T-003 item 6) gave
  90.10% raw / 91.30% dedup.
- Validation (`scripts/validate_instance_map.py`, all assigned positive instances): the share of gold fact
  values found in the assigned filing is 1.000 for exact (12,225) and exact-tie (370), 0.959 for fuzzy
  (1,062; 23 below half) and 1.000 for fuzzy-tie (12).
- Next action: none. The unassigned stratum (4.75% of rows) stays excluded from the frontier sample and is
  disclosed.

## O-012 FinNI draws on more than one 10-K per company (RESOLVED by supervisor decision T-007, owner to review)

- Status: RESOLVED (2026-10-02): the resampling cluster is the company (CIK). Phase 3 must draw the
  frontier sample so that no company is split across in/out of the sample.
- Found: 10 of 129 companies contribute two fiscal years (FY2023 and FY2024), for 139 filings in total.
  Evidence: year-specific sentences that occur only in one year's filing (T-003 item 2), plus
  `unique_exact_contexts` of at least `instance_match.min_unique_exact_for_filing` for both filings of
  each such company in `data/eval_manifest.csv`. 1,183 rows tie between two filings of the same
  company and go to that company's dominant filing (`*_tie_within_company`).
- Consequences: (1) "Bootstrap over filings" is ambiguous where a company has two filings and some
  rows tie between them. (2) The PRD's instances-per-filing (rows / 142) is a planning figure. The
  Phase 3 probe measures real per-filing tokens anyway.
- Next action: `between_filing_f1_sd` and `frontier.power_check` are computed per company, and that is
  stated next to each number. Sampling mechanics are decided in Phase 3.

## O-013 The benchmark serializer emits the same text with different labels; FinNI inherits it (RESOLVED)

- Status: RESOLVED (2026-10-02): supervisor decision (owner to review), recorded in TRADEOFFS T-010. In training, drop the negative twin.
  The eval set is untouched.
- Found: in FinTagging's `extract_text`, a `div` whose only content is a `p` yields the same text twice.
  The `div` comes out as a **negative**, because its facts are not direct children, and the `p` as a
  **positive**. Negatives never enter the seen-text set, so both survive. On the 25,770 serializer-exact
  rows, 4,746 contexts have more than one label set in their filing. This is the likely origin of the
  FinNI duplicate contexts with differing answers (204 groups per the evidence brief). Label parity against
  the gold, taking the closest candidate per context, is 25,768/25,770 identical sets with gold recall
  1.000. So the port is faithful, and the gold itself carries this ambiguity.
- Evidence: `data/manifest_work/serializer_rebuild.json` (`label_parity_on_serializer_exact`),
  `tests/test_ixbrl_facts.py::test_text_blocks_follow_notebook_rules`.
- Next action: none. `scripts/serialize_corpus.py` applies T-010 and counts the dropped twins per filing
  and in total. The format-parity script reports the divergence from the benchmark's conventions.

## O-014 The frozen accession list yields 17,094 positives, short of what the training target needs (RESOLVED)

- Status: RESOLVED (2026-10-02): the list was extended once with the same walk (TRADEOFFS T-012). 338 filings
  give 27,234 positives against a need of 25,200.
- Found: the 208 filings (sized by the eval set's planning instances-per-filing, T-011 item 3) serialize to
  321,080 instances. 17,094 are positives (10-K: 11,270 over 104 filings, mean 108.4; 10-Q: 5,824 over
  104, mean 56.0) and 303,986 are negatives after dropping 4,511 twins. At `training.negatives_fraction`,
  `target_train_instances` + `dev_slice_instances` need (40,000 + 2,000) x (1 - 0.40) = 25,200 positives.
  With all 17,094 positives the corpus tops out at about 28,490 instances, so the last
  `training.curve_points` value cannot be reached. Cause: the planning figure (about 203 instances per
  filing) counts the eval's sampled negatives, while these filings give about 82 positives per filing
  (10-Qs about half of 10-Ks).
- Evidence: `data/corpus_serialize_report.json` (per-filing counts).
- Next action: owner decision. Options: (a) extend the list by continuing the **same** seeded walk under the
  same rule and quotas, sized from the measured positives per filing (about 99 more filings), leaving the
  first 208 unchanged and recording the amendment and both sizes; (b) keep 208 and accept a smaller corpus,
  which would mean changing `training.target_train_instances` / `curve_points` (a spec change); (c) shift
  the form share toward 10-K for the extension (about twice the positives per filing; the eval is all 10-K).

## O-015 First format-parity comparison: positives match, negatives diverge (RESOLVED for negatives; see O-017)

- Status: RESOLVED (2026-10-02) for the negative divergence: after sampling (T-013), the negatives match the
  eval set. Sampled-corpus report: `data/format_parity_report_sampled.json`. The pool report
  `data/format_parity_report.json` is kept. Sampled negatives vs eval: table share 0.2041 vs 0.2042,
  tokens p10/p50/p90/p99 25/59/1,043/3,635 vs 25/60/1,132/3,656, mean 323 vs 346. The remaining gap is
  in the long tail (p90 about 8% shorter) and is left as a finding, with no further knobs. A new positive-side
  divergence caused by decontamination is O-017.
- Found (`scripts/format_parity.py` -> `data/format_parity_report.json`; eval = FinNI dedup set, train = the
  serialized corpus after twin drop):
  - Positives: table share 0.530 (eval) vs 0.516 (train). Context tokens p10/p50/p90/p99 57/293/1,792/5,684
    vs 62/273/1,739/4,845. Above `training.kaggle.max_seq_len_tokens`: 2.2% vs 1.6%. Facts per positive
    median 5 vs 4, mean 14.6 vs 12.3. Datatype mix: monetary 0.892 vs 0.878, percent 0.047 vs 0.037,
    shares 0.036 vs 0.051, perShare 0.021 vs 0.029, integer 0.004 vs 0.004.
  - HTML tag vocabulary: identical. Doubled-cell share: 0.147 (eval) vs 0.115 (train, all) / 0.141 (train
    with FinTagging's negative filters), so the serializer reproduces the artifact.
  - Negatives: fraction 0.377 (eval) vs 0.947 (train, unsampled). Table share of negatives 0.204 vs 0.034
    (train negatives passing FinTagging's filters). Negative median tokens 60 vs 50, but p90 1,132 vs 103,
    because the eval keeps long negative tables. Twin-drop divergence: 4,511 twins not written (T-010).
- Reading: the serializer and labels are at parity. The divergences are sampling divergences.
  FinTagging's sampler capped negatives per filing and kept table negatives with at least 50 digits.
  Ours has not sampled yet.
- Next action: the corpus-build sampler draws negatives to `training.negatives_fraction`, stratified by
  context type so the negatives' table share tracks the eval's. The parity script is re-run on the sampled
  corpus and both reports are kept.

## O-016 Corpus is below both caps after decontamination (RESOLVED by T-016)

- Status: RESOLVED (2026-10-02): train on 38,788 / dev 1,933. Config values are unchanged, and the
  disclosure "38,788 (cap 40,000 not reached after decontamination, O-016)" goes everywhere (T-016).
- Found: the contamination audit (T-014) flags 18,184 of 215,284 eligible instances (8.4%), including 2,801
  positives. After removal: train 23,273 positives -> 38,788 instances at `negatives_fraction`, against
  `target_train_instances` 40,000. Dev 1,160 positives -> 1,933 against `dev_slice_instances` 2,000.
  The T-012 margin (10%) covered sizing variance but not an 8-10% removal of positives.
  Without decontamination both caps were reached (v0: 40,000 / 2,000).
- Evidence: `data/corpus_report.json`, `data/corpus_report_v0_unaudited.json`,
  `data/contamination_audit/summary_pool.json`.
- Consequence: the top `training.curve_points` value (40,000) is not reachable as a prefix. All lower points
  are.
- Next action: owner decision. (a) A second extension, sized by the post-audit positive yield (T-012 said
  "extended once", so this needs the owner); (b) train on 38,788 and record the top curve point as 38,788,
  not 40,000; (c) leave as is until the Phase 2 pilot shows whether the curve has flattened before 40,000.

## O-017 Decontamination removes long table positives disproportionately (OPEN)

- Status: OPEN (finding; nothing is retuned). It is carried into the generated contamination statement and README,
  and is reported next to the table slice once the slice is measured.
- Found: positives in the pool (before the audit) matched the eval (table share 0.516 vs 0.530, median tokens
  273 vs 293). In the decontaminated sampled corpus, positives drift: table share 0.480 vs 0.530, tokens
  p50/p90/p99 236/1,438/4,309 vs 293/1,792/5,684, facts per positive mean 11.0 vs 14.6, doubled-cell share
  0.106 vs 0.147. Of the 2,801 flagged positives, 2,255 are tables. A long table has more 13-grams, so it
  is more likely to share one with an eval context (period headers, standard line items).
- Evidence: `data/format_parity_report_sampled.json`, `data/contamination_audit/flags_pool.tsv`.
- Reading: the 13-gram rule is spec (`contamination.ngram_n`) and is applied as written. The cost is a
  corpus that is somewhat shorter and less table-heavy than the eval. That bears on the tables-vs-text
  finding the project is after, so it is reported next to the table slice, not hidden.
- Next action: report it in the contamination statement and next to the table slice. No change to the audit.

## O-018 Contamination statement checklist for Phase 5 (OPEN, carried)

- Status: OPEN. The draft is generated (`docs/drafts/contamination_statement_draft.md`, also embedded in `README.md`) with
  items (1)-(4). The fresh-slice gaps and the label-noise rate stay blank until the O-019 (D9), O-022 (E4) and
  O-024 results arrive.
- The statement text must include, not only reference: (1) the fresh slice's 6 `fill=relaxed` rows
  (`data/fresh_accessions.txt`, column `fill`) and the two strata cells unfillable in the window
  (10-K | real estate & construction | large_accelerated: 5; 10-K | energy & transportation |
  large_accelerated: 1; December year-end filers file no 10-K between 2026-05-19 and 2026-10-02), T-015;
  (2) the audit counts before and after removal (18,184 of 215,284 flagged; 0 of 40,721 after) and the
  decontamination skew of positives (O-017); (3) the corpus sizes below cap (T-016); (4) the fresh-slice
  instance file pinned by sha256 in `data/fresh_selection_report.json`.
- Progress (2026-10-02): `scripts/contamination_statement.py` generates `docs/drafts/contamination_statement_draft.md`
  with items (1)-(4) from recorded files. The fresh-slice gaps and the label-noise rate stay blank until measured.

## O-019 Kaggle runs: training runs 1-5, encoder, predictions (zero-shot, full set, dev, fresh), export (OWNER-BLOCKED via O-025)

- 2026-10-02 (T-033): Kaggle is builder-run now, but blocked by O-025 (no GPU or internet until the account is
  phone-verified). The step lists below stay valid; the builder runs them with `kaggle kernels push/status/output`.

- Status: OWNER-BLOCKED (Kaggle is owner-only). The training-stack pins are filled (O-020 RESOLVED, T-019);
  only the Kaggle sessions remain. Start with owner step 1, the tarball.
- Ready: `training/kaggle_train.py` (resumable QLoRA, fixed order, curve checkpoints, T-018),
  `training/schedule.py` (tested in `tests/test_training_schedule.py`) and
  `notebooks/phase2_kaggle_train.ipynb`.
- Runs, in this order: (1) `Qwen/Qwen3-1.7B`, seed 1, `CURVE = True`; (2) `Qwen/Qwen3-1.7B`, seed 2,
  `CURVE = False`; (3) `Qwen/Qwen3-0.6B`, seed 1, `CURVE = False`. Required: the Mac pilot measured 560.6
  tokens/s, below `training.mac.min_tokens_per_sec_to_train_0p6b_locally` (O-021). Cut order: seed 2 is the first cut.
- Owner steps:
  1. Build the source tarball from the commit that filled the O-020 pins (or any later commit):
     `git archive --format=tar.gz -o /tmp/xbrl-frontier-src.tar.gz HEAD`, and upload it as a new version of
     the Kaggle dataset `xbrl-frontier-src`.
  2. Upload `data/corpus/train.jsonl`, `data/corpus/dev.jsonl` (sha256 in `data/corpus_report.json`) and
     `data/fresh/instances.jsonl` renamed `fresh_instances.jsonl` (sha256 in `data/fresh_selection_report.json`) as a private
     Kaggle dataset `xbrl-frontier-corpus`. The scripts refuse mismatching files.
  3. New notebook from `notebooks/phase2_kaggle_train.ipynb`. Accelerator GPU T4, Internet on. Attach both
     datasets, set `MODEL`, `SEED`, `CURVE` in the first code cell, and leave `RESUME = None`. Run all.
  4. When the session ends (finished or at the session limit), save the output as a Kaggle dataset, e.g.
     `xbrl-frontier-run-1p7b-s1`. To continue, open a new session, attach that dataset as well, set
     `RESUME` to its path, and run all again. Repeat until `run.json` lists `final_adapter`.
  5. Keep total use under `training.kaggle.gpu_hours_cap_total`. Each session's train time is in `run.json`.
- What I need back: the final output directory of each run (`run.json`, `final_adapter/`, `curve/`),
  unpacked into `data/phase2/kaggle/<run name>/`.
- **Run 5, curve validation** (`notebooks/phase2_kaggle_train.ipynb`, T-030). Schedule: after run 1, before run 2
  (the seed-2 run is the first item in the cut order; curve validation is the fourth, so it runs first):
  1. Same notebook and datasets as run 1. Set `MODEL = 'Qwen/Qwen3-1.7B'`, `SEED = 1`, `CURVE = False`,
     `VALIDATION = True`, `RESUME = None`. Run all, resuming across sessions as in step 4 above. The output
     directory is `run-Qwen3-1.7B-s1-validation`.
  2. Run `notebooks/kaggle_generate_predictions.ipynb` with `SET = 'dev'` and `ADAPTERS` = this run's `final_adapter`
     and run 1's `curve/prefix_10000`. Later, once the frontier sample is pinned, repeat with `SET = 'finni'` (the
     sample ids are scored with `--ids`).
  3. Return the run's `run.json` and the predictions files, unpacked into `data/phase2/kaggle/`. I run
     `scripts/score_predictions.py` on each, then `scripts/curve_validation.py`.
- **Run 4, encoder baseline** (`notebooks/phase2_kaggle_encoder.ipynb`, T-020):
  1. Attach `xbrl-frontier-src` and `xbrl-frontier-corpus`. Set `MODE = 'train'`, `RESUME = None`. Run all.
     Save the output as a dataset (e.g. `xbrl-frontier-run-encoder`). If the session ends early, start a new
     one with that dataset attached and `RESUME` set to its path. The PRD estimates about 3 T4-hours.
  2. When `run.json` lists `final_model`, start a session with `MODE = 'predict'`, `MODEL_DIR` set to that
     `final_model/` path and `SET = 'finni'`. Repeat with `SET = 'dev'` and `SET = 'fresh'`.
  3. Return `run.json` and the three `encoder_predictions_<set>.jsonl` files (+ `.meta.json`), unpacked into
     `data/phase2/kaggle/encoder/`. I score them locally with `scripts/score_predictions.py`.
- **Prediction runs for scoring** (`notebooks/kaggle_generate_predictions.ipynb`, T-021), after run 1:
  1. Attach `xbrl-frontier-src`, `xbrl-frontier-corpus` and the run-1 output dataset. Set
     `MODEL = 'Qwen/Qwen3-1.7B'`, `SET = 'dev'` and `ADAPTERS` to the run's `curve/prefix_*` directories plus
     `final_adapter`. Run all.
  2. Return the `predictions_<name>_dev.jsonl` files and `predictions_dev.meta.json`, unpacked into
     `data/phase2/kaggle/predictions/`.
- **Full-set predictions, fine-tuned students** (`kaggle_generate_predictions.ipynb`), once each run has a
  `final_adapter`: set `MODEL`, `SET = 'finni'`, `BASE = False`, `ADAPTERS = {'<run>': '<final_adapter dir>'}`. Run all.
  Return the predictions file and meta. The 1.7B seed-1 result feeds the power check (`scripts/power_check.py`),
  which unlocks the frontier sample (T-026).
- **Zero-shot base runs** (`gates.finni_f1_gain_over_zero_shot_min`; T-031), no training needed, so they can run any
  time:
  1. `MODEL = 'Qwen/Qwen3-1.7B'`, `SET = 'finni'`, `BASE = True`, `ADAPTERS = {}`. Run all and return
     `predictions_base_finni.jsonl` and its meta into `data/phase2/kaggle/predictions/zeroshot-1p7b/`.
  2. The same with `MODEL = 'Qwen/Qwen3-0.6B'`, into `.../zeroshot-0p6b/`.
  3. The full set covers the frontier sample, so the sample comparison needs no extra run: I score the same
     predictions with `scripts/score_predictions.py --ids data/frontier_sample_ids.txt` once the sample is pinned.
  4. I run `scripts/score_predictions.py --set finni` on each, then `scripts/gate_summary.py --student-full ...
     --zero-shot-full ...` (plus the `--student-sample`/`--zero-shot-sample` pair).
- **Export runs** (`notebooks/phase2_kaggle_export.ipynb`, T-023), after a training run finishes (1.7B seed 1 first;
  INT8 is the second item in the cut order):
  1. Attach `xbrl-frontier-src`, `xbrl-frontier-corpus` and the run's output dataset. Set `MODEL` and `ADAPTER`
     (the run's `final_adapter/`). Run all: merge, then int8, then awq.
  2. Save the output as a dataset (e.g. `xbrl-frontier-export-1p7b-s1`) holding `merged-fp16/`, `int8-w8a8/`,
     `awq-int4/` and `export.json`.
  3. Run `notebooks/kaggle_generate_predictions.ipynb` with `SET = 'finni'` and `CHECKPOINT = '<name>=<dir>'` for each exported variant (fp16 merged,
     INT8, AWQ) and return the predictions files to `data/phase2/kaggle/predictions/`. Then
     `scripts/score_predictions.py` and `scripts/flip_rate.py` (fp16 vs each quantized variant) decide inclusion
     under `gates.quantized_f1_drop_max` and `gates.quantized_flip_rate_max` before any load test.

## O-020 Kaggle training stack pins need approval (RESOLVED)

- Status: RESOLVED (2026-10-02): supervisor-approved (supervisor decision, owner to review) and filled (T-019).
- Proposed versions (current PyPI releases, checked 2026-10-02): transformers 5.18.0 (2026-09-30; also the
  version mlx-lm installed locally, so tokenization matches the Mac), peft 0.21.2 (2026-10-01),
  bitsandbytes 0.50.2 (2026-08-27), accelerate 1.15.0 (2026-09-09). Torch is Kaggle's preinstalled build,
  recorded in each run's `env`. `trl` and `unsloth` are not used (T-018).
- Next action: none.

## O-021 Mac pilot: 0.6B trains at 560.6 processed tokens/s, below the local-training threshold (RESOLVED)

- Status: RESOLVED (2026-10-02). Per PRD section 5 the 0.6B full run goes to Kaggle (O-019 run 3 is
  required, not conditional) and the MLX-to-PEFT converter is not built.
- Measured (`data/phase2_mac_pilot_Qwen3-0.6B.json`; M1 Pro 16 GB, mlx 0.32.3 / mlx_lm 0.32.0, 4-bit base,
  T-017 settings): 3,000 pilot instances (351 of the first 3,351 skipped as over 2,048 tokens), 187
  optimizer steps, train time 4,196 s, 2,352,306 processed tokens = **560.6 tokens/s** against
  `training.mac.min_tokens_per_sec_to_train_0p6b_locally`. Loss tokens 200,421 = 47.8 tokens/s (mlx-lm's
  figure). Peak memory 2.68 GB with gradient checkpointing.
- Settled by the pilot: the prompt format and completion-only masking run end to end. Val loss went from
  0.879 to 0.044 and train loss from 4.18 (first report) to 0.021. Format check on the 200 val instances:
  parse-failure rate 0.000, malformed items 0. Pilot-only F1 0.229 (P 0.823, R 0.133), labelled as a format
  check and not a result. Low recall after 187 optimizer steps on the shortest 90% of contexts says
  nothing yet about the full run, which the curve will show. The LR stays at `training.learning_rate`: the
  pilot ran one setting, and loss fell without instability.


## O-022 Live frontier-arm steps: probe, accuracy run, latency sample, fresh-slice control (OPEN; builder-run under T-033)

- Status: OPEN. Under owner decision T-033 the builder runs these for `gemini-3.5-flash-lite` only
  (`frontier.running_arms`), on the free tier, synchronously, with billed spend $0 (T-034). `gpt-6-luna` and
  `gpt-5.4-mini-2026-03-17` are not run (no key). Prerequisite: the frontier sample is pinned (D4: full-set
  predictions -> `scripts/power_check.py` -> `scripts/draw_frontier_sample.py`).
- Commands (builder): `.venv/bin/python -m frontier.harness run --step <probe|batch|latency|fresh_control> --arm
  gemini-3.5-flash-lite <instances> --live --approval T-033`, rerun after each `paused_daily_quota` until
  `status.json` says `complete`. Scoring: `scripts/score_predictions.py` (`--ids data/frontier_sample_ids.txt` for
  the accuracy run; `--set fresh` for the control).
- Plan: about 8,600 requests (`data/phase3/frontier_plan.json`), paced by the learned free-tier limits; the
  previous paid-arm plan is superseded.

## O-023 Phase 4 load tests on the T4 and the RTX 3050 (T4: OWNER-BLOCKED via O-025; 3050: OWNER-BLOCKED)

- Status: the T4 part is builder-run under T-033 and blocked on O-025. The 3050 part is owner-only. Prerequisites:
  exported variants that passed `gates.quantized_*` (O-019 export) and the pinned latency ids
  (`data/frontier_latency_ids.txt`, T-026).
- Ready: `serving/load_test.py`, `notebooks/phase4_kaggle_load.ipynb`, `serving/run_rtx3050_load.sh` (T-027).
- Builder (T4, RUNBOOK E5): push `notebooks/phase4_kaggle_load.ipynb` as a kernel (source from a commit that contains the
  latency ids; export datasets attached). About 6 GPU-hours in total (PRD section 9).
  Artefacts: `data/phase4/kaggle_t4/load_kaggle_t4_<model>_<variant>.json`.
- Owner action (3050, RUNBOOK E6): copy the repo (same commit) and the export directories to the laptop, then run
  `serving/run_rtx3050_load.sh <export of 0.6B> <export of 1.7B>`.
  Artefacts: `data/phase4/rtx3050_wsl2/load_rtx3050_wsl2_<model>_<variant>.json`; copy them back into the same path.
- Consumers: `scripts/cost_model.py` (T4 only), `scripts/charts.py latency` (both cards; 3050 rows carry
  `serving.hardware.rtx3050_wsl2.cost_label` and no dollars), `scripts/charts.py pareto`.

## O-024 Label audit, two hand passes (OWNER-BLOCKED)

- Status: OWNER-BLOCKED (the label audit is the owner's human instrument)
- Ready: `data/label_audit/pass1_sheet.csv` (300 instances, 12 strata) and `pass2_sheet.csv` (100 of them, blind),
  drawn by `scripts/label_audit.py draw` (T-029).
- Owner steps:
  1. For each row of `pass1_sheet.csv`, open `filing_url`, find the context, and check the gold pairs against the
     iXBRL tags. Fill `verdict (correct|incorrect)` and the counts of `missing_facts`, `spurious_facts` and
     `wrong_type_facts` (0 when none), plus `notes`. Commit the sheet. PRD section 9 estimates 6 hours for both passes.
  2. At least a week later, fill `pass2_sheet.csv` the same way without looking at pass 1, and commit it.
  3. Tell me when both are committed.
- Artefacts: the filled `data/label_audit/pass1_sheet.csv` and `pass2_sheet.csv` (committed).
- Consumers: `scripts/label_audit.py score` (refuses blank verdicts; writes `data/label_audit/audit_result.json` with the
  weighted error rate and self-agreement), then `scripts/contamination_statement.py` and `scripts/readme.py`.

## O-025 Kaggle account runs kernels without GPU or internet (OWNER-BLOCKED)

- Status: OWNER-BLOCKED (all Kaggle work stopped, per the T-033 instruction to stop if Kaggle refuses)
- Found (2026-10-02 12:22 UTC): the CLI authenticates as `adnanrashid29` and `kaggle quota` shows 30.00 h of weekly
  GPU (0.00 h used, refresh 2026-10-03T00:00). The check kernel `adnanrashid29/xbrl-frontier-gpu-check`
  (`kaggle/kernels/gpu-check/`) was pushed with `enable_gpu: true`, `enable_internet: true`,
  `machine_shape: NvidiaTeslaT4`, and the server stored those settings (`kaggle kernels pull -m`). The run
  completed **without a GPU** (`nvidia-smi` absent, `torch 2.10.0+cpu`, `cuda_available: false`) and **without
  internet** (DNS failure for huggingface.co, pypi.org and www.sec.gov). GPU quota used stayed at 0.00 h. This is
  how Kaggle treats accounts that have not completed phone verification: GPU accelerators and internet access in
  notebooks both require it, and the request is silently downgraded to CPU with no network.
- Evidence: `data/kaggle/gpu-check/gpu_check.json` and `data/kaggle/gpu-check/xbrl-frontier-gpu-check.log`.
- Exact fix (owner): sign in at https://www.kaggle.com as adnanrashid29, open **Settings**
  (https://www.kaggle.com/settings) and complete **Phone verification** (enter a phone number, then the SMS code).
  No other change is needed: the token in the environment stays as it is.
- Artefact: none (tell me it is done). Consumer: the builder re-pushes `kaggle/kernels/gpu-check` and records the
  result in `data/kaggle/gpu-check/` (RUNBOOK B, C, D, E5 follow).
- Re-check cadence (from 2026-10-02): at most once per day, in the same turn as the day's E4 report. A negative result is a
  one-line WORKLOG entry, not a commit; only a changed result (GPU or internet appears) is committed.
- Next action: after verification, I re-push the same check kernel (`.venv/bin/kaggle kernels push -p
  kaggle/kernels/gpu-check --accelerator NvidiaTeslaT4`) and confirm a T4 with internet. Then I continue with
  RUNBOOK B1, B2, C1 and D1-D11, then E5. All Gemini steps (E1-E4) wait for D4 (the pinned frontier sample).

## O-026 One unreproduced full-suite failure (RESOLVED: not reproduced)

- Status: RESOLVED (2026-10-02): the closing criterion was met. Runs 6-15 after the two hazard fixes passed 88/88
  (with `-rf`, three of them concurrent with E4). The failing test's name stays unknown (output not captured).
  If any failure recurs, it is reopened with the name and text. Enforcement note: the pre-commit hook still runs only
  the scorer subset (`tests/test_scorer_unit.py`, `test_oracles.py`, `test_score_predictions.py`, `test_bio.py`).
  Running the full suite with `-rf` before every commit is a manual discipline, not enforced by the hook.
- Found: on 2026-10-02 at about 12:10 UTC, during the T-033 step-2 work, one full-suite run (right after editing
  `tests/test_readme.py`) reported `1 failed, 87 passed`. Its output went through `tail -1`, so the test name and
  failure text were not captured. That is my error, and those details cannot be recovered. Since then 8 full-suite
  runs (3 that day, 5 in a dedicated rerun loop, all with `-rf`) passed 88/88, and 3 more are running with
  `-rf` while E4 collects in parallel.
- Hazards found while looking, both fixed in the commit that logs this entry: (1)
  `tests/test_frontier_harness.py::test_live_without_t033_refuses_before_any_key` deleted all of
  `data/phase3/live` after its subprocess run, which could also destroy real collected data. The harness output root
  is now `XBRL_FRONTIER_PHASE3_DIR` when set, and the test writes only into its own temporary directory. (2)
  `frontier/freetier.call_with_retries` bound `time.sleep` as a default argument at definition time, so the test's
  monkeypatch never applied and the 503 path slept for real (timing-dependent). Sleep now resolves at call time.
  Neither is proven to be the failure seen.
- Next action: every full-suite run now keeps its `-rf` summary in a log. If a failure recurs, its name and text go
  here and the cause gets fixed. The entry closes after the next 10 consecutive green full runs with no
  recurrence, recorded in WORKLOG.

## O-027 Gemini free tier allows 500 requests per day per project; the one-arm plan needs about 18 days (OPEN)

- Status: OPEN (scheduling finding; nothing billed)
- Found (E4, 2026-10-02): after 500 successful calls (12:34:34-13:25:25 UTC, about 10 RPM, 0 retries), call 501 got a
  per-day 429: `GenerateRequestsPerDayPerProjectPerModel-FreeTier`, quota value 500, `retryDelay` 38,068 s (to midnight
  Pacific). No per-minute 429 occurred at 10 RPM, so the per-minute limit is at least 10 and otherwise unknown. Learned values:
  `data/phase3/gemini_quota_state.json`.
- Consequence: the quota is per project and model, shared by E1-E4. The plan's 8,598 requests (probe 202, accuracy run
  4,662 at N = 23, latency 300, fresh control 3,434) need 18 free-tier days (`free_tier_days_needed` in
  `data/phase3/frontier_plan.json`). E4 has 2,934 pending, which is 6 more days. If the power check raises N (one arm:
  N up to the largest N leaving the notional reserve), the accuracy run grows by about 203 requests per extra filing
  (about 0.4 days each).
- The days-per-filing figure is scheduling information only. When the power check runs after D4, N is set by
  `gates.decision_rule` and the power check, never by the day count (T-036).
- Next action: E4 resumes daily after 07:00 UTC with the same command until complete. The schedule for E1-E3 is
  set once D4 pins the sample (O-025 first). The README states that free-tier collection spanned these dates.

## O-028 Gemini docs disagree on Batch API availability on the free tier (RESOLVED by T-037)

- Status: RESOLVED (2026-10-02): no Batch API in this project, ever (TRADEOFFS T-037). Collection stays synchronous and the
  guard is unchanged.
- Found (2026-10-02 14:10 UTC): the pricing page (last updated 2026-10-01) lists a **Batch** row for `gemini-3.5-flash-lite`
  with the free tier "Free of charge", while the rate-limits page gives Batch enqueued-token quotas for Tiers 1-3 only (T-034
  item 2). Whether a free-tier project can submit batch jobs, and whether doing so can trigger billing, is therefore
  ambiguous.
- Decision until resolved: keep `frontier.free_tier.batch_api: false`, so the guard refuses the Batch API and collection stays
  synchronous (T-033 says to stop any path that could bill). The only cost is calendar time (O-027). The measurements do not
  change: accuracy-run dollars are priced at the sheet's batch rates either way.
- Evidence: price sheet `reobservations[0]`; `/ai.google.dev/gemini-api/docs/pricing` and `/rate-limits` text as fetched.
- Next action: none from the builder. If the owner wants batch on the free tier explored, that is an owner decision (it touches
  the $0-billed rule).
