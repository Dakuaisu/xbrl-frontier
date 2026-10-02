# Findings register

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

- Status: OPEN
- Found: no public FinNI scorer exists. Set vs multiset, normalization, aggregation details and
  parse handling are unspecified in FinTagging.
- Evidence: evidence_brief section 3. Our choices are in TRADEOFFS T-002.
- Next action: the owner reviews T-002. If the FinTagging authors publish or share their scorer,
  re-score with it as a cross-check and report any difference (do not swap metrics silently).

## O-003 `context_token_bucket` slice has no bucket edges in config (RESOLVED)

- Status: RESOLVED (2026-10-02): the owner approved `tokenizers==0.23.2` and tercile edges.
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

## O-005 Kaggle T4 Phase 0 runs: throughput benchmark, smoke quantization, T4 smoke test (OWNER-BLOCKED)

- Status: OWNER-BLOCKED (Kaggle is owner-only; the software pins are filled, O-007 RESOLVED)
- Ready: `notebooks/phase0_kaggle_t4.ipynb` (thin driver), `serving/phase0_benchmark.py`,
  `serving/quantize_base.py`, `serving/smoke_test.py`, `serving/common.py`. Local analysis is in
  `scripts/phase0_cost.py`. All are covered by `tests/test_cost_and_serving.py`, which needs no GPU.
- Owner steps:
  1. Source tarball, built by the builder from the commit that filled the pins (see WORKLOG for the
     commit and sha256): `/tmp/xbrl-frontier-src.tar.gz`, made with
     `git archive --format=tar.gz -o /tmp/xbrl-frontier-src.tar.gz HEAD`. Rebuild it the same way if
     `/tmp` was cleared.
  2. On Kaggle, create a private dataset `xbrl-frontier-src` and upload that file.
  3. New notebook: import `notebooks/phase0_kaggle_t4.ipynb`. Settings: Accelerator GPU T4, Internet on.
     Add the dataset `xbrl-frontier-src`. Run all. Budget is about 1 GPU-hour (PRD section 9).
  4. Download `phase0_kaggle_t4.tar.gz`, `smoke_kaggle_t4.tar.gz` and `smoke_checkpoints.tar.gz` from
     the notebook output.
- What I need back: the three tarballs, unpacked into `data/phase0/` on the Mac
  (`data/phase0/phase0_kaggle_t4/`, `data/phase0/smoke_kaggle_t4/`, `data/phase0/smoke_checkpoints/`).
  I then run `.venv/bin/python -m scripts.phase0_cost --results data/phase0/phase0_kaggle_t4` and apply
  `serving.smoke_test.failure_action` to any failed variant.

## O-006 RTX 3050 (WSL2) smoke test (OWNER-BLOCKED)

- Status: OWNER-BLOCKED (the laptop is unreachable from here; needs O-005's `smoke_checkpoints`)
- Ready: `serving/run_rtx3050_smoke.sh`, which runs `serving.smoke_test --card rtx3050_wsl2` with
  `serving.hardware.rtx3050_wsl2.vllm_flags`.
- Owner steps (in WSL2, NVIDIA driver with WSL CUDA support installed):
  1. Copy the repo (same commit as the Kaggle run) and the unpacked `smoke_checkpoints/` to the laptop.
  2. From the repo root: `serving/run_rtx3050_smoke.sh /path/to/smoke_checkpoints`
  3. Return `data/phase0/rtx3050_wsl2.tar.gz` (the script prints its path).
- What I need back: that tarball. A variant that fails stays recorded with its error and is removed
  from the serving table per `serving.smoke_test.failure_action` (PRD risk: the 1.7B may not fit).

## O-007 Kaggle/WSL2 software pins need owner approval (RESOLVED)

- Status: RESOLVED (2026-10-02): the owner approved the three pins and they are filled in
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

## O-009 "A Schema-Aware Large Language Model for XBRL Tagging and Cross-Statement Consistency Checking" could not be read (OPEN)

- Status: OPEN (PRD section 10 risk: possible scoop)
- Found (metadata only): Zhaoyue Peng, Shuncheng Liu, Xuhao Meng, Zhe Su, Ningjing Sang; posted
  2026-07-07; DOI 10.13140/RG.2.2.19518.86080 (a ResearchGate-minted DOI); OpenAlex W7167572144 lists it
  as an article with no PDF location, no abstract and 0 citations.
- Attempts on 2026-10-02: the DOI resolves to `researchgate.net/doi/10.13140/RG.2.2.19518.86080`, which
  returns HTTP 403 ("ResearchGate - Temporarily Unavailable"). ResearchGate search also returns 403.
  Crossref has no record. Semantic Scholar does not know the DOI (its title search was rate-limited, 429).
  The arXiv title search returned nothing. An Exa library page has the metadata only.
- Evidence of scope: none beyond the title. The title suggests concept tagging plus cross-statement
  consistency checks, not a serving cost/latency frontier or a small-model learning curve. That is
  unverified until the text is read.
- Next action: the owner opens it in a logged-in browser (ResearchGate often needs an account) or emails
  the authors, saves the PDF outside the repo, and tells me the path. I then summarize it against PRD
  section 1 and reposition if needed. Retry Semantic Scholar once the rate limit clears.

## O-010 Eval manifest recovered 129 of 142 registrants: `eval_manifest_required_recovered` NOT MET (OWNER-BLOCKED)

- Status: OWNER-BLOCKED for option (a). `contamination.eval_manifest_required_recovered` is **NOT MET**
  at 129/142, and the threshold is unchanged. Owner decision (TRADEOFFS T-005): (c) accept and disclose
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
- Owner steps for (a): post the issue text in `docs/owner/fintagging_issue_draft.md` on
  `https://github.com/The-FinAI/FinTagging/issues` (add the repository link where marked).
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

## O-012 FinNI draws on more than one 10-K per company (RESOLVED by owner decision T-007)

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

- Status: RESOLVED (2026-10-02): owner decision, recorded in TRADEOFFS T-010. In training, drop the negative twin.
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

- Status: OPEN (finding; nothing is retuned)
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

- Status: OPEN until the README contamination statement is written (Phase 5).
- The statement text must include, not only reference: (1) the fresh slice's 6 `fill=relaxed` rows
  (`data/fresh_accessions.txt`, column `fill`) and the two strata cells unfillable in the window
  (10-K | real estate & construction | large_accelerated: 5; 10-K | energy & transportation |
  large_accelerated: 1; December year-end filers file no 10-K between 2026-05-19 and 2026-10-02), T-015;
  (2) the audit counts before and after removal (18,184 of 215,284 flagged; 0 of 40,721 after) and the
  decontamination skew of positives (O-017); (3) the corpus sizes below cap (T-016); (4) the fresh-slice
  instance file pinned by sha256 in `data/fresh_selection_report.json`.

## O-019 Kaggle student training sessions (OWNER-BLOCKED)

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
  2. Upload `data/corpus/train.jsonl` and `data/corpus/dev.jsonl` (sha256 in `data/corpus_report.json`)
     as a private Kaggle dataset `xbrl-frontier-corpus`. The script refuses mismatching files.
  3. New notebook from `notebooks/phase2_kaggle_train.ipynb`. Accelerator GPU T4, Internet on. Attach both
     datasets, set `MODEL`, `SEED`, `CURVE` in the first code cell, and leave `RESUME = None`. Run all.
  4. When the session ends (finished or at the session limit), save the output as a Kaggle dataset, e.g.
     `xbrl-frontier-run-1p7b-s1`. To continue, open a new session, attach that dataset as well, set
     `RESUME` to its path, and run all again. Repeat until `run.json` lists `final_adapter`.
  5. Keep total use under `training.kaggle.gpu_hours_cap_total`. Each session's train time is in `run.json`.
- What I need back: the final output directory of each run (`run.json`, `final_adapter/`, `curve/`),
  unpacked into `data/phase2/kaggle/<run name>/`.

## O-020 Kaggle training stack pins need owner approval (RESOLVED)

- Status: RESOLVED (2026-10-02): owner-approved and filled (T-019).
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
