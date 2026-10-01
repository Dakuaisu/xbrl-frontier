# PRD: Fine-tuned small-model XBRL fact extraction, measured as an accuracy / cost / latency frontier

Working name: `xbrl-frontier`. Status: v2.2, frozen 2026-10-01 (v1 re-planned for a $10 cap, no rented GPUs, FinNI only; v2.1 added the cost-gate, sampling, decision-rule, fresh-slice-control and smoke-test fixes; v2.2 added the gate-pass goodput, the corrected online break-even and the power check). Owner: solo; starts after FilingQA. All numeric gates, pins, seeds and budgets live in `config/thresholds.yaml`; all prices live in `config/prices_2026-10-01.yaml`. This document names those keys and never restates their values. Every result number in the README comes from a measured run; until then the placeholders stay blank.

## 1. Thesis and positioning

**Claim.** This project produces the first public, reproducible accuracy / cost-per-filing / latency frontier for fine-tuned 0.6B-1.7B open models on FinTagging FinNI (numeric fact identification over SEC 10-K text and tables), measured against pinned frontier APIs on identical instances, with fp16 / INT8 / INT4 serving variants on a market-priced T4 and on owned consumer hardware. The engineering deliverable is the harness and the chart, not a model.

**What it is not.** It is not the first cost or latency number in XBRL tagging, and the README must not say so. The positioning paragraph cites:

| Prior work | What it has | What it lacks (our gap) |
|---|---|---|
| FinLoRA (2505.19819) | 8B LoRA on FiNER-139/FNXL, training $ , per-prompt inference seconds vs Gemini API | sub-8B, throughput, $/query, quantized serving, p50/p95, FinTagging |
| FLaME (ACL Findings 2025) | per-dataset API $ incl. FNXL for 7-9B | zero-shot only, F1 under 0.06, no latency |
| FinTagging v5 (2505.20650) | zero-shot Qwen3-0.6B/1.7B, Llama-3.2-3B; fine-tuned BERT baselines | no fine-tuned LLM, aggregate budget only, no latency |
| FinAI Contest 2025 / HF adapters | 8B LoRA adapters for FinNI/FinCL | no published scores, cost or latency |
| FHS (2608.06614) | best FinCL linker, 32B zero-shot, calls per fact, hours on clinical split | small models, financial-split timing, $ |
| Kulkarni 2026 (2603.22651) | $/doc and p50 on 10k SEC filings | frontier/70B only, no fine-tuning, private |
| distil labs, Manulife, Olava, Mastercard, Pareto Atlas | small-vs-frontier $ and latency templates | not XBRL, not FinTagging, not a learning curve on regulatory tables |

**Finance-specific finding we are after.** Where the frontier bends: tables versus text, and the same-model zero-shot-to-fine-tuned delta on Qwen3-0.6B and 1.7B, which FinTagging already scored zero-shot on these rows. The F1-versus-training-size curve adds the question an interviewer cannot answer from a vendor post: how many auto-labelled iXBRL instances a 1.7B model needs before it stops gaining on regulatory tables.

**Resume-level outcome.** "X% of the best frontier arm's FinNI F1 at Y times lower cost per filing and Z ms p95 at W requests per second on one T4", where X, Y, Z, W are blank until measured.

## 2. Scope

In scope:
- Benchmark: `pins.eval_finni` and the fixed frontier sample drawn from it (section 7); students and the encoder are scored on the full set.
- Students: the two models in `pins.students`, LoRA fine-tuned on a self-built iXBRL corpus; the encoder in `pins.encoder_baseline` trained on the same corpus as the cost floor; one F1-versus-training-size curve for the primary student.
- Frontier arms: `pins.frontier_arms`, zero-shot with the benchmark's own prompt, batch endpoints for accuracy, synchronous calls for latency.
- Serving: vLLM on the two hardware targets in `serving.hardware`, the variants each supports natively, load tests with the sweep in `serving.request_rate_sweep_rps`.
- Cost model: dated price sheet, measured token counts, goodput-based self-host formula at the T4 market rate.

Out of scope:
- FinCL and the end-to-end triplet (no budget for frontier candidate-list runs; the FinCL protocol from v1 is parked in the decisions log).
- Period, unit, scale and sign extraction (FinTagging cannot score them; FilingQA owns that schema).
- k-shot frontier variants, hosted fine-tuned API arms, and any student above 1.7B.
- Rented GPUs of any kind; any LLM-as-judge. Scoring is exact match; the only human instrument is the label audit.

## 3. Evaluation methodology

**Instance sets.** The pinned HF revision, verified on load against `pins.eval_finni.rows_expected`. Dedup follows `eval.dedup_policy`; raw and deduplicated scores are both stored, the deduplicated one is the headline. Every instance is mapped to its source filing in `pins.eval_instance_map` (section 4) so that resampling and the company slice work. The latency sample is `frontier.sync_latency_instances` fixed instance ids, identical for every arm.

**Scorer.** Written in this repo, since no public FinNI scorer for LLM outputs exists. Set-based pair precision, recall and F1 over (normalized value string, type), aggregated per `eval.aggregation`, with per-instance means also stored. Value normalization follows `eval.value_normalization`, which mirrors the benchmark prompt. Parse failures are counted as empty predictions and reported as a separate rate, never silently replaced.

**Calibration before any model is scored.** The oracles in `eval.oracles` must hit their expected values within `eval.oracles.tolerance`: a gold-copy predictor, an empty predictor, a gold predictor with types swapped, a gold predictor with half the facts dropped. CI runs the oracles on every change to the scorer.

**Upstream forcing check.** For every gated metric, the component that could force it is measured first and reported next to it: the JSON parser can force recall to zero (parse-failure rate per arm); dedup moves F1 (both numbers); quantized F1 can hide answer churn (per-fact flip rate against the fp16 variant); the sample can flatter an arm (student scores on the shared sample and on the full set are both shown).

**Slices.** Every metric is broken out by the slices in `eval.slices`: table versus text, entity datatype, company, fiscal year, context-length bucket, and training-size point for the curve. An aggregate carried by sentences or by monetary items is reported as such.

**Uncertainty.** Bootstrap over filings, not instances, with `eval.bootstrap_resamples` resamples at `eval.ci_level`. The headline "fraction of frontier F1" is a paired bootstrap (`eval.paired_bootstrap`) of the per-filing student-minus-frontier difference on the shared sample, which is tighter than two independent intervals. `pins.seeds` training seeds for the primary student; frontier latency sampled `frontier.sync_runs` times.

**Label noise.** Before any ceiling is reported, a random stratified sample of `eval.label_audit.sample_size` evaluation instances is hand-checked against the filing; `eval.label_audit.second_pass_size` of them are re-checked a week later and the self-agreement must exceed `eval.label_audit.min_self_agreement`. The README reports the estimated auto-label error rate next to every F1.

**Fairness protocol.** Frontier arms get the benchmark's verbatim prompt, zero-shot, with thinking set per `frontier.thinking` and recorded. Students get the same output schema. Model ids are the pinned strings; runs record the response's model field. Accuracy runs use batch endpoints on the shared sample; latency runs are synchronous from `serving.frontier_latency_vantage` with network time included and stated.

## 4. Data pipeline

**Evaluation manifest and instance map (Phase 0).** The 142 FinTagging evaluation filings are not published. `research/recovered_eval_manifest.json` already holds 111 registrants recovered by exact-phrase EDGAR full-text search. Complete it to `contamination.eval_manifest_required_recovered` with the same recipe (11-word phrase, forms=10-K, the benchmark's filing window, keep queries resolving to one CIK, collapse co-registrants), or by asking the authors. Then download the 142 iXBRL documents, rebuild contexts with the benchmark's own serialization, and assign every evaluation instance to a filing by normalized exact match, then fuzzy match; the assignment rate must reach `contamination.instance_assignment_min_fraction`, and unassigned instances form their own stratum that is excluded from the frontier sample. Ship both files as `pins.eval_manifest` and `pins.eval_instance_map`.

**Training corpus.** Built from EDGAR iXBRL 10-K and 10-Q filings with the forked parser (module `ixbrl_facts/`, forked from FilingQA at a pinned commit, no runtime dependency), listed in `pins.train_accessions`, selected once and frozen. Selection rules: filing date on or before `contamination.train_filing_date_max`; CIK not in any file under `contamination.exclude_ciks_from`; stratified across sectors and filer sizes to the extent the manifest allows. The parser replicates FinTagging's conventions: ix:nonFraction leaf elements only, `us-gaap:` concepts only, the five numeric datatypes, dedup on (value, concept), tables rebuilt as raw HTML with tr/td/th, text blocks over 20 characters, and a negatives fraction per `training.negatives_fraction`. Target size `training.target_train_instances`, sized to the compute in section 9; the SEC fair-access rules (10 requests per second, declared User-Agent) are enforced in the downloader. A held-out slice of `training.dev_slice_instances` from disjoint filings is reserved for scoring curve checkpoints cheaply.

**Format parity check.** Before training, a script samples evaluation contexts and training contexts and compares token-length distributions, HTML tag vocabulary, fact-count distributions and datatype mix; divergences are logged and fixed in the serializer, not in the model.

**Contamination audit.** Every training instance is checked against every evaluation context with `contamination.ngram_n`-gram overlap (lm-eval-harness implementation) and MinHash near-duplicate detection (`contamination.minhash_num_perm`, `contamination.minhash_jaccard_threshold`). Flagged training docs are removed until the count equals `contamination.max_flagged_train_docs_after_removal`. The audit output ships with the repo.

**Fresh slice.** At least `contamination.fresh_slice_min_filings` filings dated after `contamination.fresh_slice_filing_date_min`, from CIKs outside the manifest, auto-labelled by the same parser and listed in `pins.fresh_accessions`. Students and the encoder are scored on it in full. Because the students were trained on the same parser that labels the fresh slice, a small gap could reflect convention match rather than absence of contamination, so the arm named in `frontier.fresh_slice_control_arm` is also run zero-shot on the full fresh slice as a control. The README reports the student's FinTagging-minus-fresh gap next to the control arm's gap and the difference between them: a student gap much larger than the control's is the contamination signal, similar gaps point to distribution shift. The student gap is gated by `gates.fresh_slice_f1_gap_max`.

**Contamination statement.** A model-info-sheet section in the README: manifests, exclusion rules, overlap statistics, base-model cutoffs versus eval filing dates, and the caveat that public iXBRL HTML carries the labels inline.

## 5. Models and training plan

**Students.** `pins.students`: Qwen3-0.6B and Qwen3-1.7B, the two sizes FinTagging already scored zero-shot, so the before/after lands on the benchmark's own rows. Primary is the 1.7B. Encoder baseline `pins.encoder_baseline` (SEC-BERT base) as a token classifier with the benchmark's BIO label space, trained on the same corpus; contexts longer than its window are chunked, as the benchmark's own BERT baselines were.

**Recipe.** QLoRA with `training.lora_rank`, `training.lora_alpha`, `training.learning_rate`, `training.epochs`, completion-only loss on the JSON answer, output schema identical to the benchmark's `{"result":[{"Fact","Type"}]}`. On Kaggle: Unsloth or TRL plus PEFT, `training.kaggle.dtype` (the T4 has no bf16), base quantized per `training.kaggle.quant_base`, sequence cap `training.kaggle.max_seq_len_tokens`, checkpoints every `training.kaggle.checkpoint_every_minutes` to Kaggle datasets so a run resumes across the `training.kaggle.session_hours_max` session limit. Adapters come out in PEFT format, which vLLM loads directly.

**What runs where.**
- Mac (M1 Pro, 16 GB, mlx-lm): the pilot on `training.mac.pilot_instances` at `training.mac.max_seq_len_tokens` with a 4-bit base, to settle the prompt format, loss masking and learning rate, and to measure tokens per second. Longer sequences are avoided on the Mac because mlx-lm's attention backward still materializes the full attention matrix in the released version. If the measured rate for the 0.6B exceeds `training.mac.min_tokens_per_sec_to_train_0p6b_locally`, the 0.6B full run happens on the Mac overnight and its adapter goes through the small MLX-to-PEFT converter (transpose the two adapter matrices, rename keys, set alpha to scale times rank); otherwise it also runs on Kaggle and the converter is not built. The Mac also runs the scorer, frontier batch collection, bootstrap and charts.
- Kaggle (free T4, `training.kaggle.weekly_gpu_hours_quota` per week, capped in total by `training.kaggle.gpu_hours_cap_total`): all 1.7B runs, the encoder, quantization exports, full-set student scoring, and the T4 load tests.
- RTX 3050 (4 GB, WSL2): development serving, the owned-hardware latency rows, and overnight full-set scoring of INT4 checkpoints when Kaggle quota is the bottleneck.

**Corpus sized to the compute (arithmetic).** Planning rates are `training.planning_tokens_per_sec` and are replaced by the pilot measurement. At the 1.7B planning rate, one epoch over `training.target_train_instances` instances of about 1,000 tokens each is 40M tokens, or about 7.4 hours on a T4; two epochs are about 15 hours, and two seeds about 30 hours. The 0.6B at its planning rate is about 7 hours for two epochs. The encoder at a few hundred tokens per instance is about 3 hours for three epochs. That is roughly 40 T4-hours of training, within `training.kaggle.gpu_hours_cap_total` after reserving the scoring and load-test hours in section 9. If the pilot measures a slower rate, the corpus shrinks to keep the 1.7B two-seed training inside about 30 T4-hours; the curve points stay the same relative fractions.

**F1-versus-training-size curve.** One shuffled training order is fixed; the prefixes of length `training.curve_points` are nested subsets. The first epoch of the primary student runs with the `training.curve_lr_schedule` schedule and writes a checkpoint at each curve point; every checkpoint is scored on the dev slice and on the frontier sample, and the final checkpoints also on the full set. One independent run of `training.curve_validation_run` instances checks that the nested-checkpoint curve matches a from-scratch run at that size; a gap larger than the bootstrap interval is reported as a caveat on the curve. The curve is a required chart (`gates.learning_curve_reported`), reported by table versus text as well as overall.

**Export.** Merge adapter into the base (fp16). Produce INT8 W8A8 (llm-compressor) and AWQ-INT4 (AutoAWQ, calibration on training contexts) checkpoints on Kaggle. Each checkpoint is scored on the full set before it is load-tested; `gates.quantized_f1_drop_max` and `gates.quantized_flip_rate_max` gate inclusion.

## 6. Serving and load-test plan

**Targets and native variants.** Two hardware rows, from `serving.hardware`:
- Kaggle T4 (Turing, SM 7.5): fp16, INT8 W8A8 and AWQ-INT4 are native in vLLM on Turing; FP8 is not and is not reported. This is the market-priced row.
- RTX 3050 Laptop 4 GB (Ampere, SM 8.6) under WSL2: bf16 for the 0.6B, AWQ-INT4 for both students; the 1.7B in bf16 does not fit (its weights alone are 3.4 GB). vLLM runs with the flags in `serving.hardware.rtx3050_wsl2.vllm_flags` (eager mode, short context, few sequences) and that is stated in the table. This row reports latency and throughput only and carries the cost label from the config; no dollars per filing are attached to owned hardware.

**Load tests.** GuideLLM or `vllm bench serve` (`pins.software.guidellm`) against the fixed latency sample, with `serving.warmup_requests` warm-up and `serving.requests_per_point` measured requests per point, concurrency levels per card from `serving.concurrency_levels`, and the request-rate sweep in `serving.request_rate_sweep_rps`. On Kaggle the load generator is colocated with the server, so network time is excluded and the table says so; on the 3050 the generator runs on the same laptop. Metrics: TTFT and end-to-end p50/p95, output tokens per second, request throughput, and goodput against `serving.slo`. Single-request and batched regimes are separate rows; each run records GPU, driver, vLLM flags and checkpoint hash.

**Frontier latency.** The same fixed sample, synchronous, `frontier.sync_runs` runs per arm at concurrency one, from `serving.frontier_latency_vantage`, with network time included and stated. Frontier arms therefore carry network latency the self-hosted rows do not; the README says so next to the chart rather than pretending the vantages are equal.

## 7. Cost model

**Frontier arms, priced before anything runs.** Per-filing cost is estimated from the benchmark's own token statistics (28,787 instances over 142 filings, about 203 instances per filing; 1,101.5 input and 175.2 output tokens per instance on average, from FinTagging v5 Table 4), which is about 223k input and 35.5k output tokens per filing, priced from `config/prices_2026-10-01.yaml` at batch rates:

| Arm | Batch $/M in, out | Est. $/filing | Est. $ at `frontier.sample_filings` |
|---|---|---|---|
| gpt-6-luna | 0.05, 0.25 | 0.020 | 0.46 |
| gemini-3.5-flash-lite | 0.15, 1.25 | 0.078 | 1.79 |
| gpt-5.4-mini-2026-03-17 | 0.375, 2.25 | 0.164 | 3.76 |
| Sum of the three arms | | 0.262 | 6.01 |

Then: overhead multiplier `frontier.reasoning_token_margin` on the batch total (about $6.92), plus the synchronous probe of `frontier.probe_filings` per arm at synchronous prices (about $0.52), plus the latency sample of `frontier.sync_latency_instances` instances times `frontier.sync_runs` runs per arm (about $0.77), plus the fresh-slice control arm on `contamination.fresh_slice_min_filings` filings at its batch rate with the same margin (about $0.69), totals about $8.90, leaving more than `frontier.usd_reserve_min` unspent under `budget.usd_cap_total`. The protocol is: run the probe first, replace the paper's token averages with the measured per-filing tokens and output lengths for each arm, recompute the sample size as the largest N whose priced plan still leaves the reserve, and only then submit the batches. If the mid-tier arm's measured cost breaks the plan, it is swapped per `pins.arm_swap_if_over_budget` (which, at the sheet's prices, would buy roughly 40 filings instead of 23).

**What the sample buys.** With bootstrap over filings, the 95% half-width is about 1.96 times the between-filing standard deviation of F1 divided by the square root of N. That standard deviation is measured from the students' full-set runs before any API dollar is spent, so the sizing uses a real number. For illustration only (hypothetical standard deviations, not measurements): at N of 23 a between-filing standard deviation of 0.08 gives a half-width near 0.03, of 0.12 near 0.05, of 0.16 near 0.07; at N of 40 each shrinks by about a quarter. The paired difference on the shared sample is tighter than either arm's own interval. The README shows the measured standard deviation, N, and the resulting interval next to the headline.

**Power check.** After the primary student's full-set run and before the probe, the expected half-width at the planned N is computed from the measured between-filing standard deviation (the student's own, as a proxy for the paired standard deviation, which is unknown until the frontier runs) per `frontier.power_check`. If it exceeds `gates.headline_margin_f1`, the design cannot resolve the headline gate, so the third arm in `pins.frontier_arms` is dropped and its share, together with its probe and latency-sample cost, is spent on more filings for the remaining two arms: N becomes the largest value whose priced plan still leaves `frontier.usd_reserve_min`. At the sheet's prices that is about 69 filings instead of 23, which shrinks the half-width by about 40%; the headline then reads "fraction of the best remaining arm" and the README says which arm that is.

**Sampling design.** Filings are stratified by `frontier.stratify_by` using the instance map: table-share tercile by instances-per-filing tercile, nine cells with two or three filings each. Sector cannot be a stratum at this sample size, so it is a balance check (`frontier.balance_check`): the sample's sector distribution is reported against the manifest's, and a draw that leaves a sector with more than its proportional share empty is redrawn before the ids are pinned. The sample ids are pinned before the probe and never changed.

**Self-hosted arm (T4, market rate), two regimes.** Offline: dollars per filing = (T4 $/hour / maximum throughput in filings per hour from the batched load test) / the offline utilization in `gates.cost_regimes.offline`, compared against frontier batch prices. Online: dollars per filing = (T4 $/hour / goodput in filings per hour at `serving.slo`) / the online utilization in `gates.cost_regimes.online`, compared against frontier synchronous prices. Both are also reported for every value in `serving.utilization_scenarios`. The T4 rate comes from the dated sources in `serving.hardware.kaggle_t4.market_rate_keys`, the first being the headline; self-hosted cost scales linearly with the rate, so the README shows the cost result at each source (the two sources observed on 2026-10-01 differ by about 15%), and any source left blank in the sheet is observed at pin time or dropped. The formula and the idle-cost caveat sit next to the chart. Kaggle's own quota is free and is never presented as the cost of serving.

**Comparator and break-even.** At the sheet's prices and the planning rate of about 3 requests per second, a 1.7B on a T4 costs about as much per filing as the cheapest arm at batch price, so a cost ratio against the cheapest arm is not a meaningful gate. The gate compares against the cheapest arm whose F1 lies within the student's confidence interval (`gates.cost_comparator_rule`, with the config's fallbacks for the cases where no arm falls inside it), evaluated in the regime named by `gates.cost_gate_regime`; the other regime is reported with the same rule, not gated. The full Pareto chart (F1 against dollars per filing for every arm and every serving variant, both regimes) is always shown so a reader can pick a different comparator. Before any training, Phase 0 benchmarks the base 0.6B and 1.7B on a Kaggle T4 with the FinNI token shape and states the break-even goodput: the filings per hour at which the T4 cost equals each arm's cost, which is T4 $/hour divided by (arm $/filing times utilization). At the sheet's prices against the cheapest arm, the offline break-even (batch price, offline utilization) is about 33 filings per hour and the online break-even (synchronous price, online utilization) about 29, roughly 1.9 and 1.7 requests per second at the benchmark's instances per filing. Because the gate requires `gates.cost_ratio_vs_comparator_min` times the comparator's cost, Phase 0 also reports, per arm and per regime, the goodput needed to pass the gate, which is that multiple of the break-even: against the cheapest arm about 98 filings per hour offline and 88 online, roughly 5.5 and 5.0 requests per second, well above the planning rate, which is exactly why the comparator rule exists. The measured base-model throughput therefore says, before training starts, which arms are beatable on cost and by how much.

**Owned hardware (RTX 3050).** No dollar figure. The row reports throughput, latency and the vLLM constraints under which they were measured, with the config's cost label verbatim. Electricity-based costs are not reported because they are not comparable to a market rate.

**Training cost.** Reported once, separately, as T4-hours on free quota plus Mac hours, with the market-rate equivalent at the same T4 price given for reference; never folded into per-filing cost without saying over how many filings it is amortized.

## 8. Metrics and gates

Named metrics: `finni_f1`, `finni_precision`, `finni_recall`, `parse_failure_rate`, `finni_f1_by_train_size`, `quantized_flip_rate`, `ttft_p50`, `ttft_p95`, `e2e_p50`, `e2e_p95`, `goodput`, `usd_per_filing_offline`, `usd_per_filing_online`, `usd_per_1k_facts`, `break_even_goodput`, `gate_pass_goodput`, `expected_ci_half_width`, `fresh_slice_gap`, `fresh_slice_gap_control`, `label_audit_error_rate`, `between_filing_f1_sd`. Every one is reported by slice and with a confidence interval where one applies.

The minimum defensible version ships only if every key under `gates` passes for the primary student on the deduplicated set: fraction of the best frontier arm's F1 on the shared sample (paired bootstrap), absolute gain over the same model zero-shot, quantized drop and flip rate, cost ratio against the comparator arm in the gated regime at the T4 headline rate, fresh-slice gap, and the presence of the encoder baseline and the learning curve on the charts. Every gate that rests on a bootstrap statistic follows `gates.decision_rule`: the decision is taken on the paired-bootstrap point estimate, the interval is reported beside it, and when the interval straddles the threshold the outcome is labelled inconclusive rather than pass or fail; an inconclusive headline gate ships with that label and without the claim. If a gate fails, the README reports the failure; gates are not retuned after the fact.

## 9. Build plan in hours

Human hours are estimates; the cap is `budget.hours_cap_total`. GPU hours are Kaggle quota, capped by `training.kaggle.gpu_hours_cap_total`.

| Phase | Work | Hours (est.) |
|---|---|---|
| 0. Pin and calibrate | Pin HF and model revisions; finish the eval manifest (about 2,000 scripted FTS queries, 4 h); download the 142 filings and build the instance-to-filing map (4 h); FinNI scorer with oracle tests in CI (7 h); slices and dedup policy (2 h); base-model throughput benchmark on a Kaggle T4 with the FinNI token shape, with break-even and gate-pass goodput per arm and regime (1 h); vLLM smoke test of every planned variant on the T4 and on the 3050 under WSL2 (2 h) | 20 |
| 1. Corpus | Fork and adapt the iXBRL parser (6 h); accession selection with exclusions and throttled download (4 h); serializer and format-parity script (4 h); n-gram and MinHash audit (3 h); fresh slice (2 h) | 19 |
| 2. Train | Mac pilot and tokens-per-second measurement (3 h); Kaggle training script with checkpoint-resume (5 h); run supervision across sessions (3 h); encoder baseline (3 h); curve checkpoint scoring (2 h); INT8 and AWQ export (3 h) | 19 |
| 3. Frontier arms | Prompt harness and usage logging (3 h); power check, probe, sizing, batch submission and collection (3 h); synchronous latency sample (2 h) | 8 |
| 4. Serve and load | vLLM on Kaggle T4 with GuideLLM sweeps (5 h); 3050 runs, setup already done in the smoke test (2 h); cost model, both regimes, and charts (3 h) | 10 |
| 5. Audit and write-up | Label audit, two passes (6 h); contamination statement (2 h); README, charts and blanks filled from runs (5 h); resume bullets (1 h) | 14 |
| Total | | 90 |

**Kaggle GPU hours (est.)**: training about 40 (section 5); full-set scoring of final checkpoints (two 1.7B seeds, the 0.6B, two quantized 1.7B variants, one quantized 0.6B; at a planning rate of about 3 requests per second for a 1.7B on a T4, each full pass is about 2.7 hours) about 16; curve checkpoints on the dev slice and the frontier sample about 3; quantization exports about 2; load tests about 6; Phase 0 throughput benchmark and smoke tests about 1; total about 68 against the cap, spread over three to four weeks of the weekly quota, with the 3050 absorbing INT4 full-set passes if quota runs short.

**Dollar budget**: frontier API about $8.9 as itemized in section 7 against `budget.usd_cap_total`; GPUs $0 (`budget.usd_cap_gpu`).

**Cut order** (first cut listed first): the second training seed; the INT8 variant (keep fp16 and AWQ-INT4); the 3050 rows beyond the single AWQ-INT4 1.7B run; the curve validation run (keep the nested-checkpoint curve, flag the missing check); the 0.6B student; the third frontier arm (keep the cheapest and the mid-tier). The encoder baseline, the label audit, the contamination audit, the oracle tests and the learning curve are never cut.

## 10. Risks

| Risk | Signal | Mitigation |
|---|---|---|
| Fine-tuned student does not reach `gates.finni_f1_fraction_of_best_frontier_min` | curve flattens below the frontier on the dev slice | tables-only error analysis; longer sequence cap on Kaggle; more table negatives; report honestly if the gate fails, the frontier and the curve are still the deliverable |
| Base-model throughput below the break-even goodput for every arm | Phase 0 benchmark | the comparator rule applies by design; if even the comparator is out of reach, cost is reported on the Pareto chart and the headline rests on quality and latency, not on the cost gate |
| Encoder beats the decoder on FinNI | Phase 2 scores | expected possibility; the narrative becomes "which architecture at which price" and the decoder's case is schema-valid JSON and the curve |
| Pilot tokens-per-second far below the planning rate | Mac pilot, first Kaggle session | shrink the corpus, keep curve fractions; drop the second seed first |
| Kaggle quota or session limits break runs | lost checkpoints | checkpoint-resume every `training.kaggle.checkpoint_every_minutes`; one run per session |
| Frontier arm emits thinking tokens or long outputs | probe token counts | thinking set to minimum; recompute N from measured tokens; swap the mid-tier arm per config |
| Instance-to-filing assignment incomplete | assignment rate below `contamination.instance_assignment_min_fraction` | unassigned stratum excluded from the sample; rate disclosed |
| Label noise caps the ceiling | audit error rate high | report noise-adjusted ceiling; score the audited subset separately |
| 1.7B does not fit vLLM on the 3050 even at INT4 | OOM at startup | shorter context, eager mode; if still failing, the 3050 row is 0.6B only and says so |
| T4 lacks bf16 and FP8 | n/a | fp16 training with loss scaling; FP8 not reported; stated in the serving table |
| Model or price deprecations mid-project | vendor deprecation pages | arms chosen to outlive the window; prices dated; re-observe before final runs |
| Scooped by the unread Schema-Aware XBRL Tagging preprint | read it in Phase 0 | reposition against it; the serving frontier and the curve are still unclaimed |

## 11. Resume bullets (blanks filled only from measured runs)

- Fine-tuned a __B-parameter open model on __ contamination-audited iXBRL instances that reached __% of __'s FinNI F1 on the FinTagging benchmark (paired bootstrap over __ filings) and served it with vLLM at __ ms p95 and __ requests/s on a single T4, __x cheaper per filing at market rate; total API spend $__.
- Published the F1-versus-training-size curve for regulatory table extraction: __% of final F1 at __ instances, with tables lagging text by __ points at every size.
- Recovered FinTagging's unpublished 142-filing evaluation manifest from EDGAR full-text search, mapped __% of evaluation instances to their filings, and shipped the exclusion audit (__ flagged, 0 remaining) plus a post-cutoff fresh slice with a __-point gap.
- Measured fp16 / INT8 / INT4 serving variants on Turing and Ampere consumer hardware with per-fact flip rates (__% at INT4) rather than aggregate F1 alone; every number reproducible from pinned revisions and a dated price sheet.

## 12. Decisions log

- v2 constraints: $10 total, all frontier API; no rented GPUs; Mac M1 Pro 16 GB, RTX 3050 4 GB under WSL2, free Kaggle T4; about 90 hours; starts after FilingQA.
- FinNI only; students 0.6B and 1.7B; encoder baseline kept; learning curve added as a required chart.
- Frontier arms reduced to three zero-shot arms on a pinned stratified sample of filings sized from a measured probe; students and encoder on the full set; paired bootstrap on the shared sample.
- k-shot arms, hosted SFT arm, FinCL track and the 4B student dropped; the v1 FinCL protocol (constrained generation over the 2024 numeric-concept list, frontier arms on identical BM25 candidates) is parked, not rejected.
- Primary benchmark set: `FinNI-eval-v2`; `FinTagging_Original_v2` rejected (duplicated, misaligned); published FinNI numbers are never quoted next to ours.
- Period and unit dropped from the scored schema. FilingQA parser forked at a pinned commit, no runtime dependency, separate corpora.
- No LLM judges anywhere; exact match plus one human label audit (second pass on a subset, for hours).
- T4 is the only market-priced self-hosted row; the 3050 row is latency-only and labelled as owned hardware.
- v2.1 fixes: cost gate against the cheapest arm whose F1 is within the student's interval, offline and online cost regimes, full Pareto chart, Phase 0 base-model throughput benchmark with a break-even statement; T4 priced from several dated sources with the sensitivity shown; frontier sample stratified by table-share tercile and size tercile with sector as a balance check; bootstrap gates decided on the point estimate with an inconclusive label when the interval straddles the threshold; the cheapest arm run zero-shot on the full fresh slice as a contamination control, with the sample reduced by one filing to keep the reserve; Phase 0 vLLM smoke test of every planned variant on both cards.
- v2.2 (frozen): Phase 0 reports gate-pass goodput (the gate multiple of break-even) per arm and regime; the online break-even is computed at synchronous prices; a power check after the student's full-set run drops the third arm and buys more filings if the expected half-width cannot resolve the headline margin.
- Open: whether to ask the FinTagging authors for the scorer and the missing CIKs (a public GitHub issue); whether to spend the remaining reserve on frontier runs over the fresh slice after the main sample is in.
