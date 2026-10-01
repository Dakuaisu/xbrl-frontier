# Part 1: Critical review of "fine-tuned small-model XBRL extraction frontier"

Date: 2026-10-01. Note: the budget and hardware figures in section 5 were written under the original constraints (rented GPUs, about 150 hours); they are superseded by PRD v2 (a $10 cap, no rented GPUs, about 90 hours). The verdict, positioning, evaluation methodology and contamination protocol are unchanged. Every factual statement below was verified by fetching the primary source; the per-topic reports with URLs and quotes are in `research/dossier/`, and the condensed brief is `research/evidence_brief.md`. Numbers derived by me rather than reported by a source are marked DERIVED.

## 1. Novelty

**The thesis as written is false; a narrower version is intact.**

What already exists:

- **FinLoRA** (arXiv 2505.19819, May 2025) is the closest prior work. It LoRA/QLoRA-tunes Llama-3.1-8B on FiNER-139 and FNXL, beats GPT-4o on FiNER (74.10 vs 72.29 accuracy), reports fine-tuning cost in dollars ("$14.66 to $16.54" on 4xA5000), and plots per-task inference time of the local 8B versus a fine-tuned Gemini 2.0 Flash-Lite API (DERIVED from the vector figure: FiNER 1.57 s vs 0.31 s per prompt). So "XBRL tagging work reports accuracy but not latency or cost" is wrong at 8B. What FinLoRA does not have: anything below 8B, any throughput or $/query number, any vLLM or quantized serving, any p50/p95, and nothing on FinTagging. Its latency is an unbatched HF-transformers bitsandbytes-8bit per-prompt timing with no published raw numbers.
- **FLaME** (Findings of ACL 2025) tabulates per-dataset API dollar cost including FNXL for 7-9B models, but zero-shot with F1 at or below 0.057 under its own metric. Precedent for "cost reported on FNXL", not for a usable frontier.
- **FinTagging v5 itself** already evaluates Qwen3-0.6B, Qwen3-1.7B and Llama-3.2-3B zero-shot (FinNI F1 0.0038 / 0.0541 / 0.1460; FinCL accuracy 0.04 to 0.07) plus fine-tuned BERT-large/FinBERT/SECBERT. "Nobody benchmarked small models on FinTagging" is false; "nobody fine-tuned one" is true. Its only cost line is an aggregate "500 GPU-hours and $1,500 in API usage"; it has zero latency or throughput numbers.
- **FinAI Contest 2025 Task 2** fine-tuned models of 8B or less on FinNI/FinCL subsets, and public 8-bit Llama-3.1-8B LoRA adapters for FinNI and FinCL exist on Hugging Face with no published scores, latency or cost.
- **FHS** (arXiv 2608.06614, Aug 2026, the FinTagging group) is the best published FinCL linker: Qwen3-32B zero-shot plus BM25 retrieval, accuracy 0.255 on a 2,509-fact FinTagging subset. It reports LLM calls per fact and wall-clock hours, but only on its clinical split, and no dollars. It names "model families and scales ... robustness-cost trade-off" as future work, which is a citable invitation.
- **Kulkarni & Kulkarni** (arXiv 2603.22651, Mar 2026) report $/document and p50 latency for 25-field extraction from 10,000 SEC filings with XBRL-derived labels, but only for frontier and 70B-class models, no fine-tuning, private data.
- **The generic result is well trodden.** "Fine-tuned small model matches or beats a frontier API at 10-100x lower cost and lower latency" has public, reproducible instances in 2025-2026: distil labs (Qwen3 0.6B-8B on vLLM with p50/p95/p99 and $/million requests), Manulife's bank-check study (fine-tuned 0.5B-7B VLMs vs GPT-5/Claude with $/doc and p95), Olava's SEC-contract extraction ($0.018 vs $0.149-0.456 per doc), Mastercard's 270M-8B size sweep with an F1-vs-throughput Pareto, and the Inference Engineering Pareto Atlas (arXiv 2609.17863, Sep 2026: 54 vLLM configurations of Qwen2.5-7B on L4/A100/H100 with $/M tokens and FP8/AWQ trade-offs). A README headline of "small model cheaper than GPT" will read as generic.

What is genuinely open: no work found anywhere (arXiv abstract search, Hugging Face, GitHub, Semantic Scholar citation graph, OpenAlex, the FinTagging group's own releases through 2026-09-30) that fine-tunes or distills a sub-4B model on FinTagging's FinNI and FinCL tasks and reports accuracy against per-filing dollar cost, p50/p95 latency and throughput for frontier APIs on the same instances, with quantized vLLM serving variants. No FinTagging result anywhere carries serving cost or throughput.

**Defensible claim (use this wording):** "the first public, reproducible accuracy / cost-per-filing / latency frontier for fine-tuned 0.6B-4B models on FinTagging, measured against pinned frontier APIs on identical instances, with bf16/FP8/INT4 serving variants; FinLoRA (8B, training cost, unbatched per-prompt timing), FLaME (zero-shot API cost), FinTagging (zero-shot small models, aggregate budget), FHS (32B, clinical-split timing) and Kulkarni 2026 (frontier/70B, private data) are the closest precedents."

To be non-generic the project needs a finance-specific finding, not just a cheaper model. Two are available: (a) where the frontier bends, by slice: tables versus text, and concept-frequency tail in FinCL, which is exactly where FinTagging says every model fails; (b) the same-model zero-shot-to-fine-tuned delta on Qwen3-0.6B/1.7B, which FinTagging already scored zero-shot, giving a direct before/after on the benchmark's own rows.

Scoop risk: one unread ResearchGate preprint, "A Schema-Aware Large Language Model for XBRL Tagging and Cross-Statement Consistency Checking" (Peng et al., dated 2026-07-07), could not be opened. Read it before investing hours. The Hugging Face `lm2445/FinTagging_1000*` datasets are not a competing fine-tuning project; they are Linhai Ma's (The Fin AI) data for the zero-shot FHS paper.

## 2. Data

**FinTagging is eval-only on Hugging Face.** All eight TheFinAI repos have a single `test` split. A training set exists only on GitHub (10 extra 10-Ks, 1,116 sentences and 953 tables, built for the BERT baselines), and it is leaky: four of its ten companies (AT&T, USA Compression, Ares Commercial Real Estate, and Loews as parent of CNA) also have filings in the evaluation set, and 78 of its instances are verbatim evaluation contexts. Do not train on it.

**Which files.** The canonical v5 sets are `FinNI-eval-v2` (28,787 rows) and `FinCL-eval-v2` (261,457 rows); they reproduce the paper's Table 3/4 exactly. `FinTagging_Original_v2` (46,496 rows) is a duplicated, misaligned union and must not be used as the instance set. FinNI-eval-v2 contains 1,852 duplicate-context groups, 204 with differing answers; the official dedup policy is undocumented, so yours must be written down and both raw and deduplicated scores reported.

**License.** Only `FinTagging_Original` carries a license (MIT). The other seven HF repos have no license field and the GitHub repo has no LICENSE file; the paper is CC BY 4.0; the underlying SEC text is public domain. Benchmarking is fine; redistributing derived data from the eval sets is not clearly permitted. FNXL's inner README says "Research-only use". FiNER-139 is CC-BY-SA-4.0. HiFi-KPI is CC BY 4.0.

**What the labels are.** FinTagging scores (value string, 5-way datatype, us-gaap concept) only. There is no period, unit, scale, sign, decimals, CIK, accession or fiscal year anywhere in the released data; the authors' parser discards them. Your proposed "period and unit" outputs cannot be scored on this benchmark. Drop them from the scored schema.

**Contamination.** The 142 evaluation filings are not identified in v5 (v1 listed 30 CIKs). They are recoverable: the research agents resolved 111 of 142 registrants by exact-phrase queries against EDGAR full-text search (`research/recovered_eval_manifest.json`, plus the raw query results). Finish the manifest (about 2,000 more queries, scripted, or ask the authors), then exclude every manifest CIK from all training filings at any year, and additionally restrict training filings to dates before the evaluation window. Run a 13-gram overlap audit and MinHash near-duplicate check between every training instance and every eval context, remove flagged training docs, and publish the counts. Pretraining exposure cannot be controlled: EDGAR text is in Common Crawl, BloombergGPT, Open-FinLLMs and EDGAR-CORPUS, and the base models' cutoffs (Llama 3.2: Dec 2023; Qwen3, Gemma 3: unstated) overlap the eval window. Disclose it, and add a fresh slice of filings dated after the newest pinned frontier model's cutoff so that a large FinTagging-versus-fresh gap becomes the contamination signal.

**Distribution shift.** FinTagging's "expert labels" are 99.25% unaudited iXBRL auto-labels from the same kind of parser you would write (1,949 of 261,457 entities were reviewed; kappa 0.81; corrections merged back without flags). The shift between your auto-labels and theirs is small if you replicate their conventions: ix:nonFraction leaves only, `us-gaap:` concepts only, the five datatypes, dedup on (value, concept), tables rebuilt as raw `<table><tr><td>` HTML, text blocks longer than 20 characters, roughly 43% negative instances, and the prompt-side value normalization (strip $, %, commas; never expand "million"). The real risk is label noise, not shift: the authors never report the auto-label error rate (DERIVED from kappa: roughly 10-14% flagged on a deliberately hard subset; NORA reports under 5% on a random sample of comparable iXBRL labels). Audit a random sample yourself and report a noise-adjusted ceiling.

**Alternatives.** FiNER-139 and FNXL are text-only, token-level, with 139 and 2,794 labels, from 2016-2021 filings; they are useful as secondary benchmarks (FNXL is fully public on HF, contrary to XBRLTagRec's claim), not as the primary training source for a table-aware, 17k-concept task. Your own EDGAR iXBRL pipeline is the right training source; HiFi-KPI is a CC BY 4.0 fallback with a train split if the pipeline slips.

## 3. Scope: FinNI vs FinCL

FinNI is the defensible frontier; FinCL is the interesting one; do FinNI first and FinCL second with your own protocol.

- **FinNI** is reproducible: the prompt and gold are public, the metric is defined (set-based pair P/R/F1, exact match on value string and type), and the frontier can be re-run on the same rows. It is not free: there is no public scorer for LLM outputs, so you write one and validate it with oracles, and you cannot quote the paper's 0.6932 / 0.5397 because neither the scorer nor the exact instance set behind those numbers is pinned publicly. Frontier models were zero-shot there; FinLoRA shows an 8B LoRA beating GPT-4o on the easier FiNER-139, so a fine-tuned 1.7B-4B reaching or beating the frontier on FinNI is plausible, and the headline chart is F1 versus $/filing with a latency panel.
- **FinCL as published is not reproducible**: the top-200 candidate lists live in private HF datasets, the embedding index is behind a Google sign-in, retrieval needs Elasticsearch plus OpenAI embeddings, and the scorer credits the first candidate on parse failure. Only a BM25 index over 17,388 concepts ships. Any FinCL number you produce is under your own protocol and must be labelled so.
- **Your FinCL method should not be retrieve-and-rerank.** That is XBRLTagRec and FHS. The project-owned angle is generation with constrained decoding: the fine-tuned model emits the concept name under a trie or grammar restricted to the 8,596 numeric-type concepts of the 2024 taxonomy (vLLM structured outputs via xgrammar, near-zero overhead), trained on hundreds of thousands of iXBRL facts. The published ceiling it competes with is retrieval-bound (official Acc@200 about 0.33; FHS 0.255 with a 32B model). Give the frontier arms the same BM25 top-K candidate list on a stratified sample, and report the small model both open-vocabulary and candidate-constrained. The concept-frequency-bucket slice is the finance-specific finding: expect the small model to win the head and lose the tail.
- Report the end-to-end triplet (value, type, concept) too, since that is the benchmark's headline metric and it is what the BERT baselines (SECBERT micro-F1 0.1779) are measured on.
- Include an encoder baseline (SEC-BERT token classifier trained on your corpus) and a size sweep (0.6B, 1.7B, 4B). The encoder is the real cost floor and the first question an interviewer asks.

## 4. Fair comparison

The claim survives an adversarial reader only with all of the following, written into the harness before any run:

- Same instance ids for every arm; one documented dedup policy; the full set via batch APIs for accuracy and a fixed 500-instance sample run synchronously three times for latency.
- Own scorer, validated with oracles before any model is scored: a gold-copy predictor must score exactly 1.0, an empty predictor 0.0, a gold predictor with types swapped 0.0, a predictor that drops half the gold facts recall 0.5. Then ask what upstream component could force each metric: the output parser can force FinNI recall to 0 (report parse-failure rate separately); the candidate list can force FinCL accuracy (report Acc@K of the retriever upstream of the model); dedup can move F1 by a few points (report both).
- Frontier arms zero-shot with the benchmark's own prompt, plus a k-shot variant with examples drawn from the training corpus, so "you fine-tuned and they didn't" has an answer. Add a hosted fine-tuned small-model arm if budget allows: Vertex Gemini Flash-Lite SFT is the only option (OpenAI fine-tuning is closed to new users since May 2026; Anthropic has none).
- Pinned model versions. Anthropic's dateless 4.6+ ids are fixed snapshots; use OpenAI's dated snapshots where they exist (`gpt-5.4-mini-2026-03-17`; GPT-5.6 and GPT-6 have none); Gemini stable strings are the pin. Avoid arms that die inside the project window: gpt-5-mini/nano on 2026-12-11, gpt-4.1-nano on 2026-10-23, Claude Sonnet 4.5 on 2026-11-30; Gemini 2.5 models are access-restricted for new accounts.
- Cost from real token counts in API responses, priced against the dated sheet in `config/prices_2026-10-01.yaml`, reported as $/filing and $/1,000 facts, never $/token: Anthropic's 4.7+ tokenizer yields about 30% more tokens; thinking tokens are billed as output; batch is 50% off at all three vendors; promo prices expire (Gemini 3.8 Flash on 2026-12-31).
- Self-host cost as a formula: GPU $/hour divided by goodput (requests per hour that meet the SLO) at a stated utilization, reported for three utilization scenarios, plus the idle-cost caveat. Latency from request-rate sweeps with warm-up, TTFT and end-to-end p50/p95 at several concurrency levels, tool and version pinned (`vllm bench serve` or GuideLLM). Report single-request and batched regimes separately.
- Quantization variants (bf16, FP8 only native on L4-class Ada GPUs, AWQ-INT4) with per-fact flip rate, not just aggregate F1 (small models lose more at 4-bit; Llama 3.2 3B dropped several points on structured outputs).
- Three training seeds, bootstrap confidence intervals over filings (not instances; instances within a filing are correlated), every number by slice.

## 5. Feasibility

Mac versus rented GPU: mlx-lm supports LoRA/QLoRA for all candidate models and reports tokens/s and peak memory, so the Mac handles the pilot (a few thousand examples), prompt format iteration and ablations. Two caveats: attention backward materializes a full L x L matrix until a fused kernel merged on 2026-09-29 ships in a release, so keep sequences at or below 2,048-4,096 tokens on the Mac; and mlx-lm has no PEFT export, so a short converter (transpose adapter matrices, rename keys, set alpha = scale x rank) is needed before vLLM can load the adapter. MLX 4-bit checkpoints are not servable by vLLM; quantize on the GPU side with llm-compressor or AutoAWQ. Full-scale runs (100k instances, 2 epochs) go to a rented A100 (DERIVED from the LitGPT anchor of 8.3 minutes for 4,000 example-passes at 512 tokens on an A10G: roughly 7 hours per epoch per 100k 1k-token examples on an A10G, about 2-3 hours on an A100) or to Together's managed LoRA at $0.34-0.38 per million training tokens.

Budget (all DERIVED, arithmetic in `PRD.md` section 9): about 150 hours total, 110 for the minimum defensible version, which is 9-13 weeks at 12 hours a week; roughly $300-500 in GPU and API spend, of which frontier re-runs on the full FinNI set are the largest item (31.7M input and 5.0M output tokens per model; at batch prices Sonnet 5.5 is about $57, Haiku 4.5 about $29, gpt-5.4-mini about $23, Gemini 3.8 Flash about $22, before reasoning-token overhead).

## 6. Independence from FilingQA

Fork, do not depend. Copy the iXBRL fact parser into this repo at a pinned FilingQA commit, strip offsets/period/unit from the emitted schema, and rewrite the document serialization to match FinTagging's conventions (their HTML table reconstruction and sentence selection are the eval input format, and FilingQA's are not). No runtime import of FilingQA; the two projects must be demoable separately and FilingQA's parser will change under you mid-Phase 1. Data must be separate anyway: the training corpus is a pinned accession list excluding the FinTagging manifest, and the fresh slice is another. Do not reuse FilingQA's golden-set filings for training; it adds nothing and muddies the narrative of two independent systems.

## 7. What an interviewer will use

1. "SECBERT already has the best end-to-end micro-F1 at a hundredth of the cost; why a decoder?" You need the encoder on the frontier chart, and an honest answer if it wins FinNI: the decoder's case is FinCL over 17k concepts and schema-valid JSON.
2. "Your FinCL number isn't comparable to the paper." Correct; say so, and show the frontier re-run under your protocol with the retriever's Acc@K reported upstream.
3. "Fine-tuned versus zero-shot is unfair." The k-shot arm and, if funded, the Vertex SFT arm.
4. "How much of your F1 is fitting label noise?" The 300-instance audit and noise-adjusted ceiling; agreement with the authors' kappa.
5. "The base model saw these filings in pretraining." The manifest, exclusion rules, overlap counts, cutoffs, and the fresh-slice gap.
6. "What utilization did you assume?" The three-scenario table and the goodput formula.
7. "Why 0.6B-4B rather than 8B?" FinLoRA owns 8B; the size sweep is the point; add 8B only as an anchor if budget remains.
8. "What did distil labs' post not already show?" The slice where the frontier bends, in a regulated-reporting task with a 17k-label tail.
9. "Did you dedup? Which version of the dataset? Which revision of the model?" All pinned in `config/thresholds.yaml`.
10. "p95 at what request rate, and was the GPU warm?" The sweep protocol.

## Verdict

**APPROVE WITH CHANGES.**

1. Replace the thesis with the narrowed claim in section 1 and cite FinLoRA, FLaME, FinTagging, FHS, Kulkarni 2026 and the generic SLM-versus-frontier literature explicitly; read the one unread preprint first.
2. Score (value, type, concept) only; drop period and unit from the deliverable.
3. Pin `FinNI-eval-v2` and `FinCL-eval-v2`; never use `FinTagging_Original_v2` or the GitHub training set.
4. FinNI is the headline frontier with your own oracle-validated scorer and re-run frontier arms; FinCL is a second track under your own protocol with constrained decoding over the 2024 numeric-concept list, frontier arms on a stratified sample with identical BM25 candidates, and a concept-frequency slice.
5. Build the training corpus from EDGAR iXBRL with a pinned accession list that excludes the recovered 142-filing manifest by CIK and by date; run and publish the n-gram and MinHash overlap audit; add a post-cutoff fresh slice; write a contamination statement.
6. Add an encoder baseline and a 0.6B/1.7B/4B size sweep; three seeds; bootstrap CIs over filings; every metric by slice.
7. Adopt the fairness protocol in section 4 verbatim, with the dated price sheet and the goodput-based self-host cost formula.
8. Fork the FilingQA parser at a pinned commit; no runtime dependency; separate pinned corpora.
9. Audit 300 random eval instances for label noise before reporting any ceiling.
10. Hold to the hour and dollar caps in `config/thresholds.yaml`, with the cut order in the PRD.
