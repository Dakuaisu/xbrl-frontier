# xbrl-frontier: builder working agreement

You are building `xbrl-frontier`: the first public, reproducible accuracy / cost-per-filing / latency
frontier for fine-tuned Qwen3-0.6B and 1.7B students on FinTagging FinNI. You measure them against
pinned frontier APIs on identical instances, with fp16 / INT8 / INT4 serving on a market-priced T4
and on owned hardware. The deliverable is the harness and the chart, not a model.

The spec is `docs/PRD.md` (v2.2, frozen). Read the section that covers your current work before
starting it. `docs/REVIEW.md` explains why the spec looks the way it does, and
`docs/research/evidence_brief.md` holds the verified facts behind it.

## Where things live

- The PRD and configs write paths relative to `docs/`: `config/thresholds.yaml` is
  `docs/config/thresholds.yaml`, `config/prices_2026-10-01.yaml` is
  `docs/config/prices_2026-10-01.yaml`, and `research/...` is `docs/research/...`. Don't move them.
- The findings register is `docs/OPEN.md`, the decisions log is `docs/TRADEOFFS.md`, and
  implementation notes go in `docs/WORKLOG.md`. Keep all three current.
- `data/` sits at the repo root and is git-ignored, except for the artifacts the PRD ships: the
  `pins.*` files (manifest, instance map, train and fresh accession lists) and
  `data/contamination_audit/`. Downloads, corpora, checkpoints and run outputs stay untracked. To
  ship another artifact, whitelist it in `.gitignore` and log the reason in `docs/TRADEOFFS.md`.

## Numbers

- `docs/config/thresholds.yaml` and the dated price sheet are the only places numbers live. Every
  gate, pin, seed, budget, sample size, SLO and price goes there. The PRD names keys and never
  states values, and code, docs and commit messages follow the same rule: read the key at
  runtime, never hard-code or restate a value.
- Empty pin fields (HF revisions, software versions, `fresh_slice_filing_date_min`) are filled once
  at pin time (Phase 0) and never edited after that. The blank AWS T4 price is filled only from an
  observation with its own date and source URL, never from memory. If it can't be observed, drop it.
- Every dollar figure cites the price sheet and its observed date. Report $/filing and $/1k facts,
  never $/token.
- README result blanks and resume-bullet blanks are filled only from measured runs, with the run
  that produced each number identified. Until then they stay blank. No estimate, planning rate
  or illustrative figure ever goes into a result slot.

## Measurement integrity

- No LLM-as-judge anywhere. Scoring is exact match. The only human instrument is the label audit.
- Calibrate before you score anything: the oracles in `eval.oracles` must hit their expected values
  within tolerance. The oracle tests run on every change to the scorer.
- Gates are decided per `gates.decision_rule`. When the interval straddles a threshold, the result is
  labelled **inconclusive**, not pass or fail. An inconclusive headline gate ships with that label
  and without the claim. A failed gate is reported as failed.
- Gates, thresholds and definitions are never retuned after the fact. A low number is a finding
  about the system, not a reason to change the metric, the dedup policy, the sample or the scorer.
- The deduplicated set is the headline. Raw and deduplicated scores are both stored and reported.
- A parse failure counts as an empty prediction and is reported as its own rate per arm. It is
  never silently replaced, retried until it parses, or dropped.
- For every gated metric, first measure the upstream component that could force it, and report it
  next to the metric (PRD section 3, "Upstream forcing check").
- Frontier arms get the benchmark's verbatim prompt, zero-shot, with thinking per `frontier.thinking`.
  Use the pinned model ids, and record the model field each response returns. Frontier arms run on
  the real provider APIs, because cost, tokens, latency and the served model are the measurements.
- The frontier sample ids are pinned before the probe and never changed afterwards.
- Never fabricate data, stub a metric, or fill in placeholder labels, items or results. If something
  can't be measured yet, it stays blank and gets an entry in `docs/OPEN.md`.
- When the spec and real data disagree, the data wins. Record the resolution in `docs/TRADEOFFS.md`
  with the evidence and your reasoning.

## Data hygiene (PRD section 4)

- Never use `FinTagging_Original_v2` or the GitHub-only FinTagging training set.
- Every training filing comes from `pins.train_accessions`, selected once and frozen. It obeys
  `contamination.train_filing_date_max` and excludes every CIK under
  `contamination.exclude_ciks_from`.
- The iXBRL parser in `ixbrl_facts/` is forked from FilingQA at a pinned commit, recorded in
  `docs/TRADEOFFS.md`. There is no runtime dependency on FilingQA, and nothing in FilingQA is modified.
- EDGAR access follows SEC fair-access rules (rate limit, declared User-Agent), and the downloader
  enforces them. The declared User-Agent is
  `Adnan Rashid adnanrashid230@gmail.com` (SEC rejects the GitHub noreply address with a 403). If SEC throttles or rejects
  it, stop, log it as OWNER-BLOCKED, and don't switch to a different identity or route around it.
- Format divergences found by the parity check are fixed in the serializer, not in the model.

## Scope and phases

- Work proceeds one phase at a time, in PRD section 9 order: 0 pin and calibrate, 1 corpus, 2 train,
  3 frontier arms, 4 serve and load, 5 audit and write-up. Don't build ahead into a later phase.
- Out of scope (PRD section 2): FinCL, the triplet, period/unit/scale/sign, k-shot or hosted
  fine-tuned arms, students above 1.7B, rented GPUs.
- Cuts follow the PRD's cut order exactly. The encoder baseline, label audit, contamination audit,
  oracle tests and learning curve are never cut.
- A phase is complete when its PRD section 9 work is done and verified in the repo, or blocked only
  on OWNER-BLOCKED items. Say "Phase N complete" in your report and commit.

## Work that only the owner does

You build the tooling, schemas and instructions for the items below, log each one in `docs/OPEN.md`
as **OWNER-BLOCKED** (what is ready, the exact command or steps the owner runs, and what you need
back), and move on to the next unblocked task. Never simulate, stub or model-generate their outputs.

- Both label-audit passes: hand-checking instances against the filings. The second pass happens a
  week after the first. Never have a model fill or pre-fill the audit sheets.
- Anything on the RTX 3050 under WSL2. It's another machine and can't be reached from here.
- Reading the Schema-Aware XBRL preprint, which needs a logged-in browser.
- The public GitHub issue to the FinTagging authors, until the owner approves its text.

## Kaggle and frontier calls (owner decision T-033)

- You run Kaggle yourself with the `kaggle` CLI. `KAGGLE_API_TOKEN` is in your environment. Never
  print it, never write it to a file, never read `.env`. Stay inside the `training.kaggle` quota
  and cap keys, and record the GPU-hours used by each run.
- Billed spend is $0. The only frontier key is the free-tier `GEMINI_API_KEY`, so
  `gemini-3.5-flash-lite` is the only arm that runs. Never call another provider, use a paid tier,
  or enable billing. Report dollar figures from measured tokens priced against the dated sheet.
- A rate-limit (429) or server error is retried with backoff and never scored as an empty
  prediction. Only a model reply that fails to parse counts as a parse failure.

## Engineering rules

- You run headless: nobody can answer an interactive prompt. Never use a question or ask-user tool.
  Every question goes in the `DECISIONS NEEDED:` section of your end-of-turn report. Do all the
  unblocked work you can first, then end the turn.
- Ask before adding any dependency: request it under `DECISIONS NEEDED:` with the reason and the
  version you'll pin, and don't install it until a reply approves it. Pin approved versions and log
  each one in `docs/TRADEOFFS.md`. Use the project virtualenv in `.venv/`, never a global install.
- Commit each coherent unit of work, with tests passing. Never add a remote, push, or rewrite
  history.
- Never put secrets in the repo. API keys come from the environment, and `.env` is git-ignored.
- Every run records what makes it reproducible: pinned revisions, software versions, seed, config
  hash, checkpoint hash, hardware and flags.

## Records

- `docs/OPEN.md`: a findings register. Each entry has an id, a status (OPEN, OWNER-BLOCKED,
  RESOLVED), what was found, the evidence, and the next action.
- `docs/TRADEOFFS.md`: a decisions log. Each entry gives the decision, the alternatives and why.
  Decisions made without the owner go under a heading containing
  "AUTONOMOUS DECISION - owner to review".
- `docs/WORKLOG.md`: dated implementation notes, saying what was built, how it was verified, and
  what's next.

## End-of-turn report

End every turn with: what you did, with commits and test or command output as evidence. Then a
`PROCEEDING:` line with the next concrete step. If there's a choice the PRD doesn't settle, add
`DECISIONS NEEDED:` with the options and your recommendation.
