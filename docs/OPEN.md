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

## O-003 `context_token_bucket` slice has no bucket edges in config (OPEN)

- Status: OPEN
- Found: `eval.slices` lists `context_token_bucket`, but `thresholds.yaml` defines no bucket edges or
  tokenizer, and the config is frozen. `xbrl_frontier/eval_sets.py` raises rather than invent edges.
- Evidence: `grep context_token docs/config/thresholds.yaml`.
- Next action: owner decision. Add `eval.context_token_bucket_edges` (plus which tokenizer counts the
  tokens; the pinned Qwen3 tokenizer is the natural choice), and approve the `tokenizers` dependency.
  See DECISIONS NEEDED in the turn report.

## O-004 `company` and `fiscal_year` slices wait on the instance map (OPEN)

- Status: OPEN
- Found: both slice keys come from `pins.eval_instance_map`. `eval_sets.slice_keys` reads them from that
  map and labels unmapped instances `unassigned`.
- Next action: becomes RESOLVED when the instance map ships (Phase 0, step 4).

## O-005 Kaggle T4 Phase 0 runs: throughput benchmark, smoke quantization, T4 smoke test (OWNER-BLOCKED)

- Status: OWNER-BLOCKED (Kaggle is owner-only; it also needs O-007 approval first)
- Ready: `notebooks/phase0_kaggle_t4.ipynb` (thin driver), `serving/phase0_benchmark.py`,
  `serving/quantize_base.py`, `serving/smoke_test.py`, `serving/common.py`. Local analysis is in
  `scripts/phase0_cost.py`. All are covered by `tests/test_cost_and_serving.py`, which needs no GPU.
- Owner steps:
  1. After the O-007 pins are filled and committed, build the source tarball on the Mac:
     `git archive --format=tar.gz -o /tmp/xbrl-frontier-src.tar.gz HEAD`
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

- Status: OWNER-BLOCKED (the laptop is unreachable from here; needs O-005's `smoke_checkpoints` and O-007)
- Ready: `serving/run_rtx3050_smoke.sh`, which runs `serving.smoke_test --card rtx3050_wsl2` with
  `serving.hardware.rtx3050_wsl2.vllm_flags`.
- Owner steps (in WSL2, NVIDIA driver with WSL CUDA support installed):
  1. Copy the repo (same commit as the Kaggle run) and the unpacked `smoke_checkpoints/` to the laptop.
  2. From the repo root: `serving/run_rtx3050_smoke.sh /path/to/smoke_checkpoints`
  3. Return `data/phase0/rtx3050_wsl2.tar.gz` (the script prints its path).
- What I need back: that tarball. A variant that fails stays recorded with its error and is removed
  from the serving table per `serving.smoke_test.failure_action` (PRD risk: the 1.7B may not fit).

## O-007 Kaggle/WSL2 software pins need owner approval (OPEN)

- Status: OPEN (blocks O-005 and O-006)
- Found: `pins.software` is empty. Phase 0 remote runs need vllm, llmcompressor and autoawq. Proposed
  versions (current PyPI releases, checked 2026-10-02): vllm 0.30.0 (2026-09-22), llmcompressor 0.14.0
  (2026-09-22), autoawq 0.2.9 (2025-05-11, unmaintained). Transformers, torch and compressed-tensors
  are left to vllm/llmcompressor's resolver, and the installed versions are recorded in every run's `env`.
- Next action: on approval, fill `pins.software.{vllm,llmcompressor,autoawq}` once and log them in TRADEOFFS.

## O-008 AWS g4dn.xlarge T4 price is blank in the sheet (RESOLVED)

- Status: RESOLVED (2026-10-02)
- Found: `gpu_per_hour.aws.g4dn_xlarge_T4` was null and could only be filled from a dated observation
  with a source URL.
- Resolution: observed on 2026-10-02T04:45:03Z from AWS's public on-demand price file for us-east-1 Linux
  (the file the AWS EC2 pricing page loads; publication date 2026-09-25). The value, record, rate code,
  URL and file hash are in the price sheet under `gpu_per_hour.aws_g4dn_xlarge_T4_observation`. The
  sheet's top-level `observed` date stays 2026-10-01, and this source carries its own date. All three T4
  sources now appear in `scripts/phase0_cost.py`.
