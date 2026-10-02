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
