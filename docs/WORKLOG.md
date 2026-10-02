# Work log

Dated implementation notes: what was built, how it was verified, what is next.

## 2026-10-02 Phase 0, unit 1: environment

- Built `.venv/` (Python 3.12.14) with the owner-approved set in T-001; `requirements.txt`
  (direct pins) and `requirements.lock` (pip freeze).
- Verified: `pip freeze` shows the four pins. SEC endpoints return 200 with the new User-Agent (O-001).
- Next: pin the HF revisions and verify the FinNI row count.
