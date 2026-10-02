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
3. **Parsing.** Remove `<think>…</think>`, take the first ```` ```json ```` fence if there is one, then
   `json.loads`. Failing that, parse the outermost `{…}` substring. Anything without a `result` list is a
   parse failure: an empty prediction, counted in `parse_failure_rate`. Items missing a string `Type`
   or a str/number `Fact` are dropped and counted as `malformed_items`, reported next to the parse-failure
   rate. Numeric `Fact`s are `str()`-ed. Nothing is retried or repaired. Alternative: strict
   whole-string JSON only. Rejected because it would score fence/prose wrapping as total failure and make
   the parse-failure rate measure formatting habit rather than extraction. Both the raw text and the
   parse outcome are stored, so a strict re-score is possible.
4. **Per-instance mean** (stored next to the global pooled score): an instance with empty gold and an
   empty prediction scores P=R=F1=1. Otherwise P or R is 0 when its denominator is 0.
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
