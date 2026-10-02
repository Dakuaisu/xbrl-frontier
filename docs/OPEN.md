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
