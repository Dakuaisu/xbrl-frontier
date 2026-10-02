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
