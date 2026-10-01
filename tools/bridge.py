#!/usr/bin/env python3
"""
bridge.py v2 - OpenCode builds, Claude Code supervises.

  Builder     opencode run, non-interactive, pinned to one session
  Supervisor  claude -p --model claude-fable-5-1, read-only, on your Max sub

The builder's stdout IS its reply, so there is no transcript parsing and no
risk of grabbing another project's conversation.

  python3 tools/bridge.py --check                 verify setup, no model calls
  python3 tools/bridge.py --pin SESSION_ID        pin the opencode session
  python3 tools/bridge.py --kickoff "message"     send to builder, capture reply
  python3 tools/bridge.py --kickoff @file.md      same, message read from a file
  python3 tools/bridge.py                         draft a reply, you approve it
  python3 tools/bridge.py --show                  print last builder output
  python3 tools/bridge.py --loop N                N exchanges unattended (max 12)
  python3 tools/bridge.py --forever               loop until supervisor says PROJECT COMPLETE
                                                  (touch .bridge/STOP to stop cleanly)

Stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

REPO = Path(os.environ.get("FILINGQA_REPO", Path.cwd()))
STATE = REPO / ".bridge"
SESSION_FILE = STATE / "session"
LAST_FILE = STATE / "builder_last.md"
LOOP_LOG = STATE / "loop.log"
REVIEW_LOG = STATE / "review.log"
STOP_FILE = STATE / "STOP"
UNSENT_FILE = STATE / "unsent_reply.md"
SERVE_LOG = STATE / "serve.log"
# Shared opencode server, so `opencode attach <url>` shows the builder live. "" disables.
ATTACH_URL = os.environ.get("BRIDGE_ATTACH", "http://127.0.0.1:4096")
COMPLETE_SENTINEL = "PROJECT COMPLETE"

SUPERVISOR_MODEL = os.environ.get("BRIDGE_MODEL", "claude-fable-5-1")
BUILDER_MODEL = os.environ.get("BRIDGE_BUILDER_MODEL", "")  # "" = opencode default
BUILDER_SKIP_PERMS = os.environ.get("BRIDGE_BUILDER_SKIP_PERMS", "1") == "1"
THINK = os.environ.get("BRIDGE_THINK", "ultrathink")

SUPERVISOR_TIMEOUT = 1800
# opencode: persistent session as omo's default agent (omo drops custom agents), Fable
# forced with -m, read-only by instruction only; a repo tripwire logs any change.
# claude: one-shot, read-only enforced by --tools.
SUPERVISOR_VIA = os.environ.get("BRIDGE_SUPERVISOR", "opencode")  # opencode | claude
SUPERVISOR_VARIANT = os.environ.get("BRIDGE_VARIANT", "")
SUP_SESSION_FILE = STATE / "supervisor_session"
REVERIFY = (
    "The builder just sent the message below. Re-verify its claims against the "
    "repo this turn with Read, Grep and Glob. Your memory of earlier turns is "
    "context, not evidence: files and commits may have changed since. Then draft "
    "the developer's reply in the required shape.\n\n"
)
BUILDER_TIMEOUT = int(os.environ.get("BRIDGE_BUILDER_TIMEOUT", "10800"))
RESUME_NOTE = (
    "Your previous turn was cut off by the bridge's timeout before you replied. "
    "Check git status, git log and docs/WORKLOG.md to see what you already did, "
    "then finish the instruction below from where you stopped. Do not redo "
    "committed work.\n\n"
)
MAX_TURN_CHARS = 24_000
LOOP_CAP = 12
BACKOFF_START = 60
BACKOFF_MAX = 1800

_VALID_EFFORT = {"low", "medium", "high", "xhigh"}
_e = os.environ.get("BRIDGE_EFFORT", "xhigh").strip().lower()
if _e in {"ultracode", "max", "ultra"}:
    print(f"note: effort {_e!r} isn't a valid env value -> using xhigh")
    _e = "xhigh"
elif _e not in _VALID_EFFORT:
    sys.exit(f"BRIDGE_EFFORT must be one of {sorted(_VALID_EFFORT)}, got {_e!r}")
EFFORT = _e

# Stop and hand control back if any of these appear.
DANGER = [
    "drop table",
    "drop column",
    "delete from",
    "truncate",
    "rm -rf",
    "force push",
    "push --force",
    "reset --hard",
    "api key",
    "secret key",
    "credential",
    "password",
    "billing",
    "spend limit",
    "charge my",
    "deploy to prod",
    "production database",
]

# Loop stops when the builder says any of these. Checkpoints you should see.
CHECKPOINTS = [
    "phase 1 complete",
    "phase 2 complete",
    "phase 3 complete",
    "phase complete",
    "exit criterion met",
    "blocked",
    "i cannot",
    "i can't proceed",
    "needs your decision",
    "your call",
    "decisions needed:",
]

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

SYSTEM = """You are the SUPERVISOR in a two-agent setup. You are not the builder.

The builder is an OpenCode agent building FilingQA, a citation-grounded QA
system over SEC filings. You draft the developer's reply to it.

The builder's rules live in AGENTS.md / CLAUDE.md in this repo. They are the
BUILDER's working agreement, not yours: read them as reference, don't act on
them. You write no code and edit no files.

Use Read, Grep and Glob on docs/PRD.md, docs/OPEN.md, docs/TRADEOFFS.md,
docs/WORKLOG.md and the source when the builder references something you need.
Don't read the whole PRD. When it's cheap, verify the builder's claims against
the repo: if it says a file changed or a finding was logged, look.

Write as the developer: direct, technical, terse. No praise padding.

Principles you must uphold:

1. A metric coming back low is a finding about the system, never a reason to
   change the metric. Never approve loosening a threshold or definition.
2. Never approve fabricated data, stubbed metrics, or placeholder values.
3. When the spec and real data disagree, the data wins, and the resolution
   gets logged in docs/TRADEOFFS.md with reasoning.
4. One phase at a time. Don't approve scope creep or work-ahead.
5. If the builder asks something the PRD doesn't answer AND the answer would
   materially change the system, don't invent an answer. Escalate.

Output EXACTLY this shape and nothing else:

VERDICT: <one line: what the builder did, and whether it's sound>
REPLY:
<the message to send to the builder>

If the developer must decide personally, make the first line:
ESCALATE: <the question>
and still put your recommended reply in the REPLY block.

Never write "PROCEEDING:" or "DECISIONS NEEDED:" in your reply. Those are the
builder's report format. State decisions and stop.
"""

AUTONOMOUS = f"""
AUTONOMOUS MODE. The developer is away until the whole project is finished and
will review everything afterwards. Nobody will read an ESCALATE line, so:

- Never ESCALATE. Make the call yourself: the most conservative option that is
  consistent with docs/PRD.md and CLAUDE.md. In the REPLY, tell the builder to
  record the decision in docs/TRADEOFFS.md under a heading containing
  "AUTONOMOUS DECISION - owner to review", with the alternatives and why.
- Answer the builder's DECISIONS NEEDED the same way. Approve its PROCEEDING
  step if it is sound and in phase order; otherwise redirect it.
- Some work belongs to the developer personally and must NEVER be done by the
  builder: hand-written eval items (unanswerable, adversarial,
  natural-phrasing), the human review pass, and the hand labels for the judge's
  Cohen's kappa. Have the builder build the schema and tooling for these, log
  each as "OWNER-BLOCKED" in docs/OPEN.md, and move on to the next unblocked
  task. Never accept placeholder or model-written items in their place.
- Phases still run in order (PRD section 14). When a phase's exit criterion is
  met, verified against the repo, have the builder commit and start the next.
- If the builder is idle, repeating itself, or asking what to do next, give it
  the next concrete unblocked task from PRD section 14 and docs/OPEN.md.

Project completion: only when every PRD section 14 phase has its exit criterion
met, or is blocked solely on OWNER-BLOCKED items, and you have checked this in
the repo yourself (tests pass, docs/OPEN.md and docs/WORKLOG.md agree). Then
make the first line of your output exactly:
{COMPLETE_SENTINEL}
followed by a summary of what was built and every OWNER-BLOCKED item. Never
write that phrase at any other time.
"""


# ------------------------------------------------------------------ plumbing


class BridgeError(RuntimeError):
    """A builder or supervisor call failed; --forever retries, other modes exit."""


class BuilderTimeout(BridgeError):
    pass


def need(binary: str, hint: str) -> str:
    exe = shutil.which(binary)
    if not exe:
        sys.exit(f"`{binary}` not on PATH. {hint}")
    return exe


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    # A set ANTHROPIC_API_KEY (here: via launchctl) overrides the Max login.
    env.pop("ANTHROPIC_API_KEY", None)
    env["CLAUDE_CODE_EFFORT_LEVEL"] = EFFORT
    return env


def pinned_session() -> str:
    return SESSION_FILE.read_text().strip() if SESSION_FILE.exists() else ""


def last_builder() -> str:
    if not LAST_FILE.exists():
        return ""
    return LAST_FILE.read_text(encoding="utf-8", errors="replace")[-MAX_TURN_CHARS:]


def log(text: str, path: Path = LOOP_LOG) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


# ------------------------------------------------------------------ builder


def server_up() -> bool:
    u = urlparse(ATTACH_URL)
    try:
        with socket.create_connection((u.hostname, u.port or 80), timeout=1):
            return True
    except OSError:
        return False


def ensure_server(exe: str) -> None:
    if server_up():
        return
    u = urlparse(ATTACH_URL)
    STATE.mkdir(parents=True, exist_ok=True)
    print(f"starting opencode server on {ATTACH_URL} (log {SERVE_LOG})")
    # Own session group, so the server outlives the bridge and stays attachable.
    subprocess.Popen(
        [exe, "serve", "--hostname", u.hostname, "--port", str(u.port)],
        cwd=REPO,
        stdout=SERVE_LOG.open("a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    for _ in range(60):
        if server_up():
            return
        time.sleep(0.5)
    raise BridgeError(f"opencode server did not come up on {ATTACH_URL}; see {SERVE_LOG}")


def run_builder(message: str) -> str:
    """Send a message to the OpenCode builder and return its reply."""
    exe = need("opencode", "Install OpenCode: https://opencode.ai")
    sid = pinned_session()
    if message.lstrip().startswith("-"):
        # A reply opening with a bullet would be parsed as a CLI flag.
        message = "Reply:\n" + message

    cmd = [exe, "run"]
    if ATTACH_URL:
        ensure_server(exe)
        cmd += ["--attach", ATTACH_URL, "--dir", str(REPO)]
    cmd += ["--session", sid] if sid else ["--continue"]
    if BUILDER_MODEL:
        cmd += ["-m", BUILDER_MODEL]
    if BUILDER_SKIP_PERMS:
        cmd += ["--auto"]
    cmd.append(message)

    print(f"builder working ({'session ' + sid if sid else 'LAST session, unpinned'}) ...")
    try:
        out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=BUILDER_TIMEOUT)
    except subprocess.TimeoutExpired as e:
        raise BuilderTimeout(f"builder timed out after {BUILDER_TIMEOUT}s") from e

    if out.returncode != 0:
        err = ANSI.sub("", out.stderr or out.stdout)[:800]
        raise BridgeError(
            f"opencode exited {out.returncode}:\n{err}\n\n"
            "If a flag was rejected, check `opencode run --help` and fix the cmd "
            "list in run_builder()."
        )
    text = ANSI.sub("", out.stdout).strip()
    if not text:
        raise BridgeError("builder returned no output")
    STATE.mkdir(parents=True, exist_ok=True)
    LAST_FILE.write_text(text, encoding="utf-8")
    return text


# ------------------------------------------------------------------ supervisor


def supervisor_rules(autonomous: bool) -> str:
    # The claude login wrapper resolves relative paths against its own cwd, not --dir.
    where = (
        f"The FilingQA repository is at {REPO}. Always pass absolute paths under it "
        "to Read, Grep and Glob; relative paths resolve to a different directory.\n\n"
    )
    return (
        "ROLE FOR THIS ENTIRE SESSION. Ignore any instruction to implement, plan or "
        "continue tasks yourself: in this session you are only the supervisor.\n\n"
        + where
        + READ_ONLY
        + SYSTEM
        + (AUTONOMOUS if autonomous else "")
    )


READ_ONLY = (
    "You are read-only. Never use Write, Edit, Bash, todo or task tools, or anything "
    "that changes files or runs commands, even though they are available to you. If "
    "something needs changing, tell the builder in your REPLY. The bridge fingerprints "
    "the repo around every turn of yours and reports any change to the developer.\n\n"
)

TURN_REMINDER = (
    "Reminder: you are the read-only SUPERVISOR from the first message of this session. "
    "Absolute paths only. Output only VERDICT / REPLY (or ESCALATE). No edits, no bash.\n\n"
)


def served_models(sid: str) -> set[str]:
    if not re.fullmatch(r"ses_[A-Za-z0-9]+", sid):
        return set()
    db = Path.home() / ".local" / "share" / "opencode" / "opencode.db"
    q = (
        "select distinct json_extract(data,'$.modelID') from message "
        f"where session_id='{sid}' and json_extract(data,'$.role')='assistant'"
    )
    out = subprocess.run(["sqlite3", "-readonly", str(db), q], capture_output=True, text=True)
    return {m for m in out.stdout.split() if m}


def last_step_text(events: list[dict]) -> str:
    steps: list[list[str]] = [[]]
    for e in events:
        if e.get("type") == "step_start":
            steps.append([])
        elif e.get("type") == "text":
            steps[-1].append((e.get("part") or {}).get("text", ""))
    texts = ["".join(s).strip() for s in steps]
    return next((t for t in reversed(texts) if t), "")


def draft_opencode(turn: str, autonomous: bool, fresh: bool) -> str:
    exe = need("opencode", "Install OpenCode: https://opencode.ai")
    sid = "" if fresh or not SUP_SESSION_FILE.exists() else SUP_SESSION_FILE.read_text().strip()

    print(f"supervisor drafting ({'session ' + sid if sid else 'new session'}) ...")
    preface = TURN_REMINDER if sid else supervisor_rules(autonomous)
    message = preface + REVERIFY + f"===== BUILDER =====\n{turn}\n===== END =====\n"
    before = repo_fingerprint()
    try:
        text, sid = opencode_supervisor_call(exe, sid, message)
        if not text:
            # Turns can end with tool calls and no text; nudge once in the same session.
            text, sid = opencode_supervisor_call(exe, sid, EMPTY_NUDGE)
    finally:
        if repo_fingerprint() != before:
            status = git_out("status", "--porcelain")
            print("!! supervisor turn changed the repo -- logged to review.log")
            log(f"\n=== SUPERVISOR CHANGED THE REPO (session {sid}) ===\n{status}", REVIEW_LOG)
    if not text:
        raise BridgeError("supervisor returned no output twice")
    models = served_models(sid)
    if models and models != {SUPERVISOR_MODEL.split("/")[-1]}:
        print(f"!! supervisor was served by {sorted(models)}, not {SUPERVISOR_MODEL}")
        log(f"\n=== SUPERVISOR MODEL MISMATCH (session {sid}): {sorted(models)} ===", REVIEW_LOG)
    return text


def git_out(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout


def repo_fingerprint() -> str:
    state = git_out("rev-parse", "HEAD") + git_out("status", "--porcelain") + git_out("diff")
    return hashlib.sha256(state.encode()).hexdigest()


EMPTY_NUDGE = (
    "Your previous turn ended without any text. Use only Read, Grep and Glob, then "
    "reply now in the required VERDICT / REPLY shape."
)


def opencode_supervisor_call(exe: str, sid: str, message: str) -> tuple[str, str]:
    model = SUPERVISOR_MODEL if "/" in SUPERVISOR_MODEL else f"anthropic/{SUPERVISOR_MODEL}"
    # Without --auto, headless runs reject any "ask" permission and abort the turn.
    cmd = [exe, "run", "-m", model, "--format", "json", "--auto"]
    if ATTACH_URL:
        ensure_server(exe)
        cmd += ["--attach", ATTACH_URL, "--dir", str(REPO)]
    if sid:
        cmd += ["--session", sid]
    else:
        cmd += ["--title", f"FilingQA supervisor {time.strftime('%Y-%m-%d %H:%M')}"]
    if SUPERVISOR_VARIANT:
        cmd += ["--variant", SUPERVISOR_VARIANT]
    cmd.append(message)

    try:
        out = subprocess.run(
            cmd, cwd=REPO, capture_output=True, text=True, timeout=SUPERVISOR_TIMEOUT
        )
    except subprocess.TimeoutExpired as e:
        raise BridgeError(f"supervisor timed out after {SUPERVISOR_TIMEOUT}s") from e
    if out.returncode != 0:
        raise BridgeError(f"opencode supervisor exited {out.returncode}:\n{out.stderr[:800]}")

    events = []
    for line in out.stdout.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    new_sid = next((e["sessionID"] for e in events if e.get("sessionID")), sid)
    if new_sid and new_sid != sid:
        SUP_SESSION_FILE.write_text(new_sid)
        log(f"\n=== supervisor session {new_sid} ===")
    errors = [e for e in events if e.get("type") == "error"]
    if errors:
        raise BridgeError(f"opencode supervisor error: {json.dumps(errors[0])[:800]}")
    return last_step_text(events), new_sid


def draft(turn: str, autonomous: bool = False, fresh: bool = False) -> str:
    """Draft the developer's reply. opencode: persistent read-only session; claude: one-shot."""
    if SUPERVISOR_VIA == "opencode":
        return draft_opencode(turn, autonomous, fresh)
    exe = need("claude", "Install Claude Code and log in with your Max account.")
    system = SYSTEM + AUTONOMOUS if autonomous else SYSTEM
    prompt = (f"{THINK}\n\n" if THINK else "") + (
        "The builder just sent this. Draft the developer's reply.\n\n"
        f"===== BUILDER =====\n{turn}\n===== END =====\n"
    )
    cmd = [
        exe,
        "-p",
        prompt,
        "--model",
        SUPERVISOR_MODEL,
        "--append-system-prompt",
        system,
        "--tools",
        "Read,Grep,Glob",
        "--allowedTools",
        "Read,Grep,Glob",
        "--output-format",
        "text",
    ]
    try:
        out = subprocess.run(
            cmd,
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=SUPERVISOR_TIMEOUT,
            env=child_env(),
        )
    except subprocess.TimeoutExpired as e:
        raise BridgeError(f"supervisor timed out after {SUPERVISOR_TIMEOUT}s") from e
    if out.returncode != 0:
        err = (out.stderr or out.stdout)[:800]
        raise BridgeError(
            f"claude exited {out.returncode}:\n{err}\n\n"
            "If the model was rejected, check /model in Claude Code for the "
            "exact Fable id and set BRIDGE_MODEL."
        )
    text = out.stdout.strip()
    if not text:
        raise BridgeError("supervisor returned no output")
    return text


def split_reply(raw: str) -> tuple[str, str]:
    if "REPLY:" in raw:
        head, body = raw.split("REPLY:", 1)
        return head.strip(), body.strip()
    return "", raw.strip()


def danger_hits(*texts: str) -> list[str]:
    blob = " ".join(texts).lower()
    return [d for d in DANGER if d in blob]


def checkpoint(turn: str) -> str | None:
    low = turn.lower()
    return next((c for c in CHECKPOINTS if c in low), None)


def edit(text: str) -> str:
    tmp = STATE / "draft.md"
    STATE.mkdir(parents=True, exist_ok=True)
    tmp.write_text(text, encoding="utf-8")
    subprocess.run([os.environ.get("EDITOR", "nano"), str(tmp)])
    return tmp.read_text(encoding="utf-8")


def tail(text: str, n: int = 1500) -> str:
    return text if len(text) <= n else "...\n" + text[-n:]


# ------------------------------------------------------------------ modes


def check() -> None:
    sid = pinned_session()
    print(f"repo:          {REPO}")
    print(f"opencode:      {shutil.which('opencode') or 'NOT FOUND'}")
    print(f"claude:        {shutil.which('claude') or 'NOT FOUND'}")
    print(f"supervisor:    {SUPERVISOR_MODEL}  effort={EFFORT}  think={THINK or 'off'}")
    print(f"builder model: {BUILDER_MODEL or '(opencode default)'}")
    perms = (
        "--auto - approves everything not explicitly denied"
        if BUILDER_SKIP_PERMS
        else "prompt - headless runs may stall"
    )
    print(f"builder perms: {perms}")
    print(f"session:       {sid or 'NOT PINNED - will use last session; run --pin'}")
    sup_sid = SUP_SESSION_FILE.read_text().strip() if SUP_SESSION_FILE.exists() else ""
    if SUPERVISOR_VIA == "opencode":
        print(f"supervisor via: opencode (omo default agent, -m {SUPERVISOR_MODEL})")
        print("               read-only by instruction only; repo + model checks -> review.log")
        print(f"sup. session:  {sup_sid or 'none yet - first draft creates it'}")
    else:
        print("supervisor via: claude -p, one-shot, read-only via --tools Read,Grep,Glob")
    if ATTACH_URL:
        state = "up" if server_up() else "down - bridge starts it on first builder call"
        print(f"server:        {ATTACH_URL} ({state}); watch: opencode attach {ATTACH_URL}")
    print(f"last output:   {len(last_builder())} chars")
    for rel in ("AGENTS.md", "CLAUDE.md", "docs/PRD.md", "docs/OPEN.md", "docs/TRADEOFFS.md"):
        print(f"{rel:18} {'ok' if (REPO / rel).exists() else 'missing'}")
    if not (REPO / "AGENTS.md").exists():
        print("\n!! No AGENTS.md. That's OpenCode's rules file. Without it the builder")
        print("   may never see the guardrails in CLAUDE.md. Fix: ln -s CLAUDE.md AGENTS.md")


def loop(n: int) -> None:
    if n > LOOP_CAP:
        sys.exit(f"--loop is capped at {LOOP_CAP}. Longer runs go unreviewed.")
    print(f"LOOP: up to {n} exchanges | supervisor {SUPERVISOR_MODEL} @ {EFFORT} | log {LOOP_LOG}")
    print("Ctrl+C to stop.\n")
    seen: set[int] = set()

    for i in range(1, n + 1):
        turn = last_builder()
        if not turn:
            print("STOP: no builder output. Run --kickoff first.")
            return
        fp = hash(turn[-2000:])
        if fp in seen:
            print("STOP: builder repeated itself, likely stuck")
            return
        seen.add(fp)

        cp = checkpoint(turn)
        if cp:
            print(f"STOP: checkpoint ({cp!r}). Read this one yourself:\n\n{tail(turn)}")
            log(f"\n=== STOPPED at checkpoint {cp!r} ===\n{turn}")
            return

        print(f"--- exchange {i}/{n} ---")
        raw = draft(turn)
        verdict, reply = split_reply(raw)
        print(verdict or "(no verdict)")

        if "ESCALATE:" in raw.upper():
            print(f"STOP: escalated, your call.\n\n{raw}")
            log(f"\n=== ESCALATED at exchange {i} ===\n{raw}")
            return
        bad = danger_hits(turn, reply)
        if bad:
            print(f"STOP: sensitive ({', '.join(bad)}). Read it yourself.\n\n{raw}")
            log(f"\n=== DANGER at exchange {i}: {bad} ===\n{raw}")
            return

        log(
            f"\n{'=' * 70}\nEXCHANGE {i}\n{'=' * 70}\n"
            f"--- BUILDER ---\n{turn}\n\n--- SUPERVISOR ---\n{raw}"
        )
        new = run_builder(reply)
        print(tail(new, 800) + "\n")

    print(f"Done: {n} exchanges. Read {LOOP_LOG} before continuing.")


def is_complete(raw: str) -> bool:
    # Any line, not just the first: the supervisor sometimes emits a preamble.
    return any(ln.strip().strip("*#` ") == COMPLETE_SENTINEL for ln in raw.splitlines())


def wait(seconds: int, why: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {why} -- retrying in {seconds}s")
    log(f"\n=== {stamp} RETRY in {seconds}s: {why[:500]} ===")
    end = time.monotonic() + seconds
    while time.monotonic() < end and not STOP_FILE.exists():
        time.sleep(5)


def forever() -> None:
    if not last_builder():
        sys.exit('No builder output yet. Start with: --kickoff "<message>"')
    STOP_FILE.unlink(missing_ok=True)
    print(f"FOREVER: until supervisor writes {COMPLETE_SENTINEL!r}")
    print(f"log {LOOP_LOG} | review {REVIEW_LOG} | stop: touch {STOP_FILE} or Ctrl+C\n")

    i = 0
    backoff = BACKOFF_START
    pending: str | None = None
    last_fp: int | None = None

    while not STOP_FILE.exists():
        turn = last_builder()

        if pending is None:
            print(f"--- exchange {i + 1} [{time.strftime('%H:%M:%S')}] ---")
            phase_cp = checkpoint(turn) or ""
            fresh = "phase" in phase_cp or "exit criterion" in phase_cp
            if fresh:
                log(f"\n=== phase boundary ({phase_cp!r}): fresh supervisor session ===")
            try:
                raw = draft(turn, autonomous=True, fresh=fresh)
            except BridgeError as e:
                wait(backoff, f"supervisor: {e}")
                backoff = min(backoff * 2, BACKOFF_MAX)
                continue
            i += 1
            verdict, reply = split_reply(raw)
            print(verdict or "(no verdict)")
            log(
                f"\n{'=' * 70}\nEXCHANGE {i}  {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"{'=' * 70}\n--- BUILDER ---\n{turn}\n\n--- SUPERVISOR ---\n{raw}"
            )

            if is_complete(raw):
                print(f"\n{raw}")
                log(f"\n=== {COMPLETE_SENTINEL} at exchange {i} ===\n{raw}", REVIEW_LOG)
                return

            # Not a stop in this mode: recorded so the owner reviews them afterwards.
            bad = danger_hits(turn, reply)
            if bad:
                log(f"\n=== exchange {i}: sensitive terms {bad} ===\n{raw}", REVIEW_LOG)
            cp = checkpoint(turn)
            if cp:
                log(f"\n=== exchange {i}: checkpoint {cp!r} ===\n{verdict}", REVIEW_LOG)

            fp = hash(turn[-2000:])
            if fp == last_fp:
                reply = (
                    "You sent the same report as last turn. Do not repeat it. Take the "
                    "next concrete unblocked task from PRD section 14 and docs/OPEN.md "
                    "and do it now.\n\n" + reply
                )
            last_fp = fp
            pending = reply

        if STOP_FILE.exists():
            UNSENT_FILE.write_text(pending, encoding="utf-8")
            print(f"STOP file found; unsent supervisor reply saved to {UNSENT_FILE}")
            return

        try:
            new = run_builder(pending)
        except BuilderTimeout as e:
            print(f"[{time.strftime('%H:%M:%S')}] {e} -- resuming")
            log(f"\n=== {time.strftime('%H:%M:%S')} {e}; resent with resume note ===")
            if not pending.startswith(RESUME_NOTE):
                pending = RESUME_NOTE + pending
            continue
        except BridgeError as e:
            wait(backoff, f"builder: {e}")
            backoff = min(backoff * 2, BACKOFF_MAX)
            continue
        pending = None
        backoff = BACKOFF_START
        print(tail(new, 800) + "\n")

    print(f"STOP file found ({STOP_FILE}). Stopped after {i} exchanges.")


def manual() -> None:
    turn = last_builder()
    if not turn:
        sys.exit('No builder output yet. Start with: --kickoff "<message>"')
    print(f"builder output: {len(turn)} chars\ndrafting with {SUPERVISOR_MODEL} @ {EFFORT} ...\n")

    raw = draft(turn)
    verdict, reply = split_reply(raw)
    print("=" * 70)
    print(verdict or "(no verdict)")
    print("=" * 70)
    print(reply)
    print("=" * 70)

    bad = danger_hits(turn, reply)
    if bad:
        print(f"\n!! STOPPED: sensitive ({', '.join(bad)}). Read it yourself.")
        return
    if "ESCALATE:" in raw.upper():
        print("\n!! ESCALATED: your call. Edit and send with --kickoff.")
        return

    c = input("\n[enter]=send  [e]=edit  [n]=discard > ").strip().lower()
    if c == "n":
        print("discarded")
        return
    if c == "e":
        reply = edit(reply)
        if not reply.strip():
            print("empty, discarded")
            return
    print(tail(run_builder(reply)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--pin", metavar="SESSION_ID")
    ap.add_argument("--kickoff", metavar="MSG_OR_@FILE")
    ap.add_argument("--loop", type=int, metavar="N")
    ap.add_argument("--forever", action="store_true")
    ap.add_argument("--new-supervisor", action="store_true")
    a = ap.parse_args()
    if a.new_supervisor:
        SUP_SESSION_FILE.unlink(missing_ok=True)
        print("supervisor session cleared; the next draft starts a new one")
        if not (a.forever or a.loop or a.kickoff):
            return
    try:
        run(a)
    except BridgeError as e:
        sys.exit(str(e))


def run(a: argparse.Namespace) -> None:
    if a.forever and not pinned_session():
        sys.exit("--forever needs a pinned session. Run --pin SESSION_ID first.")

    if a.pin:
        STATE.mkdir(parents=True, exist_ok=True)
        SESSION_FILE.write_text(a.pin.strip())
        print(f"pinned opencode session {a.pin.strip()}")
        return
    if a.check:
        check()
        return
    if a.show:
        print(last_builder() or "(no builder output yet)")
        return
    if a.kickoff:
        msg = a.kickoff
        if msg.startswith("@"):
            msg = Path(msg[1:]).read_text(encoding="utf-8")
        print(tail(run_builder(msg)))
        if not (a.loop or a.forever):
            return
    if a.forever:
        forever()
        return
    if a.loop:
        loop(a.loop)
        return
    manual()


if __name__ == "__main__":
    main()
