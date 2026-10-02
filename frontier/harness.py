"""Frontier-arm harness: dry-run by default; live steps only behind the T-033 guard (PRD sections 3, 7; T-024, T-034).

Steps and the instances they cover:
  probe          --accession ACC     instances of one eval filing (pins.eval_instance_map), synchronous prices
  batch          --accessions FILE   instances of the pinned sample filings, batch prices + reasoning margin
  latency        --ids FILE          fixed instance ids, frontier.sync_runs runs each, synchronous prices
  fresh_control  (fresh slice)       every fresh instance, batch prices + reasoning margin
--dry-run (default): renders the exact requests to data/phase3/requests/ and prices them from the price sheet
using token estimates (Qwen3 tokenizer as a proxy for input; the gold answer as a proxy for output). No
network, no key. --live --approval T-033: guard first (frontier.guard: Gemini free tier only, synchronous only),
then the key, then resumable synchronous calls paced by frontier.freetier (429/5xx retried, never scored empty).
Every step, including the accuracy run (`batch`) and the fresh control, is collected synchronously; it is priced
notionally at the sheet's batch rates for accuracy steps and sync rates for probe/latency. Billed spend is $0.
Live outputs: predictions in the scorer's format, usage/latency logs (request time and throttle wait separately),
the returned model field, status.json (complete / paused_daily_quota / stopped_call_failed), ledger rows.
"""
import argparse
import csv
import json
import os
import time
import urllib.request
from pathlib import Path

from frontier import guard, requests as fr
from xbrl_frontier import config, finni, sft_format

# XBRL_FRONTIER_PHASE3_DIR lets tests write outside data/phase3 (never touch real collected data).
PHASE3 = Path(os.environ.get("XBRL_FRONTIER_PHASE3_DIR") or config.REPO_ROOT / "data" / "phase3")
SYNC_STEPS = {"probe", "latency"}
KEY_ENV = {"openai_responses": "OPENAI_API_KEY", "gemini_generate_content": "GEMINI_API_KEY"}


def instances(step: str, accession: str | None, accessions: Path | None, ids: Path | None) -> list[dict]:
    if step == "fresh_control":
        return [{"id": x["id"], "context": x["text"], "answer": sft_format.answer(x["numeric_entities"])}
                for x in map(json.loads, open(PHASE3.parent / "fresh" / "instances.jsonl"))]
    rows = finni.load()
    if step == "latency":
        want = [int(l) for l in open(ids) if l.strip()]
        return [{"id": i, "context": rows[i].context, "answer": rows[i].answer} for i in want]
    accs = {accession} if step == "probe" else {l.strip() for l in open(accessions) if l.strip()}
    imap = config.docs_path(config.get("pins.eval_instance_map"))
    pick = [int(r["instance_id"]) for r in csv.DictReader(open(imap)) if r["accession"] in accs]
    return [{"id": i, "context": rows[i].context, "answer": rows[i].answer} for i in pick]


def price(arm: str, step: str, items: list[dict]) -> dict:
    from xbrl_frontier.tokens import tokenizer
    tok = tokenizer()
    p = config.prices()["api_per_million_tokens"][arm]
    sync = step in SYNC_STEPS
    runs = config.get("frontier.sync_runs") if step == "latency" else 1
    tin = sum(len(tok.encode(sft_format.query(x["context"])).ids) for x in items) * runs
    tout = sum(len(tok.encode(x["answer"]).ids) for x in items) * runs
    usd = (tin * (p["input"] if sync else p["batch_input"]) + tout * (p["output"] if sync else p["batch_output"])) / 1e6
    margin = 0.0 if sync else config.get("frontier.reasoning_token_margin")
    return {"arm": arm, "step": step, "instances": len(items), "runs": runs, "est_input_tokens": tin,
            "est_output_tokens": tout, "price_mode": "sync" if sync else "batch", "margin": margin,
            "usd_estimate": usd * (1 + margin), "price_sheet": config.get("pins.price_sheet"),
            "price_sheet_observed": config.prices()["observed"],
            "estimator": "Qwen3 tokenizer proxy for input; gold answer tokens for output; replaced by probe usage"}


def reprice_from_probe(arm: str, est: dict, probe_dir: Path | None = None) -> dict | None:
    """Scale a batch-priced proxy estimate by the arm's measured/proxy token ratios on its own probe instances.

    Returns None when the arm has no probe usage. Ratios use run 0 of the probe; measured output includes
    reasoning tokens (billed as output)."""
    from xbrl_frontier.tokens import tokenizer
    probe_dir = probe_dir or PHASE3 / "live" / f"probe_{arm}"
    usage = probe_dir / "usage.jsonl"
    if not usage.exists():
        return None
    rows = [r for r in map(json.loads, open(usage)) if r.get("run", 0) == 0 and r.get("input_tokens") is not None]
    if not rows:
        return None
    tok = tokenizer()
    fin = finni.load()
    proxy_in = sum(len(tok.encode(sft_format.query(fin[r["id"]].context)).ids) for r in rows)
    proxy_out = sum(len(tok.encode(fin[r["id"]].answer).ids) for r in rows)
    r_in = sum(r["input_tokens"] for r in rows) / proxy_in
    r_out = sum(r["output_tokens"] or 0 for r in rows) / max(proxy_out, 1)
    p = config.prices()["api_per_million_tokens"][arm]
    tin, tout = est["est_input_tokens"] * r_in, est["est_output_tokens"] * r_out
    usd = (tin * p["batch_input"] + tout * p["batch_output"]) / 1e6 * (1 + est["margin"])
    return {**est, "usd_estimate": usd, "repriced_from_probe": str(usage), "probe_instances": len(rows),
            "ratio_input_measured_over_proxy": r_in, "ratio_output_measured_over_proxy": r_out,
            "proxy_usd_estimate": est["usd_estimate"],
            "estimator": "proxy tokens scaled by this arm's measured/proxy ratios on its probe instances"}


def actual_usd(arm: str, sync: bool, input_tokens: int, output_tokens: int) -> float:
    p = config.prices()["api_per_million_tokens"][arm]
    return (input_tokens * (p["input"] if sync else p["batch_input"]) +
            output_tokens * (p["output"] if sync else p["batch_output"])) / 1e6


def _api_key(arm: str) -> str:
    return os.environ[KEY_ENV[fr.arm_spec(arm)["api"]]]


def _http(url: str, key: str, api: str, data: bytes | None = None, ctype: str = "application/json",
          method: str | None = None) -> bytes:
    if api != "gemini_generate_content":
        raise guard.GuardRefused("only the Gemini API is called (T-033)")
    h = {"x-goog-api-key": key}
    if data is not None:
        h["Content-Type"] = ctype
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def run_free_tier(arm: str, step: str, items: list[dict], key: str, out: Path, pacer=None) -> dict:
    """Resumable synchronous collection on the free tier (T-033, T-034). Completed (id, run) pairs are skipped;
    a call that fails after retries, or a daily-quota stop, leaves its instance pending (never an empty prediction)."""
    from frontier.freetier import Pacer, call_with_retries
    pacer = pacer or Pacer()
    api = fr.arm_spec(arm)["api"]
    runs = config.get("frontier.sync_runs") if step == "latency" else 1
    priced_sync = step in SYNC_STEPS
    usage_p, pred_p = out / "usage.jsonl", out / "predictions.jsonl"
    done = {(r["id"], r["run"]) for r in map(json.loads, open(usage_p))} if usage_p.exists() else set()
    priced = 0.0
    n_new = 0
    with open(pred_p, "a") as pf, open(usage_p, "a") as uf:
        for run in range(runs):
            for x in items:
                if (x["id"], run) in done:
                    continue
                b = json.dumps(fr.body(arm, sft_format.query(x["context"]))).encode()
                raw, req_s, log = call_with_retries(lambda: _http(fr.sync_endpoint(arm), key, api, b), pacer)
                r = fr.parse(arm, json.loads(raw))
                usd = actual_usd(arm, priced_sync, r["input_tokens"] or 0, r["output_tokens"] or 0)
                priced += usd
                uf.write(json.dumps({"id": x["id"], "run": run, "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                     "latency_s": req_s, **log,
                                     **{k: v for k, v in r.items() if k != "text"}, "priced_usd": usd,
                                     "priced_at": "sync" if priced_sync else "batch", "tier": "free"}) + "\n")
                uf.flush()
                if run == 0:
                    pf.write(json.dumps({"id": x["id"], "text": r["text"]}) + "\n")
                    pf.flush()
                n_new += 1
    return {"new_calls": n_new, "priced_usd": priced, "total_expected": runs * len(items),
            "done": len(done) + n_new}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run"])
    ap.add_argument("--step", required=True, choices=["probe", "batch", "latency", "fresh_control"])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--accession")
    ap.add_argument("--accessions", type=Path)
    ap.add_argument("--ids", type=Path)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--approval")
    args = ap.parse_args()
    out = PHASE3 / ("live" if args.live else "requests") / f"{args.step}_{args.arm}"
    out.mkdir(parents=True, exist_ok=True)
    items = instances(args.step, args.accession, args.accessions, args.ids)
    est = price(args.arm, args.step, items)
    rendered = fr.batch_payload(args.arm, [(x["id"], sft_format.query(x["context"])) for x in items])
    (out / ("requests.jsonl" if isinstance(rendered, str) else "requests.json")).write_text(
        rendered if isinstance(rendered, str) else json.dumps(rendered))
    (out / "estimate.json").write_text(json.dumps(est, indent=1))
    print(json.dumps(est, indent=1))
    if not args.live:
        return
    g = guard.check(args.step, args.arm, args.approval, mode="sync")
    key = _api_key(args.arm)
    from frontier.freetier import CallFailed, DailyQuotaExhausted
    status = {"step": args.step, "arm": args.arm, "guard": g, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    code = 0
    try:
        res = run_free_tier(args.arm, args.step, items, key, out)
        status.update(state="complete", **res)
    except DailyQuotaExhausted as e:
        status.update(state="paused_daily_quota", resume_after=e.resume_at, quota=e.detail)
        code = 3
    except CallFailed as e:
        status.update(state="stopped_call_failed", error=str(e))
        code = 4
    usage = [json.loads(l) for l in open(out / "usage.jsonl")] if (out / "usage.jsonl").exists() else []
    runs = config.get("frontier.sync_runs") if args.step == "latency" else 1
    got = {(u["id"], u["run"]) for u in usage}
    status["pending"] = [{"id": x["id"], "run": r} for r in range(runs) for x in items if (x["id"], r) not in got]
    status["model_fields"] = sorted({u.get("model_returned") or "" for u in usage})
    status["first_last_call_utc"] = [usage[0].get("at_utc"), usage[-1].get("at_utc")] if usage else None
    status["done"] = len(usage)
    status["expected"] = len(items) * (config.get("frontier.sync_runs") if args.step == "latency" else 1)
    status["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "status.json").write_text(json.dumps(status, indent=1))
    guard.record(args.step, args.arm, sum(u["priced_usd"] for u in usage), {"source": str(out), "kind": "free_tier_sync",
                 "state": status["state"], "calls": len(usage)})
    print(json.dumps({k: v for k, v in status.items() if k != "guard"}, indent=1))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
