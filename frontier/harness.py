"""Frontier-arm harness: dry-run by default; live steps only behind the spend guard (PRD sections 3, 7; T-024).

Steps and the instances they cover:
  probe          --accession ACC     instances of one eval filing (pins.eval_instance_map), synchronous prices
  batch          --accessions FILE   instances of the pinned sample filings, batch prices + reasoning margin
  latency        --ids FILE          fixed instance ids, frontier.sync_runs runs each, synchronous prices
  fresh_control  (fresh slice)       every fresh instance, batch prices + reasoning margin
--dry-run (default): renders the exact requests to data/phase3/requests/ and prices them from the price sheet
using token estimates (Qwen3 tokenizer as a proxy for input; the gold answer as a proxy for output). No
network, no key. --live --approval T-NNN: spend guard first (frontier.guard), then the key, then the calls.
Live outputs: predictions in the scorer's format, usage/latency logs, the returned model field, ledger rows
priced from the returned usage. `collect` fetches a submitted batch.
"""
import argparse
import csv
import json
import os
import time
import urllib.request
import uuid
from pathlib import Path

from frontier import guard, requests as fr
from xbrl_frontier import config, finni, sft_format

PHASE3 = config.REPO_ROOT / "data" / "phase3"
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


def actual_usd(arm: str, sync: bool, input_tokens: int, output_tokens: int) -> float:
    p = config.prices()["api_per_million_tokens"][arm]
    return (input_tokens * (p["input"] if sync else p["batch_input"]) +
            output_tokens * (p["output"] if sync else p["batch_output"])) / 1e6


def _api_key(arm: str) -> str:
    return os.environ[KEY_ENV[fr.arm_spec(arm)["api"]]]


def _http(url: str, key: str, api: str, data: bytes | None = None, ctype: str = "application/json",
          method: str | None = None) -> bytes:
    h = {"Authorization": f"Bearer {key}"} if api == "openai_responses" else {"x-goog-api-key": key}
    if data is not None:
        h["Content-Type"] = ctype
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def run_sync(arm: str, step: str, items: list[dict], key: str, out: Path) -> float:
    api = fr.arm_spec(arm)["api"]
    runs = config.get("frontier.sync_runs") if step == "latency" else 1
    total = 0.0
    with open(out / "predictions.jsonl", "w") as pf, open(out / "usage.jsonl", "w") as uf:
        for run in range(runs):
            for x in items:
                b = json.dumps(fr.body(arm, sft_format.query(x["context"]))).encode()
                t0 = time.perf_counter()
                resp = json.loads(_http(fr.sync_endpoint(arm), key, api, b))
                lat = time.perf_counter() - t0
                r = fr.parse(arm, resp)
                usd = actual_usd(arm, True, r["input_tokens"] or 0, r["output_tokens"] or 0)
                total += usd
                uf.write(json.dumps({"id": x["id"], "run": run, "latency_s": lat, **{k: v for k, v in r.items() if k != "text"},
                                     "usd": usd}) + "\n")
                if run == 0:
                    pf.write(json.dumps({"id": x["id"], "text": r["text"]}) + "\n")
    return total


def submit_batch(arm: str, items: list[dict], key: str, out: Path) -> dict:
    api = fr.arm_spec(arm)["api"]
    payload = fr.batch_payload(arm, [(x["id"], sft_format.query(x["context"])) for x in items])
    if api == "openai_responses":
        boundary = uuid.uuid4().hex
        data = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n"
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"batch.jsonl\"\r\n"
                f"Content-Type: application/jsonl\r\n\r\n").encode() + payload.encode() + f"\r\n--{boundary}--\r\n".encode()
        f = json.loads(_http(f"{fr.OPENAI}/files", key, api, data, f"multipart/form-data; boundary={boundary}"))
        b = json.loads(_http(f"{fr.OPENAI}/batches", key, api, json.dumps(
            {"input_file_id": f["id"], "endpoint": "/v1/responses", "completion_window": "24h"}).encode()))
        state = {"api": api, "batch_id": b["id"], "input_file_id": f["id"]}
    else:
        b = json.loads(_http(f"{fr.GEMINI}/models/{arm}:batchGenerateContent", key, api, json.dumps(payload).encode()))
        state = {"api": api, "batch_name": b["name"]}
    (out / "batch_state.json").write_text(json.dumps(state, indent=1))
    return state


def collect(arm: str, step: str, out: Path) -> None:
    state = json.loads((out / "batch_state.json").read_text())
    key = _api_key(arm)
    api = state["api"]
    rows = []
    if api == "openai_responses":
        b = json.loads(_http(f"{fr.OPENAI}/batches/{state['batch_id']}", key, api))
        if b["status"] != "completed":
            print("batch status:", b["status"], b.get("request_counts"))
            return
        for line in _http(f"{fr.OPENAI}/files/{b['output_file_id']}/content", key, api).decode().splitlines():
            r = json.loads(line)
            body = (r.get("response") or {}).get("body") or {}
            rows.append((r["custom_id"], fr.parse(arm, body) if body else None, r.get("error")))
    else:
        b = json.loads(_http(f"{fr.GEMINI}/{state['batch_name']}", key, api))
        if b.get("state") != "JOB_STATE_SUCCEEDED":
            print("batch state:", b.get("state"))
            return
        inl = (b.get("dest") or b.get("response") or {}).get("inlinedResponses")
        inl = inl.get("inlinedResponses", []) if isinstance(inl, dict) else (inl or [])
        for r in inl:
            rows.append(((r.get("metadata") or {}).get("key"), fr.parse(arm, r["response"]) if r.get("response") else None,
                         r.get("error")))
    total = 0.0
    with open(out / "predictions.jsonl", "w") as pf, open(out / "usage.jsonl", "w") as uf:
        for cid, r, err in rows:
            cid = int(cid) if str(cid).isdigit() else cid
            if r is None:
                pf.write(json.dumps({"id": cid, "text": None}) + "\n")
                uf.write(json.dumps({"id": cid, "error": err}) + "\n")
                continue
            usd = actual_usd(arm, False, r["input_tokens"] or 0, r["output_tokens"] or 0)
            total += usd
            pf.write(json.dumps({"id": cid, "text": r["text"]}) + "\n")
            uf.write(json.dumps({"id": cid, **{k: v for k, v in r.items() if k != "text"}, "usd": usd}) + "\n")
    guard.record(step, arm, total, {"source": str(out), "kind": "batch"})
    print(f"collected {len(rows)} responses, ${total:.4f} at batch prices")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run", "collect"])
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
    if args.action == "collect":
        collect(args.arm, args.step, out)
        return
    items = instances(args.step, args.accession, args.accessions, args.ids)
    est = price(args.arm, args.step, items)
    rendered = fr.batch_payload(args.arm, [(x["id"], sft_format.query(x["context"])) for x in items])
    (out / ("requests.jsonl" if isinstance(rendered, str) else "requests.json")).write_text(
        rendered if isinstance(rendered, str) else json.dumps(rendered))
    (out / "estimate.json").write_text(json.dumps(est, indent=1))
    print(json.dumps(est, indent=1))
    if not args.live:
        return
    g = guard.check(args.step, args.arm, est["usd_estimate"], args.approval)
    key = _api_key(args.arm)
    if args.step in SYNC_STEPS:
        usd = run_sync(args.arm, args.step, items, key, out)
        guard.record(args.step, args.arm, usd, {"source": str(out), "kind": "sync", **g})
        print(f"done: ${usd:.4f} at synchronous prices")
    else:
        print("submitted:", submit_batch(args.arm, items, key, out), "; run `collect` later")


if __name__ == "__main__":
    main()
