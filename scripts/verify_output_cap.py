"""Check eval.generation.output_cap against every gold answer (TRADEOFFS T-039, T-040).

For each pinned student tokenizer and every gold answer (FinNI-eval-v2, all 28,787; plus the dev slice and the fresh
slice, which are also scored), the gold completion as the student would emit it (benchmark JSON + the template's turn
end) must fit within output_cap.cap(context). Writes data/output_cap_check.json; exits non-zero if any gold exceeds.
Also reports the smallest base_tokens that would cover every gold at the configured tokens_per_numeric.
"""
import json
import sys
from pathlib import Path

from xbrl_frontier import config, finni, output_cap, sft_format

OUT = config.REPO_ROOT / "data" / "output_cap_check.json"


def golds() -> dict[str, list[tuple[str, str, str]]]:
    from training.encoder_train import fresh_instances_path
    sets = {"finni": [(str(r.idx), r.context, r.answer) for r in finni.load()]}
    dev = config.REPO_ROOT / "data" / "corpus" / "dev.jsonl"
    sets["dev"] = [(x["id"], x["text"], sft_format.answer(x["numeric_entities"])) for x in map(json.loads, open(dev))]
    sets["fresh"] = [(x["id"], x["text"], sft_format.answer(x["numeric_entities"]))
                     for x in map(json.loads, open(fresh_instances_path(None)))]
    return sets


def check(tokenizers: dict, sets: dict) -> dict:
    c = config.get("eval.generation.output_cap")
    out = {"output_cap": c, "rule": "cap = tokens_per_numeric x numeric_count(context) + base_tokens", "results": {}}
    for m, tok in tokenizers.items():
        end = len(tok.encode("<|im_end|>", add_special_tokens=False))
        for name, rows in sets.items():
            over, need, slack = [], 0, []
            for i, ctx, ans in rows:
                t = len(tok.encode(ans, add_special_tokens=False)) + end
                cap = output_cap.cap(ctx)
                need = max(need, t - c["tokens_per_numeric"] * output_cap.numeric_count(ctx))
                slack.append(cap - t)
                if t > cap:
                    over.append({"id": i, "gold_tokens": t, "cap": cap})
            slack.sort()
            out["results"][f"{m}|{name}"] = {"golds": len(rows), "exceeding": len(over), "examples": over[:10],
                                              "min_base_tokens_covering_all": need, "min_slack": slack[0],
                                              "median_slack": slack[len(slack) // 2]}
    out["all_within_cap"] = all(r["exceeding"] == 0 for r in out["results"].values())
    return out


def main() -> None:
    from transformers import AutoTokenizer
    from serving.common import pinned_snapshot
    toks = {s["id"]: AutoTokenizer.from_pretrained(pinned_snapshot(s["id"])) for s in config.get("pins.students")}
    out = check(toks, golds())
    out["config_hash"] = config.config_hash()
    OUT.write_text(json.dumps(out, indent=1))
    for k, r in out["results"].items():
        print(k, {x: r[x] for x in ("golds", "exceeding", "min_base_tokens_covering_all", "min_slack", "median_slack")})
    print("all_within_cap", out["all_within_cap"])
    sys.exit(0 if out["all_within_cap"] else 1)


if __name__ == "__main__":
    main()
