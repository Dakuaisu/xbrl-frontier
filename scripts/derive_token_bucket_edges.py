"""One-time derivation of eval.context_token_bucket_edges (owner decision 2026-10-02, TRADEOFFS T-001).

Edges = tercile boundaries of context token counts over the deduplicated pinned FinNI set, counted with
the primary student's tokenizer at its pinned revision. Edge k (k = 1, 2) is the count at sorted index
floor(k * n / 3); buckets are [0, e1), [e1, e2), [e2, inf). Refuses to overwrite an existing key.
"""
import statistics

from xbrl_frontier import config, eval_sets, finni, tokens


def main() -> None:
    text = config.THRESHOLDS_PATH.read_text()
    if "context_token_bucket_edges:" in text:
        raise SystemExit("eval.context_token_bucket_edges already set; it is computed once and never edited")
    rows = finni.load()
    keep = set(eval_sets.dedup_indices(rows))
    counts = tokens.context_token_counts([r for r in rows if r.idx in keep])
    s = sorted(counts.values())
    n = len(s)
    edges = [s[(k * n) // 3] for k in (1, 2)]
    pin = next(p for p in config.get("pins.students") if p["id"] == tokens.PRIMARY)["hf_revision"]
    line = (f"  context_token_bucket_edges:   [ {edges[0]}, {edges[1]} ]"
            f"              # terciles of context tokens over the {n} dedup instances; {tokens.PRIMARY}@{pin[:12]} "
            f"tokenizer (identical in both students); edge k = sorted[floor(k*n/3)]; scripts/derive_token_bucket_edges.py\n")
    anchor = next(l for l in text.splitlines(keepends=True) if l.lstrip().startswith("slices:"))
    config.THRESHOLDS_PATH.write_text(text.replace(anchor, anchor + line, 1))
    print("edges", edges, "n", n, "min", s[0], "median", statistics.median(s), "max", s[-1])


if __name__ == "__main__":
    main()
