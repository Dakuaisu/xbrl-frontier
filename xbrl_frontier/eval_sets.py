"""Dedup policy and instance-level slice keys from eval.*."""
from . import config

INSTANCE_SLICES = ("context_type", "company", "fiscal_year", "context_token_bucket")
RUN_SLICES = ("train_size_point",)
PAIR_SLICES = ("entity_type",)


def dedup_indices(instances) -> list[int]:
    """Indices kept under eval.dedup_policy; the first occurrence of each key wins."""
    policy = config.get("eval.dedup_policy")
    if policy != "exact_context_and_answer":
        raise ValueError(f"unknown eval.dedup_policy {policy!r}")
    seen, keep = set(), []
    for inst in instances:
        key = (inst.context, inst.answer)
        if key not in seen:
            seen.add(key)
            keep.append(inst.idx)
    return keep


def context_type(inst) -> str:
    return "table" if inst.context.lstrip().startswith("<table") else "text"


def slice_keys(instances, instance_map: dict | None = None, token_counts: dict | None = None) -> dict:
    """Per-instance keys for every slice in eval.slices that can be computed from the inputs given.

    company and fiscal_year need the instance-to-filing map; context_token_bucket needs token counts.
    Unassigned instances get the key "unassigned". Missing inputs raise rather than guess.
    """
    wanted = config.get("eval.slices")
    unknown = set(wanted) - set(INSTANCE_SLICES) - set(RUN_SLICES) - set(PAIR_SLICES)
    if unknown:
        raise ValueError(f"no implementation for eval.slices {sorted(unknown)}")
    keys = {}
    if "context_type" in wanted:
        keys["context_type"] = [context_type(i) for i in instances]
    for name, field in (("company", "cik"), ("fiscal_year", "fiscal_year")):
        if name in wanted and instance_map is not None:
            keys[name] = [instance_map.get(i.idx, {}).get(field) or "unassigned" for i in instances]
    if "context_token_bucket" in wanted and token_counts is not None:
        try:
            edges = config.get("eval.context_token_bucket_edges")
        except KeyError:
            raise KeyError("eval.context_token_bucket_edges is not defined in thresholds.yaml (docs/OPEN.md O-003)")
        keys["context_token_bucket"] = [_bucket(token_counts[i.idx], edges) for i in instances]
    return keys


def load_instance_map() -> dict[int, dict]:
    """pins.eval_instance_map joined with pins.eval_manifest: instance id -> cik, accession, fiscal_year.

    Unassigned instances are absent, so slice_keys labels them "unassigned".
    """
    import csv
    manifest = {r["accession"]: r for r in csv.DictReader(open(config.docs_path(config.get("pins.eval_manifest"))))}
    out = {}
    for r in csv.DictReader(open(config.docs_path(config.get("pins.eval_instance_map")))):
        if r["accession"]:
            out[int(r["instance_id"])] = {"cik": r["cik"], "accession": r["accession"],
                                          "fiscal_year": manifest[r["accession"]]["fiscal_year"],
                                          "method": r["method"]}
    return out


def _bucket(n: int, edges: list[int]) -> str:
    lo = 0
    for e in edges:
        if n < e:
            return f"[{lo},{e})"
        lo = e
    return f"[{lo},inf)"
