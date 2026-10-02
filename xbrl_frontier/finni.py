"""Load FinNI-eval-v2 at the pinned revision and verify it."""
import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from . import config

CACHE_DIR = config.REPO_ROOT / "data" / "hf"


@dataclass(frozen=True)
class Instance:
    idx: int
    context_index: object
    context: str
    query: str
    answer: str


class RowCountMismatch(RuntimeError):
    pass


def download() -> Path:
    pin = config.get("pins.eval_finni")
    if not pin["hf_revision"]:
        raise RuntimeError("pins.eval_finni.hf_revision is empty; pin it first")
    return Path(snapshot_download(repo_id=pin["repo"], repo_type="dataset",
                                  revision=pin["hf_revision"], cache_dir=CACHE_DIR))


def load() -> list[Instance]:
    pin = config.get("pins.eval_finni")
    root = download()
    files = sorted(root.glob(f"data/{pin['split']}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {pin['split']} parquet under {root}")
    rows = []
    for f in files:
        rows.extend(pq.read_table(f).to_pylist())
    if len(rows) != pin["rows_expected"]:
        raise RowCountMismatch(f"{len(rows)} rows, expected pins.eval_finni.rows_expected={pin['rows_expected']}")
    return [Instance(i, r["context_index"], r["context"], r["query"], r["answer"]) for i, r in enumerate(rows)]


def gold_pairs(answer: str) -> list[dict]:
    """Gold answers use the benchmark schema {"result":[{"Fact","Type"}]}."""
    return json.loads(answer)["result"]
