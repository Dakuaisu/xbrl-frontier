"""Runtime access to docs/config/*.yaml; the only place numbers live."""
import hashlib
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS = REPO_ROOT / "docs"
THRESHOLDS_PATH = DOCS / "config" / "thresholds.yaml"


@lru_cache(maxsize=None)
def thresholds() -> dict:
    return yaml.safe_load(THRESHOLDS_PATH.read_text())


@lru_cache(maxsize=None)
def prices() -> dict:
    return yaml.safe_load(docs_path(thresholds()["pins"]["price_sheet"]).read_text())


def docs_path(rel: str) -> Path:
    """Config paths under config/ and research/ are relative to docs/; data/ is at the repo root."""
    if rel.startswith("data/"):
        return REPO_ROOT / rel
    return DOCS / rel


def get(dotted: str):
    node = thresholds()
    for part in dotted.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def config_hash() -> str:
    h = hashlib.sha256(THRESHOLDS_PATH.read_bytes())
    h.update(docs_path(thresholds()["pins"]["price_sheet"]).read_bytes())
    return h.hexdigest()


def source_commit() -> str | None:
    """Commit the running code came from: SOURCE_COMMIT written into the Kaggle source archive, else git, else None."""
    f = REPO_ROOT / "SOURCE_COMMIT"
    if f.exists():
        return f.read_text().strip()
    import subprocess
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT)
    return r.stdout.strip() if r.returncode == 0 else None
