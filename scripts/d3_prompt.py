"""Create a verifiable context-free FinNI prompt artifact from the pinned local cache."""
import hashlib
import json
from pathlib import Path

from xbrl_frontier import config, finni

SENTINEL = "\x00CONTEXT\x00"


class PromptMismatch(ValueError):
    """Pinned benchmark queries do not share one unambiguous template."""


def extract(rows: list[finni.Instance]) -> str:
    template = None
    for row in rows:
        if not row.context or row.query.count(row.context) != 1:
            raise PromptMismatch("query context is not unique")
        candidate = row.query.replace(row.context, SENTINEL)
        if template is not None and candidate != template:
            raise PromptMismatch("query templates differ")
        template = candidate
    if template is None:
        raise PromptMismatch("benchmark contains no rows")
    return template


def main() -> None:
    rows = finni.load()
    template = extract(rows)
    artifact = {"dataset": config.get("pins.eval_finni.repo"), "revision": config.get("pins.eval_finni.hf_revision"),
                "rows_verified": len(rows), "sentinel": SENTINEL, "template": template,
                "sha256": hashlib.sha256(template.encode()).hexdigest()}
    path = Path("data/kaggle/d3-prompt.json")
    path.write_text(json.dumps(artifact, indent=2))
    print(json.dumps({key: value for key, value in artifact.items() if key != "template"}))


if __name__ == "__main__":
    main()
