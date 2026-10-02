"""Context token counts with the primary student's tokenizer at its pinned revision."""
from functools import lru_cache

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from . import config

PRIMARY = "Qwen/Qwen3-1.7B"


@lru_cache(maxsize=None)
def tokenizer() -> Tokenizer:
    pin = next(s for s in config.get("pins.students") if s["id"] == PRIMARY)
    path = hf_hub_download(PRIMARY, "tokenizer.json", revision=pin["hf_revision"],
                           cache_dir=str(config.REPO_ROOT / "data" / "hf"))
    return Tokenizer.from_file(path)


def context_token_counts(instances) -> dict[int, int]:
    """Tokens in the context string alone (no prompt template, no special tokens)."""
    tok = tokenizer()
    uniq = list(dict.fromkeys(i.context for i in instances))
    counts = dict(zip(uniq, (len(e.ids) for e in tok.encode_batch(uniq, add_special_tokens=False))))
    return {i.idx: counts[i.context] for i in instances}
