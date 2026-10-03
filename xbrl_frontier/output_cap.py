"""Output-length cap for every scored generation, computed from the input only (TRADEOFFS T-039).

max_tokens = eval.generation.output_cap.tokens_per_numeric x numeric_count(context) + output_cap.base_tokens,
bounded by the remaining context. numeric_count counts every span in the context that a FinNI fact can be:
digit numbers, standalone dash cells (nil), and the number / nil words that occur as gold facts.
"""
import re

from . import config

_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
          "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million "
          "billion no none not nil immaterial nominal")
NUMERIC = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?|\.\d+|(?<!\w)[—–-](?!\w)|\b(?:" + "|".join(_WORDS.split()) + r")\b",
                     re.IGNORECASE)


def numeric_count(context: str) -> int:
    return len(NUMERIC.findall(context))


def cap(context: str, remaining: int | None = None) -> int:
    c = config.get("eval.generation.output_cap")
    n = c["tokens_per_numeric"] * numeric_count(context) + c["base_tokens"]
    return n if remaining is None else min(n, remaining)
