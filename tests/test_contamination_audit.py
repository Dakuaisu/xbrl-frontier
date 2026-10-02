from scripts.contamination_audit import MinHasher, grams, lsh_params, words
from xbrl_frontier import config


def test_words_strip_tags_punctuation_and_case():
    assert words("<td>Net income, $1,234 (USD)</td><td>Total</td>") == ["net", "income", "1234", "usd", "total"]


def test_short_text_is_one_gram():
    n = config.get("contamination.ngram_n")
    assert grams(["a", "b"], n) == ["a b"]
    assert len(grams([str(i) for i in range(n + 2)], n)) == 3
    assert grams([], n) == []


def test_lsh_params_cover_threshold():
    b, r = lsh_params(config.get("contamination.minhash_jaccard_threshold"), config.get("contamination.minhash_num_perm"))
    assert b * r <= config.get("contamination.minhash_num_perm")
    assert 0.6 < (1 / b) ** (1 / r) < 0.9


def test_minhash_identical_and_disjoint():
    m = MinHasher()
    a = words("revenue increased due to higher volumes in the fourth quarter of the fiscal year " * 3)
    b = words("the board declared a quarterly dividend payable to shareholders of record in march " * 3)
    sa, sb = m.signature(a), m.signature(b)
    assert sa == m.signature(list(a))
    assert sum(x == y for x, y in zip(sa, sb)) / len(sa) < 0.2
