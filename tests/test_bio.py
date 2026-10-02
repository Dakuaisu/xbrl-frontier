import json

from ixbrl_facts import fintagging
from xbrl_frontier import bio, scorer

SPACE = {"O": 0, "B-us-gaap:Revenues": 1, "I-us-gaap:Revenues": 2,
         "B-us-gaap:NumberOfReportableSegments": 3, "I-us-gaap:NumberOfReportableSegments": 4}


def test_words_keep_grouped_numbers_and_strip_tags():
    assert bio.words("<td>Revenue $58,166</td><td>(1.5)%</td>") == ["Revenue", "$", "58,166", "(", "1.5", ")", "%"]


def test_alignment_follows_document_order_and_repeated_values():
    ws = bio.words("Revenue was 100 in 2024 and 100 in 2023; we have two segments.")
    ents = [{"value": "100", "concept": "us-gaap:Revenues"}, {"value": "100", "concept": "us-gaap:Revenues"},
            {"value": "two", "concept": "us-gaap:NumberOfReportableSegments"},
            {"value": "7", "concept": "us-gaap:Revenues"}, {"value": "5", "concept": "us-gaap:NotInSpace"}]
    labels, st = bio.label_words(ws, ents, SPACE)
    assert [w for w, l in zip(ws, labels) if l != "O"] == ["100", "100", "two"]
    assert st == {"facts": 5, "aligned": 3, "not_found": 1, "outside_label_space": 1}


def test_decode_round_trip_to_finni_pairs():
    ws = bio.words("<td>Revenues</td><td>58,166</td><td>two</td>")
    ents = [{"value": "58166", "concept": "us-gaap:Revenues"},
            {"value": "two", "concept": "us-gaap:NumberOfReportableSegments"}]
    labels, _ = bio.label_words(ws, ents, SPACE)
    pairs = bio.pairs_from_spans(bio.decode(ws, labels), fintagging.taxonomy())
    assert pairs == [{"Fact": "58166", "Type": "monetaryItemType"}, {"Fact": "two", "Type": "integerItemType"}]
    gold = scorer.gold_set(json.dumps({"result": pairs}))
    assert scorer.pair_set(pairs) == gold


def test_chunks_fit_the_window_and_cover_every_word():
    counts = [3, 200, 200, 200, 5, 600, 1]
    ch = bio.chunks(counts, 512)
    assert ch[0] == (0, 3) and ch[-1][1] == len(counts)
    assert all(b > a for a, b in ch) and [a for a, _ in ch[1:]] == [b for _, b in ch[:-1]]
    assert all(sum(counts[a:b]) <= 510 for a, b in ch if b - a > 1)


def test_benchmark_label_space_is_pinned():
    space = bio.label_space()
    assert len(space) == 34777 and space["O"] == 0 and sorted(space.values()) == list(range(len(space)))


def test_encoder_chunks_label_first_subtoken_and_respect_window():
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from training.encoder_train import encode_chunks
    from xbrl_frontier import config
    pin = config.get("pins.encoder_baseline")
    tok = AutoTokenizer.from_pretrained(snapshot_download(pin["id"], revision=pin["hf_revision"],
                                                          allow_patterns=["*.json", "vocab.txt"],
                                                          cache_dir=str(config.REPO_ROOT / "data" / "hf")))
    window = config.get("training.encoder.window_subtokens")
    ws = (["Revenues", "were", "58,166", "million"] * 300)
    labels = [7 if w == "58,166" else 0 for w in ws]
    chunks, cut = encode_chunks(tok, ws, labels, window)
    assert cut == 0 and len(chunks) > 1
    assert all(len(c["input_ids"]) <= window for c in chunks)
    assert sum(c["words"][1] - c["words"][0] for c in chunks) == len(ws)
    for c in chunks:
        assert c["input_ids"][0] == tok.cls_token_id and c["input_ids"][-1] == tok.sep_token_id
        word_labels = [c["labels"][p] for p in c["first_subtoken"]]
        assert word_labels == labels[c["words"][0]:c["words"][1]]
        assert sum(l != -100 for l in c["labels"]) == len(c["first_subtoken"])
