from training.encoder_train import encode_chunks


class Tok:
    cls_token_id, sep_token_id = 1, 2
    pieces = {"a": [10], "bb": [11, 12], "\u200b": []}

    def __call__(self, ws, is_split_into_words, add_special_tokens):
        ids, word_ids = [], []
        for i, w in enumerate(ws):
            ids += self.pieces[w]
            word_ids += [i] * len(self.pieces[w])
        return type("Enc", (dict,), {"word_ids": lambda self: word_ids})({"input_ids": ids})


def test_word_without_subtokens_keeps_labels_aligned_with_ids():
    chunks, cut = encode_chunks(Tok(), ["a", "\u200b", "bb"], [5, 6, 7], 512)
    c = chunks[0]
    assert len(c["input_ids"]) == len(c["labels"]) == 5 and cut == 0
    assert c["labels"] == [-100, 5, 7, -100, -100]
    assert c["first_subtoken"] == [1, None, 2] and c["words"] == (0, 3)


def test_truncation_keeps_word_alignment():
    chunks, cut = encode_chunks(Tok(), ["bb", "bb", "bb"], [5, 6, 7], 4)
    assert all(len(c["input_ids"]) == len(c["labels"]) for c in chunks)
