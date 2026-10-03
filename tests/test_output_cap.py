from frontier import requests as fr
from scripts.score_predictions import to_parsed
from xbrl_frontier import config, output_cap, sft_format


def test_numeric_count_covers_digits_dashes_and_number_words():
    ctx = "<td>1,234</td><td>—</td><td>-</td> the company has two segments, year-end 2024 and nil debt"
    assert output_cap.numeric_count(ctx) == 6  # 1,234 / — / - / two / 2024 / nil; the hyphen in year-end is not a cell


def test_cap_is_from_input_and_bounded_by_remaining_context():
    c = config.get("eval.generation.output_cap")
    assert output_cap.cap("no numbers here at all") == c["tokens_per_numeric"] * 1 + c["base_tokens"]  # "no" is a nil word
    assert output_cap.cap("1 2 3", remaining=10) == 10


def test_request_bodies_carry_the_cap():
    p = sft_format.query("<td>5</td>")
    g = fr.body("gemini-3.5-flash-lite", p, max_output_tokens=88)
    assert g["generationConfig"]["maxOutputTokens"] == 88
    assert "maxOutputTokens" not in fr.body("gemini-3.5-flash-lite", p)["generationConfig"]
    assert fr.body("gpt-6-luna", p, max_output_tokens=88)["max_output_tokens"] == 88


def test_cut_off_outputs_are_flagged_and_scored_as_written():
    parsed, ng, cut = to_parsed({"id": 1, "text": '{"result": [{"Fact": "5", "Ty', "finish_reason": "length"})
    assert cut and not ng and parsed.parse_failed
    assert to_parsed({"id": 1, "text": '{"result": []}'}, "MAX_TOKENS")[2]
    assert not to_parsed({"id": 1, "text": '{"result": []}'}, "STOP")[2]
