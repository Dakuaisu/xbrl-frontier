import pytest

from scripts import d3_prompt
from xbrl_frontier.finni import Instance


def test_extracts_single_shared_template():
    rows = [Instance(0, 0, "abc", "prefix abc suffix", ""),
            Instance(1, 1, "xyz", "prefix xyz suffix", "")]
    assert d3_prompt.extract(rows) == "prefix \x00CONTEXT\x00 suffix"


def test_refuses_mismatched_templates():
    rows = [Instance(0, 0, "abc", "prefix abc", ""), Instance(1, 1, "xyz", "other xyz", "")]
    with pytest.raises(ValueError):
        d3_prompt.extract(rows)
