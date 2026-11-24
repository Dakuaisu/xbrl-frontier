import copy

import pytest

from scripts import d3_10k_notebook


def test_retargets_only_the_checkpoint_and_output():
    source = "final_adapter/adapter_model.safetensors d3-pilot final= oldhash"
    original = {"cells": [{"source": [source]}]}
    result = d3_10k_notebook.retarget(original, "oldhash", "newhash")
    text = "".join(result["cells"][0]["source"])
    assert "curve/prefix_10000/adapter_model.safetensors" in text
    assert "prefix_10000=" in text and "newhash" in text and "d3-10k" in text
    assert original["cells"][0]["source"] == [source]


def test_refuses_unexpected_pilot_notebook():
    with pytest.raises(ValueError):
        d3_10k_notebook.retarget(copy.deepcopy({"cells": [{"source": ["wrong"]}]}), "old", "new")
