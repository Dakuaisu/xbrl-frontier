import pytest

from scripts import d3_20k_notebook


def test_retarget_changes_checkpoint_without_mutating_input():
    notebook = {"cells": [{"source": ["curve/prefix_10000/adapter_model.safetensors d3-10k prefix_10000= oldhash"]}]}
    new = d3_20k_notebook.retarget(notebook, "oldhash", "newhash")
    text = "".join(new["cells"][0]["source"])
    assert "curve/prefix_20000/adapter_model.safetensors" in text and "prefix_20000=" in text
    assert "d3-20k" in text and "newhash" in text
    assert "prefix_10000=" in "".join(notebook["cells"][0]["source"])


def test_retarget_refuses_unexpected_source():
    with pytest.raises(ValueError):
        d3_20k_notebook.retarget({"cells": [{"source": ["unexpected"]}]}, "old", "new")
