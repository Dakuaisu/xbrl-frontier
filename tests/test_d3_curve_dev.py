import copy

import pytest
import yaml

from scripts import d3_curve_dev
from xbrl_frontier import config


def test_retarget_changes_checkpoint_without_mutating_input():
    notebook = {"cells": [{"source": ["curve/prefix_20000/adapter_model.safetensors d3-20k prefix_20000= oldhash"]}]}
    new = d3_curve_dev.retarget(notebook, 2500, "oldhash", "newhash")
    text = "".join(new["cells"][0]["source"])
    assert "curve/prefix_2500/adapter_model.safetensors" in text and "prefix_2500=" in text
    assert "d3-2500" in text and "newhash" in text and "20000" not in text
    assert "prefix_20000=" in "".join(notebook["cells"][0]["source"])


def test_retarget_refuses_unexpected_source():
    with pytest.raises(ValueError):
        d3_curve_dev.retarget({"cells": [{"source": ["unexpected"]}]}, 5000, "old", "new")


def _hash_with_price(text: bytes) -> str:
    import hashlib
    h = hashlib.sha256(text)
    h.update(config.docs_path(config.thresholds()["pins"]["price_sheet"]).read_bytes())
    return h.hexdigest()


def test_config_differing_only_in_cap_is_equivalent():
    tree = copy.deepcopy(config.thresholds())
    tree["training"]["kaggle"]["gpu_hours_cap_total"] = -1
    text = yaml.safe_dump(tree).encode()
    assert d3_curve_dev.equivalent_config(_hash_with_price(text), text)


def test_config_differing_elsewhere_is_not_equivalent():
    tree = copy.deepcopy(config.thresholds())
    tree["eval"]["generation"]["temperature"] = 0.5
    text = yaml.safe_dump(tree).encode()
    assert not d3_curve_dev.equivalent_config(_hash_with_price(text), text)


def test_reports_require_identical_coverage():
    report = {"set": "dev", "config_hash": config.config_hash(), "instances_raw": 3, "instances_dedup": 2}
    d3_curve_dev.check_reports([report, dict(report)])
    with pytest.raises(ValueError):
        d3_curve_dev.check_reports([report, {**report, "instances_dedup": 3}])
