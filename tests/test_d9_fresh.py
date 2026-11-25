import json
from pathlib import Path

import pytest

from scripts import d9_fresh

PILOT = Path("data/kaggle/d3-supervised-pilot.ipynb")


def _toy():
    src = ("glob.glob('/kaggle/input/**/dev.jsonl', recursive=True) " + d9_fresh.DEV_SHA + " " + d9_fresh.DEV_SHA +
           " assert len(items) == 1933 and 'dev_sha256': 'dev_instances': 'provider_timeout_seconds': 7200"
           " /kaggle/working/d3-pilot '--set', 'dev' '--session-hours', '2'")
    return {"cells": [{"source": [src]}]}


def test_retarget_switches_every_dev_input_to_fresh():
    text = "".join(d9_fresh.retarget(_toy(), "freshhash", 3434)["cells"][0]["source"])
    assert d9_fresh.DEV_SHA not in text and "dev.jsonl" not in text and "'--set', 'dev'" not in text
    assert "fresh_instances.jsonl" in text and "freshhash" in text and "== 3434 and" in text
    assert "'--set', 'fresh'" in text and f"'--session-hours', '{d9_fresh.SESSION_HOURS}'" in text
    assert "/kaggle/working/d9-fresh" in text


def test_retarget_refuses_unexpected_source():
    with pytest.raises(ValueError):
        d9_fresh.retarget({"cells": [{"source": ["unexpected"]}]}, "h", 1)


@pytest.mark.skipif(not PILOT.exists(), reason="local Kaggle pilot notebook is untracked")
def test_retarget_applies_to_the_verified_pilot_notebook():
    text = "".join(d9_fresh.retarget(json.loads(PILOT.read_text()), "freshhash", 3434)["cells"][0]["source"])
    assert "d3-pilot" not in text and "'--set', 'fresh'" in text
