import json

from scripts import d3_supervised_notebook as pilot


def test_prepared_notebook_is_single_device_bounded_and_secret_free():
    notebook = pilot.build_notebook(pilot.Inputs("example-template", "prompt-digest", "dev-digest", "weight-digest", "runner-digest", 1933))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "CUDA_VISIBLE_DEVICES" in source
    assert "torch.cuda.device_count() == 1" in source
    assert "--session-hours" in source
    assert "token=False" in source
    assert "kaggle_secrets" not in source
    assert "runpy.run_module" in source
    assert "weight-digest" in source and "dev-digest" in source
    for cell in notebook["cells"]:
        compile("".join(cell["source"]), "pilot", "exec")
    assert json.loads(json.dumps(notebook))["nbformat"] == 4
