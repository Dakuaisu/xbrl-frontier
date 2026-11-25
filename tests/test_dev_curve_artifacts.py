import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs" / "paper"
LABELS = ("P2,500", "P5,000", "P10,000", "P20,000", "P37,721", "Final")
SCORES = tuple(ROOT / "data" / "scores" / name for name in (
    "d3-prefix-2500-dev.json", "d3-prefix-5000-dev.json", "d3-prefix-10000-dev.json",
    "d3-prefix-20000-dev.json", "d3-prefix-37721-dev.json", "d3-final-dev-pilot.json",
))


def _table():
    rows = []
    for line in (PAPER / "learning-curve.md").read_text().splitlines():
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == 11 and cells[0] in LABELS:
            rows.append(cells)
    return rows


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_table_matches_the_saved_score_reports_and_training_manifest():
    if not any(path.exists() for path in SCORES):
        pytest.skip("local dev score reports are not distributed")
    assert all(path.exists() for path in SCORES)
    rows = _table()
    assert [row[0] for row in rows] == list(LABELS)
    run_path = ROOT / "data/kaggle/session4-retry-output/run-Qwen3-1.7B-s1/run.json"
    run = json.loads(run_path.read_text())
    expected = [(point["instances_seen"], point["global_step"]) for point in run["curve_checkpoints"]]
    expected.append((run["trained_instances"], run["global_step"]))
    report_text = (PAPER / "learning-curve.md").read_text()
    assert _sha(run_path) in report_text
    for row, path, (seen, step) in zip(rows, SCORES, expected, strict=True):
        score = json.loads(path.read_text())
        assert score["kind"] == "xbrl_frontier.score_predictions/v1" and score["set"] == "dev"
        assert (int(row[1].replace(",", "")), int(row[2].replace(",", ""))) == (seen, step)
        values = []
        for population in ("raw", "dedup"):
            summary = score[population]["overall"]
            contexts = score[population]["slices"]["context_type"]
            values.extend(f"{value:.4f}" for value in (
                summary["finni_f1"], contexts["table"]["finni_f1"], contexts["text"]["finni_f1"]
            ))
        assert row[3:9] == values
        assert row[9] == f"{score['raw']['overall']['parse_failures']} / {score['dedup']['overall']['parse_failures']}"
        assert row[10] == f"{score['raw']['overall']['cut_off']} / {score['dedup']['overall']['cut_off']}"
        assert _sha(path) in report_text
        prediction = ROOT / score["predictions"]
        assert _sha(prediction) == score["predictions_sha256"]
        assert score["predictions_sha256"] in report_text
        preflight = json.loads((prediction.parent / "pilot_preflight.json").read_text())
        assert preflight["adapter_sha256"] in report_text


def test_svg_plots_every_deduplicated_value_at_its_actual_coordinate():
    rows = _table()
    assert [row[0] for row in rows] == list(LABELS)
    svg = ET.parse(PAPER / "checkpoint-f1.svg").getroot()
    points = [node for node in svg.iter() if node.get("data-series") and node.get("data-stage")]
    assert len(points) == len(rows) * 3
    lines = {node.attrib["data-line"]: node for node in svg.iter() if node.get("data-line")}
    bottoms = {"overall": 290, "table": 480, "text": 670}
    columns = {"overall": 6, "table": 7, "text": 8}
    assert set(lines) == set(bottoms)
    for series, bottom in bottoms.items():
        series_points = [point for point in points if point.get("data-series") == series]
        assert [point.get("data-stage") for point in series_points] == ["first-pass"] * 5 + ["final"]
        vertices = [tuple(map(float, pair.split(","))) for pair in lines[series].attrib["points"].split()]
        assert vertices == [(float(point.attrib["data-cx"]), float(point.attrib["data-cy"]))
                            for point in series_points[:-1]]
        for point, row in zip(series_points, rows, strict=True):
            seen = int(row[1].replace(",", ""))
            score = float(row[columns[series]])
            assert int(point.attrib["data-instances"]) == seen
            assert float(point.attrib["data-f1"]) == score
            assert float(point.attrib["data-cx"]) == pytest.approx(160 + 690 * seen / 40000, abs=0.02)
            assert float(point.attrib["data-cy"]) == pytest.approx(bottom - 600 * (score - 0.70), abs=0.04)
            if point.get("data-stage") == "first-pass":
                assert (point.attrib["cx"], point.attrib["cy"]) == (point.attrib["data-cx"], point.attrib["data-cy"])
            else:
                diamond = [tuple(map(float, pair.split(","))) for pair in point.attrib["points"].split()]
                assert sum(x for x, _ in diamond) / len(diamond) == pytest.approx(float(point.attrib["data-cx"]))
                assert sum(y for _, y in diamond) / len(diamond) == pytest.approx(float(point.attrib["data-cy"]))
    assert rows[-1][1] == rows[-2][1] and rows[-1][2] != rows[-2][2]
