import csv
import json

import pytest

from scripts import label_audit


def test_entity_type_and_allocation():
    assert label_audit.entity_type('{"result": []}') == "none"
    assert label_audit.entity_type('{"result": [{"Fact":"1","Type":"percentItemType"},{"Fact":"2","Type":"percentItemType"},{"Fact":"3","Type":"monetaryItemType"}]}') == "percentItemType"
    q = label_audit.allocate({"a": list(range(900)), "b": list(range(99)), "c": [1]}, 300)
    assert sum(q.values()) == 300 and q["c"] == 1 and q["a"] > q["b"]


def _sheet(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "stratum", "accession", "cik", "filing_url", "context", "gold_pairs"] + label_audit.OWNER_COLS)
        for i, stratum, verdict in rows:
            w.writerow([i, stratum, "a", "c", "u", "ctx", json.dumps([["1", "monetaryItemType"]]), verdict, "0", "1" if verdict == "incorrect" else "0", "0", ""])


def test_score_agreement_weighting_and_refusals(tmp_path, monkeypatch):
    monkeypatch.setattr(label_audit, "AUDIT", tmp_path)
    (tmp_path / "sample.json").write_text(json.dumps({"population": {"s1": 300, "s2": 100}, "first_pass": [1, 2, 3, 4], "second_pass": [1, 2]}))
    _sheet(tmp_path / "pass1_sheet.csv", [(1, "s1", "correct"), (2, "s1", "incorrect"), (3, "s2", "correct"), (4, "s2", "correct")])
    _sheet(tmp_path / "pass2_sheet.csv", [(1, "s1", "correct"), (2, "s1", "correct")])
    label_audit.score()
    r = json.loads((tmp_path / "audit_result.json").read_text())
    assert r["self_agreement"] == 0.5 and not r["self_agreement_met"]
    assert r["instance_error_rate_raw"] == 0.25 and r["label_audit_error_rate"] == pytest.approx(0.75 * 0.5)
    assert r["facts"]["spurious"] == 1
    _sheet(tmp_path / "pass2_sheet.csv", [(1, "s1", "correct"), (2, "s1", "")])
    with pytest.raises(SystemExit, match="every row must be"):
        label_audit.score()
    _sheet(tmp_path / "pass2_sheet.csv", [(1, "s1", "correct"), (3, "s2", "correct")])
    with pytest.raises(SystemExit, match="do not match the drawn ids"):
        label_audit.score()


def test_draw_refuses_to_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(label_audit, "AUDIT", tmp_path)
    (tmp_path / "sample.json").write_text("{}")
    with pytest.raises(SystemExit, match="drawn once"):
        label_audit.draw()


def test_contamination_statement_leaves_unmeasured_blank():
    from scripts.contamination_statement import build
    text = build()
    assert "__ (not yet measured: student fresh-slice run)" in text and "__ (not yet measured: label audit, two passes)" in text
    assert "cap 40,000 not reached after decontamination, O-016" in text and "fill=relaxed" in text
