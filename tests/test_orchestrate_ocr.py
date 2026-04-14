"""Tests for orchestrate_ocr.py — pure logic only, no network or subagent calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make the script importable as a module
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import orchestrate_ocr as oo  # noqa: E402


@pytest.fixture
def tmp_state(tmp_path, monkeypatch):
    """Redirect all BASE_DIR-derived paths to a temp tree."""
    state = tmp_path / "state"
    output = tmp_path / "output"
    state.mkdir()
    output.mkdir()

    monkeypatch.setattr(oo, "BASE_DIR", tmp_path)
    monkeypatch.setattr(oo, "STATE_DIR", state)
    monkeypatch.setattr(oo, "OUTPUT_DIR", output)
    monkeypatch.setattr(oo, "PREDICTIONS_FILE", state / "scanmyphotos_predictions.json")
    monkeypatch.setattr(oo, "CORRECTIONS_FILE", state / "corrections_queue.json")
    monkeypatch.setattr(oo, "RESULTS_FILE", state / "ocr_results.json")
    monkeypatch.setattr(oo, "MANUAL_QUEUE_FILE", state / "ocr_manual_queue.json")
    monkeypatch.setattr(oo, "FAILED_SHARDS_FILE", state / "failed_shards.json")
    monkeypatch.setattr(oo, "SHARDS_DIR", state / "shards")
    monkeypatch.setattr(oo, "STAGE1_SHARDS_DIR", state / "shards" / "stage1")
    monkeypatch.setattr(oo, "STAGE2_SHARDS_DIR", state / "shards" / "stage2")
    monkeypatch.setattr(oo, "STAGE1_CROPS_DIR", output / "ocr_crops_stage1")
    monkeypatch.setattr(oo, "STAGE2_CROP_DIR", output / "ocr_crops_stage2_crop")
    monkeypatch.setattr(oo, "STAGE2_FULL_DIR", output / "ocr_crops_stage2_full")
    monkeypatch.setattr(oo, "SCANMYPHOTOS_DIR", tmp_path / "scanmyphotos")
    (tmp_path / "scanmyphotos").mkdir()
    return tmp_path


def test_status_empty(tmp_state, capsys):
    rc = oo.main(["status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "YOLO predictions:    0" in out
    assert "OCR results:         0" in out
    assert "Stage-1 shards:      0 pending, 0 done" in out
    assert "Manual review queue: 0" in out
