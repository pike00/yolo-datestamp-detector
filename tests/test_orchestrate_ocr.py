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
    assert "Stage-2 shards:      0 pending, 0 done" in out
    assert "Failed shards:       0" in out


def test_pending_set_excludes_already_processed(tmp_state):
    oo.save_json(oo.PREDICTIONS_FILE, {
        "d1_1": {"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.05, "confidence": 0.8},
        "d1_2": {"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.05, "confidence": 0.8},
        "d1_3": {"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.05, "confidence": 0.8},
    })
    oo.save_json(oo.RESULTS_FILE, {"d1_2": {"text": "1 1 '99"}})
    pending = oo.compute_pending_stems()
    assert pending == ["d1_1", "d1_3"]


def test_pending_set_sorted(tmp_state):
    oo.save_json(oo.PREDICTIONS_FILE, {
        "d2_1": {"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.05, "confidence": 0.8},
        "d1_1": {"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.05, "confidence": 0.8},
    })
    pending = oo.compute_pending_stems()
    assert pending == ["d1_1", "d2_1"]


def test_load_bbox_prefers_human_correction(tmp_state):
    oo.save_json(oo.PREDICTIONS_FILE, {
        "d1_1": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.05, "confidence": 0.3},
    })
    oo.save_json(oo.CORRECTIONS_FILE, {
        "files": [
            {"stem": "d1_1", "user_correction": {
                "x": 0.9, "y": 0.9, "w": 0.2, "h": 0.1, "action": "confirmed"
            }}
        ]
    })
    bbox = oo.load_bbox_map()["d1_1"]
    assert bbox["x"] == 0.9
    assert bbox["source"] == "human"


def test_crop_box_math():
    # 1000x800 image, bbox center (0.8, 0.9), size (0.2, 0.1)
    # → bbox = 200x80, center (800, 720)
    # → bbox corners: (700, 680) to (900, 760)
    # → pad_factor 0.5 → +100 horizontal, +40 vertical → (600, 640) to (1000, 800)
    box = oo.compute_crop_box(img_w=1000, img_h=800, bbox={
        "x": 0.8, "y": 0.9, "w": 0.2, "h": 0.1
    }, pad_factor=0.5)
    assert box == (600, 640, 1000, 800)


def test_crop_box_clamps_to_image():
    box = oo.compute_crop_box(img_w=100, img_h=100, bbox={
        "x": 0.95, "y": 0.95, "w": 0.2, "h": 0.2
    }, pad_factor=0.5)
    assert box == (75, 75, 100, 100)
