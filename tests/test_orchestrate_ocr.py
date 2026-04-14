"""Tests for orchestrate_ocr.py — pure logic only, no network or subagent calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

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


def _make_test_photo(path: Path, size=(1000, 800), color=(128, 128, 128)) -> None:
    img = Image.new("RGB", size, color)
    # Paint a small distinctive patch where the stamp would be
    for x in range(780, 900):
        for y in range(720, 760):
            img.putpixel((x, y), (255, 128, 0))
    img.save(path, "JPEG", quality=90)


def test_crop_image_to_file_produces_resized_jpeg(tmp_state, tmp_path):
    src = oo.SCANMYPHOTOS_DIR / "d1_1.jpg"
    _make_test_photo(src)
    dst = tmp_path / "d1_1_crop.jpg"
    bbox = {"x": 0.84, "y": 0.9, "w": 0.12, "h": 0.05, "source": "yolo"}

    oo.crop_image_to_file(
        src=src, dst=dst, bbox=bbox,
        pad_factor=oo.STAGE1_PAD_FACTOR, max_side=oo.STAGE1_MAX_SIDE,
    )

    assert dst.exists()
    cropped = Image.open(dst)
    assert max(cropped.size) <= oo.STAGE1_MAX_SIDE
    assert cropped.format == "JPEG"


def test_crop_stage1_writes_shards_and_crops(tmp_state):
    # 3 predictions, one already processed → 2 pending → 1 shard (shard size 50)
    oo.save_json(oo.PREDICTIONS_FILE, {
        "d1_1": {"x": 0.84, "y": 0.9, "w": 0.12, "h": 0.05, "confidence": 0.9},
        "d1_2": {"x": 0.84, "y": 0.9, "w": 0.12, "h": 0.05, "confidence": 0.4},
        "d1_3": {"x": 0.84, "y": 0.9, "w": 0.12, "h": 0.05, "confidence": 0.9},
    })
    oo.save_json(oo.RESULTS_FILE, {"d1_2": {"text": "1 1 '99"}})

    for stem in ("d1_1", "d1_3"):
        _make_test_photo(oo.SCANMYPHOTOS_DIR / f"{stem}.jpg")

    rc = oo.main(["crop-stage1"])
    assert rc == 0

    shards = sorted(oo.STAGE1_SHARDS_DIR.glob("shard_*.json"))
    manifests = [s for s in shards if "_result" not in s.stem]
    assert len(manifests) == 1

    manifest = json.loads(manifests[0].read_text())
    stems_in_shard = [s["stem"] for s in manifest["stems"]]
    assert stems_in_shard == ["d1_1", "d1_3"]
    # crop files exist
    assert (oo.STAGE1_CROPS_DIR / "d1_1.jpg").exists()
    assert (oo.STAGE1_CROPS_DIR / "d1_3.jpg").exists()
    # manifest shape
    assert manifest["shard_id"] == "0000"
    assert manifest["result_path"].endswith("shard_0000_result.json")


def test_crop_stage1_limit_caps_pending(tmp_state):
    oo.save_json(oo.PREDICTIONS_FILE, {
        f"d1_{i}": {"x": 0.5, "y": 0.9, "w": 0.1, "h": 0.05, "confidence": 0.9}
        for i in range(1, 6)
    })
    for i in range(1, 6):
        _make_test_photo(oo.SCANMYPHOTOS_DIR / f"d1_{i}.jpg")

    rc = oo.main(["crop-stage1", "--limit", "3"])
    assert rc == 0

    manifest = json.loads(sorted(oo.STAGE1_SHARDS_DIR.glob("shard_*.json"))[0].read_text())
    assert len(manifest["stems"]) == 3


def test_merge_stage1_adds_entries(tmp_state):
    oo.save_json(oo.RESULTS_FILE, {"d1_0": {"text": "NONE"}})
    shard_result = oo.STAGE1_SHARDS_DIR / "shard_0000_result.json"
    oo.save_json(shard_result, {
        "shard_id": "0000",
        "stage": 1,
        "results": {
            "d1_1": {"text": "10 3 '99", "bbox_source": "yolo", "confidence": 0.9},
            "d1_2": {"text": "NONE", "bbox_source": "yolo", "confidence": 0.2},
        },
    })

    rc = oo.main(["merge-stage1", str(shard_result)])
    assert rc == 0

    merged = oo.load_json(oo.RESULTS_FILE, {})
    assert merged["d1_0"]["text"] == "NONE"  # untouched
    assert merged["d1_1"]["text"] == "10 3 '99"
    assert merged["d1_1"]["stage"] == 1
    assert merged["d1_2"]["text"] == "NONE"


def test_merge_stage1_rejects_bad_shape(tmp_state):
    shard_result = oo.STAGE1_SHARDS_DIR / "shard_0000_result.json"
    oo.save_json(shard_result, {"shard_id": "0000"})  # missing "results"
    rc = oo.main(["merge-stage1", str(shard_result)])
    assert rc == 3


def test_merge_stage1_rejects_missing_text_field(tmp_state):
    shard_result = oo.STAGE1_SHARDS_DIR / "shard_0000_result.json"
    oo.save_json(shard_result, {
        "shard_id": "0000",
        "stage": 1,
        "results": {"d1_1": {"bbox_source": "yolo"}},  # no text
    })
    rc = oo.main(["merge-stage1", str(shard_result)])
    assert rc == 3


def test_merge_stage1_rejects_results_not_dict(tmp_state):
    shard_result = oo.STAGE1_SHARDS_DIR / "shard_0000_result.json"
    oo.save_json(shard_result, {"shard_id": "0000", "results": []})  # list, not dict
    rc = oo.main(["merge-stage1", str(shard_result)])
    assert rc == 3


def test_merge_stage1_rejects_entry_not_dict(tmp_state):
    shard_result = oo.STAGE1_SHARDS_DIR / "shard_0000_result.json"
    oo.save_json(shard_result, {
        "shard_id": "0000",
        "results": {"d1_1": "10 3 '99"},  # string, not dict
    })
    rc = oo.main(["merge-stage1", str(shard_result)])
    assert rc == 3


def test_merge_stage1_rejects_invalid_json(tmp_state):
    shard_result = oo.STAGE1_SHARDS_DIR / "shard_0000_result.json"
    shard_result.parent.mkdir(parents=True, exist_ok=True)
    shard_result.write_text("{not valid json")
    rc = oo.main(["merge-stage1", str(shard_result)])
    assert rc == 3


def test_list_shards_prints_only_pending(tmp_state, capsys):
    oo.STAGE1_SHARDS_DIR.mkdir(parents=True)
    (oo.STAGE1_SHARDS_DIR / "shard_0000.json").write_text("{}")
    (oo.STAGE1_SHARDS_DIR / "shard_0000_result.json").write_text("{}")  # already done
    (oo.STAGE1_SHARDS_DIR / "shard_0001.json").write_text("{}")         # pending

    rc = oo.main(["list-shards", "stage1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "shard_0001.json" in out
    assert "shard_0000.json" not in out


def test_requeue_removes_result_file(tmp_state):
    oo.STAGE1_SHARDS_DIR.mkdir(parents=True)
    (oo.STAGE1_SHARDS_DIR / "shard_0000.json").write_text("{}")
    result = oo.STAGE1_SHARDS_DIR / "shard_0000_result.json"
    result.write_text("{}")

    rc = oo.main(["requeue", str(oo.STAGE1_SHARDS_DIR / "shard_0000.json")])
    assert rc == 0
    assert not result.exists()


@pytest.mark.parametrize("text,conf,expected", [
    ("10 3 '99", 0.9, False),          # clean, high conf
    ("NONE", 0.9, False),              # absent, high conf
    ("1? 3 '99", 0.9, True),           # contains ?
    ("10-3-99", 0.9, True),            # wrong format
    ("", 0.9, True),                   # empty
    ("10 3 '99", 0.2, True),           # low conf
    ("NONE", 0.2, True),               # low conf even if NONE
    ("10 3 '99", None, False),         # missing conf treated as high
])
def test_should_review(text, conf, expected):
    assert oo.should_review(text=text, confidence=conf) is expected


def test_select_review_stems(tmp_state):
    oo.save_json(oo.RESULTS_FILE, {
        "d1_1": {"text": "10 3 '99", "confidence": 0.9},     # no
        "d1_2": {"text": "1? 3 '99", "confidence": 0.9},     # yes (?)
        "d1_3": {"text": "10 3 '99", "confidence": 0.2},     # yes (conf)
        "d1_4": {"text": "banana", "confidence": 0.9},       # yes (format)
        "d1_5": {"text": "NONE", "confidence": 0.9},         # no
    })
    stems = oo.select_review_stems()
    assert sorted(stems) == ["d1_2", "d1_3", "d1_4"]


def test_crop_stage2_writes_two_views_per_stem(tmp_state):
    oo.save_json(oo.PREDICTIONS_FILE, {
        "d1_1": {"x": 0.84, "y": 0.9, "w": 0.12, "h": 0.05, "confidence": 0.9},
        "d1_2": {"x": 0.84, "y": 0.9, "w": 0.12, "h": 0.05, "confidence": 0.2},  # low conf → triggers
        "d1_3": {"x": 0.84, "y": 0.9, "w": 0.12, "h": 0.05, "confidence": 0.9},
    })
    oo.save_json(oo.RESULTS_FILE, {
        "d1_1": {"text": "10 3 '99", "confidence": 0.9},    # clean
        "d1_2": {"text": "1? 3 '99", "confidence": 0.2},    # triggers on ? and conf
        "d1_3": {"text": "wrong", "confidence": 0.9},       # triggers on format
    })
    for stem in ("d1_2", "d1_3"):
        _make_test_photo(oo.SCANMYPHOTOS_DIR / f"{stem}.jpg")

    rc = oo.main(["crop-stage2"])
    assert rc == 0

    # two view crops per triggered stem
    assert (oo.STAGE2_CROP_DIR / "d1_2.jpg").exists()
    assert (oo.STAGE2_FULL_DIR / "d1_2.jpg").exists()
    assert (oo.STAGE2_CROP_DIR / "d1_3.jpg").exists()
    assert (oo.STAGE2_FULL_DIR / "d1_3.jpg").exists()
    # d1_1 was clean; no stage-2 crops
    assert not (oo.STAGE2_CROP_DIR / "d1_1.jpg").exists()

    shards = [s for s in oo.STAGE2_SHARDS_DIR.glob("shard_*.json") if "_result" not in s.stem]
    assert len(shards) == 1
    manifest = json.loads(shards[0].read_text())
    stems_in_shard = sorted(s["stem"] for s in manifest["stems"])
    assert stems_in_shard == ["d1_2", "d1_3"]
    assert manifest["stems"][0]["stage1_text"] in ("1? 3 '99", "wrong")
    assert "full_path" in manifest["stems"][0]


@pytest.mark.parametrize("crop,full,expected_status,expected_text", [
    ("10 3 '99", "10 3 '99", "confirmed", "10 3 '99"),     # agree
    ("10 3 '99", "10  3 '99", "confirmed", "10 3 '99"),    # whitespace-normalized agree
    ("NONE", "NONE", "no_stamp", "NONE"),                  # both absent
    ("10 3 '99", "10 4 '99", "disagree", None),            # different dates
    ("10 3 '99", "NONE", "disagree", None),                # one sees stamp, one doesn't
])
def test_reconcile_pair(crop, full, expected_status, expected_text):
    status, text = oo.reconcile_pair(crop, full)
    assert status == expected_status
    if expected_text is not None:
        assert text == expected_text


def test_normalize_text_collapses_whitespace():
    assert oo.normalize_text("  10   3 '99 ") == "10 3 '99"
    assert oo.normalize_text("NONE") == "NONE"
    assert oo.normalize_text("") == ""


def test_merge_stage2_confirmed_overwrites_results(tmp_state):
    oo.save_json(oo.RESULTS_FILE, {
        "d1_1": {"text": "1? 3 '99", "stage": 1, "confidence": 0.9},
    })
    shard_result = oo.STAGE2_SHARDS_DIR / "shard_0000_result.json"
    oo.save_json(shard_result, {
        "shard_id": "0000",
        "stage": 2,
        "results": {
            "d1_1": {"view_crop": "10 3 '99", "view_full": "10 3 '99"},
        },
    })

    rc = oo.main(["merge-stage2", str(shard_result)])
    assert rc == 0

    merged = oo.load_json(oo.RESULTS_FILE, {})
    assert merged["d1_1"]["text"] == "10 3 '99"
    assert merged["d1_1"]["review_status"] == "confirmed"
    assert merged["d1_1"]["stage"] == 2


def test_merge_stage2_disagreement_preserves_stage1_and_queues(tmp_state):
    oo.save_json(oo.RESULTS_FILE, {
        "d1_1": {"text": "1? 3 '99", "stage": 1, "confidence": 0.4},
    })
    shard_result = oo.STAGE2_SHARDS_DIR / "shard_0000_result.json"
    oo.save_json(shard_result, {
        "shard_id": "0000",
        "stage": 2,
        "results": {
            "d1_1": {"view_crop": "10 3 '99", "view_full": "11 3 '99"},
        },
    })

    rc = oo.main(["merge-stage2", str(shard_result)])
    assert rc == 0

    merged = oo.load_json(oo.RESULTS_FILE, {})
    # stage-1 entry unchanged on disagreement
    assert merged["d1_1"]["text"] == "1? 3 '99"
    assert merged["d1_1"].get("review_status") == "disagreement"

    queue = oo.load_json(oo.MANUAL_QUEUE_FILE, [])
    assert len(queue) == 1
    assert queue[0]["stem"] == "d1_1"
    assert queue[0]["view_crop"] == "10 3 '99"
    assert queue[0]["view_full"] == "11 3 '99"
    assert queue[0]["stage1_text"] == "1? 3 '99"


def test_merge_stage2_no_stamp_overwrites(tmp_state):
    oo.save_json(oo.RESULTS_FILE, {
        "d1_1": {"text": "wrong", "stage": 1, "confidence": 0.9},
    })
    shard_result = oo.STAGE2_SHARDS_DIR / "shard_0000_result.json"
    oo.save_json(shard_result, {
        "shard_id": "0000",
        "stage": 2,
        "results": {"d1_1": {"view_crop": "NONE", "view_full": "NONE"}},
    })

    rc = oo.main(["merge-stage2", str(shard_result)])
    assert rc == 0
    merged = oo.load_json(oo.RESULTS_FILE, {})
    assert merged["d1_1"]["text"] == "NONE"
    assert merged["d1_1"]["review_status"] == "no_stamp"
