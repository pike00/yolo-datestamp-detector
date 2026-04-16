# Parallel Haiku OCR with Low-Confidence Review — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a parallel OCR pipeline that dispatches Claude Code Haiku subagents against pre-cropped scanned photos, re-reviews low-confidence results with two independent views, and produces a reconciled `ocr_results.json` plus a manual-review queue for disagreements — all running on the user's Claude Code subscription, no API key.

**Architecture:** Main Opus orchestrator does deterministic pre-cropping and sharding via a new Python CLI (`orchestrate_ocr.py`), then dispatches up to 5 background Haiku subagents in parallel using the Claude Code Task tool. Each subagent reads a shard manifest, reads the cropped JPEGs with the Read tool, transcribes date stamps into a shard result JSON, and the orchestrator merges shard results as they complete. Stage 2 repeats the dispatch pattern on flagged stems with a larger crop view and a full-image view for agreement reconciliation.

**Tech Stack:** Python 3.14 via uv (PEP 723 inline script headers), Pillow for image ops, pytest for unit tests, Claude Code Task tool for subagent dispatch, Haiku 4.5 (`claude-haiku-4-5`) for subagent inference.

**Spec:** [docs/superpowers/specs/2026-04-14-parallel-haiku-ocr-review-design.md](../specs/2026-04-14-parallel-haiku-ocr-review-design.md)

**Working directory for all commands:** `/home/will/photo_project/yolo_finetune/`

---

## File Structure

**New files:**
- `scripts/orchestrate_ocr.py` — CLI with subcommands (crop-stage1, merge-stage1, crop-stage2, merge-stage2, list-shards, requeue, status)
- `scripts/subagent_prompts/stage1_prompt.md` — prompt template for stage-1 Haiku subagents
- `scripts/subagent_prompts/stage2_prompt.md` — prompt template for stage-2 review subagents
- `tests/test_orchestrate_ocr.py` — pytest unit tests for pure logic (cropping math, triggers, reconciliation)
- `tests/__init__.py` — empty

**Modified files:**
- `.gitignore` — add `state/shards/`, `state/ocr_manual_queue.json`, `state/failed_shards.json`, `output/ocr_crops_stage1/`, `output/ocr_crops_stage2_crop/`, `output/ocr_crops_stage2_full/` (some already covered by existing `output/` rule, added explicitly for clarity)

**Runtime-created files (gitignored):**
- `output/ocr_crops_stage1/<stem>.jpg` — stage-1 crops
- `output/ocr_crops_stage2_crop/<stem>.jpg` — stage-2 large crops
- `output/ocr_crops_stage2_full/<stem>.jpg` — stage-2 full images
- `state/shards/stage1/shard_NNNN.json` — stage-1 shard manifests
- `state/shards/stage1/shard_NNNN_result.json` — stage-1 shard results (written by subagents)
- `state/shards/stage2/shard_NNNN.json` + `…_result.json` — stage-2 analogs
- `state/ocr_manual_queue.json` — disagreements for human review
- `state/failed_shards.json` — shards that failed twice

**Reused unchanged:**
- `state/scanmyphotos_predictions.json` — YOLO bboxes (input)
- `state/corrections_queue.json` — human-confirmed bboxes (input, overrides YOLO)
- `state/ocr_results.json` — final merged output (appended to, not overwritten; existing entries skip stage-1)
- `scripts/ocr_stamps.py` — untouched; kept for ad-hoc single-image runs

---

## Task 1: Skeleton script, gitignore, status subcommand

**Files:**
- Create: `scripts/orchestrate_ocr.py`
- Modify: `.gitignore`
- Create: `tests/__init__.py`
- Create: `tests/test_orchestrate_ocr.py`

- [ ] **Step 1: Update .gitignore**

Append to `yolo_finetune/.gitignore`:

```gitignore

# Parallel OCR orchestrator
state/shards/
state/ocr_manual_queue.json
state/failed_shards.json
output/ocr_crops_stage1/
output/ocr_crops_stage2_crop/
output/ocr_crops_stage2_full/
```

- [ ] **Step 2: Create empty tests/__init__.py**

Create `tests/__init__.py` with empty content.

- [ ] **Step 3: Create orchestrate_ocr.py skeleton**

Create `scripts/orchestrate_ocr.py`:

```python
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "pillow>=12.0",
# ]
# ///
"""Parallel Haiku OCR orchestrator for ScanMyPhotos date stamps.

This script is the deterministic IO half of the pipeline. It pre-crops
images, writes shard manifests, validates and merges shard results from
Haiku subagents, and reconciles stage-2 review outputs. It does NOT call
any LLM — subagent dispatch is driven by a Claude Code orchestrator
session using the Task tool.

Subcommands:
    crop-stage1 [--limit N]  Pre-crop pending stems and write stage-1 shards
    merge-stage1 <result>    Merge a stage-1 shard result into ocr_results.json
    crop-stage2              Compute triggered stems and write stage-2 shards
    merge-stage2 <result>    Reconcile a stage-2 shard result
    list-shards <stage>      List shard manifests that have no result yet
    requeue <shard>          Mark a shard as pending again
    status                   Print progress summary
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
SCANMYPHOTOS_DIR = BASE_DIR / "scanmyphotos"
STATE_DIR = BASE_DIR / "state"
OUTPUT_DIR = BASE_DIR / "output"

PREDICTIONS_FILE = STATE_DIR / "scanmyphotos_predictions.json"
CORRECTIONS_FILE = STATE_DIR / "corrections_queue.json"
RESULTS_FILE = STATE_DIR / "ocr_results.json"
MANUAL_QUEUE_FILE = STATE_DIR / "ocr_manual_queue.json"
FAILED_SHARDS_FILE = STATE_DIR / "failed_shards.json"

SHARDS_DIR = STATE_DIR / "shards"
STAGE1_SHARDS_DIR = SHARDS_DIR / "stage1"
STAGE2_SHARDS_DIR = SHARDS_DIR / "stage2"

STAGE1_CROPS_DIR = OUTPUT_DIR / "ocr_crops_stage1"
STAGE2_CROP_DIR = OUTPUT_DIR / "ocr_crops_stage2_crop"
STAGE2_FULL_DIR = OUTPUT_DIR / "ocr_crops_stage2_full"

# Cropping constants (stage 1 mirrors ocr_stamps.py exactly)
STAGE1_PAD_FACTOR = 0.5
STAGE1_MAX_SIDE = 512
STAGE2_PAD_FACTOR = 1.5
STAGE2_MAX_SIDE = 1536

# Stage-1 shard sizing
STAGE1_SHARD_SIZE = 50
# Stage-2 shard sizing (each stem costs two subagent image reads)
STAGE2_SHARD_SIZE = 25

# Stage-2 trigger rules
DATE_FORMAT_RE = re.compile(r"^\d{1,2} \d{1,2} '\d{2}$")
LOW_CONFIDENCE_THRESHOLD = 0.3


def load_json(path: Path, default):
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def cmd_status(_args) -> int:
    results = load_json(RESULTS_FILE, {})
    manual_queue = load_json(MANUAL_QUEUE_FILE, [])
    failed = load_json(FAILED_SHARDS_FILE, [])
    predictions = load_json(PREDICTIONS_FILE, {})

    stage1_pending = sum(1 for d in STAGE1_SHARDS_DIR.glob("shard_*.json")
                         if "_result" not in d.stem and not _result_path(d).exists()) if STAGE1_SHARDS_DIR.exists() else 0
    stage1_done = sum(1 for d in STAGE1_SHARDS_DIR.glob("shard_*_result.json")) if STAGE1_SHARDS_DIR.exists() else 0
    stage2_pending = sum(1 for d in STAGE2_SHARDS_DIR.glob("shard_*.json")
                         if "_result" not in d.stem and not _result_path(d).exists()) if STAGE2_SHARDS_DIR.exists() else 0
    stage2_done = sum(1 for d in STAGE2_SHARDS_DIR.glob("shard_*_result.json")) if STAGE2_SHARDS_DIR.exists() else 0

    print(f"YOLO predictions:    {len(predictions)}")
    print(f"OCR results:         {len(results)}")
    print(f"Stage-1 shards:      {stage1_pending} pending, {stage1_done} done")
    print(f"Stage-2 shards:      {stage2_pending} pending, {stage2_done} done")
    print(f"Manual review queue: {len(manual_queue)}")
    print(f"Failed shards:       {len(failed)}")
    return 0


def _result_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(manifest_path.stem + "_result.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="Print progress summary")

    args = parser.parse_args(argv)
    if args.cmd == "status":
        return cmd_status(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Write failing test for status command**

Create `tests/test_orchestrate_ocr.py`:

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/will/photo_project/yolo_finetune && uv run --with pytest --with pillow pytest tests/test_orchestrate_ocr.py -v`

Expected: `test_status_empty PASSED`

- [ ] **Step 6: Smoke-test the real status command**

Run: `cd /home/will/photo_project/yolo_finetune && uv run scripts/orchestrate_ocr.py status`

Expected: prints real counts from existing `state/scanmyphotos_predictions.json` (6458) and `state/ocr_results.json` (whatever exists).

- [ ] **Step 7: Commit**

```bash
cd /home/will/photo_project/yolo_finetune && git add .gitignore scripts/orchestrate_ocr.py tests/__init__.py tests/test_orchestrate_ocr.py && git commit -m "feat(ocr): scaffold parallel orchestrator with status command"
```

---

## Task 2: Pending set and crop math (pure logic, TDD)

**Files:**
- Modify: `scripts/orchestrate_ocr.py`
- Modify: `tests/test_orchestrate_ocr.py`

- [ ] **Step 1: Write failing tests for pending set**

Append to `tests/test_orchestrate_ocr.py`:

```python
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
    x1, y1, x2, y2 = box
    assert x1 >= 0 and y1 >= 0
    assert x2 <= 100 and y2 <= 100
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/will/photo_project/yolo_finetune && uv run --with pytest --with pillow pytest tests/test_orchestrate_ocr.py -v`

Expected: four new tests FAIL with `AttributeError: module 'orchestrate_ocr' has no attribute 'compute_pending_stems'` (and similar).

- [ ] **Step 3: Implement pending set and bbox logic**

Insert into `scripts/orchestrate_ocr.py` above `cmd_status`:

```python
def load_predictions() -> dict[str, dict]:
    """YOLO predictions, tagged with source."""
    raw = load_json(PREDICTIONS_FILE, {})
    return {k: {**v, "source": "yolo"} for k, v in raw.items()}


def load_corrections() -> dict[str, dict]:
    """Human-confirmed bboxes from the corrections queue."""
    data = load_json(CORRECTIONS_FILE, {})
    boxes: dict[str, dict] = {}
    for entry in data.get("files", []):
        corr = entry.get("user_correction")
        if not corr or corr.get("x") is None:
            continue
        if corr.get("action") in ("confirmed", "corrected"):
            boxes[entry["stem"]] = {
                "x": corr["x"], "y": corr["y"],
                "w": corr["w"], "h": corr["h"],
                "source": "human",
            }
    return boxes


def load_bbox_map() -> dict[str, dict]:
    """Unified bbox map. Human corrections override YOLO for the same stem."""
    yolo = load_predictions()
    human = load_corrections()
    return {**yolo, **human}


def compute_pending_stems() -> list[str]:
    """Stems that have a YOLO prediction but no entry in ocr_results.json."""
    predictions = load_json(PREDICTIONS_FILE, {})
    results = load_json(RESULTS_FILE, {})
    pending = [s for s in predictions if s not in results]
    return sorted(pending)


def compute_crop_box(img_w: int, img_h: int, bbox: dict, pad_factor: float) -> tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) in pixel coords, clamped to the image."""
    cx = bbox["x"] * img_w
    cy = bbox["y"] * img_h
    bw = bbox["w"] * img_w
    bh = bbox["h"] * img_h

    pad_x = bw * pad_factor
    pad_y = bh * pad_factor

    x1 = max(0, int(cx - bw / 2 - pad_x))
    y1 = max(0, int(cy - bh / 2 - pad_y))
    x2 = min(img_w, int(cx + bw / 2 + pad_x))
    y2 = min(img_h, int(cy + bh / 2 + pad_y))
    return (x1, y1, x2, y2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/will/photo_project/yolo_finetune && uv run --with pytest --with pillow pytest tests/test_orchestrate_ocr.py -v`

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/will/photo_project/yolo_finetune && git add scripts/orchestrate_ocr.py tests/test_orchestrate_ocr.py && git commit -m "feat(ocr): pending set and crop math with tests"
```

---

## Task 3: crop-stage1 subcommand (image IO)

**Files:**
- Modify: `scripts/orchestrate_ocr.py`
- Modify: `tests/test_orchestrate_ocr.py`

- [ ] **Step 1: Write failing test for crop_image_to_file**

Append to `tests/test_orchestrate_ocr.py`:

```python
from PIL import Image


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/will/photo_project/yolo_finetune && uv run --with pytest --with pillow pytest tests/test_orchestrate_ocr.py::test_crop_image_to_file_produces_resized_jpeg -v`

Expected: FAIL with `AttributeError: module 'orchestrate_ocr' has no attribute 'crop_image_to_file'`.

- [ ] **Step 3: Implement crop_image_to_file**

Insert into `scripts/orchestrate_ocr.py` below `compute_crop_box`:

```python
def crop_image_to_file(
    src: Path, dst: Path, bbox: dict, pad_factor: float, max_side: int,
) -> None:
    """Crop one photo to its padded bbox and save as JPEG."""
    from PIL import Image

    img = Image.open(src).convert("RGB")
    box = compute_crop_box(img.width, img.height, bbox, pad_factor)
    crop = img.crop(box)
    if max(crop.size) > max_side:
        crop.thumbnail((max_side, max_side), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    crop.save(dst, "JPEG", quality=90)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/will/photo_project/yolo_finetune && uv run --with pytest --with pillow pytest tests/test_orchestrate_ocr.py::test_crop_image_to_file_produces_resized_jpeg -v`

Expected: PASS.

- [ ] **Step 5: Write failing test for cmd_crop_stage1**

Append to `tests/test_orchestrate_ocr.py`:

```python
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
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `cd /home/will/photo_project/yolo_finetune && uv run --with pytest --with pillow pytest tests/test_orchestrate_ocr.py -v -k crop_stage1`

Expected: both tests FAIL because `crop-stage1` subcommand does not exist yet.

- [ ] **Step 7: Implement cmd_crop_stage1 and register subcommand**

Insert into `scripts/orchestrate_ocr.py` below `crop_image_to_file`:

```python
def cmd_crop_stage1(args) -> int:
    pending = compute_pending_stems()
    if args.limit is not None:
        pending = pending[: args.limit]

    if not pending:
        print("No pending stems.")
        return 0

    bbox_map = load_bbox_map()
    STAGE1_SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    STAGE1_CROPS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Pre-cropping {len(pending)} stems...")
    shard_index = 0
    shard_entries: list[dict] = []

    def flush_shard() -> None:
        nonlocal shard_entries, shard_index
        if not shard_entries:
            return
        shard_id = f"{shard_index:04d}"
        manifest_path = STAGE1_SHARDS_DIR / f"shard_{shard_id}.json"
        result_path = STAGE1_SHARDS_DIR / f"shard_{shard_id}_result.json"
        # Paths inside the manifest are relative to BASE_DIR for portability
        manifest = {
            "shard_id": shard_id,
            "stage": 1,
            "result_path": str(result_path.relative_to(BASE_DIR)),
            "stems": shard_entries,
        }
        save_json(manifest_path, manifest)
        shard_index += 1
        shard_entries = []

    for stem in pending:
        bbox = bbox_map.get(stem)
        if not bbox:
            print(f"  SKIP {stem}: no bbox")
            continue
        src = SCANMYPHOTOS_DIR / f"{stem}.jpg"
        if not src.exists():
            print(f"  SKIP {stem}: source image missing")
            continue
        dst = STAGE1_CROPS_DIR / f"{stem}.jpg"
        try:
            crop_image_to_file(
                src=src, dst=dst, bbox=bbox,
                pad_factor=STAGE1_PAD_FACTOR, max_side=STAGE1_MAX_SIDE,
            )
        except Exception as e:
            print(f"  SKIP {stem}: crop failed ({e})")
            continue

        shard_entries.append({
            "stem": stem,
            "crop_path": str(dst.relative_to(BASE_DIR)),
            "bbox_source": bbox.get("source", "yolo"),
            "confidence": bbox.get("confidence"),
        })
        if len(shard_entries) >= STAGE1_SHARD_SIZE:
            flush_shard()

    flush_shard()
    print(f"Wrote {shard_index} shard manifest(s) to {STAGE1_SHARDS_DIR.relative_to(BASE_DIR)}/")
    return 0
```

Then update `main()` to register the subcommand:

```python
    p_crop1 = sub.add_parser("crop-stage1", help="Pre-crop pending stems")
    p_crop1.add_argument("--limit", type=int, help="Cap pending set size (pilot runs)")
```

and in the dispatch:

```python
    if args.cmd == "crop-stage1":
        return cmd_crop_stage1(args)
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd /home/will/photo_project/yolo_finetune && uv run --with pytest --with pillow pytest tests/test_orchestrate_ocr.py -v`

Expected: all tests PASS.

- [ ] **Step 9: Commit**

```bash
cd /home/will/photo_project/yolo_finetune && git add scripts/orchestrate_ocr.py tests/test_orchestrate_ocr.py && git commit -m "feat(ocr): crop-stage1 subcommand pre-crops pending stems"
```

---

## Task 4: Subagent prompts

**Files:**
- Create: `scripts/subagent_prompts/stage1_prompt.md`
- Create: `scripts/subagent_prompts/stage2_prompt.md`

- [ ] **Step 1: Create stage1_prompt.md**

Create `scripts/subagent_prompts/stage1_prompt.md`:

```markdown
# Stage-1 Date Stamp OCR Worker

You are a worker in a parallel OCR pipeline. You will receive one **shard manifest path** from the dispatcher. Your only job is to transcribe date stamps from the pre-cropped images listed in that manifest, then write the results to a single JSON file.

## Inputs you will be given

- `SHARD_MANIFEST_PATH`: absolute path to a JSON file of the form

```json
{
  "shard_id": "0042",
  "stage": 1,
  "result_path": "state/shards/stage1/shard_0042_result.json",
  "stems": [
    {"stem": "d1_00000133", "crop_path": "output/ocr_crops_stage1/d1_00000133.jpg", "bbox_source": "yolo", "confidence": 0.87}
  ]
}
```

- `BASE_DIR`: absolute path to prepend to the relative `crop_path` and `result_path` values.

## Procedure

1. Read the manifest JSON.
2. For each entry in `stems`:
   a. Use the Read tool to read the image at `{BASE_DIR}/{crop_path}`.
   b. Look at the image and transcribe the date stamp using the RULES below. Do not write anything else to the conversation — just hold the transcription in working memory.
3. After processing every stem, use the Write tool to write a single JSON file to `{BASE_DIR}/{result_path}` with this exact shape:

```json
{
  "shard_id": "0042",
  "stage": 1,
  "results": {
    "d1_00000133": {"text": "10 3 '99", "bbox_source": "yolo", "confidence": 0.87}
  }
}
```

Every stem from the manifest MUST appear as a key in `results`. Preserve `bbox_source` and `confidence` from the manifest.

## Transcription rules

Look at each image. It may contain a date stamp — small digits in orange, red, amber, or yellow, typically imprinted by a camera in the corner of a photo.

- If you see a date stamp, the `text` field is ONLY the exact characters visible. No reformatting, no guessing missing digits, no converting to a standard date format.
- Preserve the original spacing, punctuation, and apostrophes exactly as they appear. For example: `10 3 '99`, not `10/3/1999`.
- If digits are partially obscured or unclear, use `?` for each uncertain character. Example: `1? 3 '99`.
- If there is no date stamp visible, the `text` field is exactly `NONE`.

## Hard rules

- Do NOT modify any file other than the shard result path.
- Do NOT call the Bash tool.
- Do NOT skip stems. Every stem in the manifest must appear in `results`.
- Do NOT write explanations, logs, or progress prints — your only output is the result JSON file.
- When you are done, your final reply to the dispatcher should be one line: `DONE shard_id=<id> stems=<count>`.
```

- [ ] **Step 2: Create stage2_prompt.md**

Create `scripts/subagent_prompts/stage2_prompt.md`:

```markdown
# Stage-2 Date Stamp Review Worker

You are a review worker in a parallel OCR pipeline. You will receive one **shard manifest path** pointing at stems that failed the stage-1 confidence filter. For each stem you read TWO images (a larger crop of the stamp region and a full-image view) and transcribe each one independently. The dispatcher reconciles them later.

## Inputs you will be given

- `SHARD_MANIFEST_PATH`: absolute path to a JSON file of the form

```json
{
  "shard_id": "0017",
  "stage": 2,
  "result_path": "state/shards/stage2/shard_0017_result.json",
  "stems": [
    {
      "stem": "d1_00000133",
      "crop_path": "output/ocr_crops_stage2_crop/d1_00000133.jpg",
      "full_path": "output/ocr_crops_stage2_full/d1_00000133.jpg",
      "stage1_text": "1? 3 '99",
      "confidence": 0.22
    }
  ]
}
```

- `BASE_DIR`: absolute path to prepend to relative paths.

## Procedure

1. Read the manifest JSON.
2. For each stem:
   a. Read `{BASE_DIR}/{crop_path}` and transcribe it — call that `view_crop`.
   b. Read `{BASE_DIR}/{full_path}` and transcribe it independently — call that `view_full`.
   c. Do NOT let one influence the other. Treat each read as a fresh transcription.
3. Write the result JSON to `{BASE_DIR}/{result_path}`:

```json
{
  "shard_id": "0017",
  "stage": 2,
  "results": {
    "d1_00000133": {"view_crop": "10 3 '99", "view_full": "10 3 '99"}
  }
}
```

Every stem from the manifest MUST appear as a key in `results`.

## Transcription rules (identical to stage 1)

- If you see a date stamp, the text is ONLY the exact characters visible. No reformatting, no guessing missing digits, no converting to a standard date format.
- Preserve the original spacing, punctuation, and apostrophes exactly as they appear (e.g., `10 3 '99`, not `10/3/1999`).
- Uncertain characters become `?` (e.g., `1? 3 '99`).
- No stamp visible → the text is exactly `NONE`.

## Hard rules

- Do NOT modify any file other than the shard result path.
- Do NOT call the Bash tool.
- Do NOT skip stems.
- Do NOT reconcile or combine the two views — report both verbatim.
- Final reply to dispatcher: `DONE shard_id=<id> stems=<count>`.
```

- [ ] **Step 3: Commit**

```bash
cd /home/will/photo_project/yolo_finetune && git add scripts/subagent_prompts/ && git commit -m "feat(ocr): stage-1 and stage-2 subagent prompts"
```

---

## Task 5: merge-stage1 subcommand (validated merge)

**Files:**
- Modify: `scripts/orchestrate_ocr.py`
- Modify: `tests/test_orchestrate_ocr.py`

- [ ] **Step 1: Write failing tests for merge_stage1**

Append to `tests/test_orchestrate_ocr.py`:

```python
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
    assert rc != 0


def test_merge_stage1_rejects_missing_text_field(tmp_state):
    shard_result = oo.STAGE1_SHARDS_DIR / "shard_0000_result.json"
    oo.save_json(shard_result, {
        "shard_id": "0000",
        "stage": 1,
        "results": {"d1_1": {"bbox_source": "yolo"}},  # no text
    })
    rc = oo.main(["merge-stage1", str(shard_result)])
    assert rc != 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/will/photo_project/yolo_finetune && uv run --with pytest --with pillow pytest tests/test_orchestrate_ocr.py -v -k merge_stage1`

Expected: all three FAIL (subcommand not registered).

- [ ] **Step 3: Implement merge_stage1**

Insert into `scripts/orchestrate_ocr.py` below `cmd_crop_stage1`:

```python
def _validate_stage1_shard_result(data: dict) -> tuple[bool, str]:
    if not isinstance(data, dict):
        return False, "shard result is not an object"
    if "results" not in data or not isinstance(data["results"], dict):
        return False, "missing or non-object 'results' field"
    for stem, entry in data["results"].items():
        if not isinstance(entry, dict):
            return False, f"entry for {stem} is not an object"
        if "text" not in entry or not isinstance(entry["text"], str):
            return False, f"entry for {stem} missing string 'text'"
    return True, ""


def cmd_merge_stage1(args) -> int:
    shard_path = Path(args.shard_result).resolve()
    if not shard_path.exists():
        print(f"ERROR: shard result not found: {shard_path}", file=sys.stderr)
        return 2

    data = load_json(shard_path, None)
    ok, err = _validate_stage1_shard_result(data)
    if not ok:
        print(f"ERROR: invalid shard result {shard_path.name}: {err}", file=sys.stderr)
        return 3

    results = load_json(RESULTS_FILE, {})
    added = 0
    for stem, entry in data["results"].items():
        results[stem] = {**entry, "stage": 1}
        added += 1
    save_json(RESULTS_FILE, results)
    print(f"Merged {added} stems from {shard_path.name}")
    return 0
```

Register in `main()`:

```python
    p_merge1 = sub.add_parser("merge-stage1", help="Merge a stage-1 shard result into ocr_results.json")
    p_merge1.add_argument("shard_result", help="Path to shard_NNNN_result.json")
```

and dispatch:

```python
    if args.cmd == "merge-stage1":
        return cmd_merge_stage1(args)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/will/photo_project/yolo_finetune && uv run --with pytest --with pillow pytest tests/test_orchestrate_ocr.py -v -k merge_stage1`

Expected: all three PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/will/photo_project/yolo_finetune && git add scripts/orchestrate_ocr.py tests/test_orchestrate_ocr.py && git commit -m "feat(ocr): merge-stage1 subcommand with schema validation"
```

---

## Task 6: list-shards and requeue subcommands

**Files:**
- Modify: `scripts/orchestrate_ocr.py`
- Modify: `tests/test_orchestrate_ocr.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_orchestrate_ocr.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/will/photo_project/yolo_finetune && uv run --with pytest --with pillow pytest tests/test_orchestrate_ocr.py -v -k "list_shards or requeue"`

Expected: FAIL.

- [ ] **Step 3: Implement list-shards and requeue**

Insert into `scripts/orchestrate_ocr.py` below `cmd_merge_stage1`:

```python
def cmd_list_shards(args) -> int:
    stage_dir = STAGE1_SHARDS_DIR if args.stage == "stage1" else STAGE2_SHARDS_DIR
    if not stage_dir.exists():
        return 0
    for manifest in sorted(stage_dir.glob("shard_*.json")):
        if "_result" in manifest.stem:
            continue
        if _result_path(manifest).exists():
            continue
        print(manifest)
    return 0


def cmd_requeue(args) -> int:
    manifest = Path(args.shard_path).resolve()
    if not manifest.exists():
        print(f"ERROR: manifest not found: {manifest}", file=sys.stderr)
        return 2
    result = _result_path(manifest)
    if result.exists():
        result.unlink()
        print(f"Removed {result.name}; shard is now pending again")
    else:
        print(f"No result file to remove; shard was already pending")
    return 0
```

Register in `main()`:

```python
    p_list = sub.add_parser("list-shards", help="List shard manifests with no result yet")
    p_list.add_argument("stage", choices=["stage1", "stage2"])

    p_requeue = sub.add_parser("requeue", help="Mark a shard as pending again")
    p_requeue.add_argument("shard_path", help="Path to shard_NNNN.json")
```

and dispatch:

```python
    if args.cmd == "list-shards":
        return cmd_list_shards(args)
    if args.cmd == "requeue":
        return cmd_requeue(args)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/will/photo_project/yolo_finetune && uv run --with pytest --with pillow pytest tests/test_orchestrate_ocr.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/will/photo_project/yolo_finetune && git add scripts/orchestrate_ocr.py tests/test_orchestrate_ocr.py && git commit -m "feat(ocr): list-shards and requeue subcommands"
```

---

## Task 7: Stage-2 trigger detection (pure logic, TDD)

**Files:**
- Modify: `scripts/orchestrate_ocr.py`
- Modify: `tests/test_orchestrate_ocr.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_orchestrate_ocr.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/will/photo_project/yolo_finetune && uv run --with pytest --with pillow pytest tests/test_orchestrate_ocr.py -v -k "should_review or select_review"`

Expected: FAIL with AttributeError on `should_review`.

- [ ] **Step 3: Implement should_review and select_review_stems**

Insert into `scripts/orchestrate_ocr.py` below `compute_crop_box`:

```python
def should_review(text: str, confidence: float | None) -> bool:
    """Return True if a stage-1 result should be re-reviewed in stage 2."""
    if confidence is not None and confidence < LOW_CONFIDENCE_THRESHOLD:
        return True
    if text is None:
        return True
    if "?" in text:
        return True
    if text == "NONE":
        return False
    if not DATE_FORMAT_RE.match(text):
        return True
    return False


def select_review_stems() -> list[str]:
    results = load_json(RESULTS_FILE, {})
    flagged: list[str] = []
    for stem, entry in results.items():
        text = entry.get("text", "")
        conf = entry.get("confidence")
        if should_review(text, conf):
            flagged.append(stem)
    return sorted(flagged)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/will/photo_project/yolo_finetune && uv run --with pytest --with pillow pytest tests/test_orchestrate_ocr.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/will/photo_project/yolo_finetune && git add scripts/orchestrate_ocr.py tests/test_orchestrate_ocr.py && git commit -m "feat(ocr): stage-2 trigger detection"
```

---

## Task 8: crop-stage2 subcommand

**Files:**
- Modify: `scripts/orchestrate_ocr.py`
- Modify: `tests/test_orchestrate_ocr.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_orchestrate_ocr.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/will/photo_project/yolo_finetune && uv run --with pytest --with pillow pytest tests/test_orchestrate_ocr.py::test_crop_stage2_writes_two_views_per_stem -v`

Expected: FAIL — subcommand not registered.

- [ ] **Step 3: Implement cmd_crop_stage2**

Insert into `scripts/orchestrate_ocr.py` below `cmd_requeue`:

```python
def _save_full_image(src: Path, dst: Path, max_side: int) -> None:
    from PIL import Image
    img = Image.open(src).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, "JPEG", quality=90)


def cmd_crop_stage2(_args) -> int:
    flagged = select_review_stems()
    if not flagged:
        print("No stems trigger stage-2 review.")
        return 0

    bbox_map = load_bbox_map()
    results = load_json(RESULTS_FILE, {})

    STAGE2_SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    STAGE2_CROP_DIR.mkdir(parents=True, exist_ok=True)
    STAGE2_FULL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Preparing stage-2 review for {len(flagged)} stems...")
    shard_index = 0
    shard_entries: list[dict] = []

    def flush_shard() -> None:
        nonlocal shard_entries, shard_index
        if not shard_entries:
            return
        shard_id = f"{shard_index:04d}"
        manifest_path = STAGE2_SHARDS_DIR / f"shard_{shard_id}.json"
        result_path = STAGE2_SHARDS_DIR / f"shard_{shard_id}_result.json"
        save_json(manifest_path, {
            "shard_id": shard_id,
            "stage": 2,
            "result_path": str(result_path.relative_to(BASE_DIR)),
            "stems": shard_entries,
        })
        shard_index += 1
        shard_entries = []

    for stem in flagged:
        src = SCANMYPHOTOS_DIR / f"{stem}.jpg"
        if not src.exists():
            print(f"  SKIP {stem}: source missing")
            continue
        bbox = bbox_map.get(stem)
        if not bbox:
            print(f"  SKIP {stem}: no bbox")
            continue

        crop_dst = STAGE2_CROP_DIR / f"{stem}.jpg"
        full_dst = STAGE2_FULL_DIR / f"{stem}.jpg"
        try:
            crop_image_to_file(
                src=src, dst=crop_dst, bbox=bbox,
                pad_factor=STAGE2_PAD_FACTOR, max_side=STAGE2_MAX_SIDE,
            )
            _save_full_image(src=src, dst=full_dst, max_side=STAGE2_MAX_SIDE)
        except Exception as e:
            print(f"  SKIP {stem}: crop failed ({e})")
            continue

        shard_entries.append({
            "stem": stem,
            "crop_path": str(crop_dst.relative_to(BASE_DIR)),
            "full_path": str(full_dst.relative_to(BASE_DIR)),
            "stage1_text": results.get(stem, {}).get("text", ""),
            "confidence": bbox.get("confidence"),
        })
        if len(shard_entries) >= STAGE2_SHARD_SIZE:
            flush_shard()

    flush_shard()
    print(f"Wrote {shard_index} stage-2 shard manifest(s)")
    return 0
```

Register:

```python
    sub.add_parser("crop-stage2", help="Prepare stage-2 review shards")
```

and dispatch:

```python
    if args.cmd == "crop-stage2":
        return cmd_crop_stage2(args)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/will/photo_project/yolo_finetune && uv run --with pytest --with pillow pytest tests/test_orchestrate_ocr.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/will/photo_project/yolo_finetune && git add scripts/orchestrate_ocr.py tests/test_orchestrate_ocr.py && git commit -m "feat(ocr): crop-stage2 subcommand with two-view crops"
```

---

## Task 9: Stage-2 reconciliation (pure logic, TDD)

**Files:**
- Modify: `scripts/orchestrate_ocr.py`
- Modify: `tests/test_orchestrate_ocr.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_orchestrate_ocr.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/will/photo_project/yolo_finetune && uv run --with pytest --with pillow pytest tests/test_orchestrate_ocr.py -v -k "reconcile or normalize"`

Expected: FAIL.

- [ ] **Step 3: Implement reconcile_pair and normalize_text**

Insert into `scripts/orchestrate_ocr.py` below `select_review_stems`:

```python
def normalize_text(text: str) -> str:
    """Collapse whitespace for agreement comparison."""
    return " ".join(text.split())


def reconcile_pair(view_crop: str, view_full: str) -> tuple[str, str | None]:
    """Compare two views.

    Returns (status, final_text):
        ("confirmed", text) - both views agree on a non-NONE answer
        ("no_stamp", "NONE") - both views say NONE
        ("disagree", None)  - any disagreement
    """
    nc = normalize_text(view_crop)
    nf = normalize_text(view_full)
    if nc == "NONE" and nf == "NONE":
        return "no_stamp", "NONE"
    if nc == nf:
        return "confirmed", nc
    return "disagree", None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/will/photo_project/yolo_finetune && uv run --with pytest --with pillow pytest tests/test_orchestrate_ocr.py -v`

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/will/photo_project/yolo_finetune && git add scripts/orchestrate_ocr.py tests/test_orchestrate_ocr.py && git commit -m "feat(ocr): stage-2 reconciliation logic"
```

---

## Task 10: merge-stage2 subcommand

**Files:**
- Modify: `scripts/orchestrate_ocr.py`
- Modify: `tests/test_orchestrate_ocr.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_orchestrate_ocr.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/will/photo_project/yolo_finetune && uv run --with pytest --with pillow pytest tests/test_orchestrate_ocr.py -v -k merge_stage2`

Expected: FAIL.

- [ ] **Step 3: Implement merge-stage2**

Insert into `scripts/orchestrate_ocr.py` below `cmd_crop_stage2`:

```python
def _validate_stage2_shard_result(data: dict) -> tuple[bool, str]:
    if not isinstance(data, dict) or "results" not in data:
        return False, "missing 'results' field"
    if not isinstance(data["results"], dict):
        return False, "'results' is not an object"
    for stem, entry in data["results"].items():
        if not isinstance(entry, dict):
            return False, f"entry for {stem} is not an object"
        for field in ("view_crop", "view_full"):
            if field not in entry or not isinstance(entry[field], str):
                return False, f"entry for {stem} missing string '{field}'"
    return True, ""


def cmd_merge_stage2(args) -> int:
    shard_path = Path(args.shard_result).resolve()
    if not shard_path.exists():
        print(f"ERROR: shard result not found: {shard_path}", file=sys.stderr)
        return 2

    data = load_json(shard_path, None)
    ok, err = _validate_stage2_shard_result(data)
    if not ok:
        print(f"ERROR: invalid shard result {shard_path.name}: {err}", file=sys.stderr)
        return 3

    results = load_json(RESULTS_FILE, {})
    manual_queue = load_json(MANUAL_QUEUE_FILE, [])

    confirmed = no_stamp = disagree = 0
    for stem, entry in data["results"].items():
        status, final_text = reconcile_pair(entry["view_crop"], entry["view_full"])
        if status == "confirmed":
            results[stem] = {
                **results.get(stem, {}),
                "text": final_text,
                "stage": 2,
                "review_status": "confirmed",
            }
            confirmed += 1
        elif status == "no_stamp":
            results[stem] = {
                **results.get(stem, {}),
                "text": "NONE",
                "stage": 2,
                "review_status": "no_stamp",
            }
            no_stamp += 1
        else:  # disagree
            stage1_entry = results.get(stem, {})
            stage1_text = stage1_entry.get("text", "")
            results[stem] = {
                **stage1_entry,
                "review_status": "disagreement",
            }
            manual_queue.append({
                "stem": stem,
                "stage1_text": stage1_text,
                "view_crop": entry["view_crop"],
                "view_full": entry["view_full"],
                "confidence": stage1_entry.get("confidence"),
            })
            disagree += 1

    save_json(RESULTS_FILE, results)
    save_json(MANUAL_QUEUE_FILE, manual_queue)
    print(f"Merged {shard_path.name}: {confirmed} confirmed, {no_stamp} no-stamp, {disagree} disagreements")
    return 0
```

Register:

```python
    p_merge2 = sub.add_parser("merge-stage2", help="Merge and reconcile a stage-2 shard result")
    p_merge2.add_argument("shard_result")
```

and dispatch:

```python
    if args.cmd == "merge-stage2":
        return cmd_merge_stage2(args)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/will/photo_project/yolo_finetune && uv run --with pytest --with pillow pytest tests/test_orchestrate_ocr.py -v`

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/will/photo_project/yolo_finetune && git add scripts/orchestrate_ocr.py tests/test_orchestrate_ocr.py && git commit -m "feat(ocr): merge-stage2 with reconciliation and manual queue"
```

---

## Task 11: Pilot — stage-1 pre-crop and dispatch

**This task is manual orchestration. No code to write.** The orchestrator Claude session runs the commands and dispatches subagents.

- [ ] **Step 1: Pre-crop the pilot slice**

Run: `cd /home/will/photo_project/yolo_finetune && uv run scripts/orchestrate_ocr.py crop-stage1 --limit 200`

Expected:
- Prints `Pre-cropping 200 stems...`
- Prints `Wrote 4 shard manifest(s) to state/shards/stage1/`
- Files exist at `output/ocr_crops_stage1/` (200 JPEGs)
- Files exist at `state/shards/stage1/` (`shard_0000.json` through `shard_0003.json`)

- [ ] **Step 2: Verify shard count with list-shards**

Run: `cd /home/will/photo_project/yolo_finetune && uv run scripts/orchestrate_ocr.py list-shards stage1`

Expected: prints 4 pending shard paths.

- [ ] **Step 3: Read the stage-1 prompt file**

Use the Read tool on `scripts/subagent_prompts/stage1_prompt.md`. Hold the contents in context — you will include them verbatim in each subagent dispatch.

- [ ] **Step 4: Dispatch all 4 pilot shards in parallel background**

Use the Agent tool with `subagent_type: "general-purpose"`, `model: "haiku"`, `run_in_background: true`. Issue FOUR Agent tool calls in a single message so they start concurrently.

For each shard, the prompt must contain:
1. The full verbatim contents of `stage1_prompt.md` from Step 3
2. A trailing section that fills in the inputs:

```
## Inputs for this run

SHARD_MANIFEST_PATH: /home/will/photo_project/yolo_finetune/state/shards/stage1/shard_NNNN.json
BASE_DIR: /home/will/photo_project/yolo_finetune

Begin now. Remember: your only output to me is the single line `DONE shard_id=<id> stems=<count>`.
```

Replace `NNNN` with `0000`, `0001`, `0002`, `0003` for the four calls.

- [ ] **Step 5: Wait for notifications**

The harness will notify as each background agent completes. Do not poll. When a subagent reports DONE, proceed to merge.

- [ ] **Step 6: Merge each completed shard as it arrives**

For each completed shard, run: `cd /home/will/photo_project/yolo_finetune && uv run scripts/orchestrate_ocr.py merge-stage1 state/shards/stage1/shard_NNNN_result.json`

Expected: `Merged <N> stems from shard_NNNN_result.json`

If merge-stage1 returns a non-zero exit code (malformed shard result), run `requeue` on that shard and re-dispatch it once:
```
uv run scripts/orchestrate_ocr.py requeue state/shards/stage1/shard_NNNN.json
```
Then dispatch a fresh subagent for that shard. If it fails a second time, append the stem list to `state/failed_shards.json` manually and continue.

- [ ] **Step 7: Verify pilot stage-1 completion**

Run: `cd /home/will/photo_project/yolo_finetune && uv run scripts/orchestrate_ocr.py status`

Expected:
- `Stage-1 shards: 0 pending, 4 done`
- `OCR results` count increased by 200 (minus any skipped)

Do not commit yet — state files are gitignored.

---

## Task 12: Pilot — stage-2 pre-crop, dispatch, and reconciliation

**Manual orchestration, no code.**

- [ ] **Step 1: Run crop-stage2 on pilot results**

Run: `cd /home/will/photo_project/yolo_finetune && uv run scripts/orchestrate_ocr.py crop-stage2`

Expected:
- Prints `Preparing stage-2 review for N stems...` where N is the count of pilot stems that triggered A/B/C rules (expect 30-60% of 200 → 60-120 stems)
- Shard count = ceil(N / 25)
- Files exist at `output/ocr_crops_stage2_crop/` and `output/ocr_crops_stage2_full/`
- Shard manifests at `state/shards/stage2/`

- [ ] **Step 2: Read the stage-2 prompt file**

Use the Read tool on `scripts/subagent_prompts/stage2_prompt.md`. Hold contents in context.

- [ ] **Step 3: Dispatch all stage-2 shards in parallel**

List the pending shards: `uv run scripts/orchestrate_ocr.py list-shards stage2`.

Dispatch up to 5 concurrent Agent calls (one per shard). Each prompt = the full verbatim stage2_prompt.md followed by:

```
## Inputs for this run

SHARD_MANIFEST_PATH: /home/will/photo_project/yolo_finetune/state/shards/stage2/shard_NNNN.json
BASE_DIR: /home/will/photo_project/yolo_finetune

Begin now.
```

If there are more than 5 shards, dispatch 5, wait for at least one to complete, then dispatch the next one — keep 5 in flight until all are done.

- [ ] **Step 4: Merge each completed shard**

For each completed stage-2 shard, run: `cd /home/will/photo_project/yolo_finetune && uv run scripts/orchestrate_ocr.py merge-stage2 state/shards/stage2/shard_NNNN_result.json`

Requeue-and-retry once on any malformed result.

- [ ] **Step 5: Print final pilot summary**

Run: `cd /home/will/photo_project/yolo_finetune && uv run scripts/orchestrate_ocr.py status`

Then report to the user:
- total OCR results
- how many stage-1 triggered stage-2 review
- how many were `confirmed` / `no_stamp` / `disagreement` after review
- manual queue size
- failed shards count

---

## Task 13: Pilot gate — inspect and decide

**Manual review checkpoint. Do not proceed without user approval.**

- [ ] **Step 1: Spot-check 10-20 confirmed results against source photos**

Pick 10-20 stems at random from `ocr_results.json` where `review_status == "confirmed"` (or stage-1 clean). For each:
- Open the source JPEG at `/mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/Photos/img/Photos/ScanMyPhotos/Disc N/<stem>.jpg`
- Read the date stamp visually
- Compare to the text in `ocr_results.json`

Report any mismatches.

- [ ] **Step 2: Inspect the manual review queue**

Read `state/ocr_manual_queue.json`. For each disagreement entry, note both views' texts and assess whether either appears correct based on the crop paths.

- [ ] **Step 3: Present summary to user**

Report:
- Pilot hit rate: `<stamps found>` / 200 (vs YOLO's 6458/7455 detection rate)
- Accuracy spot check: <correct>/<checked> matches
- Review trigger rate: `<review count>` / 200
- Post-review confirmed: `<confirmed>` / `<review count>`
- Disagreements in manual queue: `<count>`
- Any failed shards, dispatch issues, or unexpected prompts

- [ ] **Step 4: Await go/no-go for full run**

If user says GO: re-run Tasks 11-12 without the `--limit 200` flag to process the remaining ~6,258 photos. The `compute_pending_stems()` logic will correctly skip the pilot 200 that are already in `ocr_results.json`.

If user wants fixes: apply them first, add new tasks to this plan as needed, re-run pilot.

---

## Self-Review Notes

Spec sections mapped to tasks:
- Architecture / data flow → Tasks 1-10 (code) + 11-12 (orchestration)
- `orchestrate_ocr.py` subcommands → Tasks 1, 3, 5, 6, 8, 10
- Subagent prompts → Task 4
- Shard manifest schemas → Tasks 3 (stage 1 shape), 8 (stage 2 shape)
- Pre-cropping rules → Tasks 2, 3 (stage 1), 8 (stage 2 two views)
- Stage-2 triggers → Task 7
- Reconciliation rules → Tasks 9, 10
- Parallelism / in-flight limit of 5 → Task 11 step 4, Task 12 step 3
- Pilot gate → Task 13
- File layout / gitignore → Task 1
- Failure handling (requeue once then failed_shards.json) → Task 6 + Tasks 11-12 steps on merge errors
