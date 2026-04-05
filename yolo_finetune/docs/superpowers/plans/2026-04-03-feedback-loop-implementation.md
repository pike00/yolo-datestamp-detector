# Feedback Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a semi-automated feedback loop where users can quickly review and confirm model-predicted bounding boxes, accumulating corrections for continuous model improvement.

**Architecture:** 
- `feedback.py` orchestrates the workflow: `prepare` (runs inference, saves predictions), `finalize` (moves annotated images to corrections), and `status` (shows progress)
- `annotate.py` gains a `--mode=correct` flag that loads model predictions from `corrections_meta.json` and displays them pre-drawn on the canvas
- `train.py` automatically discovers and merges corrections into the training dataset before each retraining
- Predictions stored in JSON format with confidence scores for visibility during review

**Tech Stack:** ultralytics (inference), pathlib, json, yaml, existing annotate.py canvas infrastructure

---

## File Structure

**Create:**
- `feedback.py` — orchestrates feedback workflow (prepare, finalize, status commands)

**Modify:**
- `annotate.py` — add `--mode=correct` flag, load predictions from JSON, display pre-drawn boxes
- `train.py` — merge corrections into dataset before splitting

**Created at runtime:**
- `dataset/to_annotate/` — staging directory for current feedback cycle
- `dataset/corrections/` — accumulates all feedback annotations (persists across cycles)
- `corrections_meta.json` — stores model predictions (x, y, w, h, confidence) per image

---

## Task 1: Create feedback.py with prepare command

**Files:**
- Create: `feedback.py`

- [ ] **Step 1: Create feedback.py with imports and constants**

```python
#!/usr/bin/env python3
"""Feedback loop orchestration for YOLO continuous improvement."""

import json
import sys
from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
CORRECTIONS_DIR = DATASET_DIR / "corrections"
TO_ANNOTATE_DIR = DATASET_DIR / "to_annotate"
CORRECTIONS_META_FILE = CORRECTIONS_DIR / "corrections_meta.json"
MODEL_PATH = BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
INFER_OUTPUT_DIR = BASE_DIR / "infer_output"


def load_corrections_meta():
    """Load existing predictions metadata, return empty dict if file doesn't exist."""
    if CORRECTIONS_META_FILE.exists():
        with open(CORRECTIONS_META_FILE) as f:
            return json.load(f)
    return {}


def save_corrections_meta(meta):
    """Save predictions metadata to JSON."""
    CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CORRECTIONS_META_FILE, "w") as f:
        json.dump(meta, f, indent=2)
```

- [ ] **Step 2: Implement prepare() command**

```python
def prepare():
    """
    Select images from infer_output with missed detections.
    Run inference on them and save predicted boxes to corrections_meta.json.
    """
    # Ensure directories exist
    TO_ANNOTATE_DIR.mkdir(parents=True, exist_ok=True)
    CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Train a model first with: python train.py")
        return

    if not INFER_OUTPUT_DIR.exists():
        print(f"Error: {INFER_OUTPUT_DIR} not found. Run infer.py first.")
        return

    # List available inference outputs
    infer_images = sorted(INFER_OUTPUT_DIR.glob("*_detected.jpg"))
    if not infer_images:
        print(f"No inference outputs found in {INFER_OUTPUT_DIR}")
        return

    print("\nInference results available:")
    for i, img in enumerate(infer_images, 1):
        # Extract original stem (remove _detected suffix)
        stem = img.stem.replace("_detected", "")
        print(f"  {i}. {stem}")

    # Prompt user for selections
    print("\nEnter image filenames with MISSED detections (comma-separated, or 'all'):")
    print("Example: img001, img002")
    user_input = input("> ").strip()

    if user_input.lower() == "all":
        selected_stems = [img.stem.replace("_detected", "") for img in infer_images]
    else:
        selected_stems = [s.strip() for s in user_input.split(",") if s.strip()]

    if not selected_stems:
        print("No images selected.")
        return

    # Find source images in photo_mapping_samples
    source_dir = BASE_DIR.parent / "photo_mapping_samples"
    source_images = {p.stem: p for ext in ("*.jpg", "*.JPG") for p in source_dir.glob(ext)}

    # Copy selected images to to_annotate and run inference
    print(f"\nPreparing {len(selected_stems)} images for annotation...")
    model = YOLO(str(MODEL_PATH))
    corrections_meta = load_corrections_meta()

    copied_count = 0
    for stem in selected_stems:
        if stem not in source_images:
            print(f"  Warning: source image {stem} not found in {source_dir}")
            continue

        src_img = source_images[stem]
        dst_img = TO_ANNOTATE_DIR / src_img.name

        # Copy image
        if not dst_img.exists():
            dst_img.write_bytes(src_img.read_bytes())
        copied_count += 1

        # Run inference and save prediction
        results = model(str(src_img), imgsz=384, conf=0.3, device="cpu", verbose=False)
        result = results[0]

        if len(result.boxes) > 0:
            # Take first box (most confident)
            box = result.boxes[0]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])

            # Convert to normalized center + width/height format
            img_h, img_w = result.orig_shape
            center_x = ((x1 + x2) / 2) / img_w
            center_y = ((y1 + y2) / 2) / img_h
            box_w = (x2 - x1) / img_w
            box_h = (y2 - y1) / img_h

            corrections_meta[stem] = {
                "x": round(center_x, 4),
                "y": round(center_y, 4),
                "w": round(box_w, 4),
                "h": round(box_h, 4),
                "confidence": round(conf, 4),
            }
        else:
            # Model found no detection on this image
            corrections_meta[stem] = {"x": None, "y": None, "w": None, "h": None, "confidence": 0.0}

    save_corrections_meta(corrections_meta)

    print(f"\n✓ Copied {copied_count} images to {TO_ANNOTATE_DIR}")
    print(f"✓ Saved predictions to {CORRECTIONS_META_FILE}")
    print(f"\nNext: python annotate.py --mode=correct")
```

- [ ] **Step 3: Commit**

```bash
git add feedback.py
git commit -m "feat: add feedback.py prepare command with inference"
```

---

## Task 2: Add finalize and status commands to feedback.py

**Files:**
- Modify: `feedback.py`

- [ ] **Step 1: Implement finalize() command**

```python
def finalize():
    """
    Move annotated images from to_annotate to corrections.
    Check that each image has a corresponding label file.
    """
    if not TO_ANNOTATE_DIR.exists() or not list(TO_ANNOTATE_DIR.glob("*")):
        print(f"No images in {TO_ANNOTATE_DIR} to finalize.")
        return

    images = sorted(TO_ANNOTATE_DIR.glob("*.jpg")) + sorted(TO_ANNOTATE_DIR.glob("*.JPG"))
    if not images:
        print("No images found.")
        return

    print(f"Finalizing {len(images)} images from {TO_ANNOTATE_DIR}...")

    CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    labeled_count = 0
    skipped_count = 0

    for img in images:
        stem = img.stem
        dst_img = CORRECTIONS_DIR / img.name

        # Move image
        dst_img.write_bytes(img.read_bytes())
        img.unlink()

        # Check for label file
        label_file = BASE_DIR / "dataset" / "labels" / f"{stem}.txt"
        if label_file.exists():
            dst_label = CORRECTIONS_DIR / f"{stem}.txt"
            dst_label.write_bytes(label_file.read_bytes())
            labeled_count += 1
        else:
            # Image was skipped (no box) — will be negative example
            skipped_count += 1

    print(f"✓ Moved {labeled_count} labeled images to {CORRECTIONS_DIR}")
    print(f"✓ Moved {skipped_count} skipped (negative) images to {CORRECTIONS_DIR}")
    print(f"✓ Cleared {TO_ANNOTATE_DIR}")
```

- [ ] **Step 2: Implement status() command**

```python
def status():
    """Show current feedback loop status."""
    corrections_in_dir = sum(1 for p in CORRECTIONS_DIR.glob("*.txt")) if CORRECTIONS_DIR.exists() else 0
    skipped_in_dir = sum(1 for p in CORRECTIONS_DIR.glob("*.jpg")) - corrections_in_dir if CORRECTIONS_DIR.exists() else 0
    to_annotate_count = sum(1 for p in TO_ANNOTATE_DIR.glob("*.jpg")) if TO_ANNOTATE_DIR.exists() else 0

    print("\n=== Feedback Loop Status ===")
    print(f"Accumulated corrections: {corrections_in_dir} labeled + {skipped_in_dir} skipped (negative)")
    print(f"Pending annotation: {to_annotate_count} images in {TO_ANNOTATE_DIR}")

    if CORRECTIONS_META_FILE.exists():
        meta = load_corrections_meta()
        with_boxes = sum(1 for v in meta.values() if v.get("x") is not None)
        print(f"Model predictions saved: {with_boxes} with boxes, {len(meta) - with_boxes} no detection")
```

- [ ] **Step 3: Add main() function and CLI**

```python
def main():
    """Parse command-line arguments and dispatch."""
    if len(sys.argv) < 2:
        print("Usage: python feedback.py {prepare|finalize|status}")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "prepare":
        prepare()
    elif command == "finalize":
        finalize()
    elif command == "status":
        status()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Commit**

```bash
git add feedback.py
git commit -m "feat: add feedback.py finalize and status commands"
```

---

## Task 3: Modify train.py to merge corrections

**Files:**
- Modify: `train.py`

- [ ] **Step 1: Update setup_dataset() to merge corrections**

In `train.py`, find the `setup_dataset()` function. Add this code **before** the line `labeled = sorted(p.stem for p in LABELS_DIR.glob("*.txt"))`:

```python
def setup_dataset():
    """Split labeled + skipped images into train/val with YOLO directory structure."""
    
    # Merge corrections into main dataset before collecting
    CORRECTIONS_DIR = BASE_DIR / "dataset" / "corrections"
    if CORRECTIONS_DIR.exists():
        print(f"Merging corrections from {CORRECTIONS_DIR}...")
        
        # Merge labeled images (copy .txt files)
        for label_file in CORRECTIONS_DIR.glob("*.txt"):
            dst_label = LABELS_DIR / label_file.name
            if not dst_label.exists():
                dst_label.write_bytes(label_file.read_bytes())
        
        # Merge skipped images (add to skipped.txt)
        correction_images = {p.stem for p in CORRECTIONS_DIR.glob("*.jpg")}
        correction_images |= {p.stem for p in CORRECTIONS_DIR.glob("*.JPG")}
        correction_labeled = {p.stem for p in CORRECTIONS_DIR.glob("*.txt")}
        correction_skipped = correction_images - correction_labeled
        
        if correction_skipped:
            existing_skipped = set()
            if SKIPPED_FILE.exists():
                existing_skipped = {
                    line.strip() for line in SKIPPED_FILE.read_text().splitlines()
                }
            
            new_skipped = existing_skipped | correction_skipped
            SKIPPED_FILE.write_text("\n".join(sorted(new_skipped)) + "\n")

    # Collect labeled images (those with a .txt in labels/)
    labeled = sorted(p.stem for p in LABELS_DIR.glob("*.txt"))
    # ... rest of function unchanged
```

- [ ] **Step 2: Verify train.py still starts with setup_dataset call**

Check that `main()` calls `setup_dataset()` first:

```python
def main():
    print("Setting up dataset...")
    data_yaml = setup_dataset()
    print("\nStarting YOLOv8 fine-tuning (CPU, this will take a while)...\n")
    train(data_yaml)
```

- [ ] **Step 3: Commit**

```bash
git add train.py
git commit -m "feat: train.py merges corrections before setup"
```

---

## Task 4: Modify annotate.py to add --mode=correct flag

**Files:**
- Modify: `annotate.py`

- [ ] **Step 1: Add argument parsing to annotate.py**

At the top of `annotate.py`, after imports, add:

```python
import argparse

# ... existing imports and constants ...

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Annotate bounding boxes on images")
    parser.add_argument("--mode", choices=["annotate", "correct"], default="annotate",
                        help="annotate: label new images; correct: review model predictions")
    return parser.parse_args()

ARGS = parse_args()
```

- [ ] **Step 2: Update IMAGE_SOURCE selection based on mode**

In the file, find where `IMAGE_SOURCE` is set. Modify it to:

```python
# Determine image source based on mode
if ARGS.mode == "correct":
    IMAGE_SOURCE = BASE_DIR / "dataset" / "to_annotate"
    CORRECTIONS_METADATA_PATH = BASE_DIR / "dataset" / "corrections" / "corrections_meta.json"
else:
    IMAGE_SOURCE = BASE_DIR.parent / "photo_mapping_samples"
    CORRECTIONS_METADATA_PATH = None
```

- [ ] **Step 3: Load predictions metadata at startup**

Add this function near the top of annotate.py:

```python
def load_predictions_metadata():
    """Load model predictions from corrections_meta.json if in correct mode."""
    if ARGS.mode == "correct" and CORRECTIONS_METADATA_PATH and CORRECTIONS_METADATA_PATH.exists():
        with open(CORRECTIONS_METADATA_PATH) as f:
            return json.load(f)
    return {}

PREDICTIONS_META = load_predictions_metadata()
```

And add `import json` to the imports.

- [ ] **Step 4: Modify progress.json loading to handle correct mode separately**

Find the line `PROGRESS_FILE = BASE_DIR / "progress.json"`. Change it to:

```python
if ARGS.mode == "correct":
    PROGRESS_FILE = BASE_DIR / "progress_correct.json"
else:
    PROGRESS_FILE = BASE_DIR / "progress.json"
```

This keeps correction sessions separate from main annotation sessions.

- [ ] **Step 5: Commit**

```bash
git add annotate.py
git commit -m "feat: add --mode=correct flag to annotate.py"
```

---

## Task 5: Load and display model predictions in correct mode

**Files:**
- Modify: `annotate.py`

- [ ] **Step 1: Update load_progress() to pre-load predictions in correct mode**

Find the `load_progress()` function. After loading the JSON, add:

```python
def load_progress():
    """Load annotation progress from JSON."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            data = json.load(f)
    else:
        data = {"current_index": 0, "completed": [], "last_box": None}

    # In correct mode, pre-load predictions for all images
    if ARGS.mode == "correct":
        files = sorted(IMAGE_SOURCE.glob("*.jpg")) + sorted(IMAGE_SOURCE.glob("*.JPG"))
        for img_file in files:
            stem = img_file.stem
            if stem not in data["completed"] and stem in PREDICTIONS_META:
                pred = PREDICTIONS_META[stem]
                if pred["x"] is not None:
                    # Pre-populate with model prediction
                    data[f"prediction_{stem}"] = pred
    
    return data
```

- [ ] **Step 2: Update the box initialization to use predictions in correct mode**

In the HTTP handler that serves the current state (search for `get_current_state` or similar), modify it to:

```python
def get_current_state():
    """Return current annotation state as JSON."""
    progress = load_progress()
    current_idx = progress.get("current_index", 0)
    
    files = sorted(IMAGE_SOURCE.glob("*.jpg")) + sorted(IMAGE_SOURCE.glob("*.JPG"))
    if current_idx >= len(files):
        return {"done": True}
    
    current_file = files[current_idx]
    stem = current_file.stem
    
    # Load box: use prediction if in correct mode, else use last_box or default
    box = progress.get("last_box") or {"x": 0.4, "y": 0.5, "w": 0.2, "h": 0.1}
    
    if ARGS.mode == "correct" and stem in PREDICTIONS_META:
        pred = PREDICTIONS_META[stem]
        if pred["x"] is not None:
            box = {
                "x": pred["x"],
                "y": pred["y"],
                "w": pred["w"],
                "h": pred["h"],
            }
            box["model_confidence"] = pred["confidence"]  # Mark as from model
    
    return {
        "done": False,
        "filename": current_file.name,
        "index": current_idx,
        "total": len(files),
        "box": box,
        "mode": ARGS.mode,
    }
```

- [ ] **Step 3: Commit**

```bash
git add annotate.py
git commit -m "feat: load model predictions in correct mode"
```

---

## Task 6: Update index.html to display model predictions with visual distinction

**Files:**
- Modify: `index.html`

- [ ] **Step 1: Add CSS for model-predicted boxes (different color)**

In the `<style>` section of `index.html`, find the canvas styling. Add:

```css
/* For distinguishing model predictions from user corrections */
#canvas {
    cursor: crosshair;
}

.box-type {
    font-size: 12px;
    margin-top: 5px;
    padding: 5px;
    border-radius: 4px;
}

.box-type.model {
    background-color: #ffd700;
    color: #333;
}

.box-type.user {
    background-color: #00aa00;
    color: white;
}
```

- [ ] **Step 2: Update canvas drawing to show different color for model predictions**

Find the `drawBox()` function in the JavaScript section. Modify it to:

```javascript
function drawBox() {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();

    // Clear and redraw image
    ctx.drawImage(imageElement, 0, 0, canvas.width, canvas.height);

    // Determine box color based on source
    const isModelPrediction = currentState.box.model_confidence !== undefined;
    const boxColor = isModelPrediction ? '#FFD700' : '#00FF00';  // Yellow for model, green for user
    const lineWidth = isModelPrediction ? 3 : 2;

    // Draw box
    const x = currentState.box.x * canvas.width;
    const y = currentState.box.y * canvas.height;
    const w = currentState.box.w * canvas.width;
    const h = currentState.box.h * canvas.height;

    ctx.strokeStyle = boxColor;
    ctx.lineWidth = lineWidth;
    ctx.strokeRect(x - w / 2, y - h / 2, w, h);

    // Draw corners
    ctx.fillStyle = boxColor;
    const cornerSize = 5;
    [
        [x - w / 2, y - h / 2],
        [x + w / 2, y - h / 2],
        [x - w / 2, y + h / 2],
        [x + w / 2, y + h / 2],
    ].forEach(([cx, cy]) => {
        ctx.fillRect(cx - cornerSize / 2, cy - cornerSize / 2, cornerSize, cornerSize);
    });

    // Show confidence if model prediction
    if (isModelPrediction) {
        const conf = currentState.box.model_confidence;
        ctx.fillStyle = '#FFD700';
        ctx.font = '14px monospace';
        ctx.fillText(`Model: ${(conf * 100).toFixed(0)}%`, 10, 30);
    }
}
```

- [ ] **Step 3: Update status display to show mode**

Find the status bar update code (where it shows "N/M"). Modify to add mode indicator:

```javascript
function updateStatusBar() {
    const filename = currentState.filename;
    const index = currentState.index + 1;
    const total = currentState.total;
    const mode = currentState.mode === 'correct' ? '(CORRECT MODE)' : '';
    
    document.getElementById('status-bar').textContent = 
        `${filename} [${index}/${total}] ${mode}`;
}
```

- [ ] **Step 4: Commit**

```bash
git add index.html
git commit -m "feat: show model predictions in yellow, user corrections in green"
```

---

## Task 7: Integration test — verify full feedback loop

**Files:**
- Test: manual workflow validation

- [ ] **Step 1: Test feedback.py prepare command**

```bash
cd /home/will/photo_project/yolo_finetune
python feedback.py status
# Expected: Shows "Accumulated corrections: 0 labeled + 0 skipped"

# Run infer.py first if not done
python infer.py

# Now prepare
python feedback.py prepare
# Select a few images from the list
# Verify corrections_meta.json is created with predictions
cat dataset/corrections/corrections_meta.json | head -20
```

- [ ] **Step 2: Test annotate.py --mode=correct**

```bash
python annotate.py --mode=correct
# Browser should open with first image from to_annotate/
# Yellow box should be pre-drawn from model prediction
# Confidence score should display
# Press Enter to confirm box, or arrows to adjust
# When done, press Q to quit
```

- [ ] **Step 3: Test feedback.py finalize**

```bash
python feedback.py finalize
# Should move images from to_annotate to corrections/
# Check: ls dataset/corrections/ | head
# Should see both .jpg images and .txt label files
```

- [ ] **Step 4: Test train.py integration**

```bash
python train.py
# Should print "Merging corrections from dataset/corrections/..."
# Should show updated train/val split including corrections
# Should resume from previous best.pt
# Should complete training
```

- [ ] **Step 5: Verify model improvement**

```bash
python infer.py
# Run inference again on sample images
# Check infer_output/ for improved detections
# Repeat steps 1-4 if more improvements needed
```

- [ ] **Step 6: Final status check**

```bash
python feedback.py status
# Should show accumulated corrections from feedback cycles
```

---

## Self-Review

**Spec coverage:**
- ✅ feedback.py prepare: runs inference, saves predictions to JSON
- ✅ feedback.py finalize: moves annotated images to corrections
- ✅ feedback.py status: shows progress
- ✅ annotate.py --mode=correct: loads predictions, displays pre-drawn boxes
- ✅ Visual distinction: yellow for model, green for user adjustments
- ✅ Confidence display: shown in top-left of canvas
- ✅ train.py integration: merges corrections before training
- ✅ Workflow: prepare → annotate → finalize → train

**Placeholder scan:** All tasks contain complete code, no "TBD"s, exact file paths and commands.

**Type/name consistency:** `corrections_meta`, `PREDICTIONS_META`, `CORRECTIONS_METADATA_PATH` used consistently. JSON structure: `{"stem": {"x": float, "y": float, "w": float, "h": float, "confidence": float}}`
