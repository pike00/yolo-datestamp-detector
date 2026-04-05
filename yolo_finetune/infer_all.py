#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["ultralytics>=8.0", "opencv-python-headless"]
# ///
"""Batch inference on all ScanMyPhotos images. Writes predictions + worker status."""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
SCANMYPHOTOS_DIR = BASE_DIR / "scanmyphotos"
LABELS_DIR = BASE_DIR / "dataset" / "labels"
SKIPPED_FILE = BASE_DIR / "skipped.txt"
PREDICTIONS_FILE = BASE_DIR / "scanmyphotos_predictions.json"
STATUS_FILE = BASE_DIR / "worker_status.json"
MODEL_PATH = BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"

INFERENCE_SIZE = 384
CONF_THRESHOLD = 0.3


def write_status(phase, progress, total, message):
    """Write worker status for dashboard polling."""
    with open(STATUS_FILE, "w") as f:
        json.dump({"phase": phase, "progress": progress, "total": total, "message": message}, f)


def get_pending_files():
    """Get list of files that need inference (no label, not skipped)."""
    labeled_stems = {p.stem for p in LABELS_DIR.glob("*.txt")}

    skipped_stems = set()
    if SKIPPED_FILE.exists():
        skipped_stems = {line.strip() for line in SKIPPED_FILE.read_text().splitlines() if line.strip()}

    reviewed = labeled_stems | skipped_stems

    all_images = sorted(SCANMYPHOTOS_DIR.glob("*.jpg"))
    pending = [img for img in all_images if img.stem not in reviewed]
    return pending


def run_inference():
    """Run YOLO inference on all pending files."""
    from ultralytics import YOLO

    if not MODEL_PATH.exists():
        write_status("error", 0, 0, f"Model not found: {MODEL_PATH}")
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return

    pending = get_pending_files()
    total = len(pending)

    if total == 0:
        write_status("done", 0, 0, "No pending files to infer.")
        print("No pending files to infer.")
        return

    write_status("inference", 0, total, f"Loading model...")
    print(f"Running inference on {total} pending files...")

    model = YOLO(str(MODEL_PATH))

    # Load existing predictions to merge with
    predictions = {}
    if PREDICTIONS_FILE.exists():
        with open(PREDICTIONS_FILE) as f:
            predictions = json.load(f)

    for i, img_path in enumerate(pending):
        stem = img_path.stem

        results = model(
            str(img_path),
            imgsz=INFERENCE_SIZE,
            conf=CONF_THRESHOLD,
            device="cpu",
            verbose=False,
        )

        result = results[0]
        if len(result.boxes) > 0:
            # Take highest confidence detection
            best_idx = result.boxes.conf.argmax()
            box = result.boxes[best_idx]

            # Convert to YOLO normalized format (center x, center y, w, h)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            img_h, img_w = result.orig_shape
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h
            conf = float(box.conf[0])

            predictions[stem] = {
                "x": round(float(cx), 6),
                "y": round(float(cy), 6),
                "w": round(float(bw), 6),
                "h": round(float(bh), 6),
                "confidence": round(conf, 4),
            }

        if (i + 1) % 50 == 0 or (i + 1) == total:
            write_status("inference", i + 1, total, f"{i + 1}/{total} inferred")
            # Save predictions incrementally every 50
            with open(PREDICTIONS_FILE, "w") as f:
                json.dump(predictions, f, indent=2)
            print(f"  [{i + 1}/{total}] {stem}")

    # Final save
    with open(PREDICTIONS_FILE, "w") as f:
        json.dump(predictions, f, indent=2)

    write_status("done", total, total, f"Inference complete. {len(predictions)} predictions saved.")
    print(f"Done. {len(predictions)} total predictions in {PREDICTIONS_FILE.name}")


if __name__ == "__main__":
    run_inference()
