# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "ultralytics>=8.0",
#     "opencv-python-headless",
# ]
# ///
"""Batch inference on pending ScanMyPhotos images. Writes predictions + worker status."""

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
CONF_THRESHOLD = 0.01


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


BATCH_SIZE = 32
STATUS_EVERY = 200


def extract_best_prediction(result):
    """Extract best detection from a single result, or None."""
    if len(result.boxes) == 0:
        return None
    best_idx = result.boxes.conf.argmax()
    box = result.boxes[best_idx]
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    img_h, img_w = result.orig_shape
    return {
        "x": round(((x1 + x2) / 2) / img_w, 6),
        "y": round(((y1 + y2) / 2) / img_h, 6),
        "w": round((x2 - x1) / img_w, 6),
        "h": round((y2 - y1) / img_h, 6),
        "confidence": round(float(box.conf[0]), 4),
    }


def run_inference():
    """Run YOLO inference on all pending files."""
    import torch
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

    write_status("inference", 0, total, "Loading model...")
    print(f"Running inference on {total} pending files...")

    # Use half precision on CPU if supported (PyTorch >= 2.x handles this well)
    model = YOLO(str(MODEL_PATH))

    # Set thread count to use all available cores for inference
    torch.set_num_threads(torch.get_num_threads() or 12)

    # Load existing predictions to merge with
    predictions = {}
    if PREDICTIONS_FILE.exists():
        with open(PREDICTIONS_FILE) as f:
            predictions = json.load(f)

    # Process in batches -- YOLO handles batched inputs natively
    processed = 0
    for batch_start in range(0, total, BATCH_SIZE):
        batch_paths = [str(p) for p in pending[batch_start : batch_start + BATCH_SIZE]]

        results = model.predict(
            batch_paths,
            imgsz=INFERENCE_SIZE,
            conf=CONF_THRESHOLD,
            device="cpu",
            verbose=False,
            batch=BATCH_SIZE,
        )

        for j, result in enumerate(results):
            stem = pending[batch_start + j].stem
            pred = extract_best_prediction(result)
            if pred is not None:
                predictions[stem] = pred

        processed += len(batch_paths)

        if processed % STATUS_EVERY < BATCH_SIZE or processed == total:
            write_status("inference", processed, total, f"{processed}/{total} inferred")
            with open(PREDICTIONS_FILE, "w") as f:
                json.dump(predictions, f, separators=(",", ":"))
            print(f"  [{processed}/{total}]")

    # Final save (pretty-printed for human readability)
    with open(PREDICTIONS_FILE, "w") as f:
        json.dump(predictions, f, indent=2)

    write_status("done", total, total, f"Inference complete. {len(predictions)} predictions saved.")
    print(f"Done. {len(predictions)} total predictions in {PREDICTIONS_FILE.name}")


if __name__ == "__main__":
    run_inference()
