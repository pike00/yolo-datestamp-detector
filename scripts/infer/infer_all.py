# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "ultralytics>=8.0",
#     "opencv-python-headless",
#     "psycopg[binary]>=3.1.0",
# ]
# ///
"""Batch inference on pending ScanMyPhotos images.

Predictions and the no-stamp set are persisted in Postgres
(stamp_predictions, stamp_no_stamp). Worker progress is written to a small
JSON status file so the dashboard can poll it without hitting the DB.
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from _db import (  # noqa: E402
    get_predicted_stems,
    load_skipped_stems,
    upsert_predictions,
)

SCANMYPHOTOS_DIR = BASE_DIR / "scanmyphotos"
LABELS_DIR = BASE_DIR / "dataset" / "labels"
STATUS_FILE = BASE_DIR / "state" / "worker_status.json"
MODEL_PATH = BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"

INFERENCE_SIZE = 384
CONF_THRESHOLD = 0.01
BATCH_SIZE = 32
STATUS_EVERY = 200


def write_status(phase, progress, total, message):
    """Write worker status for dashboard polling."""
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, "w") as f:
        json.dump(
            {"phase": phase, "progress": progress, "total": total, "message": message},
            f,
        )


def get_pending_files():
    """Return the list of images that have neither a label nor a no-stamp marker."""
    labeled_stems = {p.stem for p in LABELS_DIR.glob("*.txt")}
    skipped_stems = load_skipped_stems()
    reviewed = labeled_stems | skipped_stems

    all_images = sorted(SCANMYPHOTOS_DIR.glob("*.jpg"))
    return [img for img in all_images if img.stem not in reviewed]


def extract_best_prediction(result):
    """Extract the highest-confidence detection from a single result, or None."""
    if len(result.boxes) == 0:
        return None
    best_idx = result.boxes.conf.argmax()
    box = result.boxes[best_idx]
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    img_h, img_w = result.orig_shape
    return (
        round(((x1 + x2) / 2) / img_w, 6),
        round(((y1 + y2) / 2) / img_h, 6),
        round((x2 - x1) / img_w, 6),
        round((y2 - y1) / img_h, 6),
        round(float(box.conf[0]), 4),
    )


def run_inference():
    """Run YOLO inference on all pending files and upsert into stamp_predictions."""
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

    model = YOLO(str(MODEL_PATH))
    torch.set_num_threads(torch.get_num_threads() or 12)

    pending_buffer: list[tuple] = []
    written = 0
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
                pending_buffer.append((stem, *pred))

        processed += len(batch_paths)

        if processed % STATUS_EVERY < BATCH_SIZE or processed == total:
            if pending_buffer:
                upsert_predictions(pending_buffer)
                written += len(pending_buffer)
                pending_buffer.clear()
            write_status("inference", processed, total, f"{processed}/{total} inferred")
            print(f"  [{processed}/{total}]")

    if pending_buffer:
        upsert_predictions(pending_buffer)
        written += len(pending_buffer)

    total_in_db = len(get_predicted_stems())
    write_status(
        "done",
        total,
        total,
        f"Inference complete. {written} new/updated, {total_in_db} total in stamp_predictions.",
    )
    print(f"Done. {written} new/updated, {total_in_db} total rows in stamp_predictions")


if __name__ == "__main__":
    run_inference()
