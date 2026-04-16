#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "ultralytics>=8.0",
#     "opencv-python-headless",
#     "psycopg[binary]>=3.1.0",
# ]
# ///
"""Re-run inference with new weights and flag photos whose prediction drifted.

Compares the predictions currently in stamp_predictions against a fresh
inference run using NEW_WEIGHTS, and upserts the comparison into the
stamp_prediction_drift table.

A prediction is flagged "drift" if IoU(old, new) < IOU_THRESHOLD.
"gone" means the old model found a stamp and the new model did not.
"""
import argparse
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from _db import load_predictions, upsert_drift  # noqa: E402

SCANMYPHOTOS_DIR = BASE_DIR / "scanmyphotos"
NEW_WEIGHTS = BASE_DIR / "runs" / "detect" / "gpu-40ep" / "weights" / "best.pt"

INFERENCE_SIZE = 384
CONF_THRESHOLD = 0.01
BATCH_SIZE = 32
IOU_THRESHOLD = 0.5  # below this, flag as drift


def xywh_to_xyxy(p):
    """Normalized center-x/y + w/h -> x1,y1,x2,y2 (still normalized)."""
    return (p["x"] - p["w"] / 2, p["y"] - p["h"] / 2,
            p["x"] + p["w"] / 2, p["y"] + p["h"] / 2)


def iou(a, b):
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def extract_best(result):
    if len(result.boxes) == 0:
        return None
    best_idx = result.boxes.conf.argmax()
    box = result.boxes[best_idx]
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    h, w = result.orig_shape
    return {
        "x": round(((x1 + x2) / 2) / w, 6),
        "y": round(((y1 + y2) / 2) / h, 6),
        "w": round((x2 - x1) / w, 6),
        "h": round((y2 - y1) / h, 6),
        "confidence": round(float(box.conf[0]), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="process only N stems (debug)")
    ap.add_argument("--iou-threshold", type=float, default=IOU_THRESHOLD)
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    if not NEW_WEIGHTS.exists():
        raise SystemExit(f"new weights not found at {NEW_WEIGHTS}")

    old = load_predictions()
    stems = sorted(old.keys())
    if args.limit:
        stems = stems[: args.limit]

    missing = [s for s in stems if not (SCANMYPHOTOS_DIR / f"{s}.jpg").is_file()]
    if missing:
        print(f"WARN: {len(missing)} stems have no matching jpg, skipping those")
        stems = [s for s in stems if s not in set(missing)]

    total = len(stems)
    print(f"comparing {total} stems (iou_threshold={args.iou_threshold})")
    print(f"loading new model: {NEW_WEIGHTS}")
    model = YOLO(str(NEW_WEIGHTS))
    torch.set_num_threads(torch.get_num_threads() or 12)

    drift: dict[str, dict] = {}
    pending_rows: list[tuple] = []
    t0 = time.time()
    for batch_start in range(0, total, BATCH_SIZE):
        batch_stems = stems[batch_start : batch_start + BATCH_SIZE]
        batch_paths = [str(SCANMYPHOTOS_DIR / f"{s}.jpg") for s in batch_stems]

        results = model.predict(
            batch_paths,
            imgsz=INFERENCE_SIZE,
            conf=CONF_THRESHOLD,
            device="cpu",
            verbose=False,
            batch=BATCH_SIZE,
        )

        for stem, result in zip(batch_stems, results):
            new_pred = extract_best(result)
            old_pred = old[stem]
            if new_pred is None:
                flag = "gone"
                iou_val = 0.0
            else:
                iou_val = iou(old_pred, new_pred)
                flag = "drift" if iou_val < args.iou_threshold else "stable"
            iou_rounded = round(iou_val, 4)
            drift[stem] = {"old": old_pred, "new": new_pred, "iou": iou_rounded, "flag": flag}
            pending_rows.append((stem, old_pred, new_pred, iou_rounded, flag))

        done = batch_start + len(batch_stems)
        if done % (BATCH_SIZE * 5) == 0 or done == total:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(f"  [{done}/{total}]  {rate:.1f} img/s  eta {eta:.0f}s")
            if pending_rows:
                upsert_drift(pending_rows)
                pending_rows.clear()

    if pending_rows:
        upsert_drift(pending_rows)

    flagged = [s for s, v in drift.items() if v["flag"] in ("drift", "gone")]
    drifted = sum(1 for v in drift.values() if v["flag"] == "drift")
    gone = sum(1 for v in drift.values() if v["flag"] == "gone")
    stable = sum(1 for v in drift.values() if v["flag"] == "stable")

    print()
    print(f"total:   {total}")
    print(f"stable:  {stable}")
    print(f"drift:   {drifted} (IoU < {args.iou_threshold})")
    print(f"gone:    {gone} (new model found nothing)")
    print(f"flagged: {len(flagged)}")
    print("report:  stamp_prediction_drift table")


if __name__ == "__main__":
    main()
