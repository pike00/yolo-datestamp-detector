# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pillow>=10.0",
#     "psycopg[binary]>=3.1.0",
# ]
# ///
"""Re-crop the VLM bench corpus using the current stamp_predictions bboxes.

Preserves the 200-stem selection in state/bench/manifest.json (so any frozen
ground truth stays aligned) — only the crop pixels and per-stem bbox in the
manifest get refreshed.

Writes:
    state/bench/crops/<stem>.jpg   - refreshed JPEG crops (PAD_FACTOR=0.5, max side 512)
    state/bench/manifest.json      - bbox + yolo_confidence updated in-place

Usage:
    uv run scripts/ocr/recrop_bench_from_db.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from _db import get_db  # noqa: E402

SCANMYPHOTOS_DIR = BASE_DIR / "scanmyphotos"
BENCH_DIR = BASE_DIR / "state" / "bench"
CROPS_DIR = BENCH_DIR / "crops"
MANIFEST_PATH = BENCH_DIR / "manifest.json"

PAD_FACTOR = 0.5
CROP_MAX_SIDE = 512


def assign_bucket(confidence: float) -> str:
    if confidence < 0.3:
        return "[0.0, 0.3)"
    if confidence < 0.6:
        return "[0.3, 0.6)"
    if confidence < 0.85:
        return "[0.6, 0.85)"
    return "[0.85, 1.0]"


def crop_stamp(img: Image.Image, bbox: dict) -> Image.Image:
    w_img, h_img = img.size
    cx, cy = bbox["x"] * w_img, bbox["y"] * h_img
    bw, bh = bbox["w"] * w_img, bbox["h"] * h_img
    pad_x = bw * PAD_FACTOR
    pad_y = bh * PAD_FACTOR
    x1 = max(0, int(cx - bw / 2 - pad_x))
    y1 = max(0, int(cy - bh / 2 - pad_y))
    x2 = min(w_img, int(cx + bw / 2 + pad_x))
    y2 = min(h_img, int(cy + bh / 2 + pad_y))
    return img.crop((x1, y1, x2, y2))


def main() -> None:
    mf = json.loads(MANIFEST_PATH.read_text())
    stems = [s["stem"] for s in mf["stems"]]
    print(f"Manifest stems: {len(stems)}")

    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT stem, x, y, w, h, confidence
            FROM stamp_predictions
            WHERE stem = ANY(%s) AND w > 0 AND h > 0
            """,
            (stems,),
        ).fetchall()
    bbox_by_stem = {
        r[0]: {
            "x": float(r[1]),
            "y": float(r[2]),
            "w": float(r[3]),
            "h": float(r[4]),
            "conf": float(r[5]),
        }
        for r in rows
    }
    print(f"Fresh predictions for: {len(bbox_by_stem)}/{len(stems)}")
    missing = [s for s in stems if s not in bbox_by_stem]
    if missing:
        print(f"  missing bboxes (kept old crop + bbox): {len(missing)}")

    refreshed = 0
    for entry in mf["stems"]:
        stem = entry["stem"]
        fresh = bbox_by_stem.get(stem)
        if fresh is None:
            continue
        src = SCANMYPHOTOS_DIR / f"{stem}.jpg"
        if not src.exists():
            print(f"  skip {stem}: source missing")
            continue
        try:
            img = Image.open(src).convert("RGB")
        except Exception as e:
            print(f"  skip {stem}: open failed ({e})")
            continue
        crop = crop_stamp(
            img, {"x": fresh["x"], "y": fresh["y"], "w": fresh["w"], "h": fresh["h"]}
        )
        if max(crop.size) > CROP_MAX_SIDE:
            crop.thumbnail((CROP_MAX_SIDE, CROP_MAX_SIDE), Image.LANCZOS)
        dst = CROPS_DIR / f"{stem}.jpg"
        crop.save(dst, format="JPEG", quality=90)

        entry["bbox"] = {
            "x": fresh["x"], "y": fresh["y"], "w": fresh["w"], "h": fresh["h"],
        }
        entry["yolo_confidence"] = fresh["conf"]
        entry["confidence_bucket"] = assign_bucket(fresh["conf"])
        refreshed += 1

    mf["recropped_at"] = datetime.now(timezone.utc).isoformat()
    mf["recropped_source"] = "stamp_predictions (yolo26m gpu-40ep)"
    MANIFEST_PATH.write_text(json.dumps(mf, indent=2))
    print(f"Refreshed {refreshed}/{len(stems)} crops")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
