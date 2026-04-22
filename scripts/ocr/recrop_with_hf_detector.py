# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "ultralytics>=8.3",
#     "huggingface_hub>=0.20",
#     "pillow>=10.0",
#     "psycopg[binary]>=3.1.0",
# ]
# ///
"""Re-crop the bench corpus using pike00/yolo-date-stamp-detector from HF.

Loads the first N stems from state/bench/manifest.json, runs the HF YOLOv8
detector on the full-resolution source image, and writes padded crops to
state/bench/crops_hf/<stem>.jpg for the VLM bench to consume.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from _db import get_db  # noqa: E402

BENCH_DIR = BASE_DIR / "state" / "bench"
CORPUS_DIR = BENCH_DIR / "corpus"
HF_CROPS_DIR = BENCH_DIR / "crops_hf"
MANIFEST_PATH = BENCH_DIR / "manifest.json"

PAD_FACTOR = 0.5
CROP_MAX_SIDE = 512


def pad_crop(img: Image.Image, x1: float, y1: float, x2: float, y2: float) -> Image.Image:
    w_img, h_img = img.size
    bw = x2 - x1
    bh = y2 - y1
    pad_x = bw * PAD_FACTOR
    pad_y = bh * PAD_FACTOR
    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(w_img, int(x2 + pad_x))
    y2 = min(h_img, int(y2 + pad_y))
    return img.crop((x1, y1, x2, y2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    print("Downloading weights from HF...")
    weights_path = hf_hub_download("pike00/yolo-date-stamp-detector", "best.pt")
    print(f"  {weights_path}")

    model = YOLO(weights_path)

    manifest = json.loads(MANIFEST_PATH.read_text())
    all_stems = [s["stem"] for s in manifest["stems"]]
    with get_db() as conn:
        rows = conn.execute(
            "SELECT stem FROM stamp_ocr WHERE model='sonnet' AND parsed_date IS NOT NULL AND stem = ANY(%s)",
            (all_stems,),
        ).fetchall()
    gt_stems = {r[0] for r in rows}
    stems = [s for s in all_stems if s in gt_stems][: args.n]
    print(f"Selected {len(stems)} stems with Sonnet-parsed ground truth")

    HF_CROPS_DIR.mkdir(parents=True, exist_ok=True)

    results_log = []
    for i, stem in enumerate(stems):
        src = CORPUS_DIR / f"{stem}.jpg"
        if not src.exists():
            print(f"  [{i+1}/{len(stems)}] {stem}: MISSING source")
            continue

        res = model.predict(str(src), conf=args.conf, verbose=False)[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            print(f"  [{i+1}/{len(stems)}] {stem}: NO DETECTION")
            results_log.append({"stem": stem, "boxes": 0})
            continue

        best_idx = int(boxes.conf.argmax())
        xyxy = boxes.xyxy[best_idx].tolist()
        conf = float(boxes.conf[best_idx])

        img = Image.open(src).convert("RGB")
        crop = pad_crop(img, *xyxy)
        if max(crop.size) > CROP_MAX_SIDE:
            crop.thumbnail((CROP_MAX_SIDE, CROP_MAX_SIDE), Image.LANCZOS)
        out = HF_CROPS_DIR / f"{stem}.jpg"
        crop.save(out, format="JPEG", quality=90)
        print(
            f"  [{i+1}/{len(stems)}] {stem}: conf={conf:.2f} "
            f"bbox=({xyxy[0]:.0f},{xyxy[1]:.0f},{xyxy[2]:.0f},{xyxy[3]:.0f})"
        )
        results_log.append(
            {"stem": stem, "boxes": len(boxes), "top_conf": conf, "xyxy": xyxy}
        )

    log_path = HF_CROPS_DIR / "_detections.json"
    log_path.write_text(json.dumps(results_log, indent=2))
    print(f"\nCrops: {HF_CROPS_DIR}")
    print(f"Log:   {log_path}")


if __name__ == "__main__":
    main()
