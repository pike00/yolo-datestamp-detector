# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "pillow>=12.0",
#     "psycopg[binary]>=3.1.0",
# ]
# ///
"""Build the portable VLM bench corpus: 200 stems stratified by YOLO
confidence bucket, pre-cropped to the stamp region.

Writes:
    state/bench/corpus/<stem>.jpg      - full source images (copied)
    state/bench/crops/<stem>.jpg       - padded stamp crops (JPEG)
    state/bench/manifest.json          - stem metadata + stratification info

Usage:
    uv run scripts/ocr/build_bench_corpus.py [--per-bucket 50] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from _db import get_db  # noqa: E402

SCANMYPHOTOS_DIR = BASE_DIR / "scanmyphotos"
BENCH_DIR = BASE_DIR / "state" / "bench"
CORPUS_DIR = BENCH_DIR / "corpus"
CROPS_DIR = BENCH_DIR / "crops"
MANIFEST_PATH = BENCH_DIR / "manifest.json"

PAD_FACTOR = 0.5
CROP_MAX_SIDE = 512

CONFIDENCE_BUCKETS = ["[0.0, 0.3)", "[0.3, 0.6)", "[0.6, 0.85)", "[0.85, 1.0]"]


def assign_bucket(confidence: float) -> str:
    if confidence < 0.3:
        return "[0.0, 0.3)"
    if confidence < 0.6:
        return "[0.3, 0.6)"
    if confidence < 0.85:
        return "[0.6, 0.85)"
    return "[0.85, 1.0]"


def stratified_sample(
    rows: list[tuple[str, float]],
    per_bucket: int,
    seed: int,
) -> tuple[list[tuple[str, float]], dict]:
    """Return (sampled_rows, skew_report).

    `rows` is a list of (stem, confidence). Target is `per_bucket` stems per
    confidence bucket. If a bucket is underfilled, take all it has and
    redistribute the deficit across over-filled buckets round-robin.

    `skew_report` maps bucket name -> {"target": int, "actual": int} for
    every underfilled bucket; empty dict if all buckets hit target.
    """
    rng = random.Random(seed)
    by_bucket: dict[str, list[tuple[str, float]]] = {b: [] for b in CONFIDENCE_BUCKETS}
    for stem, conf in rows:
        by_bucket[assign_bucket(conf)].append((stem, conf))
    for b in CONFIDENCE_BUCKETS:
        rng.shuffle(by_bucket[b])

    target = per_bucket
    picked: dict[str, list[tuple[str, float]]] = {b: [] for b in CONFIDENCE_BUCKETS}
    skew: dict[str, dict] = {}
    deficit = 0
    for b in CONFIDENCE_BUCKETS:
        available = by_bucket[b]
        if len(available) >= target:
            picked[b] = available[:target]
        else:
            picked[b] = available[:]
            deficit += target - len(available)
            skew[b] = {"target": target, "actual": len(available)}

    # Redistribute deficit round-robin across over-filled buckets.
    over_filled = [b for b in CONFIDENCE_BUCKETS if len(by_bucket[b]) > len(picked[b])]
    idx = 0
    while deficit > 0 and over_filled:
        b = over_filled[idx % len(over_filled)]
        remaining_in_bucket = by_bucket[b][len(picked[b]):]
        if remaining_in_bucket:
            picked[b].append(remaining_in_bucket[0])
            deficit -= 1
        else:
            over_filled.remove(b)
            continue
        idx += 1

    flat = [item for b in CONFIDENCE_BUCKETS for item in picked[b]]
    return flat, skew


def load_predictions_from_db() -> list[tuple[str, float, float, float, float, float]]:
    """Return list of (stem, x, y, w, h, confidence) from stamp_predictions."""
    with get_db() as conn:
        return conn.execute(
            """
            SELECT stem, x, y, w, h, confidence
            FROM stamp_predictions
            WHERE w > 0 AND h > 0
            """
        ).fetchall()


def crop_stamp(img: Image.Image, bbox: dict) -> Image.Image:
    """Pad + crop the stamp region using normalized (cx, cy, w, h)."""
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


def build_corpus(per_bucket: int, seed: int) -> dict:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    preds = load_predictions_from_db()
    pred_rows = [(r[0], float(r[5])) for r in preds]
    bbox_map = {r[0]: {"x": float(r[1]), "y": float(r[2]), "w": float(r[3]), "h": float(r[4])} for r in preds}

    sampled, skew = stratified_sample(pred_rows, per_bucket=per_bucket, seed=seed)
    print(f"Sampled {len(sampled)} stems across {len(CONFIDENCE_BUCKETS)} buckets.")
    if skew:
        print(f"Underfilled buckets: {skew}")

    manifest_stems = []
    confidence_lookup = {stem: conf for stem, conf in sampled}

    for stem, conf in sampled:
        src = SCANMYPHOTOS_DIR / f"{stem}.jpg"
        if not src.exists():
            print(f"  MISSING: {src}")
            continue

        dst_source = CORPUS_DIR / f"{stem}.jpg"
        if not dst_source.exists():
            shutil.copy2(src, dst_source)

        bbox = bbox_map[stem]
        img = Image.open(src).convert("RGB")
        crop = crop_stamp(img, bbox)
        if max(crop.size) > CROP_MAX_SIDE:
            crop.thumbnail((CROP_MAX_SIDE, CROP_MAX_SIDE), Image.LANCZOS)
        dst_crop = CROPS_DIR / f"{stem}.jpg"
        crop.save(dst_crop, format="JPEG", quality=90)

        manifest_stems.append(
            {
                "stem": stem,
                "bbox": bbox,
                "bbox_source": "yolo",
                "yolo_confidence": conf,
                "confidence_bucket": assign_bucket(conf),
                "crop_path": str(dst_crop.relative_to(BASE_DIR)),
                "source_path": str(dst_source.relative_to(BASE_DIR)),
            }
        )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "per_bucket_target": per_bucket,
        "stratification": {"confidence_buckets": CONFIDENCE_BUCKETS},
        "skew": skew,
        "ground_truth_frozen": False,
        "stems": manifest_stems,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest: {MANIFEST_PATH}")
    print(f"Corpus:   {CORPUS_DIR}")
    print(f"Crops:    {CROPS_DIR}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build VLM bench corpus")
    parser.add_argument("--per-bucket", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_corpus(per_bucket=args.per_bucket, seed=args.seed)


if __name__ == "__main__":
    main()
