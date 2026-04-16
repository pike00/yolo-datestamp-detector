#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "Pillow>=10",
#     "psycopg[binary]>=3.1.0",
# ]
# ///
"""Render example drift visualizations for README.

Picks one stable, one drift, one gone example from the stamp_prediction_drift
table and writes cropped old-vs-new box overlays to examples/drift_*.jpg.

The crop is tight around the union of old and new boxes (with padding) so the
rest of the family photo stays private.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from _db import load_drift  # noqa: E402

IMG_DIR = ROOT / "scanmyphotos"
OUT_DIR = ROOT / "examples"

PICKS = {
    "stable": "d2_00000303",
    "drift": "d1_00000917",
    "gone": "d4_00000296",
}

OLD_COLOR = (220, 60, 60)
NEW_COLOR = (60, 200, 120)
PAD_FRAC = 0.6
MIN_CROP_FRAC = 0.28


def yolo_to_abs(box: dict, W: int, H: int) -> tuple[int, int, int, int]:
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    cx, cy = x * W, y * H
    bw, bh = w * W, h * H
    return int(cx - bw / 2), int(cy - bh / 2), int(cx + bw / 2), int(cy + bh / 2)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def render(stem: str, entry: dict, flag: str, out: Path) -> None:
    img = Image.open(IMG_DIR / f"{stem}.jpg").convert("RGB")
    W, H = img.size
    boxes = []
    if entry.get("old"):
        boxes.append(("old", yolo_to_abs(entry["old"], W, H), entry["old"].get("confidence")))
    if entry.get("new"):
        boxes.append(("new", yolo_to_abs(entry["new"], W, H), entry["new"].get("confidence")))

    xs = [b[1][0] for b in boxes] + [b[1][2] for b in boxes]
    ys = [b[1][1] for b in boxes] + [b[1][3] for b in boxes]
    bx0, by0, bx1, by1 = min(xs), min(ys), max(xs), max(ys)
    bw, bh = bx1 - bx0, by1 - by0
    pad_x = max(int(bw * PAD_FRAC), int(W * 0.03))
    pad_y = max(int(bh * PAD_FRAC), int(H * 0.03))
    cx0, cy0 = max(bx0 - pad_x, 0), max(by0 - pad_y, 0)
    cx1, cy1 = min(bx1 + pad_x, W), min(by1 + pad_y, H)

    min_w = int(W * MIN_CROP_FRAC)
    min_h = int(H * MIN_CROP_FRAC)
    if cx1 - cx0 < min_w:
        extra = (min_w - (cx1 - cx0)) // 2
        cx0 = max(cx0 - extra, 0)
        cx1 = min(cx1 + extra, W)
    if cy1 - cy0 < min_h:
        extra = (min_h - (cy1 - cy0)) // 2
        cy0 = max(cy0 - extra, 0)
        cy1 = min(cy1 + extra, H)

    crop = img.crop((cx0, cy0, cx1, cy1)).copy()
    draw = ImageDraw.Draw(crop)
    width = max(3, int(min(crop.size) * 0.008))
    font = load_font(max(14, int(min(crop.size) * 0.035)))

    for kind, (x0, y0, x1, y1), conf in boxes:
        color = OLD_COLOR if kind == "old" else NEW_COLOR
        draw.rectangle(
            (x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0),
            outline=color,
            width=width,
        )
        label = f"{kind} {conf:.2f}" if conf is not None else kind
        ty = (y0 - cy0) - int(font.size * 1.3) - 2
        if ty < 2:
            ty = (y1 - cy0) + 4
        tx = max(2, x0 - cx0)
        tb = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle(
            (tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2),
            fill=color,
        )
        draw.text((tx, ty), label, fill=(255, 255, 255), font=font)

    iou = entry.get("iou")
    if iou is not None:
        legend = f"{flag} | IoU {iou:.2f}" if flag != "gone" else "gone | new model detects nothing"
    else:
        legend = flag
    lb = draw.textbbox((8, 8), legend, font=font)
    draw.rectangle((lb[0] - 4, lb[1] - 4, lb[2] + 4, lb[3] + 4), fill=(0, 0, 0))
    draw.text((8, 8), legend, fill=(255, 255, 255), font=font)

    out.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out, "JPEG", quality=88)
    try:
        rel = out.relative_to(ROOT)
    except ValueError:
        rel = out
    print(f"wrote {rel} ({crop.size[0]}x{crop.size[1]})")


def main() -> None:
    drift = load_drift()
    for flag, stem in PICKS.items():
        entry = drift[stem]
        render(stem, entry, flag, OUT_DIR / f"drift_{flag}.jpg")


if __name__ == "__main__":
    main()
