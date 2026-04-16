# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "anthropic>=0.52.0",
#     "pillow>=12.0",
#     "psycopg[binary]>=3.1.0",
# ]
# ///
"""OCR date stamps from scanned photos using Claude Haiku.

Uses human-confirmed bounding boxes from corrections_queue.json when available,
falls back to YOLO predictions in stamp_predictions, then to a bottom-right
quadrant crop. OCR results are upserted into stamp_ocr (model='haiku').

Tracks token usage per-image and writes a cost summary.

Usage:
    uv run ocr_stamps.py                        # process all confirmed stamps
    uv run ocr_stamps.py --limit 10             # process first 10
    uv run ocr_stamps.py --image d1_00000133    # single image by stem
    uv run ocr_stamps.py --resume               # skip already-processed
    uv run ocr_stamps.py --full-image           # send full image, no crop
    uv run ocr_stamps.py --source yolo          # use YOLO predictions only
"""

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import anthropic
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from _db import (  # noqa: E402
    load_ocr_results,
    load_predictions as db_load_predictions,
    upsert_ocr_result,
)

SCANMYPHOTOS_DIR = BASE_DIR / "scanmyphotos"
CORRECTIONS_FILE = BASE_DIR / "state" / "corrections_queue.json"
COST_LOG_FILE = BASE_DIR / "state" / "ocr_cost_log.json"

MODEL = "claude-haiku-4-5-20251001"

# Haiku 4.5 pricing (per million tokens)
PRICE_INPUT_PER_MTOK = 0.80
PRICE_OUTPUT_PER_MTOK = 4.00
PRICE_CACHE_WRITE_PER_MTOK = 1.00
PRICE_CACHE_READ_PER_MTOK = 0.08

PROMPT = """\
Look at this image. It may contain a date stamp — small digits in orange, red, amber, or yellow, typically imprinted by a camera.

Rules:
- If you see a date stamp, respond with ONLY the exact characters visible. No reformatting, no guessing missing digits, no converting to a standard date format.
- Preserve the original spacing, punctuation, and apostrophes exactly as they appear (e.g., "10 3 '99" not "10/3/1999").
- If digits are partially obscured or unclear, use ? for each uncertain character (e.g., "1? 3 '99").
- If there is no date stamp visible, respond with exactly: NONE

Output nothing else — no explanation, no labels, no prefix."""

# Padding multiplier around YOLO bbox for crop
PAD_FACTOR = 0.5


def load_corrections() -> dict[str, dict]:
    """Load human-confirmed bounding boxes from corrections queue.

    Returns dict of stem -> {x, y, w, h, source: "human"} for confirmed stamps.
    """
    if not CORRECTIONS_FILE.exists():
        return {}
    with open(CORRECTIONS_FILE) as f:
        data = json.load(f)
    boxes = {}
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


def load_predictions() -> dict[str, dict]:
    """Load YOLO predictions, tagged with source."""
    raw = db_load_predictions()
    return {k: {**v, "source": "yolo"} for k, v in raw.items()}


def load_bboxes(source: str) -> dict[str, dict]:
    """Build unified bbox map. Human corrections take priority over YOLO."""
    yolo = load_predictions()
    if source == "yolo":
        return yolo
    human = load_corrections()
    merged = {**yolo, **human}  # human overwrites yolo for same stem
    return merged


def load_results() -> dict:
    return load_ocr_results()


def crop_stamp_region(img: Image.Image, pred: dict) -> Image.Image:
    """Crop image to YOLO-predicted stamp region with padding."""
    w_img, h_img = img.size
    cx, cy = pred["x"] * w_img, pred["y"] * h_img
    bw, bh = pred["w"] * w_img, pred["h"] * h_img

    pad_x = bw * PAD_FACTOR
    pad_y = bh * PAD_FACTOR

    x1 = max(0, int(cx - bw / 2 - pad_x))
    y1 = max(0, int(cy - bh / 2 - pad_y))
    x2 = min(w_img, int(cx + bw / 2 + pad_x))
    y2 = min(h_img, int(cy + bh / 2 + pad_y))

    return img.crop((x1, y1, x2, y2))


def crop_bottom_right(img: Image.Image) -> Image.Image:
    """Fallback: crop bottom-right quadrant."""
    w, h = img.size
    return img.crop((w // 2, int(h * 0.75), w, h))


def image_to_base64(img: Image.Image, max_side: int = 512) -> tuple[str, str]:
    """Resize if needed and encode to base64 JPEG. Returns (b64_data, media_type)."""
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode(), "image/jpeg"


def compute_cost(usage) -> dict:
    """Extract token counts and compute cost from API usage object."""
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens
    cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0

    cost = (
        (input_tokens / 1_000_000) * PRICE_INPUT_PER_MTOK
        + (output_tokens / 1_000_000) * PRICE_OUTPUT_PER_MTOK
        + (cache_creation / 1_000_000) * PRICE_CACHE_WRITE_PER_MTOK
        + (cache_read / 1_000_000) * PRICE_CACHE_READ_PER_MTOK
    )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": cache_creation,
        "cache_read_input_tokens": cache_read,
        "cost_usd": round(cost, 8),
    }


def ocr_single(
    client: anthropic.Anthropic,
    img_path: Path,
    pred: dict | None,
    full_image: bool,
) -> tuple[str, dict]:
    """Send one image to Haiku, return (text, usage_dict)."""
    img = Image.open(img_path).convert("RGB")

    if full_image:
        crop = img
    elif pred:
        crop = crop_stamp_region(img, pred)
    else:
        crop = crop_bottom_right(img)

    b64_data, media_type = image_to_base64(crop)

    response = client.messages.create(
        model=MODEL,
        max_tokens=64,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": b64_data},
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
    )

    text = response.content[0].text.strip()
    usage = compute_cost(response.usage)
    return text, usage


def main():
    parser = argparse.ArgumentParser(description="OCR date stamps via Claude Haiku")
    parser.add_argument("--limit", type=int, help="Max images to process")
    parser.add_argument("--image", type=str, help="Single image stem (e.g. d1_00000133)")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed images")
    parser.add_argument("--full-image", action="store_true", help="Send full image instead of crop")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Min YOLO confidence to include")
    parser.add_argument("--source", choices=["auto", "yolo"], default="auto",
                        help="Bbox source: 'auto' prefers human corrections, 'yolo' uses predictions only")
    args = parser.parse_args()

    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
    bboxes = load_bboxes(args.source)
    results = load_results() if args.resume else {}

    # Build work list
    if args.image:
        stems = [args.image]
    else:
        stems = sorted(bboxes.keys())
        if args.min_confidence > 0:
            stems = [s for s in stems if bboxes[s].get("confidence", 1.0) >= args.min_confidence]

    if args.resume:
        stems = [s for s in stems if s not in results]

    if args.limit:
        stems = stems[: args.limit]

    if not stems:
        print("Nothing to process.")
        return

    human_count = sum(1 for s in stems if bboxes.get(s, {}).get("source") == "human")
    yolo_count = len(stems) - human_count
    print(f"Processing {len(stems)} images with model {MODEL}")
    print(f"Bbox sources: {human_count} human-confirmed, {yolo_count} YOLO-only")
    if not args.full_image:
        print("Mode: cropped stamp region (use --full-image to send full image)")
    print()

    totals = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "processed": 0, "stamps_found": 0}
    cost_log = []

    for i, stem in enumerate(stems):
        img_path = SCANMYPHOTOS_DIR / f"{stem}.jpg"
        if not img_path.exists():
            print(f"  SKIP {stem}: file not found")
            continue

        bbox = bboxes.get(stem)

        try:
            text, usage = ocr_single(client, img_path, bbox, args.full_image)
        except anthropic.RateLimitError:
            print(f"  Rate limited at {stem}, saving progress and exiting.")
            break
        except anthropic.APIError as e:
            print(f"  API error on {stem}: {e}")
            continue

        upsert_ocr_result(
            stem,
            text,
            bbox_source=bbox["source"] if bbox else None,
            confidence=bbox.get("confidence") if bbox else None,
            stage=1,
        )

        totals["input_tokens"] += usage["input_tokens"]
        totals["output_tokens"] += usage["output_tokens"]
        totals["cost_usd"] += usage["cost_usd"]
        totals["processed"] += 1
        if text != "NONE":
            totals["stamps_found"] += 1

        cost_log.append({"stem": stem, **usage})

        status = text if text != "NONE" else "-"
        print(f"  [{i + 1}/{len(stems)}] {stem}: {status}  ({usage['input_tokens']}in/{usage['output_tokens']}out/${usage['cost_usd']:.6f})")

        # Respect rate limits: small delay between calls
        time.sleep(0.1)

    with open(COST_LOG_FILE, "w") as f:
        json.dump({"summary": totals, "per_image": cost_log}, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Processed:    {totals['processed']}")
    print(f"Stamps found: {totals['stamps_found']}")
    print(f"Input tokens: {totals['input_tokens']:,}")
    print(f"Output tokens:{totals['output_tokens']:,}")
    print(f"Total cost:   ${totals['cost_usd']:.4f}")
    print(f"Avg cost/img: ${totals['cost_usd'] / max(totals['processed'], 1):.6f}")
    print("\nResults stored in stamp_ocr (model='haiku')")
    print(f"Cost log: {COST_LOG_FILE}")


if __name__ == "__main__":
    main()
