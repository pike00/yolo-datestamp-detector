#!/usr/bin/env python3
# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "ultralytics",
#   "opencv-python",
#   "pytesseract",
#   "transformers",
#   "torch",
#   "Pillow",
# ]
# ///
"""
Extract date stamps from scanned photos using YOLO detection + OCR.

Pipeline:
1. YOLO detects stamp bounding boxes
2. Crop the stamp region
3. Run Tesseract and TrOCR on the crop
4. Parse dates from OCR output
"""
import re
import json
import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

# ── YOLO detection ──────────────────────────────────────────────────

def detect_stamps(model, image_path, conf=0.3, imgsz=384):
    """Run YOLO inference and return list of (x1, y1, x2, y2, conf) tuples."""
    results = model(str(image_path), imgsz=imgsz, conf=conf, device="cpu", verbose=False)
    boxes = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        c = float(box.conf[0])
        boxes.append((int(x1), int(y1), int(x2), int(y2), c))
    return boxes


def crop_stamp(image_path, box, pad_pct=0.1):
    """Crop stamp region from image with padding. Returns PIL Image."""
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    x1, y1, x2, y2, _ = box
    # Add padding
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad_pct), int(bh * pad_pct)
    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(w, x2 + px)
    y2 = min(h, y2 + py)
    crop = img[y1:y2, x1:x2]
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


# ── OCR engines ─────────────────────────────────────────────────────

def ocr_tesseract(crop_pil):
    """Run Tesseract on a PIL crop. Returns raw text."""
    import pytesseract
    # Convert to grayscale, threshold for better OCR
    gray = cv2.cvtColor(np.array(crop_pil), cv2.COLOR_RGB2GRAY)
    # Adaptive threshold to handle varying stamp brightness
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Also try inverted
    inverted = cv2.bitwise_not(binary)

    results = []
    for img in [gray, binary, inverted]:
        text = pytesseract.image_to_string(
            img,
            config="--psm 7 -c tessedit_char_whitelist=0123456789/ ",
        ).strip()
        if text:
            results.append(text)
    return results


def ocr_trocr(crop_pil, processor, model):
    """Run TrOCR on a PIL crop. Returns text."""
    pixel_values = processor(images=crop_pil, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, max_new_tokens=20)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()


# ── Date parsing ────────────────────────────────────────────────────

DATE_PATTERNS_MD_FIRST = [
    # M D 'YY or MM DD 'YY (with any quote character)
    re.compile(r"(\d{1,2})\s+(\d{1,2})\s*['\u2018\u2019\u201c\u201d\"\x60](\d{2})"),
    # M/D/YY or MM/DD/YY
    re.compile(r"(\d{1,2})/(\d{1,2})/(\d{2,4})"),
    # M-D-YY
    re.compile(r"(\d{1,2})-(\d{1,2})-(\d{2,4})"),
    # M.D'YY (TrOCR sometimes uses dots)
    re.compile(r"(\d{1,2})\.(\d{1,2})['\u2018\u2019\u201c\u201d\"\x60](\d{2})"),
    # M D YY (no quote, just digits and spaces)
    re.compile(r"(\d{1,2})\s+(\d{1,2})\s+(\d{2,4})"),
    # MMDDYY run together (6 digits)
    re.compile(r"(\d{1,2})(\d{2})(\d{2})$"),
]

DATE_PATTERNS_YR_FIRST = [
    # 'YY M D (year-first with quote prefix)
    re.compile(r"['\u2018\u2019\u201c\u201d\"\x60](\d{2})\s+(\d{1,2})\s+(\d{1,2})"),
    # "YY M D (with double quote prefix like TrOCR sometimes outputs)
    re.compile(r"[\"'\u201c\u201d](\d{2})\s+(\d{1,2})\s+(\d{1,2})"),
]


def parse_date(text):
    """Try to extract a date from OCR text. Returns (month, day, year) or None."""
    # Normalize unicode quotes/chars
    normalized = text.replace("\u2018", "'").replace("\u2019", "'")
    normalized = normalized.replace("\u201c", '"').replace("\u201d", '"')

    # Try year-first patterns (common in TrOCR output like "'94 10 15")
    for pattern in DATE_PATTERNS_YR_FIRST:
        m = pattern.search(normalized)
        if m:
            year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if year < 100:
                year += 1900 if year > 50 else 2000
            if 1 <= month <= 12 and 1 <= day <= 31 and 1980 <= year <= 2015:
                return (month, day, year)

    # Try month-day-year patterns
    for pattern in DATE_PATTERNS_MD_FIRST:
        m = pattern.search(normalized)
        if m:
            month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if year < 100:
                year += 1900 if year > 50 else 2000
            if 1 <= month <= 12 and 1 <= day <= 31 and 1980 <= year <= 2015:
                return (month, day, year)
            # Try swapped month/day
            if 1 <= day <= 12 and 1 <= month <= 31 and 1980 <= year <= 2015:
                return (day, month, year)

    # Try extracting from Tesseract's digit-only output (e.g. "62893" -> 6/28/93)
    digits = re.sub(r"[^\d]", "", normalized)
    if 5 <= len(digits) <= 6:
        # Try parsing as MDDYY or MMDDYY
        for split in [(1, 3), (2, 4), (1, 2), (2, 3)]:
            try:
                month = int(digits[:split[0]])
                day = int(digits[split[0]:split[1]])
                year = int(digits[split[1]:])
                if year < 100:
                    year += 1900 if year > 50 else 2000
                if 1 <= month <= 12 and 1 <= day <= 31 and 1980 <= year <= 2015:
                    return (month, day, year)
            except (ValueError, IndexError):
                continue

    return None


def format_date(parsed):
    """Format parsed date tuple as YYYY-MM-DD."""
    if parsed is None:
        return None
    month, day, year = parsed
    return f"{year:04d}-{month:02d}-{day:02d}"


# ── Main pipeline ──────────────────────────────────────────────────

def main():
    from ultralytics import YOLO

    model_path = Path("yolo_finetune/runs/detect/train/weights/best.pt")
    sample_dir = Path("photo_mapping_samples")
    output_file = Path("ocr_results.json")

    if not model_path.exists():
        print(f"Model not found at {model_path}")
        sys.exit(1)

    print("Loading YOLO model...")
    yolo = YOLO(str(model_path))

    print("Loading TrOCR model (first run downloads ~350MB)...")
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    trocr_proc = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    print("Models loaded.\n")

    images = sorted(sample_dir.glob("*.jpg"))
    # Filter out _boxes files
    images = [p for p in images if "_boxes" not in p.name]
    print(f"Processing {len(images)} images...\n")

    results = []
    detected_count = 0
    parsed_count = 0

    for idx, img_path in enumerate(images, 1):
        boxes = detect_stamps(yolo, img_path)

        if not boxes:
            results.append({
                "file": img_path.name,
                "stamp_detected": False,
            })
            print(f"[{idx:3d}/{len(images)}] {img_path.name:25s}  -- no stamp")
            continue

        detected_count += 1
        # Use highest confidence box
        best_box = max(boxes, key=lambda b: b[4])
        crop = crop_stamp(img_path, best_box)

        # Run both OCR engines
        tess_texts = ocr_tesseract(crop)
        trocr_text = ocr_trocr(crop, trocr_proc, trocr_model)

        # Try to parse dates -- prefer TrOCR over Tesseract
        trocr_date = parse_date(trocr_text)
        tess_dates = [parse_date(t) for t in tess_texts]
        tess_dates = [(d, t) for d, t in zip(tess_dates, tess_texts) if d]

        best_date = None
        best_source = None
        if trocr_date:
            best_date = trocr_date
            best_source = f"trocr: {trocr_text}"
            parsed_count += 1
        elif tess_dates:
            best_date = tess_dates[0][0]
            best_source = f"tesseract: {tess_dates[0][1]}"
            parsed_count += 1

        result = {
            "file": img_path.name,
            "stamp_detected": True,
            "confidence": best_box[4],
            "bbox": list(best_box[:4]),
            "tesseract_raw": tess_texts,
            "trocr_raw": trocr_text,
            "parsed_date": format_date(best_date),
            "date_source": best_source,
        }
        results.append(result)

        date_str = format_date(best_date) or "UNPARSED"
        conf_str = f"conf={best_box[4]:.2f}"
        tess_preview = tess_texts[0] if tess_texts else "-"
        print(
            f"[{idx:3d}/{len(images)}] {img_path.name:25s}  "
            f"{conf_str}  tess='{tess_preview}'  trocr='{trocr_text}'  => {date_str}"
        )

    # Summary
    print(f"\n{'='*60}")
    print(f"Stamps detected: {detected_count}/{len(images)}")
    print(f"Dates parsed:    {parsed_count}/{detected_count} detected stamps")
    print(f"Overall yield:   {parsed_count}/{len(images)} images have extracted dates")
    print(f"{'='*60}")

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
