"""
Compare TrOCR vs Tesseract on stamp crops from the detection pipeline.
Runs both engines on each crop and reports results side-by-side.
"""

import json
import re
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import pytesseract


def load_trocr():
    """Load TrOCR model and processor (cached after first call)."""
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    return processor, model


def trocr_read(image: Image.Image, processor, model) -> str:
    """Run TrOCR on a PIL image, return predicted text."""
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, max_new_tokens=32)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def tesseract_read(image_path: str) -> str:
    """Run Tesseract on an image file, return predicted text."""
    import pytesseract
    img = Image.open(image_path)
    # PSM 7 = single text line, which is what date stamps are
    text = pytesseract.image_to_string(img, config="--psm 7 -c tessedit_char_whitelist=0123456789/ .-")
    return text.strip()


def looks_like_date(text: str) -> bool:
    """Check if OCR output resembles a date stamp."""
    # Remove common OCR noise
    cleaned = text.strip().replace("\n", " ")
    if len(cleaned) < 3:
        return False
    # Must contain at least 2 digits
    digits = re.findall(r"\d", cleaned)
    if len(digits) < 2:
        return False
    # Common date patterns: M/D/YY, M D 'YY, MM-DD-YY, etc.
    date_patterns = [
        r"\d{1,2}\s+\d{1,2}\s*'\s*\d{2}",     # 10 3 '99
        r"\d{1,2}/\d{1,2}/\d{2,4}",              # 10/3/99
        r"\d{1,2}-\d{1,2}-\d{2,4}",              # 10-3-99
        r"\d{1,2}\.\d{1,2}\.\d{2,4}",            # 10.3.99
        r"\d{1,2}\s+\d{1,2}\s+\d{2,4}",          # 10 3 99
    ]
    for pat in date_patterns:
        if re.search(pat, cleaned):
            return True
    return False


def isolate_stamp_text(crop_path: str) -> np.ndarray:
    """Better preprocessing: isolate orange digits via HSV, extract V channel."""
    img = cv2.imread(crop_path)
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 60, 80]), np.array([30, 255, 255]))
    mask |= cv2.inRange(hsv, np.array([155, 60, 80]), np.array([180, 255, 255]))
    v = hsv[:, :, 2]
    isolated = np.where(mask > 0, v, 0).astype(np.uint8)
    inverted = 255 - isolated  # black digits on white
    if inverted.shape[1] < 384:
        scale = 384 / inverted.shape[1]
        inverted = cv2.resize(inverted, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(inverted, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)


def run_comparison(debug_dir: str, limit: int | None = None):
    """Run TrOCR and Tesseract on all crops in a debug directory."""
    debug_dir = Path(debug_dir)
    results_path = debug_dir / "detection_results.json"

    with open(results_path) as f:
        detections = json.load(f)

    found = [d for d in detections if d.get("found") and d.get("ocr_crop_path")]
    if limit:
        found = found[:limit]

    print(f"Running OCR comparison on {len(found)} crops...")
    print(f"Loading TrOCR model...")
    processor, model = load_trocr()
    print(f"Model loaded.\n")

    results = []
    for i, det in enumerate(found):
        filename = det["filename"]
        crop_path = det.get("debug_crop_path", "")

        # TrOCR on raw crop (upscaled)
        trocr_raw = ""
        if crop_path and Path(crop_path).exists():
            crop_img = Image.open(crop_path).convert("RGB")
            trocr_raw = trocr_read(crop_img, processor, model)

        # TrOCR on isolated orange text
        trocr_iso = ""
        if crop_path and Path(crop_path).exists():
            iso = isolate_stamp_text(crop_path)
            if iso is not None:
                trocr_iso = trocr_read(Image.fromarray(iso), processor, model)

        # Tesseract on isolated text
        tess_text = ""
        if crop_path and Path(crop_path).exists():
            iso = isolate_stamp_text(crop_path)
            if iso is not None:
                tess_text = pytesseract.image_to_string(
                    Image.fromarray(iso), config="--psm 7").strip()

        result = {
            "filename": filename,
            "position": det.get("position"),
            "sensitivity": det.get("sensitivity"),
            "trocr_raw": trocr_raw,
            "trocr_isolated": trocr_iso,
            "tesseract": tess_text,
            "trocr_raw_is_date": looks_like_date(trocr_raw),
            "trocr_iso_is_date": looks_like_date(trocr_iso),
            "tess_is_date": looks_like_date(tess_text),
        }
        results.append(result)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(found)}")

    # Print comparison table
    print(f"\n{'File':>16} | {'TrOCR (raw)':>20} | {'TrOCR (isolated)':>20} | {'Tesseract':>20} | Date?")
    print("-" * 110)
    raw_dates = 0
    iso_dates = 0
    tess_dates = 0
    for r in results:
        tr = r["trocr_raw"][:20]
        ti = r["trocr_isolated"][:20]
        ts = r["tesseract"][:20].replace("\n", " ")
        marks = (f"{'R' if r['trocr_raw_is_date'] else '.'}"
                 f"{'I' if r['trocr_iso_is_date'] else '.'}"
                 f"{'T' if r['tess_is_date'] else '.'}")
        print(f"{r['filename']:>16} | {tr:>20} | {ti:>20} | {ts:>20} | {marks}")
        raw_dates += r["trocr_raw_is_date"]
        iso_dates += r["trocr_iso_is_date"]
        tess_dates += r["tess_is_date"]

    print(f"\nDate-like outputs: TrOCR(raw)={raw_dates}/{len(results)}, "
          f"TrOCR(isolated)={iso_dates}/{len(results)}, "
          f"Tesseract={tess_dates}/{len(results)}")

    # Save results
    out_path = debug_dir / "ocr_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

    return results


if __name__ == "__main__":
    import sys
    debug_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/will/photo_project/debug_stamps_v5"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run_comparison(debug_dir, limit)
