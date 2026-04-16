"""
Stamp detection and cropping for scanned photos with camera date imprints.
Detects orange/red/amber date stamps in photo edges by finding horizontal
clusters of digit-sized contours, not just warm-colored blobs.
"""

import cv2
import numpy as np
from pathlib import Path
import json


def _warm_mask(hsv: np.ndarray, s_min: int = 100, v_min: int = 150) -> np.ndarray:
    """Create a binary mask of warm (orange/red/amber) pixels."""
    m1 = cv2.inRange(hsv, np.array([0, s_min, v_min]), np.array([25, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([160, s_min, v_min]), np.array([180, 255, 255]))
    return m1 | m2


def _find_digit_cluster(mask: np.ndarray, region_h: int, region_w: int) -> dict | None:
    """
    Find a horizontal cluster of digit-sized contours in a binary mask.
    Returns the best cluster as {bbox, count, aspect, score} or None.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Collect digit-sized contours.
    digits = []
    for c in contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area < 5:
            continue
        max_w = min(80, region_w * 0.08)
        max_h = min(80, region_h * 0.7)
        if 3 <= bw <= max_w and 3 <= bh <= max_h:
            digits.append((bx, by, bw, bh, area))

    if len(digits) < 3:
        return None

    # Group digits by horizontal alignment: similar y-center
    best_cluster = None
    best_score = 0

    for anchor in digits:
        ay_center = anchor[1] + anchor[3] // 2
        a_h = anchor[3]
        tolerance = max(a_h * 0.7, 12)

        group = []
        for d in digits:
            dy_center = d[1] + d[3] // 2
            if abs(dy_center - ay_center) <= tolerance:
                # Heights should be roughly similar (within 3x)
                h_ratio = max(d[3], a_h) / max(min(d[3], a_h), 1)
                if h_ratio < 3.0:
                    group.append(d)

        if len(group) < 3:
            continue

        # X-continuity: split into sub-groups where gaps are too large.
        # Date stamps have wide spaces between components (e.g. "10  3  '99").
        # Measured gaps: 59-76px even for small digits (median_w=5).
        # Use generous gap to avoid splitting real stamps.
        group.sort(key=lambda d: d[0])
        widths = [d[2] for d in group]
        median_w = sorted(widths)[len(widths) // 2]
        max_gap = max(median_w * 8, 100)

        sub_groups = [[group[0]]]
        for d in group[1:]:
            prev = sub_groups[-1][-1]
            gap = d[0] - (prev[0] + prev[2])
            if gap > max_gap:
                sub_groups.append([d])
            else:
                sub_groups[-1].append(d)

        # Evaluate each contiguous sub-group
        for sg in sub_groups:
            if len(sg) < 3:
                continue

            gx1 = min(d[0] for d in sg)
            gy1 = min(d[1] for d in sg)
            gx2 = max(d[0] + d[2] for d in sg)
            gy2 = max(d[1] + d[3] for d in sg)
            gw = gx2 - gx1
            gh = gy2 - gy1
            aspect = gw / max(gh, 1)

            # Stamp cluster constraints
            if aspect < 2.0 or aspect > 12.0:
                continue
            if gw > 300 or gw > region_w * 0.30:
                continue
            if gh > region_h * 0.15 and gh > 70:
                continue

            total_digit_area = sum(d[4] for d in sg)
            fill_ratio = total_digit_area / max(gw * gh, 1)
            if fill_ratio > 0.7 or fill_ratio < 0.02:
                continue
            if total_digit_area < 80:
                continue

            # Per-digit fill: real LED digits are well-filled (0.3-0.6)
            per_digit_fills = [d[4] / max(d[2] * d[3], 1) for d in sg]
            mean_digit_fill = np.mean(per_digit_fills)
            if mean_digit_fill < 0.25:
                continue

            # Y-spread: digits sit on the same baseline
            y_centers = [d[1] + d[3] // 2 for d in sg]
            y_range = max(y_centers) - min(y_centers)
            mean_h = np.mean([d[3] for d in sg])
            if y_range > mean_h * 2.0:
                continue

            score = len(sg) * aspect * (1 + mean_digit_fill)
            if score > best_score:
                best_score = score
                best_cluster = {
                    "bbox": (gx1, gy1, gw, gh),
                    "count": len(sg),
                    "aspect": round(aspect, 2),
                    "score": round(score, 2),
                    "fill_ratio": round(fill_ratio, 3),
                    "total_area": int(total_digit_area),
                    "digit_fill": round(mean_digit_fill, 3),
                    "y_range": int(y_range),
                }

    return best_cluster


def detect_stamp_region(image_path: str, debug_dir: str | None = None) -> dict:
    """
    Detect and crop a camera date stamp from a scanned photo.

    Strategy: search edge regions for horizontal clusters of small, bright,
    warm-colored contours (LED/LCD digit shapes). This avoids false positives
    from warm-colored objects (furniture, skin, toys, film edge artifacts).

    Returns dict with:
      - found: bool
      - crop: numpy array of the cropped stamp region (or None)
      - position: where the stamp was found
      - digit_count: number of digit-like contours in the cluster
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"found": False, "error": f"Could not read {image_path}"}

    h, w = img.shape[:2]

    # Edge regions to search — bottom/top 10%, sides 10% (for rotated photos)
    edge_pct = 0.10
    regions = [
        ("bottom-right", int(h * (1 - edge_pct)), h, int(w * 0.5), w),
        ("bottom-left", int(h * (1 - edge_pct)), h, 0, int(w * 0.5)),
        ("bottom-full", int(h * (1 - edge_pct)), h, 0, w),
        ("top-right", 0, int(h * edge_pct), int(w * 0.5), w),
        ("top-left", 0, int(h * edge_pct), 0, int(w * 0.5)),
        ("right-bottom", int(h * 0.5), h, int(w * (1 - edge_pct)), w),
        ("left-bottom", int(h * 0.5), h, 0, int(w * edge_pct)),
        ("right-top", 0, int(h * 0.5), int(w * (1 - edge_pct)), w),
        ("left-top", 0, int(h * 0.5), 0, int(w * edge_pct)),
    ]

    best_result = None
    best_score = 0

    for name, y1, y2, x1, x2 in regions:
        region = img[y1:y2, x1:x2]
        rh, rw = region.shape[:2]
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Try two sensitivity levels — tight first, then moderate.
        # Loose (S>=70) catches more stamps but has unacceptable FP rate;
        # those cases are better handled by the LLM second pass.
        for s_min, v_min, label in [(120, 160, "tight"), (100, 150, "moderate")]:
            mask = _warm_mask(hsv, s_min, v_min)
            if cv2.countNonZero(mask) < 20:
                continue

            cluster = _find_digit_cluster(mask, rh, rw)
            if cluster is None:
                continue

            score = cluster["score"]
            # Prefer tighter threshold matches (more confident)
            if label == "tight":
                score *= 1.5

            if score > best_score:
                bx, by, bw, bh = cluster["bbox"]
                # Pad the crop for OCR context
                pad_x = max(15, int(bw * 0.3))
                pad_y = max(10, int(bh * 0.5))
                crop_y1 = max(0, by - pad_y)
                crop_y2 = min(rh, by + bh + pad_y)
                crop_x1 = max(0, bx - pad_x)
                crop_x2 = min(rw, bx + bw + pad_x)

                crop = region[crop_y1:crop_y2, crop_x1:crop_x2]

                best_result = {
                    "found": True,
                    "crop": crop,
                    "position": name,
                    "digit_count": cluster["count"],
                    "cluster_aspect": cluster["aspect"],
                    "cluster_fill": cluster["fill_ratio"],
                    "cluster_area": cluster["total_area"],
                    "sensitivity": label,
                    "bbox_in_region": (int(bx), int(by), int(bw), int(bh)),
                    "abs_bbox": (x1 + bx, y1 + by, bw, bh),
                }
                best_score = score

    if best_result is None:
        return {"found": False}

    # Contrast check: real stamps are bright digits on a dark background.
    # Reject if the background is as bright as or brighter than the digits.
    crop = best_result["crop"]
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    warm = _warm_mask(hsv_crop, s_min=60, v_min=80)
    bg = cv2.bitwise_not(warm)
    v_ch = hsv_crop[:, :, 2]
    bg_vals = v_ch[bg > 0]
    fg_vals = v_ch[warm > 0]
    if len(fg_vals) > 0 and len(bg_vals) > 0:
        contrast = int(np.median(fg_vals)) - int(np.median(bg_vals))
        best_result["contrast"] = contrast
        if contrast < 30:
            return {"found": False}

    # Save debug visualization if requested
    if debug_dir and best_result["found"]:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem

        crop_path = str(debug_path / f"{stem}_crop.png")
        cv2.imwrite(crop_path, best_result["crop"])

        debug_img = img.copy()
        ax, ay, aw, ah = best_result["abs_bbox"]
        # Use same padding as the crop so the box matches what gets sent to OCR
        pad_x = max(15, int(aw * 0.3))
        pad_y = max(10, int(ah * 0.5))
        cv2.rectangle(debug_img, (ax - pad_x, ay - pad_y), (ax + aw + pad_x, ay + ah + pad_y), (0, 255, 0), 3)
        anno_path = str(debug_path / f"{stem}_annotated.jpg")
        cv2.imwrite(anno_path, debug_img)

        best_result["debug_crop_path"] = crop_path
        best_result["debug_annotated_path"] = anno_path

    return best_result


def prepare_crop_for_ocr(crop: np.ndarray) -> np.ndarray:
    """
    Process a stamp crop to maximize OCR readability.
    Returns a clean binary image suitable for TrOCR or Tesseract.
    """
    if crop is None or crop.size == 0:
        return None

    # Upscale small crops for better OCR (TrOCR works best at ~384px wide)
    h, w = crop.shape[:2]
    if w < 200:
        scale = 384 / w
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Isolate the orange/warm text
    masks = [
        cv2.inRange(hsv, np.array([0, 50, 100]), np.array([30, 255, 255])),
        cv2.inRange(hsv, np.array([155, 50, 100]), np.array([180, 255, 255])),
    ]
    mask = masks[0] | masks[1]

    # Dilate slightly to fill gaps in digits
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Create black text on white background (what OCR expects)
    binary = 255 - mask

    # Add white border padding
    padded = cv2.copyMakeBorder(binary, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

    return padded


def ocr_trocr(crop: np.ndarray, processor=None, model=None) -> str:
    """Run TrOCR on a stamp crop. Returns recognized text."""
    from PIL import Image

    if processor is None or model is None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
        model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')

    # TrOCR works on RGB PIL images — use the raw crop (not binarized)
    if len(crop.shape) == 2:
        pil_img = Image.fromarray(crop)
    else:
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    # Upscale small crops
    w, h = pil_img.size
    if w < 200:
        scale = 384 / w
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, max_new_tokens=20)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()


def ocr_tesseract(crop: np.ndarray) -> str:
    """Run Tesseract on a binarized stamp crop. Returns recognized text."""
    import subprocess, tempfile
    binary = prepare_crop_for_ocr(crop)
    if binary is None:
        return ""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        cv2.imwrite(f.name, binary)
        result = subprocess.run(
            ["tesseract", f.name, "stdout", "--psm", "7", "-c", "tessedit_char_whitelist=0123456789/' "],
            capture_output=True, text=True, timeout=10,
        )
        Path(f.name).unlink(missing_ok=True)
    return result.stdout.strip()


def validate_date_text(text: str) -> dict | None:
    """
    Check if OCR text matches a date stamp pattern.
    Returns parsed date dict {month, day, year} or None.
    """
    import re
    text = text.strip().replace("'", "'").replace("`", "'")

    # Common patterns: "M D 'YY", "M D YY", "MM DD 'YY", "M/D/YY", etc.
    patterns = [
        r"(\d{1,2})\s+(\d{1,2})\s*['\u2019]?\s*(\d{2,4})",   # 12 3 '99 or 12 3 99
        r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})",             # 12/3/99
        r"(\d{1,2})\s*['\u2019]\s*(\d{1,2})\s*['\u2019]\s*(\d{2,4})",  # 12'3'99
    ]

    for pat in patterns:
        m = re.search(pat, text)
        if m:
            a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
            # Normalize year
            year = c if c > 100 else (1900 + c if c > 50 else 2000 + c)
            if 1980 <= year <= 2015 and 1 <= a <= 12 and 1 <= b <= 31:
                return {"month": a, "day": b, "year": year, "raw": text}
            # Try swapped (DD/MM)
            if 1980 <= year <= 2015 and 1 <= b <= 12 and 1 <= a <= 31:
                return {"month": b, "day": a, "year": year, "raw": text, "ambiguous": True}
    return None


def batch_detect_stamps(image_dir: str, debug_dir: str | None = None, limit: int | None = None) -> list[dict]:
    """Run stamp detection on all images in a directory."""
    image_dir = Path(image_dir)
    results = []

    files = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.JPG"))
    if limit:
        files = files[:limit]

    for i, img_path in enumerate(files):
        result = detect_stamp_region(str(img_path), debug_dir=debug_dir)
        result["file"] = str(img_path)
        result["filename"] = img_path.name

        # Remove numpy array from serializable result
        if "crop" in result:
            crop = result.pop("crop")
            result["has_crop"] = crop is not None
            if crop is not None and debug_dir:
                ocr_ready = prepare_crop_for_ocr(crop)
                if ocr_ready is not None:
                    ocr_path = Path(debug_dir) / f"{img_path.stem}_ocr.png"
                    cv2.imwrite(str(ocr_path), ocr_ready)
                    result["ocr_crop_path"] = str(ocr_path)

        results.append(result)

        if (i + 1) % 25 == 0:
            found = sum(1 for r in results if r.get("found"))
            print(f"  Processed {i+1}/{len(files)} — {found} stamps found so far")

    return results


if __name__ == "__main__":
    import sys

    image_dir = sys.argv[1] if len(sys.argv) > 1 else "/mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/Photos/img/Photos/ScanMyPhotos/Disc 2"
    debug_dir = sys.argv[2] if len(sys.argv) > 2 else "/home/will/photo_project/debug_stamps_v2"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    print(f"Detecting stamps in {image_dir} (limit={limit})...")
    results = batch_detect_stamps(image_dir, debug_dir=debug_dir, limit=limit)

    found = sum(1 for r in results if r.get("found"))
    print(f"\nDone: {found}/{len(results)} stamps detected")

    # Show details of detections
    for r in results:
        if r.get("found"):
            print(f"  {r['filename']:>16}: pos={r['position']}, digits={r.get('digit_count')}, "
                  f"aspect={r.get('cluster_aspect')}, fill={r.get('cluster_fill')}, sens={r.get('sensitivity')}")

    # Save results
    out_path = Path(debug_dir) / "detection_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")
