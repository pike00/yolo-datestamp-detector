# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "opencv-python-headless",
#     "pillow>=12.0",
#     "numpy",
# ]
# ///
"""Contrast enhancement experiments for date stamp OCR.

Tries multiple techniques to isolate orange/red LED digits from photo backgrounds.
Outputs a comparison grid for visual evaluation.

Usage:
    uv run enhance_stamps.py                         # 12 random samples, grid
    uv run enhance_stamps.py --image d1_00000133     # single image, all methods
    uv run enhance_stamps.py --count 20              # 20 random samples
    uv run enhance_stamps.py --method hsv            # single method on all samples
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).parent
CROP_DIR = BASE_DIR / "crop_preview"
OUTPUT_DIR = BASE_DIR / "enhance_preview"


def red_minus_blue_green(img: np.ndarray) -> np.ndarray:
    """Subtract max(blue, green) from red channel. Isolates warm-colored pixels."""
    b, g, r = cv2.split(img)
    diff = cv2.subtract(r, cv2.max(b, g))
    _, binary = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    return binary


def hsv_orange_mask(img: np.ndarray) -> np.ndarray:
    """HSV thresholding for orange/red/amber range."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Orange/amber: H 5-25
    mask = cv2.inRange(hsv, (5, 60, 60), (25, 255, 255))
    # Red low: H 0-5
    mask |= cv2.inRange(hsv, (0, 60, 60), (5, 255, 255))
    # Red high (wraps): H 170-179
    mask |= cv2.inRange(hsv, (170, 60, 60), (179, 255, 255))
    return mask


def clahe_red_channel(img: np.ndarray) -> np.ndarray:
    """CLAHE on red channel only, then threshold."""
    r = img[:, :, 2]  # BGR order
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(r)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def clahe_then_hsv(img: np.ndarray) -> np.ndarray:
    """CLAHE on L channel (LAB), then HSV orange mask."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return hsv_orange_mask(enhanced_bgr)


def adaptive_red(img: np.ndarray) -> np.ndarray:
    """Adaptive threshold on red channel."""
    r = img[:, :, 2]
    binary = cv2.adaptiveThreshold(
        r, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5
    )
    return binary


def combined_pipeline(img: np.ndarray) -> np.ndarray:
    """Best-of: CLAHE + HSV mask + morphological cleanup."""
    # Step 1: CLAHE on LAB L-channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Step 2: HSV color mask
    mask = hsv_orange_mask(enhanced_bgr)

    # Step 3: also add red-minus-blue signal
    b_ch, g_ch, r_ch = cv2.split(enhanced_bgr)
    diff = cv2.subtract(r_ch, cv2.max(b_ch, g_ch))
    _, diff_bin = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_or(mask, diff_bin)

    # Step 4: morphological close to connect digit fragments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Step 5: small dilate to thicken thin strokes
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def red_bg_tuned(img: np.ndarray) -> np.ndarray:
    """Red-minus-bg with CLAHE boost and morphological cleanup.
    Tuned for faint stamps on light backgrounds."""
    # Boost contrast first
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b_ch])
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    b, g, r = cv2.split(enhanced_bgr)
    diff = cv2.subtract(r, cv2.max(b, g))

    # Lower threshold to catch faint stamps
    _, binary = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

    # Clean up small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # Close gaps in digit segments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary


def hsv_tight(img: np.ndarray) -> np.ndarray:
    """HSV mask with tighter saturation/value to reduce skin tone false positives."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Tighter ranges: require higher saturation and value (actual LED glow)
    mask = cv2.inRange(hsv, (5, 100, 120), (25, 255, 255))
    mask |= cv2.inRange(hsv, (0, 100, 120), (5, 255, 255))
    mask |= cv2.inRange(hsv, (170, 100, 120), (179, 255, 255))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


METHODS = {
    "original": None,
    "red-bg": red_minus_blue_green,
    "red-bg+": red_bg_tuned,
    "hsv_mask": hsv_orange_mask,
    "hsv_tight": hsv_tight,
    "clahe+hsv": clahe_then_hsv,
    "combined": combined_pipeline,
}


def process_single(img_path: Path, methods: dict) -> dict[str, np.ndarray]:
    """Run all methods on one image, return name->result."""
    img = cv2.imread(str(img_path))
    if img is None:
        return {}

    # Upscale 2x for better detail
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    results = {"original": img}
    for name, fn in methods.items():
        if fn is None:
            continue
        result = fn(img)
        # Convert single-channel to BGR for grid assembly
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        results[name] = result
    return results


def make_comparison_row(results: dict[str, np.ndarray], stem: str, cell_w: int = 280, cell_h: int = 100) -> np.ndarray:
    """Create a single row showing all methods for one image."""
    method_names = list(results.keys())
    n = len(method_names)

    row = np.ones((cell_h + 25, cell_w * n, 3), dtype=np.uint8) * 255

    for i, name in enumerate(method_names):
        img = results[name]
        # Resize to fit cell
        h, w = img.shape[:2]
        scale = min(cell_w / w, cell_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Center in cell
        x_off = i * cell_w + (cell_w - new_w) // 2
        y_off = (cell_h - new_h) // 2
        row[y_off : y_off + new_h, x_off : x_off + new_w] = resized

        # Label
        label = name if i > 0 else f"{stem}"
        cv2.putText(row, label, (i * cell_w + 5, cell_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    return row


def make_single_comparison(results: dict[str, np.ndarray], stem: str, cell_w: int = 400, cell_h: int = 150) -> np.ndarray:
    """Vertical layout for single-image mode -- larger cells."""
    method_names = list(results.keys())
    n = len(method_names)
    label_h = 30

    canvas = np.ones(((cell_h + label_h) * n, cell_w, 3), dtype=np.uint8) * 255

    for i, name in enumerate(method_names):
        img = results[name]
        h, w = img.shape[:2]
        scale = min(cell_w / w, cell_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        y_base = i * (cell_h + label_h)
        x_off = (cell_w - new_w) // 2
        y_off = y_base + (cell_h - new_h) // 2
        canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized

        cv2.putText(canvas, name, (5, y_base + cell_h + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Stamp contrast enhancement comparison")
    parser.add_argument("--image", type=str, help="Single image stem (e.g. d1_00000133)")
    parser.add_argument("--count", type=int, default=12, help="Number of random samples (default 12)")
    parser.add_argument("--method", type=str, choices=list(METHODS.keys()),
                        help="Show only this method")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    methods = METHODS
    if args.method:
        methods = {"original": None, args.method: METHODS[args.method]}

    if args.image:
        # Single image mode -- vertical comparison
        candidates = list(CROP_DIR.glob(f"{args.image}_*.jpg"))
        if not candidates:
            print(f"No crop found for {args.image}")
            return
        img_path = candidates[0]
        results = process_single(img_path, methods)
        if not results:
            print(f"Failed to read {img_path}")
            return
        canvas = make_single_comparison(results, args.image)
        out = OUTPUT_DIR / f"{args.image}_compare.jpg"
        cv2.imwrite(str(out), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"Saved: {out}")
    else:
        # Grid mode -- multiple images, each as a row
        all_crops = sorted(CROP_DIR.glob("*.jpg"))
        random.seed(args.seed)
        samples = random.sample(all_crops, min(args.count, len(all_crops)))
        samples.sort(key=lambda p: p.stem)

        rows = []
        for img_path in samples:
            stem = img_path.stem
            results = process_single(img_path, methods)
            if results:
                row = make_comparison_row(results, stem)
                rows.append(row)
                print(f"  {stem}: processed")

        if not rows:
            print("No images processed.")
            return

        grid = np.vstack(rows)
        out = OUTPUT_DIR / "comparison_grid.jpg"
        cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"\nSaved grid ({len(rows)} images x {len(methods)} methods): {out}")
        print(f"Methods: {', '.join(methods.keys())}")


if __name__ == "__main__":
    main()
