# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pillow>=12.0",
#     "numpy",
# ]
# ///
"""Generate hard-case augmentations from labeled images.

Takes images with known bounding box labels and creates augmented copies
that mimic the failure modes identified in model analysis:
- Bright backgrounds (where stamps wash out)
- Color temperature shifts (warm/cool tint)
- Low contrast (faded scans)
- Overexposure (blown highlights)

Bounding box labels are copied unchanged since all transforms are global
(no spatial distortion).

Usage:
    uv run augment_hard_cases.py                        # augment all labeled images
    uv run augment_hard_cases.py --limit 200            # augment first 200
    uv run augment_hard_cases.py --preview d1_00000437  # preview augments for one image
    uv run augment_hard_cases.py --clean                # remove all augmented files
"""

import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).parent
SCANMYPHOTOS_DIR = BASE_DIR / "scanmyphotos"
LABELS_DIR = BASE_DIR / "dataset" / "labels"
AUG_IMAGES_DIR = BASE_DIR / "dataset" / "augmented"
AUG_LABELS_DIR = BASE_DIR / "dataset" / "augmented_labels"

AUG_PREFIX = "aug_"

# Pre-computed gamma LUT (avoids per-pixel float math)
_GAMMA_LUT = np.empty(256, dtype=np.uint8)
for _i in range(256):
    _GAMMA_LUT[_i] = int(round(255.0 * (_i / 255.0) ** 0.6))


def apply_all_augmentations(arr, aug_names):
    """Apply all requested augmentations to a numpy array (H,W,3 uint8).

    Returns dict of {suffix: augmented_array}. Operates entirely in numpy
    to avoid repeated PIL<->numpy conversions.
    """
    results = {}
    # Pre-compute float32 array once for transforms that need it
    need_float = {"warm", "cool"} & set(aug_names)
    arr_f = arr.astype(np.float32) if need_float else None

    for name in aug_names:
        if name == "bright":
            # Brightness 1.6x: blend toward white
            out = np.clip(arr.astype(np.int16) * 160 // 100, 0, 255).astype(np.uint8)
        elif name == "vbright":
            # Brightness 2.0x
            out = np.clip(arr.astype(np.int16) * 2, 0, 255).astype(np.uint8)
        elif name == "lowcon":
            # Contrast 0.5x: blend toward mean gray
            mean = arr.mean(dtype=np.float32)
            out = np.clip((arr.astype(np.float32) - mean) * 0.5 + mean, 0, 255).astype(np.uint8)
        elif name == "brlowcon":
            # Brightness 1.5x then contrast 0.6x
            bright = np.clip(arr.astype(np.int16) * 150 // 100, 0, 255).astype(np.float32)
            mean = bright.mean()
            out = np.clip((bright - mean) * 0.6 + mean, 0, 255).astype(np.uint8)
        elif name == "warm":
            out = np.empty_like(arr)
            out[:, :, 0] = np.clip(arr_f[:, :, 0] * 1.12, 0, 255)
            out[:, :, 1] = np.clip(arr_f[:, :, 1] * 1.05, 0, 255)
            out[:, :, 2] = np.clip(arr_f[:, :, 2] * 0.85, 0, 255)
        elif name == "cool":
            out = np.empty_like(arr)
            out[:, :, 0] = np.clip(arr_f[:, :, 0] * 0.88, 0, 255)
            out[:, :, 1] = arr[:, :, 1]
            out[:, :, 2] = np.clip(arr_f[:, :, 2] * 1.15, 0, 255)
        elif name == "gamma":
            # LUT-based gamma -- no float math per pixel
            out = _GAMMA_LUT[arr]
        else:
            continue
        results[name] = out

    return results


def process_stem(stem, aug_names, img_dir, lbl_dir, out_img_dir, out_lbl_dir):
    """Process a single image: read once, apply all augments, write all outputs."""
    src_img = img_dir / f"{stem}.jpg"
    src_lbl = lbl_dir / f"{stem}.txt"

    if not src_img.exists() or not src_lbl.exists():
        return 0

    img = Image.open(src_img)
    arr = np.array(img)
    label_text = src_lbl.read_bytes()

    augmented = apply_all_augmentations(arr, aug_names)

    count = 0
    for suffix, aug_arr in augmented.items():
        aug_stem = f"{AUG_PREFIX}{suffix}_{stem}"
        out_path = out_img_dir / f"{aug_stem}.jpg"
        lbl_path = out_lbl_dir / f"{aug_stem}.txt"

        Image.fromarray(aug_arr).save(out_path, "JPEG", quality=85)
        lbl_path.write_bytes(label_text)
        count += 1

    return count


def get_labeled_stems():
    """Get stems that have both a label file and a source image."""
    label_stems = {p.stem for p in LABELS_DIR.glob("*.txt") if p.stem.startswith("d")}
    image_stems = {p.stem for p in SCANMYPHOTOS_DIR.glob("*.jpg")}
    return sorted(label_stems & image_stems)


def clean():
    """Remove all augmented files."""
    removed = 0
    for d in [AUG_IMAGES_DIR, AUG_LABELS_DIR]:
        if d.exists():
            for f in d.iterdir():
                f.unlink()
                removed += 1
            d.rmdir()
    print(f"Removed {removed} augmented files.")


ALL_AUGMENTATIONS = ["bright", "vbright", "lowcon", "brlowcon", "warm", "cool", "gamma"]
DEFAULT_AUGMENTATIONS = "bright,vbright,brlowcon,warm,gamma"


def main():
    parser = argparse.ArgumentParser(description="Generate hard-case augmentations")
    parser.add_argument("--limit", type=int, help="Max images to augment")
    parser.add_argument("--preview", type=str, help="Preview augmentations for one stem")
    parser.add_argument("--clean", action="store_true", help="Remove all augmented files")
    parser.add_argument("--augs", type=str, default=DEFAULT_AUGMENTATIONS,
                        help=f"Comma-separated augmentation names (default: {DEFAULT_AUGMENTATIONS})")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4),
                        help="Number of parallel workers (default: min(8, cpu_count))")
    args = parser.parse_args()

    if args.clean:
        clean()
        return

    aug_names = [a for a in args.augs.split(",") if a in ALL_AUGMENTATIONS]
    if not aug_names:
        print(f"No valid augmentations in: {args.augs}")
        print(f"Available: {', '.join(ALL_AUGMENTATIONS)}")
        return

    if args.preview:
        print(f"Preview augmentations for {args.preview}:")
        for name in aug_names:
            print(f"  {AUG_PREFIX}{name}_{args.preview}")
        return

    stems = get_labeled_stems()
    if args.limit:
        stems = stems[:args.limit]

    if not stems:
        print("No labeled images found.")
        return

    AUG_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    AUG_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    total = len(stems) * len(aug_names)
    print(f"Generating {len(aug_names)} augmentations x {len(stems)} images = {total} augmented images")
    print(f"Augmentations: {', '.join(aug_names)}")
    print(f"Workers: {args.workers}")
    print()

    created = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_stem, stem, aug_names,
                SCANMYPHOTOS_DIR, LABELS_DIR, AUG_IMAGES_DIR, AUG_LABELS_DIR,
            ): stem
            for stem in stems
        }
        done = 0
        for future in as_completed(futures):
            created += future.result()
            done += 1
            if done % 200 == 0 or done == len(stems):
                print(f"  [{done}/{len(stems)}] {created} augmented images created")

    print(f"\nDone. {created} augmented images in {AUG_IMAGES_DIR}")


if __name__ == "__main__":
    main()
