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
    uv run augment_hard_cases.py                    # augment all labeled images
    uv run augment_hard_cases.py --limit 200        # augment first 200
    uv run augment_hard_cases.py --preview d1_00000437  # preview augments for one image
    uv run augment_hard_cases.py --clean             # remove all augmented files
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

BASE_DIR = Path(__file__).parent
SCANMYPHOTOS_DIR = BASE_DIR / "scanmyphotos"
LABELS_DIR = BASE_DIR / "dataset" / "labels"
AUG_IMAGES_DIR = BASE_DIR / "dataset" / "augmented"
AUG_LABELS_DIR = BASE_DIR / "dataset" / "augmented_labels"

# Prefix for augmented files so they're easy to identify
AUG_PREFIX = "aug_"

# Augmentation definitions: (suffix, transform_function)
# Each produces one augmented image from one source image.
AUGMENTATIONS = {}


def aug_bright(img):
    """Increase brightness to simulate bright/washed-out scans."""
    return ImageEnhance.Brightness(img).enhance(1.6)


def aug_very_bright(img):
    """Strong brightness increase -- mimics overexposed scans."""
    return ImageEnhance.Brightness(img).enhance(2.0)


def aug_low_contrast(img):
    """Reduce contrast -- mimics faded/aged prints."""
    return ImageEnhance.Contrast(img).enhance(0.5)


def aug_bright_low_contrast(img):
    """Bright + low contrast compound -- the hardest failure mode."""
    img = ImageEnhance.Brightness(img).enhance(1.5)
    return ImageEnhance.Contrast(img).enhance(0.6)


def aug_warm_tint(img):
    """Warm color temperature -- yellowish cast from aged photos."""
    arr = np.array(img, dtype=np.float32)
    arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.12, 0, 255)  # boost red
    arr[:, :, 1] = np.clip(arr[:, :, 1] * 1.05, 0, 255)  # slight green
    arr[:, :, 2] = np.clip(arr[:, :, 2] * 0.85, 0, 255)  # reduce blue
    return Image.fromarray(arr.astype(np.uint8))


def aug_cool_tint(img):
    """Cool color temperature -- bluish cast."""
    arr = np.array(img, dtype=np.float32)
    arr[:, :, 0] = np.clip(arr[:, :, 0] * 0.88, 0, 255)  # reduce red
    arr[:, :, 2] = np.clip(arr[:, :, 2] * 1.15, 0, 255)  # boost blue
    return Image.fromarray(arr.astype(np.uint8))


def aug_gamma_high(img):
    """High gamma -- lightens midtones, simulates scanner overexposure."""
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.power(arr, 0.6)  # gamma < 1 = brighter midtones
    return Image.fromarray((arr * 255).astype(np.uint8))


AUGMENTATIONS = {
    "bright": aug_bright,
    "vbright": aug_very_bright,
    "lowcon": aug_low_contrast,
    "brlowcon": aug_bright_low_contrast,
    "warm": aug_warm_tint,
    "cool": aug_cool_tint,
    "gamma": aug_gamma_high,
}


def get_labeled_stems():
    """Get stems that have both a label file and a source image."""
    label_stems = {p.stem for p in LABELS_DIR.glob("*.txt") if p.stem.startswith("d")}
    image_stems = {p.stem for p in SCANMYPHOTOS_DIR.glob("*.jpg")}
    return sorted(label_stems & image_stems)


def augment_image(stem, augs, preview=False):
    """Generate all augmentations for a single image. Returns list of created files."""
    src_img_path = SCANMYPHOTOS_DIR / f"{stem}.jpg"
    src_label_path = LABELS_DIR / f"{stem}.txt"

    if not src_img_path.exists() or not src_label_path.exists():
        return []

    img = Image.open(src_img_path)
    label_text = src_label_path.read_text()
    created = []

    for suffix, transform_fn in augs.items():
        aug_stem = f"{AUG_PREFIX}{suffix}_{stem}"
        aug_img_path = AUG_IMAGES_DIR / f"{aug_stem}.jpg"
        aug_label_path = AUG_LABELS_DIR / f"{aug_stem}.txt"

        if preview:
            print(f"  {aug_stem}")
            continue

        aug_img = transform_fn(img)
        aug_img.save(aug_img_path, "JPEG", quality=92)
        aug_label_path.write_text(label_text)
        created.append(aug_stem)

    return created


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


def main():
    parser = argparse.ArgumentParser(description="Generate hard-case augmentations")
    parser.add_argument("--limit", type=int, help="Max images to augment")
    parser.add_argument("--preview", type=str, help="Preview augmentations for one stem")
    parser.add_argument("--clean", action="store_true", help="Remove all augmented files")
    parser.add_argument("--augs", type=str, default="bright,vbright,brlowcon,warm,gamma",
                        help="Comma-separated augmentation names (default: bright,vbright,brlowcon,warm,gamma)")
    args = parser.parse_args()

    if args.clean:
        clean()
        return

    # Filter to requested augmentations
    selected_augs = {k: v for k, v in AUGMENTATIONS.items() if k in args.augs.split(",")}
    if not selected_augs:
        print(f"No valid augmentations in: {args.augs}")
        print(f"Available: {', '.join(AUGMENTATIONS.keys())}")
        return

    active_augs = selected_augs

    if args.preview:
        print(f"Preview augmentations for {args.preview}:")
        augment_image(args.preview, active_augs, preview=True)
        return

    stems = get_labeled_stems()
    if args.limit:
        stems = stems[:args.limit]

    if not stems:
        print("No labeled images found.")
        return

    AUG_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    AUG_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    n_augs = len(active_augs)
    total = len(stems) * n_augs
    print(f"Generating {n_augs} augmentations x {len(stems)} images = {total} augmented images")
    print(f"Augmentations: {', '.join(active_augs.keys())}")
    print(f"Output: {AUG_IMAGES_DIR}")
    print()

    created = 0
    for i, stem in enumerate(stems):
        files = augment_image(stem, active_augs)
        created += len(files)
        if (i + 1) % 100 == 0 or i == len(stems) - 1:
            print(f"  [{i + 1}/{len(stems)}] {created} augmented images created")

    print(f"\nDone. {created} augmented images in {AUG_IMAGES_DIR}")
    print(f"Labels in {AUG_LABELS_DIR}")
    print(f"\nNext: update train.py to include augmented data, then run 'just train'")


if __name__ == "__main__":
    main()
