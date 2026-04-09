#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pillow>=10.0",
# ]
# ///
"""
Stratified random sampling from multiple image source directories.
Samples N images per source, avoiding sequential bias.

Configure source directories via DISC_DIRS environment variable (colon-separated):
    export DISC_DIRS="/path/to/disc1:/path/to/disc2:/path/to/disc3:/path/to/disc4"
"""
import os
import random
from pathlib import Path
import shutil

# Configuration: source directories (colon-separated env var or defaults)
_default_dirs = [
    "/data/scanned_photos/disc_1",
    "/data/scanned_photos/disc_2",
    "/data/scanned_photos/disc_3",
    "/data/scanned_photos/disc_4",
]
DISCS = os.environ.get("DISC_DIRS", ":".join(_default_dirs)).split(":")
SAMPLES_PER_DISC = 25  # Total: 100 images (25 × 4)
OUTPUT_DIR = Path("stratified_samples")

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

random.seed(42)  # Reproducible

print(f"Stratified sampling: {SAMPLES_PER_DISC} images per disc")
print(f"Total target: {SAMPLES_PER_DISC * len(DISCS)} images\n")

all_sampled = []

for disc_num, disc_path in enumerate(DISCS, 1):
    disc_p = Path(disc_path)

    # Get all JPG files (skip non-image files)
    jpg_files = sorted([f for f in disc_p.glob("*.jpg") if f.is_file()])
    jpg_files += sorted([f for f in disc_p.glob("*.JPG") if f.is_file()])

    if not jpg_files:
        print(f"❌ Disc {disc_num}: No JPG files found in {disc_path}")
        continue

    # Random sample without replacement
    sampled = random.sample(jpg_files, min(SAMPLES_PER_DISC, len(jpg_files)))
    all_sampled.extend(sampled)

    print(f"✓ Disc {disc_num}: Sampled {len(sampled)} from {len(jpg_files)} images")

    # Copy to output directory with disc prefix
    for idx, src in enumerate(sampled, 1):
        dst = OUTPUT_DIR / f"D{disc_num}_{idx:03d}.jpg"
        shutil.copy2(src, dst)

print(f"\n✅ Total sampled: {len(all_sampled)} images")
print(f"📁 Saved to: {OUTPUT_DIR.absolute()}")
print(f"\nDistribution:")
for i in range(1, 5):
    count = len(list(OUTPUT_DIR.glob(f"D{i}_*.jpg")))
    print(f"  Disc {i}: {count} images")
