#!/usr/bin/env python3
"""Feedback loop orchestration for YOLO continuous improvement."""

import json
import sys
from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "dataset"
CORRECTIONS_DIR = DATASET_DIR / "corrections"
TO_ANNOTATE_DIR = DATASET_DIR / "to_annotate"
CORRECTIONS_META_FILE = CORRECTIONS_DIR / "corrections_meta.json"
MODEL_PATH = BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
INFER_OUTPUT_DIR = BASE_DIR / "output" / "infer"


def load_corrections_meta():
    """Load existing predictions metadata, return empty dict if file doesn't exist."""
    if CORRECTIONS_META_FILE.exists():
        with open(CORRECTIONS_META_FILE) as f:
            return json.load(f)
    return {}


def save_corrections_meta(meta):
    """Save predictions metadata to JSON."""
    CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CORRECTIONS_META_FILE, "w") as f:
        json.dump(meta, f, indent=2)


def prepare():
    """Prepare feedback images: select missed detections and run inference."""
    # Create directories
    TO_ANNOTATE_DIR.mkdir(parents=True, exist_ok=True)
    CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Check model exists
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        sys.exit(1)

    # List available inference outputs (images ending with _detected.jpg)
    detected_images = sorted(INFER_OUTPUT_DIR.glob("*_detected.jpg"))

    if not detected_images:
        print(f"No inference outputs found in {INFER_OUTPUT_DIR}")
        return

    print(f"\nFound {len(detected_images)} inference outputs:")
    print("=" * 60)

    for i, img in enumerate(detected_images, 1):
        print(f"{i:3d}. {img.stem}")

    print("=" * 60)
    print("\nEnter image numbers with missed detections (comma-separated),")
    print("or press Enter to skip:")
    user_input = input("> ").strip()

    if not user_input:
        print("No images selected.")
        return

    # Parse selections
    try:
        indices = [int(x.strip()) for x in user_input.split(",")]
    except ValueError:
        print("Invalid input. Please enter comma-separated numbers.")
        return

    # Validate indices
    selected = []
    for idx in indices:
        if 1 <= idx <= len(detected_images):
            selected.append(detected_images[idx - 1])
        else:
            print(f"Warning: Index {idx} out of range, skipping.")

    if not selected:
        print("No valid selections.")
        return

    print(f"\nSelected {len(selected)} images for correction.")

    # Copy images to to_annotate/
    print("\nCopying selected images to dataset/to_annotate/...")
    import shutil
    for detected_img in selected:
        # Get stem (e.g., "00000080" from "00000080_detected.jpg")
        stem = detected_img.stem.replace("_detected", "")

        # Find the original source image in dataset/images/
        source_candidates = list((DATASET_DIR / "images").rglob(f"{stem}.*"))
        if not source_candidates:
            print(f"  Warning: Could not find source image for {stem}, skipping.")
            continue

        source_img = source_candidates[0]
        dest_img = TO_ANNOTATE_DIR / source_img.name
        shutil.copy2(source_img, dest_img)
        print(f"  Copied {source_img.name}")

    # Load model and run inference
    print("\nLoading model...")
    model = YOLO(str(MODEL_PATH))

    # Run inference on each selected image and save predictions
    print("\nRunning inference on selected images...")
    corrections_meta = load_corrections_meta()

    for source_img in (TO_ANNOTATE_DIR).glob("*"):
        if source_img.is_file() and source_img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            print(f"  Inferring on {source_img.name}...", end=" ")
            results = model.predict(str(source_img), conf=0.5, verbose=False)

            stem = source_img.stem
            predictions = []

            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # Get normalized bounding box (x_center, y_center, width, height)
                        x_center, y_center, width, height = box.xywhn[0].tolist()
                        confidence = float(box.conf[0])

                        predictions.append({
                            "x": x_center,
                            "y": y_center,
                            "w": width,
                            "h": height,
                            "confidence": confidence
                        })

            corrections_meta[stem] = predictions if predictions else {"found": False}
            print(f"Found {len(predictions)} detection(s)")

    # Save predictions metadata
    save_corrections_meta(corrections_meta)
    print(f"\nSaved predictions to {CORRECTIONS_META_FILE}")
    print("Ready for annotation. Run: python annotate.py")


def finalize():
    """
    Move annotated images from to_annotate to corrections.
    Check that each image has a corresponding label file.
    """
    if not TO_ANNOTATE_DIR.exists() or not list(TO_ANNOTATE_DIR.glob("*")):
        print(f"No images in {TO_ANNOTATE_DIR} to finalize.")
        return

    images = sorted(TO_ANNOTATE_DIR.glob("*.jpg")) + sorted(TO_ANNOTATE_DIR.glob("*.JPG"))
    if not images:
        print("No images found.")
        return

    print(f"Finalizing {len(images)} images from {TO_ANNOTATE_DIR}...")

    CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    labeled_count = 0
    skipped_count = 0

    for img in images:
        stem = img.stem
        dst_img = CORRECTIONS_DIR / img.name

        # Move image
        dst_img.write_bytes(img.read_bytes())
        img.unlink()

        # Check for label file
        label_file = BASE_DIR / "dataset" / "labels" / f"{stem}.txt"
        if label_file.exists():
            dst_label = CORRECTIONS_DIR / f"{stem}.txt"
            dst_label.write_bytes(label_file.read_bytes())
            labeled_count += 1
        else:
            # Image was skipped (no box) — will be negative example
            skipped_count += 1

    print(f"✓ Moved {labeled_count} labeled images to {CORRECTIONS_DIR}")
    print(f"✓ Moved {skipped_count} skipped (negative) images to {CORRECTIONS_DIR}")
    print(f"✓ Cleared {TO_ANNOTATE_DIR}")


def status():
    """Show current feedback loop status."""
    corrections_in_dir = sum(1 for p in CORRECTIONS_DIR.glob("*.txt")) if CORRECTIONS_DIR.exists() else 0
    skipped_in_dir = sum(1 for p in CORRECTIONS_DIR.glob("*.jpg")) - corrections_in_dir if CORRECTIONS_DIR.exists() else 0
    to_annotate_count = sum(1 for p in TO_ANNOTATE_DIR.glob("*.jpg")) if TO_ANNOTATE_DIR.exists() else 0

    print("\n=== Feedback Loop Status ===")
    print(f"Accumulated corrections: {corrections_in_dir} labeled + {skipped_in_dir} skipped (negative)")
    print(f"Pending annotation: {to_annotate_count} images in {TO_ANNOTATE_DIR}")

    if CORRECTIONS_META_FILE.exists():
        meta = load_corrections_meta()
        with_boxes = sum(1 for v in meta.values() if v.get("x") is not None)
        print(f"Model predictions saved: {with_boxes} with boxes, {len(meta) - with_boxes} no detection")


def main():
    """Parse command-line arguments and dispatch."""
    if len(sys.argv) < 2:
        print("Usage: python feedback.py {prepare|finalize|status}")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "prepare":
        prepare()
    elif command == "finalize":
        finalize()
    elif command == "status":
        status()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
