#!/usr/bin/env python3
"""Feedback loop orchestration for YOLO continuous improvement."""

import json
import sys
from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
CORRECTIONS_DIR = DATASET_DIR / "corrections"
TO_ANNOTATE_DIR = DATASET_DIR / "to_annotate"
CORRECTIONS_META_FILE = CORRECTIONS_DIR / "corrections_meta.json"
MODEL_PATH = BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
INFER_OUTPUT_DIR = BASE_DIR / "infer_output"


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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python feedback.py <command>")
        print("Commands: prepare")
        sys.exit(1)

    command = sys.argv[1]
    if command == "prepare":
        prepare()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
