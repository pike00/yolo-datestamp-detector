"""YOLO fine-tuning on annotated bounding box data."""

import random
import shutil
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
SKIPPED_FILE = BASE_DIR / "skipped.txt"
IMAGE_SOURCE = BASE_DIR.parent / "photo_mapping_samples"

SEED = 42
VAL_RATIO = 0.2


def setup_dataset():
    """Split labeled + skipped images into train/val with YOLO directory structure."""
    # Collect labeled images (those with a .txt in labels/)
    labeled = sorted(p.stem for p in LABELS_DIR.glob("*.txt"))
    if not labeled:
        print("No labels found in dataset/labels/. Run annotate.py first.")
        raise SystemExit(1)

    # Collect skipped images (negative examples)
    skipped = []
    if SKIPPED_FILE.exists():
        skipped = [
            line.strip() for line in SKIPPED_FILE.read_text().splitlines()
            if line.strip()
        ]
    skipped_stems = [Path(f).stem for f in skipped]

    print(f"Found {len(labeled)} labeled images, {len(skipped_stems)} skipped (negative examples)")

    # Combine all stems for splitting
    all_stems = labeled + skipped_stems
    random.seed(SEED)
    random.shuffle(all_stems)

    val_count = max(1, int(len(all_stems) * VAL_RATIO))
    val_stems = set(all_stems[:val_count])
    train_stems = set(all_stems[val_count:])

    # Create train/val directory structure
    for split in ("train", "val"):
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Clear old split symlinks
    for split in ("train", "val"):
        for f in (DATASET_DIR / "images" / split).iterdir():
            f.unlink()
        for f in (DATASET_DIR / "labels" / split).iterdir():
            f.unlink()

    labeled_set = set(labeled)

    for stem in train_stems | val_stems:
        split = "val" if stem in val_stems else "train"

        # Find source image
        src_img = None
        for ext in (".jpg", ".JPG"):
            candidate = IMAGE_SOURCE / f"{stem}{ext}"
            if candidate.exists():
                src_img = candidate
                break

        if src_img is None:
            print(f"  Warning: no source image for {stem}, skipping")
            continue

        # Symlink image
        dst_img = DATASET_DIR / "images" / split / src_img.name
        if not dst_img.exists():
            dst_img.symlink_to(src_img.resolve())

        # Copy label if this is a labeled image (not a negative example)
        if stem in labeled_set:
            src_label = LABELS_DIR / f"{stem}.txt"
            dst_label = DATASET_DIR / "labels" / split / f"{stem}.txt"
            if src_label.exists() and not dst_label.exists():
                shutil.copy2(src_label, dst_label)
        # Negative examples: image present, no label file — YOLO handles this correctly

    # Write data.yaml
    data_yaml = {
        "path": str(DATASET_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "target"},
    }
    yaml_path = DATASET_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    train_labeled = len(train_stems & labeled_set)
    val_labeled = len(val_stems & labeled_set)
    train_neg = len(train_stems) - train_labeled
    val_neg = len(val_stems) - val_labeled
    print(f"Train: {train_labeled} labeled + {train_neg} negative = {len(train_stems)}")
    print(f"Val:   {val_labeled} labeled + {val_neg} negative = {len(val_stems)}")
    print(f"Dataset config: {yaml_path}")

    return yaml_path


def train(data_yaml):
    """Run YOLOv8 fine-tuning."""
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(data_yaml),
        epochs=100,
        patience=10,
        imgsz=640,
        batch=8,
        device="cpu",
        project=str(BASE_DIR / "runs" / "detect"),
        name="train",
        exist_ok=True,
        verbose=True,
    )

    best = BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
    if best.exists():
        print(f"\nTraining complete! Best weights: {best}")
    else:
        print("\nTraining finished but no best.pt found — check logs above.")


def main():
    print("Setting up dataset...")
    data_yaml = setup_dataset()
    print("\nStarting YOLOv8 fine-tuning (CPU, this will take a while)...\n")
    train(data_yaml)


if __name__ == "__main__":
    main()
