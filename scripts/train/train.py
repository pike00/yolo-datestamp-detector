# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "ultralytics>=8.3",
#     "pyyaml",
#     "tensorboard",
#     "apprise",
#     "psycopg[binary]>=3.1.0",
# ]
# ///
"""YOLO fine-tuning on annotated bounding box data."""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from _db import load_skipped_stems  # noqa: E402

DATASET_DIR = BASE_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
IMAGE_SOURCE = BASE_DIR / "scanmyphotos"
AUG_IMAGES_DIR = DATASET_DIR / "augmented"
AUG_LABELS_DIR = DATASET_DIR / "augmented_labels"

SEED = 42
VAL_RATIO = 0.2


def setup_dataset():
    """Split labeled + skipped images into train/val with YOLO directory structure."""
    # Migrate old-style labels (00000080.txt) to new disc-prefixed names (d1_00000080.txt)
    old_labels = [p for p in LABELS_DIR.glob("*.txt") if not p.stem.startswith("d")]
    if old_labels:
        print(f"Migrating {len(old_labels)} old-style labels to disc-prefixed names...")
        for label_path in old_labels:
            stem = label_path.stem
            # Find matching image in scanmyphotos/
            matches = list(IMAGE_SOURCE.glob(f"d*_{stem}.jpg"))
            if matches:
                new_stem = matches[0].stem  # e.g., d1_00000080
                new_label = LABELS_DIR / f"{new_stem}.txt"
                if not new_label.exists():
                    shutil.copy2(label_path, new_label)
                print(f"  {stem} -> {new_stem}")

    # Collect labeled images (those with a .txt in labels/)
    labeled = sorted(p.stem for p in LABELS_DIR.glob("*.txt") if p.stem.startswith("d"))
    if not labeled:
        print("No labels found in dataset/labels/. Run annotate.py first.")
        raise SystemExit(1)

    # Collect skipped images (negative examples) from stamp_no_stamp
    skipped_stems = sorted(load_skipped_stems())

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
        src_img = IMAGE_SOURCE / f"{stem}.jpg"
        if not src_img.exists():
            src_img = None

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

    # Include augmented hard cases (only in training set)
    aug_count = 0
    if AUG_IMAGES_DIR.exists() and AUG_LABELS_DIR.exists():
        for aug_img in AUG_IMAGES_DIR.glob("*.jpg"):
            aug_stem = aug_img.stem
            aug_label = AUG_LABELS_DIR / f"{aug_stem}.txt"
            if not aug_label.exists():
                continue
            # Only add to train split (augmented data shouldn't be in validation)
            dst_img = DATASET_DIR / "images" / "train" / aug_img.name
            dst_label = DATASET_DIR / "labels" / "train" / f"{aug_stem}.txt"
            if not dst_img.exists():
                dst_img.symlink_to(aug_img.resolve())
            if not dst_label.exists():
                shutil.copy2(aug_label, dst_label)
            aug_count += 1
        if aug_count:
            print(f"Added {aug_count} augmented hard cases to training set")

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


def notify(url, title, body):
    """Send a notification via apprise."""
    import apprise

    ap = apprise.Apprise()
    ap.add(url)
    ap.notify(title=title, body=body)


def make_epoch_callback(notify_url, every=10):
    """Return a callback that notifies every N epochs."""
    def on_train_epoch_end(trainer):
        epoch = trainer.epoch + 1
        if epoch % every != 0:
            return
        metrics = trainer.metrics
        box_loss = trainer.loss_items[0].item() if trainer.loss_items is not None else None
        msg_parts = [f"Epoch {epoch}/{trainer.epochs}"]
        if box_loss is not None:
            msg_parts.append(f"box_loss: {box_loss:.4f}")
        for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)"):
            if key in metrics:
                short = key.split("/")[1]
                msg_parts.append(f"{short}: {metrics[key]:.4f}")
        notify(notify_url, "YOLO Training", "\n".join(msg_parts))

    return on_train_epoch_end


def make_done_callback(notify_url):
    """Return a callback that notifies when training finishes."""
    def on_train_end(trainer):
        metrics = trainer.metrics
        parts = ["Training complete"]
        for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)"):
            if key in metrics:
                short = key.split("/")[1]
                parts.append(f"{short}: {metrics[key]:.4f}")
        parts.append(f"Best weights: {trainer.best}")
        notify(notify_url, "YOLO Training Done", "\n".join(parts))

    return on_train_end


def train(data_yaml, notify_url=None):
    """Run YOLOv8 fine-tuning. Resumes from previous best.pt if available."""
    from ultralytics import YOLO

    # Enable TensorBoard logging
    os.environ["COMET_MODE"] = "disabled"

    best_pt = BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
    if best_pt.exists():
        print(f"Resuming fine-tune from previous best: {best_pt}")
        model = YOLO(str(best_pt))
    else:
        print("No previous model found, starting from yolo26s.pt")
        model = YOLO("yolo26s.pt")

    from ultralytics import settings
    settings.update({"tensorboard": True})

    if notify_url:
        model.add_callback("on_train_epoch_end", make_epoch_callback(notify_url))
        model.add_callback("on_train_end", make_done_callback(notify_url))
        print(f"Notifications enabled (every 10 epochs)")

    model.train(
        data=str(data_yaml),
        epochs=100,
        patience=10,
        imgsz=640,
        batch=4,
        device="cpu",
        project=str(BASE_DIR / "runs" / "detect"),
        name="train",
        exist_ok=True,
        verbose=True,
        # Brightness/contrast augmentation to handle bright backgrounds
        # where date stamps wash out (default hsv_v=0.4, we increase it)
        hsv_h=0.015,  # hue jitter (default)
        hsv_s=0.7,    # saturation jitter (default)
        hsv_v=0.6,    # value/brightness jitter (increased from 0.4)
    )

    if best_pt.exists():
        print(f"\nTraining complete! Best weights: {best_pt}")
        print(f"TensorBoard: uv run tensorboard --logdir {BASE_DIR / 'runs' / 'detect'}")
    else:
        print("\nTraining finished but no best.pt found -- check logs above.")


DEFAULT_APPRISE_URL = "mmost://mattermost:8065/918rokyemjboifstiwqaqht7fy"


def main():
    parser = argparse.ArgumentParser(description="YOLO fine-tuning")
    parser.add_argument(
        "--notify", nargs="?", const=DEFAULT_APPRISE_URL, default=DEFAULT_APPRISE_URL,
        help="Apprise URL for notifications (default: Mattermost). Pass --no-notify to disable.",
    )
    parser.add_argument("--no-notify", action="store_true", help="Disable notifications")
    args = parser.parse_args()

    notify_url = None if args.no_notify else (os.environ.get("APPRISE_URL") or args.notify)

    print("Setting up dataset...")
    data_yaml = setup_dataset()
    print("\nStarting YOLOv8 fine-tuning (CPU, this will take a while)...\n")
    train(data_yaml, notify_url=notify_url)


if __name__ == "__main__":
    main()
