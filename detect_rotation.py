#!/usr/bin/env python3
# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "torch",
#   "torchvision",
#   "Pillow",
#   "psycopg[binary]",
#   "pillow-heif",
#   "apprise",
# ]
# ///
"""
Detect image orientation using DuarteBarbosa/deep-image-orientation-detection.

EfficientNetV2-S fine-tuned on 189K images, 98.82% accuracy.
Stores all predictions (including per-class probabilities) in PostgreSQL
rotation_predictions table for downstream use.

Classes:
  0: already correct (0 deg)
  1: needs 90 CW rotation
  2: needs 180 rotation
  3: needs 90 CCW rotation (270 CW)
"""
import hashlib
import os
import sys
import time
from pathlib import Path

import pillow_heif
import torch
from PIL import Image

pillow_heif.register_heif_opener()
from torchvision import models, transforms

MODEL_PATH = Path("models/orientation_model_v2_0.9882.pth")
MODEL_NAME = "efficientnetv2_s_orientation_v2"
CLASSES = {0: 0, 1: 90, 2: 180, 3: 270}  # class -> degrees CW to correct

# DB config (matches dedup/.env defaults)
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "dedup")
DB_USER = os.getenv("DB_USER", "dedup")
DB_PASSWORD = os.getenv("DB_PASSWORD", "dedup_local_dev")

APPRISE_URLS = os.getenv("APPRISE_URLS", "")


_apobj = None

def _get_apprise():
    global _apobj
    if _apobj is None and APPRISE_URLS:
        import apprise
        _apobj = apprise.Apprise()
        for url in APPRISE_URLS.split():
            _apobj.add(url)
    return _apobj

def notify(message: str):
    """Send notification via apprise SDK."""
    ap = _get_apprise()
    if not ap:
        return
    try:
        ap.notify(body=message)
    except Exception as e:
        print(f"Notification failed: {e}")


def load_model(model_path: Path):
    model = models.efficientnet_v2_s(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 4)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def predict_rotation(model, transform, image_path: Path) -> dict:
    """Returns dict with predicted class, rotation, confidence, and all probabilities."""
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        cls = torch.argmax(probs).item()
    return {
        "predicted_class": cls,
        "rotation_needed": CLASSES[cls],
        "confidence": float(probs[cls]),
        "prob_0": float(probs[0]),
        "prob_90": float(probs[1]),
        "prob_180": float(probs[2]),
        "prob_270": float(probs[3]),
    }


def file_sha256(path: Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_db_connection():
    import psycopg
    return psycopg.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASSWORD,
    )


def get_already_predicted(conn, model_name: str) -> set[str]:
    """Return set of file_paths already predicted with this model."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT file_path FROM rotation_predictions WHERE model_name = %s",
            (model_name,),
        )
        return {row[0] for row in cur.fetchall()}


def insert_predictions(conn, rows: list[dict]):
    """Batch insert prediction rows."""
    if not rows:
        return
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO rotation_predictions
               (file_path, filename, sha256, model_name,
                predicted_class, rotation_needed, confidence,
                prob_0, prob_90, prob_180, prob_270)
               VALUES (%(file_path)s, %(filename)s, %(sha256)s, %(model_name)s,
                       %(predicted_class)s, %(rotation_needed)s, %(confidence)s,
                       %(prob_0)s, %(prob_90)s, %(prob_180)s, %(prob_270)s)""",
            rows,
        )
    conn.commit()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Detect image rotation and store in DB")
    parser.add_argument("input", help="Image file or directory of images")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip files already predicted in the DB")
    parser.add_argument("--no-hash", action="store_true",
                        help="Skip SHA-256 computation (faster, no DB linkage)")
    parser.add_argument("--only-rotated", action="store_true",
                        help="Only print images that need rotation")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="DB insert batch size (default: 100)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_file():
        images = [input_path]
    elif input_path.is_dir():
        images = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic"}
        )
    else:
        print(f"Not found: {input_path}")
        sys.exit(1)

    # Connect to DB
    conn = get_db_connection()
    print(f"Connected to {DB_HOST}:{DB_PORT}/{DB_NAME}")

    # Skip already-predicted files
    if args.skip_existing:
        already_done = get_already_predicted(conn, MODEL_NAME)
        before = len(images)
        images = [p for p in images if str(p.resolve()) not in already_done]
        print(f"Skipping {before - len(images)} already-predicted files")

    if not images:
        print("Nothing to process.")
        conn.close()
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    transform = get_transform()
    total = len(images)
    print(f"Processing {total} images...\n")
    notify(f":camera: Rotation detection started: {total} images to process")

    batch = []
    rotated_count = 0
    error_count = 0
    start = time.time()

    for idx, img_path in enumerate(images, 1):
        try:
            pred = predict_rotation(model, transform, img_path)
        except Exception as e:
            print(f"[{idx:4d}/{len(images)}] {img_path.name:30s}  ERROR: {e}")
            error_count += 1
            continue

        sha = None if args.no_hash else file_sha256(img_path)

        row = {
            "file_path": str(img_path.resolve()),
            "filename": img_path.name,
            "sha256": sha,
            "model_name": MODEL_NAME,
            **pred,
        }
        batch.append(row)

        if pred["rotation_needed"] != 0:
            rotated_count += 1

        if not args.only_rotated or pred["rotation_needed"] != 0:
            label = "OK" if pred["rotation_needed"] == 0 else f"ROTATE {pred['rotation_needed']} CW"
            print(f"[{idx:4d}/{len(images)}] {img_path.name:30s}  {label:15s}  conf={pred['confidence']:.3f}")

        # Notify every 1000 images
        if idx % 1000 == 0:
            elapsed_so_far = time.time() - start
            rate = idx / elapsed_so_far
            eta_min = (total - idx) / rate / 60
            notify(
                f":camera: Rotation detection progress: {idx}/{total} "
                f"({idx*100//total}%) | {rotated_count} need rotation | "
                f"{error_count} errors | {rate:.1f} img/s | ~{eta_min:.0f}m remaining"
            )

        # Flush batch
        if len(batch) >= args.batch_size:
            insert_predictions(conn, batch)
            batch.clear()

    # Final flush
    if batch:
        insert_predictions(conn, batch)

    conn.close()

    elapsed = time.time() - start
    processed = len(images) - error_count
    print(f"\n{'='*50}")
    print(f"Total: {processed} images in {elapsed:.1f}s ({elapsed/processed:.2f}s/img)")
    print(f"Correctly oriented: {processed - rotated_count}")
    print(f"Need rotation: {rotated_count}")
    if error_count:
        print(f"Errors: {error_count}")
    print(f"Results stored in {DB_NAME}.rotation_predictions")
    print(f"{'='*50}")
    notify(
        f":white_check_mark: Rotation detection complete: {processed} images "
        f"in {elapsed:.1f}s ({elapsed/processed:.2f}s/img) | "
        f"{rotated_count} need rotation | {error_count} errors"
    )


if __name__ == "__main__":
    main()
