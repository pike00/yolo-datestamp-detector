# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "torch",
#     "torchvision",
#     "pillow>=12.0",
# ]
# ///
"""Detect rotation for every scanmyphotos image, write predictions to JSON.

Uses the project's pre-trained EfficientNetV2-S orientation classifier
(98.82% accuracy on a 4-class 0/90/180/270 problem). Runs on CPU.

Output schema (state/rotation_predictions.json):

{
  "d1_00000133": {
    "rotation": 0,         # degrees CW required to make the photo upright
    "confidence": 0.98,    # confidence in the chosen class
    "probs": {"0": 0.98, "90": 0.01, "180": 0.005, "270": 0.005}
  }
}

Usage:
    uv run scripts/detect_rotation_batch.py
    uv run scripts/detect_rotation_batch.py --limit 100
    uv run scripts/detect_rotation_batch.py --resume
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

YOLO_DIR = Path(__file__).parent.parent
PHOTO_PROJECT_DIR = YOLO_DIR.parent
SCANMYPHOTOS_DIR = YOLO_DIR / "scanmyphotos"
MODEL_PATH = PHOTO_PROJECT_DIR / "models" / "orientation_model_v2_0.9882.pth"
OUTPUT_FILE = YOLO_DIR / "state" / "rotation_predictions.json"

# Class index -> degrees CW required to correct
CLASSES = {0: 0, 1: 90, 2: 180, 3: 270}


def load_model():
    import torch
    from torchvision import models

    model = models.efficientnet_v2_s(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 4)
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    # Inference mode (equivalent to model.eval())
    model.train(False)
    return model


def get_transform():
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def predict_one(model, transform, img_path: Path) -> dict:
    import torch
    from PIL import Image

    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        cls = int(torch.argmax(probs).item())
    return {
        "rotation": CLASSES[cls],
        "confidence": round(float(probs[cls]), 4),
        "probs": {
            "0": round(float(probs[0]), 4),
            "90": round(float(probs[1]), 4),
            "180": round(float(probs[2]), 4),
            "270": round(float(probs[3]), 4),
        },
    }


def load_existing() -> dict[str, dict]:
    if OUTPUT_FILE.exists():
        return json.loads(OUTPUT_FILE.read_text())
    return {}


def save_atomic(data: dict[str, dict]) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(OUTPUT_FILE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Cap the number of images to process")
    parser.add_argument("--resume", action="store_true", help="Skip stems already in the output file")
    parser.add_argument("--save-every", type=int, default=200, help="Save partial results every N images")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print(f"ERROR: model not found at {MODEL_PATH}", file=sys.stderr)
        sys.exit(2)

    images = sorted(SCANMYPHOTOS_DIR.glob("*.jpg"))
    if not images:
        print(f"No images in {SCANMYPHOTOS_DIR}", file=sys.stderr)
        sys.exit(1)

    results = load_existing() if args.resume else {}
    if args.resume:
        before = len(images)
        images = [p for p in images if p.stem not in results]
        print(f"Resuming: {before - len(images)} already done, {len(images)} remaining")

    if args.limit is not None:
        images = images[: args.limit]

    if not images:
        print("Nothing to do.")
        return

    print(f"Loading model from {MODEL_PATH.relative_to(PHOTO_PROJECT_DIR)}...")
    model = load_model()
    transform = get_transform()

    total = len(images)
    print(f"Processing {total} images on CPU...\n")
    start = time.time()
    rotated_count = 0
    error_count = 0

    for idx, img_path in enumerate(images, 1):
        try:
            pred = predict_one(model, transform, img_path)
        except Exception as e:
            print(f"[{idx:5d}/{total}] {img_path.name}: ERROR {e}", file=sys.stderr)
            error_count += 1
            continue

        results[img_path.stem] = pred
        if pred["rotation"] != 0:
            rotated_count += 1

        if idx % 50 == 0 or idx == total:
            elapsed = time.time() - start
            rate = idx / max(elapsed, 0.001)
            eta_min = (total - idx) / rate / 60 if rate > 0 else 0
            print(
                f"[{idx:5d}/{total}] {img_path.stem} -> rot={pred['rotation']} "
                f"conf={pred['confidence']:.2f}  "
                f"({rate:.1f} img/s, ~{eta_min:.0f}m left, {rotated_count} rotated)"
            )

        if idx % args.save_every == 0:
            save_atomic(results)

    save_atomic(results)
    elapsed = time.time() - start
    print()
    print(f"Done in {elapsed:.0f}s ({elapsed/total:.2f}s/img)")
    print(f"Total: {total - error_count} processed, {rotated_count} need rotation, {error_count} errors")
    print(f"Wrote {OUTPUT_FILE.relative_to(YOLO_DIR)}")


if __name__ == "__main__":
    main()
