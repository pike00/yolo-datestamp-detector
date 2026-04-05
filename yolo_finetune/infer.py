#!/usr/bin/env python3
"""
Run inference on sample images and draw red bounding boxes on detections.
Optimized for CPU inference speed.
"""
import cv2
from pathlib import Path
from ultralytics import YOLO

# Paths
MODEL_PATH = "runs/detect/train/weights/best.pt"
SAMPLE_DIR = Path("../photo_mapping_samples")
OUTPUT_DIR = Path("infer_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load model
print(f"Loading model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

# Inference settings for speed (CPU optimization)
# Smaller image size = faster, but less accurate
# Adjust imgsz down (320, 384) for more speed; up (640) for accuracy
INFERENCE_SIZE = 384  # Faster than 640, good balance
CONF_THRESHOLD = 0.3  # Only keep detections above 30% confidence

print(f"Running inference on {SAMPLE_DIR} (imgsz={INFERENCE_SIZE}, conf={CONF_THRESHOLD})...\n")

image_files = sorted(SAMPLE_DIR.glob("*.jpg")) + sorted(SAMPLE_DIR.glob("*.JPG"))
print(f"Found {len(image_files)} images\n")

for idx, img_path in enumerate(image_files, 1):
    # Run inference
    results = model(
        str(img_path),
        imgsz=INFERENCE_SIZE,
        conf=CONF_THRESHOLD,
        device='cpu',
        verbose=False
    )

    # Read original image
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    # Draw bounding boxes in red
    result = results[0]
    box_count = 0

    for box in result.boxes:
        # Get box coordinates (normalized → pixel coords)
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        conf = float(box.conf[0])

        # Draw red box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # BGR: red

        # Draw confidence label
        cv2.putText(img, f"{conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        box_count += 1

    # Save output
    out_path = OUTPUT_DIR / f"{img_path.stem}_detected.jpg"
    cv2.imwrite(str(out_path), img)

    status = f"✓ {box_count} box" if box_count else "○ no detection"
    print(f"[{idx:3d}/{len(image_files)}] {img_path.name:40s} {status}")

print(f"\nDone! Results saved to {OUTPUT_DIR}/")
