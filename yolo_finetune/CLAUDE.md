# YOLO Fine-Tune: Date Stamp Detector

## Purpose
Train YOLOv8 to detect camera date stamp regions on scanned photos, replacing the
unreliable OpenCV heuristics in `../stamp_detect.py`.

## Architecture
- `annotate.py` — HTTP server + REST API for keyboard-only bounding box annotation
- `index.html` — Browser-based annotation UI (vanilla JS + Canvas, no framework)
- `train.py` — YOLO fine-tuning using `ultralytics` library
- `dataset/` — YOLO-format training data (images/, labels/, data.yaml)
- Design spec: `docs/superpowers/specs/2026-04-03-bbox-annotator-yolo-finetune-design.md`

## Training Config
- Base model: `yolov8n.pt` (nano — CPU training friendly)
- Single class: `0` = date stamp region
- CPU-only training (no discrete GPU)
- Image size 640, batch 8, epochs 100 with early stopping patience=10

## Annotation Source
- Sample images in `../photo_mapping_samples/` (100 JPGs)
- Skipped photos (no stamp) become negative examples in training
- Output: YOLO-format normalized bbox labels
