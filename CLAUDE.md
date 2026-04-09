# YOLO Date Stamp Detector

## Overview

Fine-tuned YOLOv8 to detect camera date stamp regions on scanned photographs.
Single class detection: bounding box around the date stamp area.

## Environment

- Python 3.12+ via uv -- deps managed via inline PEP 723 script headers
- CPU-only training and inference (no GPU required)
- PostgreSQL optional (for corrections dashboard rotation tracking)
- Task runner: `just` (run `just` to list all recipes)

## Architecture

- `annotate.py` -- HTTP server + REST API for bounding box annotation (:8888)
- `index.html` -- Browser annotation UI (vanilla JS + Canvas)
- `train.py` -- YOLO fine-tuning using `ultralytics`
- `infer_all.py` -- Batch inference on pending images
- `corrections_dashboard.py` -- Review/correct predictions (:8889)
- `dashboard.html` / `batch_review.html` -- Dashboard UIs
- `feedback.py` -- Feedback loop orchestration (prepare/finalize/status)
- `ocr_stamps.py` -- OCR via Claude Haiku (requires ANTHROPIC_API_KEY)
- `setup_scanmyphotos.py` -- Optional: import images from dedup database
- `stratified_sample.py` -- Stratified sampling across image sources

## Configuration

Several scripts accept configuration via environment variables:
- `DATABASE_URL` -- PostgreSQL connection (corrections_dashboard.py)
- `IMAGE_DIR` -- Source image directory for annotation (annotate.py)
- `ORIGINALS_DIR` -- Deduplicated originals path (setup_scanmyphotos.py)
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` -- DB config (setup_scanmyphotos.py)
- `DISC_DIRS` -- Colon-separated source directories (stratified_sample.py)

## Training Config

- Base model: `yolov8n.pt` (nano, CPU-friendly)
- Single class: `0` = date stamp region (called "target" in data.yaml)
- Image size 640, batch 8, epochs 100 with early stopping patience=10
- Labels: YOLO-format normalized bbox in `dataset/labels/`

## Date Stamp Characteristics

- Orange/red/amber LED digits imprinted by camera, typically bottom edge
- Format: `M D 'YY` (e.g., "10 3 '99"), spanning ~1986-2010
- Many photos have NO stamp -- detector must handle absence
- Some photos are rotated 90/180/270 -- stamps may appear on side edges

## Key Data Files

- `corrections_queue.json` -- Full review queue with statuses (gitignored)
- `scanmyphotos_predictions.json` -- YOLO predictions (gitignored)
- `scanmyphotos_manifest.json` -- Source file mapping (gitignored)
- `skipped.txt` -- Stems of images confirmed to have no date stamp
- `dataset/labels/*.txt` -- YOLO bounding box labels
- `status.json` -- Summary stats (run `just update-status` to refresh)
