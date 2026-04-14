# YOLO Date Stamp Detector

## Overview

Fine-tuned YOLOv8 to detect camera date stamp regions on scanned photographs.
Single class detection: bounding box around the date stamp area.

## Environment

- Python 3.12+ via uv -- deps managed via inline PEP 723 script headers
- CPU-only training and inference (no GPU required)
- PostgreSQL optional (for corrections dashboard rotation tracking)
- Task runner: `just` (run `just` to list all recipes)

## Project Structure

- `scripts/` -- All Python scripts (training, inference, annotation, OCR, etc.)
- `ui/` -- Browser UIs (annotation, corrections dashboard, batch review)
- `state/` -- Runtime state files (JSON queues, progress, predictions, skipped.txt)
- `output/` -- Generated outputs (inference visualizations, crops, enhancements)
- `docker/` -- Dockerfiles and compose configs
- `dataset/` -- Training data (images, labels, augmented, corrections)
- `runs/` -- Model training artifacts and weights
- `scanmyphotos/` -- Source images (gitignored)
- `examples/` -- README images and sample detections

## Architecture

- `scripts/annotate.py` -- HTTP server + REST API for bounding box annotation (:8888)
- `scripts/train.py` -- YOLO fine-tuning using `ultralytics`
- `scripts/infer_all.py` -- Batch inference on pending images
- `scripts/corrections_dashboard.py` -- Review/correct predictions (:8889)
- `scripts/feedback.py` -- Feedback loop orchestration (prepare/finalize/status)
- `scripts/ocr_stamps.py` -- OCR via Claude Haiku (requires ANTHROPIC_API_KEY)
- `scripts/ocr_gemma.py` -- OCR via local Gemma4 (requires Ollama)
- `scripts/augment_hard_cases.py` -- Data augmentation for failure modes
- `scripts/enhance_stamps.py` -- Stamp enhancement experiments
- `scripts/setup_scanmyphotos.py` -- Optional: import images from dedup database
- `scripts/stratified_sample.py` -- Stratified sampling across image sources
- `ui/index.html` -- Annotation UI (vanilla JS + Canvas)
- `ui/dashboard.html` / `ui/batch_review.html` -- Dashboard UIs

## Configuration

Several scripts accept configuration via environment variables:
- `DATABASE_URL` -- PostgreSQL connection (corrections_dashboard.py)
- `IMAGE_DIR` -- Source image directory for annotation (annotate.py)
- `ORIGINALS_DIR` -- Deduplicated originals path (setup_scanmyphotos.py)
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` -- DB config (setup_scanmyphotos.py)
- `DISC_DIRS` -- Colon-separated source directories (stratified_sample.py)

## Training Config

- Base model: `yolo26m.pt` (medium, 20.4M params)
- Single class: `0` = date stamp region (called "target" in data.yaml)
- Image size 640, batch 4, epochs 100 with early stopping patience=10
- Labels: YOLO-format normalized bbox in `dataset/labels/`

## Date Stamp Characteristics

- Orange/red/amber LED digits imprinted by camera, typically bottom edge
- Format: `M D 'YY` (e.g., "10 3 '99"), spanning ~1986-2010
- Many photos have NO stamp -- detector must handle absence
- Some photos are rotated 90/180/270 -- stamps may appear on side edges

## Key Data Files

- `state/corrections_queue.json` -- Full review queue with statuses (gitignored)
- `state/scanmyphotos_predictions.json` -- YOLO predictions (gitignored)
- `state/scanmyphotos_manifest.json` -- Source file mapping (gitignored)
- `state/skipped.txt` -- Stems of images confirmed to have no date stamp
- `state/status.json` -- Summary stats (run `just update-status` to refresh)
- `dataset/labels/*.txt` -- YOLO bounding box labels
