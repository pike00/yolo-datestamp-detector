# Photo Project

## Overview
Consolidating ~77K media files (467 GB) into deduplicated, organized, metadata-enriched collection.
Current focus: extracting date stamps from ~7,500 scanned 4x6 photos (ScanMyPhotos Discs 1-4).

## Critical Constraints
- **NEVER modify files on HDD** at `/mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/` — read-only source
- **No external API calls** (Gemini, OpenAI, Anthropic) without explicit user approval
- Local LLM via Ollama is OK if installed

## Source Photo Paths
- Disc 1: `/mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/Photos/img/Photos/ScanMyPhotos/Disc 1/` (1,775 files)
- Disc 2: `/mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/Photos/img/Photos/ScanMyPhotos/Disc 2/` (2,040 files)
- Disc 3: `/mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/Photos/img/Photos/ScanMyPhotos/Disc 3/` (2,076 files)
- Disc 4: `/mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/Photos/img/Photos/ScanMyPhotos/Disc 4/` (1,576 files)
- Samples: `data/samples/` (100 pre-selected JPGs across discs)

## Environment
- Python 3.14 via uv — activate venv: `source .venv/bin/activate`
- CPU-only (AMD Ryzen 12 cores, 27GB RAM, integrated AMD Radeon Vega — no discrete GPU)
- Tesseract 5.3.4 installed system-wide
- Deps managed in `pyproject.toml` with uv

- Python 3.12+ via uv -- deps managed via inline PEP 723 script headers
- CPU-only training and inference (no GPU required)
- PostgreSQL optional (for corrections dashboard rotation tracking)
- Task runner: `just` (run `just` to list all recipes)

## Project Structure

- `scripts/` -- Python scripts, grouped by role (train/infer/annotate/ocr/data)
- `ui/` -- Browser UIs (annotation, corrections dashboard, batch review)
- `state/` -- Runtime state files (JSON queues, progress, shard manifests). Predictions, OCR results, drift, and the no-stamp set live in Postgres now.
- `output/` -- Generated outputs (inference visualizations, crops, enhancements, pilot_review.html)
- `docker/` -- Dockerfiles and compose configs
- `dataset/` -- Training data (images, labels, augmented, corrections)
- `runs/` -- Model training artifacts and weights
- `scanmyphotos/` -- Source images (gitignored)
- `examples/` -- README images and sample detections
- `tests/` -- Pytest suite

## Architecture

- `scripts/train/train.py` -- YOLO fine-tuning using `ultralytics`
- `scripts/train/gpu_bench_one_epoch.py` -- AWS GPU spot one-epoch bench
- `scripts/train/regen_val_plots.py` -- Refresh validation plots in `examples/`
- `scripts/infer/infer_all.py` -- Batch inference on pending images
- `scripts/infer/compare_predictions.py` -- Diff old vs new model predictions
- `scripts/infer/render_drift_examples.py` -- Render drift visualization crops
- `scripts/annotate/annotate.py` -- HTTP server + REST API for bounding box annotation (:8888)
- `scripts/annotate/corrections_dashboard.py` -- Review/correct predictions (:8889)
- `scripts/annotate/feedback.py` -- Feedback loop orchestration (prepare/finalize/status)
- `scripts/ocr/orchestrate_ocr.py` -- Parallel Haiku OCR orchestrator (crop/merge/reconcile)
- `scripts/ocr/ocr_stamps.py` -- OCR via Claude Haiku (requires ANTHROPIC_API_KEY)
- `scripts/ocr/ocr_gemma.py` -- OCR via local Gemma4 (requires Ollama)
- `scripts/ocr/ocr_ollama_bench.py` -- Local Ollama vision accuracy bench
- `scripts/ocr/build_pilot_review_html.py` -- Render OCR pilot review HTML
- `scripts/data/setup_scanmyphotos.py` -- Optional: import images from dedup database
- `scripts/data/stratified_sample.py` -- Stratified sampling across image sources
- `scripts/data/augment_hard_cases.py` -- Data augmentation for failure modes
- `scripts/data/detect_rotation_batch.py` -- Pre-compute rotation predictions
- `scripts/data/enhance_stamps.py` -- Stamp enhancement experiments
- `ui/index.html` -- Annotation UI (vanilla JS + Canvas)
- `ui/dashboard.html` / `ui/batch_review.html` -- Dashboard UIs

## Configuration

Several scripts accept configuration via environment variables:
- `DATABASE_URL` -- PostgreSQL connection (defaults to `postgresql://dedup:dedup_local_dev@localhost:5432/dedup`); used by every script that touches `stamp_predictions`, `stamp_ocr`, `stamp_prediction_drift`, or `stamp_no_stamp` via `scripts/_db.py`
- `YOLO_MODEL_LABEL` -- Label written into `stamp_predictions.model` for new infer runs (default `yolo26m-best`)
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
- Orange/red/amber LED digits imprinted by camera, typically bottom edge of photo
- Format: `M D 'YY` (e.g., "10 3 '99"), spanning ~1986-2010
- Many photos have NO stamp — detector must handle `found: False`
- Some photos are rotated 90° — stamps may appear on side edges

## Key Data Stores

Postgres tables (`dedup` database, see `scripts/_db.py`):
- `stamp_predictions` -- YOLO bbox predictions per stem (model label tracked in `model` column)
- `stamp_ocr` -- OCR results, composite PK `(stem, model)`; haiku and gemma both live here
- `stamp_prediction_drift` -- old vs new bbox diff with iou and flag
- `stamp_no_stamp` -- stems confirmed to have no date stamp
- `stamp_rotations` -- user-confirmed rotations from corrections_dashboard
- `rotation_predictions` -- rotation classifier output (managed by the dedup pipeline, joined on sha256)

Files still on disk:
- `state/corrections_queue.json` -- Full review queue with statuses (gitignored)
- `state/scanmyphotos_manifest.json` -- Source file mapping (gitignored)
- `state/rotation_predictions.json` -- Local rotation cache keyed by stem (gitignored)
- `state/shards/` -- Shard manifests for the parallel OCR orchestrator
- `state/status.json` -- Summary stats (run `just update-status` to refresh)
- `dataset/labels/*.txt` -- YOLO bounding box labels
