# YOLO Date Stamp Detector

Fine-tuned YOLOv8 model to detect camera date stamp regions on scanned photographs.

Many consumer cameras from the 1980s-2000s imprinted date stamps directly onto film --
small orange/red/amber LED digits (typically `M D 'YY`, e.g. "10 3 '99") burned into
the bottom edge of each photo. When these photos are later bulk-scanned, the date stamps
become the only reliable source of temporal metadata.

This project trains a single-class object detector to locate these stamp regions, enabling
downstream OCR to extract actual dates and write them back as EXIF metadata.

## Results

Trained on ~3,000 hand-labeled scanned photos, the model achieves:

| Metric | Value |
|--------|-------|
| Precision | 95.3% |
| Recall | 95.8% |
| mAP@50 | 95.0% |
| mAP@50-95 | 73.8% |

Training converged in 27 epochs (early-stopped at 37/100). The lower mAP@50-95 reflects
some imprecision in tight bounding box localization, which is acceptable since the box
only needs to roughly locate the stamp region for OCR cropping.

### Confidence Distribution

Batch inference on ~7,500 images produced 6,458 detections:

- **85%** of detections have confidence >= 0.50
- **71%** have confidence >= 0.70
- Bimodal distribution: strong peak at 0.7-0.9 (true positives) and secondary cluster at 0.3-0.4 (ambiguous/borderline cases needing manual review)

## Approach

### Why YOLO?

Initial attempts used OpenCV heuristics (color filtering for orange digits, edge detection)
but these proved unreliable -- date stamps vary in color, brightness, position, and some
photos have orange-tinted content that triggers false positives. A learned detector
generalizes far better from labeled examples.

YOLOv8-nano was chosen because:
- Single-class detection is a simple task; nano is sufficient
- All training runs on CPU (no GPU required)
- Fast inference (~50ms/image on CPU) enables batch processing thousands of photos

### Pipeline

```
 Annotate          Train           Infer           Review          OCR
 (browser UI) --> (YOLOv8n) --> (batch pred) --> (dashboard) --> (LLM/Tesseract)
      |               |              |               |               |
  human labels    fine-tune     predictions    corrections      date strings
  (bbox + skip)   on labels    (confidence)   (confirm/edit)   (EXIF-ready)
```

1. **Annotate** -- Browser-based labeling UI for drawing bounding boxes around date stamps. Keyboard-driven workflow (arrow keys to navigate, click-drag to draw). Skipped photos become negative training examples.

2. **Train** -- YOLOv8-nano fine-tuned on labeled data. Automatic train/val split (80/20). Resumes from previous best weights if available. Early stopping prevents overfitting.

3. **Infer** -- Batch inference on all unlabeled photos. Low confidence threshold (0.01) to catch all candidates. Predictions saved as JSON for review.

4. **Review** -- Corrections dashboard for reviewing predictions. Supports confirm, edit bbox, mark as no-stamp, and skip. Bulk approve for high-confidence batches. Handles rotated photos.

5. **OCR** -- Crops detected stamp regions and sends to an LLM (Claude Haiku) or Tesseract for text extraction. Tracks token usage and cost.

The pipeline is iterative: corrections from step 4 feed back into training data for the next training round, improving the model over time.

### Handling Rotation

Some scanned photos are rotated 90/180/270 degrees. The corrections dashboard supports
rotation during review, and bounding box coordinates are transformed back to the original
image coordinate space before saving labels. Rotation metadata is stored in PostgreSQL
for downstream processing.

## Setup

### Prerequisites

- Python 3.12+ via [uv](https://github.com/astral-sh/uv)
- [just](https://github.com/casey/just) task runner
- PostgreSQL (optional, for corrections dashboard rotation tracking)
- No GPU required

### Quick Start

```sh
git clone https://github.com/pike00/yolo-datestamp-detector.git
cd yolo-datestamp-detector

# uv handles dependencies automatically via inline PEP 723 script headers.
# No pip install or requirements.txt needed.

# Place scanned photo JPGs in scanmyphotos/ with naming: d{disc}_{number}.jpg
# Or use your own images -- any JPGs in scanmyphotos/ work for annotation.

# Start annotating
just annotate

# Train the model (after labeling some images)
just train

# Run batch inference
just infer

# Review predictions
just dashboard
```

### Data Setup

Source images go in `scanmyphotos/` (gitignored). The naming convention is
`d{disc}_{filename}.jpg` (e.g. `d1_00000133.jpg`), but any JPGs will work.

If you have a PostgreSQL database with file metadata, `setup_scanmyphotos.py` can
query it and copy images automatically. Configure via environment variables:

```sh
export ORIGINALS_DIR=/path/to/deduplicated/originals
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=dedup
export DB_USER=dedup
export DB_PASSWORD=changeme
just setup-scanmyphotos
```

## Usage

```sh
just                    # List all commands
just annotate           # Label bounding boxes (browser UI, :8888)
just train              # Train model (resumes from best.pt)
just infer              # Batch inference on pending images
just cycle              # Train then infer
just dashboard          # Corrections dashboard (:8889)
just ocr                # OCR detected stamps (requires ANTHROPIC_API_KEY)
just stats              # Dataset statistics
just update-status      # Refresh status.json
just tensorboard        # Training metrics
just infer-one <photo>  # Single-image inference
```

## Project Structure

```
.
|-- annotate.py              # Annotation server + REST API
|-- index.html               # Browser annotation UI (vanilla JS + Canvas)
|-- train.py                 # YOLO fine-tuning with train/val split
|-- infer_all.py             # Batch inference with progress tracking
|-- corrections_dashboard.py # Prediction review/correction server
|-- dashboard.html           # Corrections dashboard UI
|-- batch_review.html        # Bulk review UI for high-confidence predictions
|-- feedback.py              # Feedback loop orchestration
|-- ocr_stamps.py            # Date stamp OCR via Claude Haiku
|-- setup_scanmyphotos.py    # Optional: import images from dedup database
|-- stratified_sample.py     # Stratified sampling across image sources
|-- dataset/
|   |-- data.yaml            # YOLO dataset config
|   |-- labels/              # YOLO-format bounding box labels
|   |-- corrections/         # Corrected labels from feedback loop
|   `-- to_annotate/         # Staging area for correction annotation
|-- examples/                # Sample photos for reference (committed)
|-- scanmyphotos/            # Working image directory (gitignored)
|-- runs/                    # Training runs + model weights (gitignored)
`-- status.json              # Current project statistics
```

## Examples

The `examples/` directory contains sample photos:

- `stamp_*.jpg` -- Photos with visible date stamps, with `.txt` YOLO label files
- `no_stamp_*.jpg` -- Photos without date stamps (negative examples)

## Model Details

| Parameter | Value |
|-----------|-------|
| Base model | YOLOv8-nano (3M params, 8.2 GFLOPs) |
| Classes | 1 ("target" = date stamp region) |
| Training image size | 640px |
| Inference image size | 384px |
| Batch size | 8 |
| Early stopping | patience=10 |
| Confidence threshold | 0.01 (inference), 0.30 (recommended operational) |

## Docker

```sh
docker compose up cycle    # Train + infer in container
```

Mounts `dataset/`, `scanmyphotos/`, and `runs/` as volumes.

## License

MIT
