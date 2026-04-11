# YOLO Date Stamp Detector

Fine-tuned YOLOv8 model to detect camera date stamp regions on scanned photographs.

Many consumer cameras from the 1980s-2000s imprinted date stamps directly onto film --
small orange/red/amber LED digits (typically `M D 'YY`, e.g. "10 3 '99") burned into
the bottom edge of each photo. When these photos are later bulk-scanned, the date stamps
become the only reliable source of temporal metadata.

This project trains a single-class object detector to locate these stamp regions, enabling
downstream OCR to extract actual dates and write them back as EXIF metadata.

### Detection Example

| Input scan | Model output (conf 0.82) |
|:---:|:---:|
| ![Original photo](examples/stamp_golden_gate.jpg) | ![Detection result](examples/detection_example.jpg) |

The model draws a bounding box around the orange "4 23 '95" date stamp in the bottom-right corner.

## Results

Trained on ~3,000 hand-labeled scanned photos, the model achieves:

| Metric | Value |
|--------|-------|
| Precision | 95.3% |
| Recall | 95.8% |
| mAP@50 | 95.0% |
| mAP@50-95 | 73.8% |
| F1 (optimal) | 0.96 @ conf 0.37 |

Training converged in 27 epochs (early-stopped at 37/100). The lower mAP@50-95 reflects
some imprecision in tight bounding box localization, which is acceptable since the box
only needs to roughly locate the stamp region for OCR cropping.

### Training Curves

![Training curves](examples/training_curves.png)

All losses (box, classification, DFL) decrease smoothly across training and validation sets.
Precision, recall, and mAP metrics stabilize around epoch 20, confirming convergence without
overfitting.

### Precision-Recall Curve

![Precision-Recall curve](examples/precision_recall_curve.png)

The PR curve hugs the top-right corner with 0.950 mAP@0.5 -- near-perfect precision is
maintained across almost the entire recall range before dropping off at the very tail.

### F1-Confidence Curve

![F1-Confidence curve](examples/f1_confidence_curve.png)

Peak F1 of 0.96 at confidence threshold 0.37. The broad plateau from 0.1-0.7 means the
model is robust to threshold selection -- you don't need to fine-tune the threshold to
get good results.

### Confusion Matrix

![Confusion matrix](examples/confusion_matrix.png)

96% true positive rate with only 4% of stamps missed. Zero false positives on background
images -- the model never hallucinates a stamp where none exists.

### Confidence Distribution

![Confidence distribution](examples/confidence_distribution.png)

Batch inference on ~7,500 images produced 6,458 detections. The bimodal distribution shows
a strong high-confidence peak (0.7-0.9, true positives) and a secondary cluster around
0.3-0.4 (borderline cases requiring manual review).

## Approach

### Why YOLO?

Initial attempts used OpenCV heuristics (color filtering for orange digits, edge detection)
but these proved unreliable -- date stamps vary in color, brightness, position, and some
photos have orange-tinted content that triggers false positives. A learned detector
generalizes far better from labeled examples.

YOLO (You Only Look Once) is a family of object detection models that process an entire
image in a single forward pass, predicting bounding box locations and class labels
simultaneously. This project uses YOLOv8-nano (3.2M parameters), the smallest variant
from [Ultralytics](https://github.com/ultralytics/ultralytics). It starts from weights
pre-trained on the COCO dataset (330K images, 80 object classes), then fine-tunes on a
few thousand labeled date stamp examples -- a transfer learning approach that converges
quickly even on CPU without a GPU.

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
just update-status      # Refresh state/status.json
just tensorboard        # Training metrics
just infer-one <photo>  # Single-image inference
```

## Project Structure

```
.
|-- scripts/
|   |-- train.py                 # YOLO fine-tuning with train/val split
|   |-- infer_all.py             # Batch inference with progress tracking
|   |-- annotate.py              # Annotation server + REST API
|   |-- corrections_dashboard.py # Prediction review/correction server
|   |-- feedback.py              # Feedback loop orchestration
|   |-- ocr_stamps.py            # Date stamp OCR via Claude Haiku
|   |-- ocr_gemma.py             # Date stamp OCR via local Gemma4
|   |-- augment_hard_cases.py    # Data augmentation for failure modes
|   |-- enhance_stamps.py        # Stamp enhancement experiments
|   |-- setup_scanmyphotos.py    # Import images from dedup database
|   `-- stratified_sample.py     # Stratified sampling across image sources
|-- ui/
|   |-- index.html               # Browser annotation UI (vanilla JS + Canvas)
|   |-- dashboard.html           # Corrections dashboard UI
|   `-- batch_review.html        # Bulk review UI for high-confidence predictions
|-- state/                       # Runtime state files (gitignored, except skipped.txt)
|-- output/                      # Inference visualizations and previews (gitignored)
|-- docker/                      # Dockerfiles and compose configs
|-- dataset/
|   |-- data.yaml                # YOLO dataset config
|   |-- labels/                  # YOLO-format bounding box labels
|   |-- corrections/             # Corrected labels from feedback loop
|   `-- to_annotate/             # Staging area for correction annotation
|-- examples/                    # Sample photos and model evaluation plots
|-- scanmyphotos/                # Working image directory (gitignored)
`-- runs/                        # Training runs + model weights (gitignored)
```

## Model Details

| Parameter | Value |
|-----------|-------|
| Base model | YOLOv8-nano (3M params, 8.2 GFLOPs) |
| Classes | 1 ("target" = date stamp region) |
| Training image size | 640px |
| Inference image size | 384px |
| Batch size | 8 |
| Early stopping | patience=10 |
| Confidence threshold | 0.01 (batch inference), 0.35 (recommended operational) |

## Docker

```sh
docker compose -f docker/docker-compose.yml up cycle    # Train + infer in container
```

Mounts `dataset/`, `scanmyphotos/`, and `runs/` as volumes.

## License

MIT
