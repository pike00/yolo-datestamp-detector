# Bounding Box Annotator + YOLO Fine-Tune

**Date:** 2026-04-03
**Status:** Approved

## Goal

Build a keyboard-only bounding box annotation UI and a YOLO fine-tuning pipeline. The annotator lets a user label regions of interest on ~100 scanned photos, producing training data. The trainer fine-tunes YOLOv8 on that data to detect the labeled regions automatically on the remaining ~7,400 photos.

## Context

The parent project (`photo_project/`) is consolidating ~77K media files. ~7,500 scanned 4x6 photos (ScanMyPhotos Discs 1-4) have orange camera date stamps that need detection for date extraction. An OpenCV-based detector (`stamp_detect.py`) has an unacceptable false positive rate. Rather than continuing to tune heuristics, we train a YOLO model on human-annotated ground truth.

Source photos: `../photo_mapping_samples/` (100 JPGs, pre-selected sample across discs).

## Architecture

Single Python package with two entry points:

```
yolo_finetune/
  annotate.py      # HTTP server + REST API for annotation
  train.py         # YOLO fine-tuning using ultralytics
  index.html       # Browser-based annotation UI (single page app)
  dataset/
    images/        # Symlinks to source photos (only labeled ones)
    labels/        # YOLO-format .txt files (one per image)
    data.yaml      # YOLO dataset config
  progress.json    # Annotation session state (resume support)
  skipped.txt      # Filenames the user skipped (no region of interest)
```

## Annotation UI

### Technology

Browser-based: Python `http.server` backend serves the UI and a REST API. The frontend is a single `index.html` with vanilla JS and Canvas rendering. No build step, no framework dependencies.

### Workflow

1. Run `python annotate.py` — starts server on `localhost:8888`, opens browser.
2. First photo loads with a centered default box (20% image width, 10% height).
3. User adjusts the box with keyboard shortcuts and presses Enter to confirm, or S to skip.
4. Next photo loads with the box in the same position/size as the last confirmed annotation.
5. User can go back (Z) to fix mistakes, quit (Q) and resume later.

### Keyboard Controls

| Key | Action |
|-----|--------|
| Arrow keys | Move box (20px steps) |
| Shift + arrows | Fine move (3px steps) |
| `[` / `]` | Shrink / grow width |
| `{` / `}` | Shrink / grow height |
| Enter | Confirm annotation, advance to next |
| S | Skip photo (no region of interest), advance |
| Z | Go back to previous photo |
| R | Reset box to default position |
| Q | Quit and save progress |

### Display

- Dark background, photo centered and scaled to fit viewport.
- Green bounding box overlay with corner handles.
- Top status bar: filename, progress (N/M), labeled count, skipped count.
- Bottom bar: keyboard shortcut reference.

### Persistence

- `progress.json` stores: current image index, last box position (normalized x, y, w, h), list of completed filenames.
- On startup, if `progress.json` exists, resume from where the user left off.
- Each confirm/skip writes immediately (no batch save).

## Output Format

Annotations are written directly in YOLO format — no conversion step.

### Per-image label file (`dataset/labels/{filename}.txt`)

```
0 0.85 0.92 0.18 0.06
```

Format: `class_id center_x center_y width height` — all values normalized to [0, 1] relative to image dimensions.

Single class: `0` (the annotated region of interest).

### Dataset config (`dataset/data.yaml`)

```yaml
path: .
train: images/train
val: images/val
names:
  0: target
```

### Train/val split

When `train.py` runs, it splits labeled images 80/20 into `images/train`, `images/val` (and corresponding `labels/train`, `labels/val`) using random shuffle with a fixed seed for reproducibility.

### Skipped photos (`skipped.txt`)

One filename per line. These are photos the user confirmed have no region of interest. During training, `train.py` symlinks these into the dataset as negative examples — images present in `images/` with no corresponding file in `labels/`. This teaches the model that not every photo contains a target.

## YOLO Fine-Tuning

### Model

- Base: `yolov8n.pt` (YOLOv8 nano — smallest, fastest, suitable for CPU training)
- Library: `ultralytics` (pip install)
- Single class detection

### Training config

- Epochs: 100 (with early stopping patience=10)
- Image size: 640
- Batch size: 8 (conservative for 27GB RAM, CPU-only)
- Device: cpu (no discrete GPU available)
- Augmentation: ultralytics defaults (mosaic, flip, scale, translate)

### Output

- Best weights: `runs/detect/train/weights/best.pt`
- Training metrics and plots in `runs/detect/train/`
- The trained model can then be used to run inference on the full ~7,500 photo set

### Usage after training

```python
from ultralytics import YOLO
model = YOLO("runs/detect/train/weights/best.pt")
results = model.predict("path/to/photo.jpg", conf=0.5)
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    # crop and OCR the detected region
```

## Dependencies

Add to `pyproject.toml`:
```
ultralytics>=8.0
```

The existing venv already has `torch`, `torchvision`, `pillow`, and `opencv` (via `stamp_detect.py` deps).

## What This Does NOT Include

- Multi-class annotation (single class only)
- Mouse-based interaction (keyboard only)
- Cloud training or GPU offload
- Integration with `stamp_detect.py` (that happens after training, outside this scope)
