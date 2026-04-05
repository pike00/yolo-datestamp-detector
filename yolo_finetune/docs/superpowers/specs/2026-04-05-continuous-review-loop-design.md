# Continuous Review Loop for ScanMyPhotos Date Stamp Detection

## Purpose

Enable continuous improvement of the YOLOv8 date stamp detector across all 7,455 ScanMyPhotos images. The dashboard auto-queues every file, walks the user through review in a continuous loop, and supports background training + re-inference so the model improves as you work.

## Scope

- Extends the existing corrections dashboard (not a rewrite)
- Targets all 7,455 deduplicated ScanMyPhotos JPGs across 4 discs
- Single `uv run corrections_dashboard.py` to start, browser-based review
- Background training and inference via subprocesses

## File Setup

### Source Data

- 7,455 unique ScanMyPhotos JPGs copied from `originals/` (content-hash dedup collection) to `yolo_finetune/scanmyphotos/` with disc-prefixed names: `d{N}_{stem}.jpg` (e.g., `d1_00000001.jpg`, `d3_00001642.jpg`)
- Manifest at `scanmyphotos_manifest.json` maps disc number, original filename, and SHA-256 hash for each file
- Setup script `setup_scanmyphotos.py` queries PostgreSQL dedup database for the ScanMyPhotos subset, copies files (never moves), and writes the manifest
- Files are read-only copies from `originals/` -- source files are never modified

### Naming Convention

- Working name format: `d{disc}_{original_stem}` (e.g., `d1_00000001`)
- Display format in UI: `D1 - 00000001`
- This resolves filename overlaps across discs (each disc starts from 00000001.jpg)

## Continuous Review Loop

### Startup

1. Dashboard loads all 7,455 files into the queue from `scanmyphotos/`
2. Loads any existing predictions from `scanmyphotos_predictions.json`
3. Loads existing labels from `dataset/labels/` and `skipped.txt` to mark already-reviewed files
4. Filters default to "Pending" -- user sees only unreviewed files
5. First pending file auto-selects and loads in the canvas

### Review Cycle

For each file, the user sees the photo with any prediction box overlaid on canvas:

- **Enter** -- confirm the box as-is (saves YOLO label, auto-advances to next pending)
- **Arrow keys** -- move box (Shift for fine movement), then Enter to confirm
- **[ ] { }** -- resize box width/height, then Enter to confirm
- **S** -- mark as "no stamp" (adds to skipped.txt as negative example, auto-advances)
- **Skip button** -- defer for later review (auto-advances)

### Progress Display

Header bar shows real-time stats:
```
Reviewed: 142 / 7,455 (1.9%) | Labels: 98 | No stamp: 44 | Skipped: 12
```

### Image Serving

Dashboard always serves the clean original image from `scanmyphotos/`. Predictions and user corrections render as canvas overlays (not baked into the image), so boxes are always editable.

## Background Training + Re-inference

### Trigger

"Train + Re-infer" button in the action panel. Disabled while a worker is already running.

### Execution

1. Dashboard spawns a subprocess that runs training then inference sequentially
2. Training: `train.py` runs YOLOv8 fine-tuning with all accumulated labels
3. Inference: `infer_all.py` (new script) runs the updated model on all pending (unreviewed) files
4. User continues reviewing during both phases -- the dashboard stays responsive

### Progress Communication

Subprocess writes progress to `worker_status.json`:
```json
{"phase": "training", "progress": 45, "total": 100, "message": "Epoch 45/100"}
```
```json
{"phase": "inference", "progress": 1200, "total": 6500, "message": "1200/6500 inferred"}
```
```json
{"phase": "done", "message": "Training complete. 6500 files inferred."}
```

Dashboard JavaScript polls `GET /api/worker-status` every 2 seconds and updates a status indicator in the UI.

### Prediction Storage

`infer_all.py` writes predictions to `scanmyphotos_predictions.json`:
```json
{
  "d1_00000001": {"x": 0.82, "y": 0.93, "w": 0.15, "h": 0.05, "confidence": 0.87},
  "d3_00001642": {"x": 0.78, "y": 0.91, "w": 0.18, "h": 0.06, "confidence": 0.92}
}
```

After inference completes, dashboard reloads predictions on next queue refresh. Previously-pending files now show predicted boxes for quick confirm-or-edit.

## infer_all.py

New script for batch inference on the full ScanMyPhotos collection:

- Input: `scanmyphotos/` directory (7,455 JPGs)
- Model: `runs/detect/train/weights/best.pt` (latest trained model)
- Skips files that already have labels in `dataset/labels/` or stems in `skipped.txt`
- Writes normalized YOLO-format predictions to `scanmyphotos_predictions.json`
- Writes progress to `worker_status.json` for dashboard polling
- CPU-only inference, imgsz=384, conf=0.3 (matching existing infer.py settings)

## File Changes

### Create

| File | Purpose |
|------|---------|
| `infer_all.py` | Batch inference on scanmyphotos/, writes predictions + worker status |

### Modify

| File | Changes |
|------|---------|
| `corrections_dashboard.py` | Swap SAMPLE_DIR to scanmyphotos/, add `/api/worker-status` endpoint, update `train()` to chain train+infer as background subprocess, add stats to `/api/queue` response, serve images from scanmyphotos/ |
| `dashboard.html` | Auto-select first pending file on load, add progress bar header, worker status polling/display, default filter to "pending", replace "Train Now" with "Train + Re-infer", load predictions for canvas overlay |
| `train.py` | Update `setup_dataset()` to pull images from `scanmyphotos/` instead of `photo_mapping_samples/`. Label files use matching stems (e.g., image `d1_00000001.jpg` pairs with label `d1_00000001.txt`). |

### Runtime (created/updated at runtime)

| File | Purpose |
|------|---------|
| `scanmyphotos_predictions.json` | Model predictions for pending files |
| `worker_status.json` | Training/inference progress for UI polling |
| `corrections_queue.json` | Persistent queue state (reset for new file set) |

### Unchanged

`setup_scanmyphotos.py`, `annotate.py`, `index.html`, `feedback.py`, `scanmyphotos_manifest.json`

## Queue Structure

Queue entries use working_name stem as identifier:

```json
{
  "files": [
    {
      "stem": "d1_00000001",
      "disc": 1,
      "display_name": "D1 - 00000001",
      "status": "pending",
      "created_at": "2026-04-05T...",
      "last_reviewed_at": null,
      "prediction": {"x": 0.82, "y": 0.93, "w": 0.15, "h": 0.05, "confidence": 0.87},
      "user_correction": null
    }
  ],
  "stats": {
    "total": 7455,
    "pending": 7313,
    "labeled": 98,
    "no_stamp": 44,
    "skipped": 0
  }
}
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| Arrow keys | Move box (20px steps) |
| Shift + arrows | Fine move (3px steps) |
| `[` / `]` | Shrink/grow width |
| `{` / `}` | Shrink/grow height |
| Enter | Confirm box and advance to next |
| S | Mark no stamp and advance to next |
