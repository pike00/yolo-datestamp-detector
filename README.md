# yolo-datestamp-detector

Fine-tuned YOLO detector for orange LED date stamps in scanned 4x6 photos. Stamps are camera-imprinted in format `M D 'YY` (e.g., `10 3 '99`), spanning roughly 1986-2010. Handles rotated photos and missing stamps.

## What it does

1. **Detect** — YOLO (yolo26m fine-tuned) localizes the stamp region in each photo
2. **OCR** — Claude Haiku or local Gemma4 reads the date from the cropped region
3. **Review** — Browser-based UIs for annotating, correcting, and confirming results

## Quick start

```bash
# Install deps
uv sync

# Annotate images (label stamp regions)
just annotate          # :8888

# Train the model
just train

# Run batch inference
just infer

# Run OCR on detected stamps (requires ANTHROPIC_API_KEY)
just ocr

# Full cycle: train → infer
just cycle
```

Run `just` to list all recipes.

## Architecture

```
scripts/
  train/
    train.py                 # YOLO fine-tuning (yolo26m, CPU or GPU)
    regen_val_plots.py       # Refresh validation plots
  infer/
    infer_all.py             # Batch inference on pending images
    compare_predictions.py   # Diff old vs new model predictions
    render_drift_examples.py # Visualize prediction drift
  annotate/
    annotate.py              # Annotation server (:8888)
    corrections_dashboard.py # Review/correct predictions (:8889)
    feedback.py              # Feedback loop: prepare → correct → finalize
  ocr/
    ocr_stamps.py            # OCR via Claude Haiku (ANTHROPIC_API_KEY required)
    ocr_gemma.py             # OCR via local Gemma4 (Ollama required)
    orchestrate_ocr.py       # Parallel Haiku OCR orchestrator (sharded)
    build_pilot_review_html.py  # Render OCR pilot review HTML
  _db.py                     # Shared Postgres helpers

docker/
  docker-compose.yml         # Train/infer Docker services
  docker-compose.ocr.yml     # Gemma4 OCR container

dataset/
  data.yaml                  # YOLO dataset config (single class: "target")
  labels/                    # YOLO-format bounding box labels
  corrections/               # Corrected labels from feedback loop

state/                       # Runtime state (gitignored)
runs/                        # Training artifacts and weights (gitignored)
scanmyphotos/                # Source images (gitignored)
```

## Data stores

PostgreSQL (`dedup` database, see `scripts/_db.py`):

| Table | Contents |
|---|---|
| `stamp_predictions` | YOLO bbox per stem, model label tracked in `model` column |
| `stamp_ocr` | OCR results, composite PK `(stem, model)` |
| `stamp_prediction_drift` | Old vs new bbox diff with IoU |
| `stamp_no_stamp` | Stems confirmed to have no date stamp |
| `stamp_rotations` | User-confirmed rotations |

Default connection: `postgresql://dedup:dedup_local_dev@localhost:5432/dedup`  
Override with `DATABASE_URL`.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgresql://dedup:dedup_local_dev@localhost:5432/dedup` | Postgres connection |
| `YOLO_MODEL_LABEL` | `yolo26m-best` | Label written into `stamp_predictions.model` |
| `IMAGE_DIR` | — | Source image directory for annotation |
| `DISC_DIRS` | — | Colon-separated source directories (stratified sampling) |

## Training config

- Base model: `yolo26m.pt` (20.4M params, medium)
- Single class `0` = stamp region (called `target` in data.yaml)
- `imgsz=640`, `batch=4`, `epochs=100`, `patience=10`, `device=cpu`

## Stamp characteristics

- Color: orange/red/amber LED digits, camera-imprinted
- Location: typically bottom edge; rotated photos may have stamps on side edges
- Format: `M D 'YY` — e.g., `10 3 '99`
- Era: ~1986-2010
- ~30% of photos have no stamp
