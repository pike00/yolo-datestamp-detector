# yolo-datestamp-detector

YOLO fine-tuned detector for orange LED date stamps in scanned 4x6 photos, plus an OCR
pipeline (Claude Haiku or local Gemma4) that reads the cropped stamp. See `README.md` for
the user-facing summary; this file captures non-obvious facts for Claude Code.

## Critical Constraints

- **NEVER modify files on HDD** at `/mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/` (read-only source).
- **No external API calls** (Gemini, OpenAI, Anthropic) without explicit user approval.
  The Haiku OCR path is the exception and gates on `ANTHROPIC_API_KEY` being set.
- Local LLM via Ollama is fine when installed (used by `ocr_gemma.py` and the bench harness).

## Repo Layout vs Parent

This repo lives at `/home/will/photo_project/yolo-datestamp-detector/`. Some directories
the scripts expect live at the **parent** `/home/will/photo_project/`, not inside the repo:

- `scanmyphotos/` -- source images (gitignored; symlinks/copies from dedup originals)
- `runs/` -- training artifacts and weights (gitignored)
- `originals/`, `organized/`, `needs_date/` -- dedup pipeline scratch

Scripts compute `BASE_DIR = scripts/<role>/<script>.py -> ../../..`, which resolves to the
**repo root**, so `BASE_DIR / "runs"` and `BASE_DIR / "scanmyphotos"` are expected inside
the repo. If they are missing, symlink from the parent or set env-var overrides (below).

## Environment

- Python 3.14 (`.python-version`). No top-level `pyproject.toml`. Every script under
  `scripts/` uses an inline PEP 723 header (`# /// script ... # ///`) and is run via
  `uv run`. `just train`, `just infer`, etc. wrap this.
- CPU-only on workstation (AMD Ryzen, no discrete GPU). GPU paths exist for spot bench
  (`scripts/train/gpu_bench_one_epoch.py`) and the OCR fleet container.
- Postgres `dedup` DB on localhost:5432, user `dedup`, default password `dedup_local_dev`.
  Override with `DATABASE_URL`. Schema in `schema/*.sql`. Nightly pg_dump via the
  `dedup-db-backup` sidecar in Homelab infra (added 2026-04-20).
- Task runner: `just` (run `just` to list all recipes).

## Key Env Vars

- `YOLO_WEIGHTS` -- absolute path to weights; default `runs/detect/gpu-40ep/weights/best.pt`.
  Set this when iterating across model versions without copying files (`infer_all.py:33`).
- `YOLO_MODEL_LABEL` -- label written into `stamp_predictions.model` (default `yolo26m-best`).
- `INFER_FORCE` -- if set, re-runs inference even on stems already in `stamp_predictions`.
- `DATABASE_URL` -- Postgres connection; used by every script that touches stamp tables.
- `IMAGE_DIR` -- source dir for `annotate.py`.
- `DISC_DIRS` -- colon-separated source dirs for `stratified_sample.py`.
- `ANTHROPIC_API_KEY` -- required by `ocr_stamps.py` and the Haiku orchestrator.
- `APPRISE_URL` -- training notifications (Mattermost / etc.); `--no-notify` to silence.

## Training Config

- Base: `yolo26m.pt` (20.4M params, medium).
- Single class `0` = stamp region (called `target` in `dataset/data.yaml`).
- Training: `imgsz=640`, `batch=4`, `epochs=100`, `patience=10`, `device=cpu`.
- **Inference uses `imgsz=384`, `conf=0.01`** (`infer_all.py:36-37`). The lower conf
  feeds the no-stamp sweep that writes `stamp_no_stamp` rows with `source='auto'`.
- `dataset/labels/{train,val}/*.txt` IS tracked in git (human-curated, ~1.5 MB,
  non-regenerable, lost once on 2026-04-16). Do not add `dataset/labels/` to gitignore.
- `*.pt` weights and `runs/` are gitignored. Weights live on the host filesystem only.

## Date Stamp Characteristics

- Orange/red/amber LED digits, camera-imprinted, typically bottom edge of photo.
- Format: `M D 'YY` (e.g., `10 3 '99`), spanning ~1986-2010.
- ~30% of photos have no stamp -- detector must handle `found: False` cleanly.
- Rotated photos (90/180/270) may have stamps on side or top edges;
  `scripts/data/detect_rotation_batch.py` pre-computes rotation predictions.

## Postgres Tables (in `dedup` DB)

See `scripts/_db.py` for helpers and `schema/stamp_tables.sql` for DDL.

- `stamp_predictions` -- YOLO bbox per stem, `model` column tracks model label.
- `stamp_ocr` -- OCR results, composite PK `(stem, model)`; `stage`, `host_label`,
  `parsed_date`, `parse_error`, `review_status` drive the two-stage pipeline.
- `stamp_prediction_drift` -- old vs new bbox diff with IoU and flag.
- `stamp_no_stamp` -- stems confirmed to have no date stamp (includes auto-swept
  `source='auto'` rows from the conf < 0.05 pass).
- `stamp_rotations` -- user-confirmed rotations from corrections_dashboard.
- `scanmyphotos_manifest` -- stem -> sha256/disc/source_path for the 7,454 ScanMyPhotos scans.
- `exif_dates`, `video_dates` -- PIL/ffprobe extracted timestamps per sha256.
- Views: `media_dates` (union over exif/video/stamp_ocr), `media_has_date`.

## State Files Still on Disk

Bench results and a few orchestrator inputs persist as JSON, not Postgres:

- `state/corrections_queue.json` -- review queue (gitignored).
- `state/scanmyphotos_manifest.json` -- source-file map (gitignored).
- `state/rotation_predictions.json` -- local rotation cache keyed by stem (gitignored).
- `state/shards/` -- shard manifests for parallel OCR orchestrator; input shards are
  gitignored, `*_result.json` IS tracked (expensive OCR output, ~2 KB/shard).
- `state/status.json` -- summary stats (`just update-status`).
- `state/bench/` -- VLM bench corpus, ground truth, per-model runs (gitignored).

## OCR Pipeline

- `ocr_stamps.py` -- single-host Haiku runner.
- `orchestrate_ocr.py` -- sharded parallel Haiku orchestrator (crop -> merge -> reconcile).
- `ocr_fleet.py` -- containerized fleet runner; can post milestone crops to Mattermost
  (see commits 977aaf1, d815450).
- `ocr_gemma.py` -- local Gemma4 via Ollama (`docker compose -f docker/docker-compose.ocr.yml`).
- VLM bench: `bench_vlm_ocr.py`, `bench_vlm_litellm.py`, profile wrappers in
  `bench_profiles/{ares-cpu,m2pro,cloud-gpu,ollama-cloud}.sh`. Build corpus once with
  `just bench-build`, freeze ground truth with `just bench-seed-{plan,review,freeze}`,
  run with `just bench-run <profile> <model>`, report with `just bench-report`.

## Common Commands

```bash
just                       # list all recipes
just train                 # uv run scripts/train/train.py
just infer                 # batch inference; honors YOLO_WEIGHTS
just cycle                 # train + infer
just annotate              # :8888 annotation UI
just dashboard             # :8889 corrections review UI
just ocr                   # Haiku OCR (needs ANTHROPIC_API_KEY)
just ocr-gemma             # Gemma4 OCR via Docker + Ollama
just stats                 # dataset stats (reads Postgres)
just update-status         # refresh state/status.json
just infer-one <path>      # single-image inference (imgsz=384, conf=0.35 default)
just tensorboard           # serves runs/detect over uvx tensorboard
```

## Superpowers Skill Save Paths

Override defaults when using superpowers skills in this repo:

- `brainstorming` saves specs to `docs/specs/` (not `docs/superpowers/specs/`).
- `writing-plans` saves plans to `docs/plans/` (not `docs/superpowers/plans/`).
