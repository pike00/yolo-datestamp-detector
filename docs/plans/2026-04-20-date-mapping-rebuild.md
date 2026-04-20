---
date: 2026-04-20
status: proposed
owner: will
---

# Date-Mapping Rebuild (ScanMyPhotos stamp OCR → unified media dates)

## Why

The undated-media gallery (10,378 photos + 1,090 videos) is currently dominated by the ~7,455 ScanMyPhotos scans whose dates live in printed date stamps, not EXIF. The YOLO→crop→OCR pipeline that previously extracted those dates was keyed in `stamp_predictions` / `stamp_ocr` / `stamp_no_stamp` / `stamp_prediction_drift` tables in `dedup-postgres`. Those tables were **wiped on 2026-04-16 22:25 UTC** when the dedup-postgres volume was reinitialized for the pgvector:pg18 upgrade — the handoff for that work explicitly called the missing stamp tables out as a pre-existing concern and they were never backed up (confirmed by exhaustive search on 2026-04-19 across Postgres, kopia, HDD dumps, rclone remotes, git, and Loki).

Goal: rebuild the stamp pipeline end-to-end, populate a unified media-dates view that joins EXIF / ffprobe / stamp OCR, and regenerate the undated gallery against it.

## Pre-flight: What survives

| Asset | Location | Status |
|---|---|---|
| YOLO weights (best) | [runs/detect/train/weights/best.pt](runs/detect/train/weights/best.pt) | 60 MB, Apr 16 — kept |
| YOLO weights (40ep) | [runs/detect/gpu-40ep/weights/best.pt](runs/detect/gpu-40ep/weights/best.pt) | 44 MB, Apr 16 — alt |
| ScanMyPhotos symlinks | [scanmyphotos/](scanmyphotos/) | 7,455 disc-prefixed symlinks (`d{1..4}_NNNNNNNN.jpg`) → HDD source |
| Inference script | [scripts/infer/infer_all.py](scripts/infer/infer_all.py) | Ready; expects `runs/detect/train/weights/best.pt`, writes `stamp_predictions` |
| OCR orchestrator | [scripts/ocr/orchestrate_ocr.py](scripts/ocr/orchestrate_ocr.py) | Stage-1 + stage-2 crop/merge; subagent-driven Haiku |
| OCR scripts | [scripts/ocr/ocr_stamps.py](scripts/ocr/ocr_stamps.py), [ocr_gemma.py](scripts/ocr/ocr_gemma.py) | Haiku (cloud) + Gemma (local Ollama) |
| Corrections dashboard | [scripts/annotate/corrections_dashboard.py](scripts/annotate/corrections_dashboard.py) | :8889, reads/writes `stamp_predictions`/`stamp_ocr`/`stamp_rotations` |
| Annotation UI | [scripts/annotate/annotate.py](scripts/annotate/annotate.py) | :8888, for bbox relabelling |
| DB abstraction | [scripts/_db.py](scripts/_db.py) | Complete — all helper functions present, just missing tables to write into |
| Video dates | [state/video_dates.json](state/video_dates.json) | 4,206 with creation_time, 6 with `date:null`, out of 4,212 probed |
| Dedup dump backup | `dedup_db_backup` volume + `/mnt/823.../backups/dedup/` | Set up 2026-04-20 so this doesn't happen again |

## What's lost and needs rebuilding

1. **All four stamp_* tables** (stamp_predictions, stamp_ocr, stamp_no_stamp, stamp_prediction_drift) — schema needs to be re-emitted. `_db.py` is the reference for column sets.
2. **`scanmyphotos_manifest`** — mapping disc+stem → source path → organized-media sha256. Never existed as a first-class object; the previous pipeline keyed by `stem` alone and didn't cross-link to sha256. This is the biggest gap to close.
3. **`dataset/labels/`** — only 3 txt files remain out of 3,124 training labels. Not blocking for inference on existing weights, but blocks retraining.
4. **state/shards/** — only `state/shards/stage2/` exists (empty). Stage-1 shard history is gone. OK because `orchestrate_ocr.py crop-stage1` will re-emit.
5. **state/corrections_queue.json, rotation_predictions.json** — regenerable from `stamp_predictions` + `corrections_dashboard`.

## Design decisions

### D1. Key by `stem`, not `sha256`

Keep the existing `stem`-keyed schema (`d1_00000549`) unchanged. All existing code, UI, and scripts assume this. A separate small `scanmyphotos_manifest` table maps stem → sha256 and source path; joining through it is cheap (7,455 rows).

Rationale: rewriting every stamp script and the corrections dashboard to be sha256-keyed is a huge blast radius for marginal benefit. The stem-based pipeline is battle-tested; we only need the join table.

### D2. Compute sha256 manifest once, in Python, parallelized

For each `scanmyphotos/d{N}_NNNNNNNN.jpg` symlink, resolve to HDD source, compute sha256 of file contents (matches the `originals/media/{sha256}.ext` naming convention used elsewhere). Store `(stem, disc, source_path, sha256)` in `scanmyphotos_manifest`.

Cost: 7,455 files × ~3MB avg = ~22 GB read off spinning disk. At ~100 MB/s sustained read that's ~4 minutes. Hash is not the bottleneck. 12-way parallel with `concurrent.futures` to overlap IO + CPU.

### D3. Unified `media_dates` view

A view, not a materialized table, so it always reflects current stamp_ocr + manifest + video_dates state:

```sql
CREATE VIEW media_dates AS
-- EXIF-only photos
SELECT pe.sha256, 'exif' AS source, ...  -- TBD: need an exif_dates table OR compute on-the-fly
FROM photo_embeddings pe
WHERE pe.media_type = 'photo' AND <has_exif_check>

UNION ALL
-- ScanMyPhotos via stamp OCR
SELECT m.sha256, 'stamp_ocr', so.parsed_date, so.confidence, so.review_status
FROM scanmyphotos_manifest m
JOIN stamp_ocr so ON so.stem = m.stem AND so.model = 'haiku'
WHERE so.parsed_date IS NOT NULL

UNION ALL
-- Videos via ffprobe creation_time
SELECT sha256, 'ffprobe', date, NULL, NULL
FROM video_dates  -- imported from state/video_dates.json
;
```

Open question: EXIF dates are currently not stored in Postgres at all (scan done on-demand via PIL). Task 8 decides: import to an `exif_dates` table (one row per sha256) or keep on-disk.

### D4. Gemma (local) first-pass + Haiku for low-confidence

Memory notes Haiku confuses 5/9 in LED digits; stage-2 dual-crop verification exists for that. But cost and privacy favour running Gemma first on all 7,455 images, then only routing low-confidence results to Haiku. Validated in `project_local_vlm_benchmark.md`: "Sonnet decisively outperforms gemma4:e4b and qwen3-vl:4b for date stamp OCR; use cloud models for production." So skip Gemma first pass — go straight to Haiku. Keep Gemma as a smoke/sanity check.

### D5. Parsed-date column

Add `parsed_date DATE` and `parse_error TEXT` to `stamp_ocr` so downstream joins don't have to reparse `raw_text`. Parse once at merge time.

## Tasks

Each task has a verification step. Stop and escalate on any failure.

### T1. Recreate `stamp_*` schema — 15 min

**Why**: Nothing else works until these tables exist.

**What**: Emit `schema/stamp_tables.sql` with the full DDL (inferred from [_db.py](scripts/_db.py) column usage). Columns + constraints:

- `stamp_predictions`: stem PK (text), x/y/w/h int, confidence real, model text, updated_at timestamptz default now()
- `stamp_no_stamp`: stem PK (text), source text default 'user', added_at timestamptz default now()
- `stamp_prediction_drift`: stem PK (text), old_{x,y,w,h,confidence}, new_{x,y,w,h,confidence}, iou real, flag text, updated_at
- `stamp_ocr`: (stem text, model text) composite PK, raw_text text, bbox_source text, confidence real, stage int, review_status text, parsed_date date, parse_error text, updated_at
- `stamp_rotations` (already referenced by corrections_dashboard): stem PK, rotation int CHECK (0/90/180/270), confirmed_at timestamptz

Apply via psql to dedup-postgres. Also add `schema/` to the project, register in a `just schema-apply` recipe.

**Verify**: `\dt` on dedup-postgres shows all 5 stamp_* tables. `SELECT count(*)` returns 0 on each.

### T2. Build `scanmyphotos_manifest` table + populate — 10 min code + 5 min run

**Why**: Closes the biggest gap — the stem↔sha256 link. Without it, the stamp pipeline is isolated from organized media.

**What**:
1. Schema: `scanmyphotos_manifest (stem text PK, disc int, source_path text, sha256 text NOT NULL, indexed on sha256, size_bytes bigint, mtime timestamptz)`.
2. `scripts/data/build_scanmyphotos_manifest.py`: walks `scanmyphotos/*.jpg` symlinks, resolves realpath, computes sha256 with `hashlib.file_digest` (3.11+), 12-way ProcessPoolExecutor, upserts to `scanmyphotos_manifest`. Skips stems already present unless `--force`.
3. Cross-check against `originals/media/` — every sha256 from the manifest **must** exist as `originals/media/{sha256}.jpg`. If not, flag with a warning; means the dedup pipeline missed a file.

**Verify**:
- `SELECT count(*) FROM scanmyphotos_manifest` = 7,455.
- Every (stem.jpg symlink target) hashed match `originals/media/{sha256}.jpg` size.
- Spot-check: pick 3 random rows, `sha256sum` the source path manually, compare.

### T3. YOLO batch inference on all 7,455 scans — ~1–2 hr on CPU

**Why**: Populates `stamp_predictions` with bbox + confidence for every image.

**What**: Run [scripts/infer/infer_all.py](scripts/infer/infer_all.py) — already wired to read from `scanmyphotos/`, skip stems already in `stamp_predictions` or `stamp_no_stamp`, and write to Postgres. Uses `runs/detect/train/weights/best.pt`.

Start in background (tmux or `just infer-bg`). Tail `state/worker_status.json` for progress.

**Verify**:
- `SELECT count(*) FROM stamp_predictions` in 6,500–7,400 range (some will be below the conf threshold and end up with `found=false` — check exact semantics in _db.py before finalizing the number).
- Confidence histogram: `SELECT round(confidence::numeric, 1) AS conf, count(*) FROM stamp_predictions GROUP BY 1 ORDER BY 1` — expect a bimodal shape (one peak near 0 for no-stamp, one peak ≥0.5 for real stamps).

### T4. Stage-1 OCR via Haiku subagents — ~4–6 hr wall clock, ~$5–15 API spend

**Why**: Extract date text from each predicted bbox.

**What**:
1. `orchestrate_ocr.py crop-stage1 --limit 7500` — crops and writes shard manifests to `state/shards/stage1/`. Skips stems in `stamp_no_stamp`.
2. Dispatch shards to Haiku subagents per [project_parallel_haiku_ocr.md](~/.claude/memory/project_parallel_haiku_ocr.md) (proven: 6,458 photos in 126 shards, 87.3% clean).
3. After each shard returns, `orchestrate_ocr.py merge-stage1 <result.json>` writes to `stamp_ocr` with `model='haiku'`, `stage=1`.
4. **New**: `merge-stage1` also parses `raw_text` into `parsed_date DATE` where possible; leaves `parse_error` set on unparseable rows. Parse grammar already exists in `ocr_stamps.py` (formats `M D 'YY`, `YY M D`, etc. per commit history).

**Verify**:
- Stage-1 coverage: `SELECT count(*) FROM stamp_ocr WHERE stage=1 AND model='haiku'` ≈ `SELECT count(*) FROM stamp_predictions WHERE confidence > <thresh>`.
- Parse success rate ≥ 70% on first pass.
- Cost check: cross-reference Anthropic console usage against estimate before triggering stage-2.

### T5. Stage-2 verification on flagged stage-1 results — ~1 hr

**Why**: 5/9 digit confusion, `?` markers, and low-confidence results get a second look via dual-crop prompt.

**What**: `orchestrate_ocr.py crop-stage2` identifies candidates (confidence < threshold, `?` in text, or parse failure). Dispatch stage-2 shards. `merge-stage2` reconciles: if stage-1 and stage-2 agree → confident, if disagree → flag for manual review.

**Verify**:
- `SELECT review_status, count(*) FROM stamp_ocr GROUP BY 1`. Status values from [_db.py](scripts/_db.py): `auto_accepted`, `needs_review`, `rejected`, `manual`. Need clean mix.
- Manual queue size ≤ 500 (realistic for one-person review).

### T6. Human review via corrections dashboard — as-available

**Why**: Items flagged in T5 need eyeballs. Dashboard lets you correct bbox, rotation, or final date in one pass.

**What**: Start dashboard: `just dashboard` (or `python scripts/annotate/corrections_dashboard.py`). Port 8889. Review queue ordered by `review_status='needs_review'`, lowest confidence first.

Corrections write back to `stamp_ocr` with `review_status='manual'` and a signed-off `parsed_date`.

**Verify**: `SELECT count(*) FROM stamp_ocr WHERE review_status='needs_review'` drops to 0 over sessions.

### T7. Populate `stamp_no_stamp` for confirmed blanks — batch

**Why**: Photos genuinely without a stamp need to be marked so they don't keep getting reprocessed and don't show up as "date recoverable".

**What**: Combine signals: YOLO confidence < 0.05 AND stage-1 OCR says `NONE`/empty → auto-add to `stamp_no_stamp` with `source='auto'`. Human-confirmed blanks from T6 get `source='user'`.

**Verify**: Count consistent with historical `state/status.json` claim of `confirmed_no_stamp: 278` plus whatever fraction of the `pending_review: 4128` converts to no-stamp.

### T8. Decide: EXIF-dates storage + `media_dates` view — 30 min + 20 min run

**Why**: Unifying date sources into one view is what makes the undated gallery sharper and supports future work (chronological browsing, timeline UI, etc.).

**Decision to make** (flag for user):
- **(A)** Add an `exif_dates (sha256 PK, date_taken timestamptz, source text)` table, populated once by a PIL scan of `originals/media/` and updated on new imports. Join in `media_dates` view.
- **(B)** Keep EXIF reads on-the-fly per request. Cheaper now, slower for bulk queries later.

Recommendation: **(A)**. The gallery builder already reads every photo; storing the result costs ~8 MB and earns us fast queries. Add `scripts/data/extract_exif_dates.py` mirroring `extract_video_dates.py`.

**What**: Emit the `media_dates` view DDL joining exif_dates + stamp_ocr (via manifest) + video_dates.

**Verify**:
- `SELECT source, count(*) FROM media_dates GROUP BY 1` — expect `exif` ~30K, `stamp_ocr` ~6.5K, `ffprobe` ~4.2K.
- `SELECT count(*) FROM photo_embeddings pe LEFT JOIN media_dates md ON pe.sha256 = md.sha256 WHERE md.sha256 IS NULL` → undated count.

### T9. Rebuild undated gallery against `media_dates` — 10 min

**Why**: Close the loop. The current [scripts/find_undated_media.py](scripts/find_undated_media.py) only checks EXIF + `video_dates.json`, which is why the ScanMyPhotos scans incorrectly dominate the gallery.

**What**: Replace the body of `find_undated_media.py` with a single DB query against `media_dates`, producing the same JSON shape the gallery builder expects. Gallery builder untouched.

**Verify**:
- Total undated count drops by thousands (from 10,378 to ~2,800–3,500 photos + however many videos still lack creation_time).
- Spot-check: the user's Apr 19 test photo `004e39c9...jpg` no longer appears in the gallery because it now has a stamp OCR date.

### T10. Commit schema + scripts, update CLAUDE.md — 15 min

**Why**: Keep the repo an accurate record. CLAUDE.md currently documents the stamp_* tables as if they exist; that claim needs reconciliation.

**What**:
- Commit `schema/stamp_tables.sql`, `scripts/data/build_scanmyphotos_manifest.py`, `scripts/data/extract_exif_dates.py`, view DDL, `find_undated_media.py` updates.
- Update `CLAUDE.md`:
  - Under "Key Data Stores": note the new `scanmyphotos_manifest`, `exif_dates`, `media_dates` view.
  - Under "Critical Constraints": add "dedup-postgres is now backed up nightly via Homelab/infra/backup; do NOT rely on this as a substitute for writing a new migration script when re-provisioning."
- Write a handoff for the session.

## Open questions (need user input before starting)

1. **EXIF-dates storage**: option A (persistent table) or B (on-the-fly)? See T8.
2. **Training data recovery**: the 3,121 missing labels in `dataset/labels/` are fine for inference now, but if YOLO accuracy is lower than expected on re-run (we can't A/B against old predictions because those are gone), do we want to budget time to relabel? Deferred — start with inference, revisit if accuracy is visibly worse than `state/status.json` claimed (predictions_total: 6,679).
3. **Haiku budget ceiling**: previous run was 6,458 photos for ~$5–15 equivalent. Confirm OK to spend similar again, or set a hard cap.
4. **Gemma sanity check**: skip, or run on a 200-photo sample first to confirm Haiku is still the right choice (might have gotten worse/better with model updates)?

## Risks

- **Haiku API changes**: the subagent-driven parallel OCR relied on Claude Code's Task tool behaviour. If prompts or rate limits have shifted, stage-1 may need tuning. Mitigation: run 1 shard first, inspect output before dispatching the remaining 125.
- **YOLO drift**: the on-disk weights are from Apr 16 — same weights that produced 6,679 predictions originally. Should be stable. But a bad model file could silently produce garbage; the confidence histogram in T3 is the sentinel.
- **sha256 collisions with existing media**: every ScanMyPhotos file should hash to one of the `originals/media/{sha256}.jpg` files. If the dedup import ever renamed/recompressed files, sha256s won't match. T2's cross-check covers this; escalate on any miss.
- **Re-running this whole thing**: now that pgdump + tarball backups are wired up (see `Homelab/infra/backup/`), a volume wipe won't destroy the work again. Confirmed 2026-04-20: first dump is 329 MB gz, tarball is 345 MB in `/mnt/823.../backups/dedup/20260419_195800/`.

## Estimated wall-clock

| Phase | Time |
|---|---|
| T1–T2 (schema + manifest) | ~30 min |
| T3 (YOLO batch) | ~2 hr CPU |
| T4 (Haiku stage-1) | ~4–6 hr (mostly async) |
| T5 (stage-2) | ~1 hr |
| T6 (human review) | open-ended, ~2–4 hr first pass |
| T7–T9 (no-stamp, views, gallery) | ~1 hr |
| T10 (commit + handoff) | ~15 min |
| **Total active** | ~10–15 hr, spread across multiple sessions |
