---
summary: "Date-mapping rebuild -- T1/T2 done (+committed), T3 YOLO running in bg, T8 schema+video import done"
---

# Handoff: Date-Mapping Rebuild (ScanMyPhotos stamp OCR)

**Date:** 2026-04-19 (updated)
**Goal:** Rebuild the ScanMyPhotos stamp OCR pipeline after the 2026-04-16 pgvector volume wipe, and produce a unified `media_dates` view so the undated gallery reflects stamp dates + EXIF + ffprobe.

## Current Status

**Plan:** [docs/plans/2026-04-20-date-mapping-rebuild.md](docs/plans/2026-04-20-date-mapping-rebuild.md) — 10 tasks.

Done & **committed** (photo_project `9260004`, `29ecbfc`; Homelab `918b83d`):
- Exhaustive recovery search confirmed stamp_* data irrecoverable (wiped 2026-04-16 22:25 UTC).
- **Backup wired up end-to-end:** `dedup-db-backup` sidecar, no-sops branch in driver, `backup_dedup()` in `homelab-volumes.sh`, `justfile pgdump` includes dedup. First dump 329 MB gz / 345 MB tarball at `/mnt/823.../backups/dedup/20260419_195800/`. Nightly timers (01:30 pg-dump, 02:00 tar) pick it up automatically; Kopia snapshots the tarball on its next cycle.
- **Undated-media scanner + gallery** committed ([scripts/find_undated_media.py](scripts/find_undated_media.py), [scripts/build_undated_gallery.py](scripts/build_undated_gallery.py)). Gallery serves at `http://ares:8890/output/undated_gallery/` during the session (bg task dies with session).
- **T1 done**: [schema/stamp_tables.sql](schema/stamp_tables.sql) applied. 5 tables in dedup-postgres, all populated below. `stamp_ocr` gained `parsed_date DATE` + `parse_error TEXT` beyond the pre-wipe schema.
- **T2 done**: [scripts/data/build_scanmyphotos_manifest.py](scripts/data/build_scanmyphotos_manifest.py) hashed 7,454 ScanMyPhotos scans in 64 s off HDD (12 workers). Cross-check passed — every sha256 exists in `originals/media/`. Disc totals match `state/status.json` exactly (1774/2034/2071/1575).

Done but **uncommitted**:
- **T8 (partial)**: [schema/media_dates.sql](schema/media_dates.sql) applied — tables `exif_dates`, `video_dates`, views `media_dates` and `media_has_date`. `video_dates` populated via [scripts/data/import_video_dates.py](scripts/data/import_video_dates.py) (4,206 rows, 2008–2023).
- **T8 pending run**: [scripts/data/extract_exif_dates.py](scripts/data/extract_exif_dates.py) written but **not run** — deferred until T3 YOLO finishes to avoid CPU contention.
- **T9 draft**: [scripts/find_undated_media.py](scripts/find_undated_media.py) rewritten to query `media_has_date` view (modified, not staged). The old on-the-fly PIL EXIF scan is replaced with a single SQL query.

**In flight right now:**
- **T3 YOLO batch inference** running in background shell task `b0kymxopb`. Log: [state/infer_run.log](state/infer_run.log), status: [state/worker_status.json](state/worker_status.json). **Last seen 1024/7454 inferred, 952 rows in stamp_predictions.** ETA ~25 more minutes from that snapshot. Uses `runs/detect/train/weights/best.pt`, writes to `stamp_predictions` via psycopg.
- Monitor task `b82zf16jw` is tailing the log for "inferred|ERROR" lines. Both bg tasks die with this session.

Table state snapshot (from during this session):
| Table | Rows |
|---|---|
| stamp_predictions | 952 (growing; YOLO still running) |
| stamp_no_stamp | 0 |
| stamp_prediction_drift | 0 |
| stamp_ocr | 0 |
| scanmyphotos_manifest | 7,454 |
| video_dates | 4,206 |
| exif_dates | 0 (pending T8 run) |

## Next Steps

### Immediate on resume (before anything else)

1. **Check whether T3 YOLO finished.** `cat state/worker_status.json` — `phase: done` means done. Also `docker exec dedup-postgres psql -U dedup -d dedup -c "SELECT count(*) FROM stamp_predictions;"` — should be in the 6,500–7,400 range. If it's still less than 7,454 with phase `inference`, the bg shell died when this session ended; re-launch with:
   ```
   source .venv/bin/activate && python scripts/infer/infer_all.py 2>&1 | tee -a state/infer_run.log
   ```
   The script is idempotent (skips stems already in `stamp_predictions`), so resuming is safe.

2. **Verify T3 confidence histogram:**
   ```sql
   SELECT round(confidence::numeric, 1) AS conf, count(*)
   FROM stamp_predictions GROUP BY 1 ORDER BY 1;
   ```
   Expect a bimodal shape (low-conf no-stamp peak near 0.0–0.2, real-stamp peak ≥ 0.5). If unimodal or flat, something regressed vs the old weights.

### Then proceed through the plan

3. **Run T8 EXIF dates extraction** (~30 s):
   ```
   source .venv/bin/activate && python scripts/data/extract_exif_dates.py
   ```
   Expect ~30K hits (the 47K photos minus ~10K ScanMyPhotos scans + ~7K non-EXIF digital).

4. **Commit T8 work**: `schema/media_dates.sql`, `scripts/data/import_video_dates.py`, `scripts/data/extract_exif_dates.py`, plus `scripts/find_undated_media.py` rewrite. Suggested message:
   ```
   feat: exif_dates + video_dates + media_dates view; rebuild undated-media query
   ```

5. **Before T4, resolve 4 open questions** from [docs/plans/2026-04-20-date-mapping-rebuild.md](docs/plans/2026-04-20-date-mapping-rebuild.md) (the user declined an AskUserQuestion round this session):
   - Haiku budget (same ~$15 as last time / $5 hard cap / 200-photo smoke first).
   - Gemma sanity check (skip / 50-image bench first).
   - EXIF dates storage — **effectively answered**: the table exists and the extractor is ready, so just run it in step 3.
   - Training data recovery — deferred unless T3 histogram looks bad.

6. **Kick off T4 Haiku stage-1** via `orchestrate_ocr.py crop-stage1`, dispatch shards per `~/.claude/memory/project_parallel_haiku_ocr.md`. Write parsed dates into `stamp_ocr.parsed_date` at merge time (new: column didn't exist pre-wipe; merge-stage1 code needs a ~10-line addition to parse raw_text → date).

7. **T5–T9** per plan. T9 can be tested early — once stamp_ocr has any rows with `parsed_date IS NOT NULL`, `scripts/find_undated_media.py` will show the impact on the gallery.

## Key Context

- **Design decision (unchanged):** stem-keyed stamp_* schema + `scanmyphotos_manifest` join table. No sha256 refactor.
- **New columns beyond pre-wipe:** `stamp_ocr.parsed_date DATE` + `stamp_ocr.parse_error TEXT`. The `merge-stage1` path in `orchestrate_ocr.py` doesn't populate these yet; add a parse step there.
- **Design decision (T8):** multi-row `media_dates` view (one row per source kind per sha256). A companion `media_has_date` view gives the trivial "is this sha256 dated at all?" set used by `find_undated_media.py`.
- **HDD is quiet now.** `docker-face_detect-1` finished ~40 min before this session. The face + embedding pipelines aren't competing for IO.
- **Gotcha:** running the EXIF extractor **while YOLO is still going** will oversubscribe the CPU (torch has 12 threads, PIL extractor adds 12 more). Wait for YOLO to finish, or drop `--workers 4`.
- **Gotcha:** `ls | wc -l` reports 7,455 in `scanmyphotos/` because the user's `eza` alias emits a header line. Real count: 7,454 symlinks, confirmed by manifest row count + disc totals.
- **CLAUDE.md is still stale** about stamp_* tables being live; T10 reconciles. Add the new tables/views (`scanmyphotos_manifest`, `exif_dates`, `video_dates`, `media_dates`, `media_has_date`) when you update it.

## Files Touched

**Committed (photo_project `9260004`, `29ecbfc`):**
- [docs/plans/2026-04-20-date-mapping-rebuild.md](docs/plans/2026-04-20-date-mapping-rebuild.md)
- [scripts/find_undated_media.py](scripts/find_undated_media.py) — *further modified this session, see uncommitted*
- [scripts/build_undated_gallery.py](scripts/build_undated_gallery.py)
- [schema/stamp_tables.sql](schema/stamp_tables.sql)
- [scripts/data/build_scanmyphotos_manifest.py](scripts/data/build_scanmyphotos_manifest.py)
- [docs/handoff/2026-04-19 Handoff - Date Mapping Rebuild.md](docs/handoff/2026-04-19%20Handoff%20-%20Date%20Mapping%20Rebuild.md) — *this file, being updated in-place*

**Committed (Homelab `918b83d`):**
- `infra/backup/docker-compose.yml`, `infra/backup/justfile`, `infra/backup/scripts/{homelab-volumes,pg-backup-driver}.sh`

**Uncommitted in photo_project** (applied/run, needs staging):
- [schema/media_dates.sql](schema/media_dates.sql) — applied to DB
- [scripts/data/import_video_dates.py](scripts/data/import_video_dates.py) — 4,206 rows populated
- [scripts/data/extract_exif_dates.py](scripts/data/extract_exif_dates.py) — written, not yet run
- [scripts/find_undated_media.py](scripts/find_undated_media.py) — rewritten to use `media_has_date` view

**Runtime artifacts (gitignored):**
- `state/worker_status.json` — YOLO progress, poll during resume
- `state/infer_run.log` — YOLO stdout log (teed)
- `state/undated_media.json` — old shape from pre-view run; will be overwritten by the new `find_undated_media.py` on next invocation

## Blockers

- **T4 still gated on Haiku budget approval.** User declined the AskUserQuestion round this session; either re-ask or proceed with the 200-photo smoke option as the lowest-risk default.
- **T3 (YOLO) completion is pending.** If this session dies before YOLO finishes, the bg shell goes with it — resume per "Immediate on resume" step 1.
