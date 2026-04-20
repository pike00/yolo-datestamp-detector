---
summary: "Date-mapping pipeline rebuild -- backup set up, schema applied, manifest smoke passed; T2 full run rejected"
---

# Handoff: Date-Mapping Rebuild (ScanMyPhotos stamp OCR)

**Date:** 2026-04-19
**Goal:** Rebuild the ScanMyPhotos stamp OCR pipeline after its Postgres tables were wiped on 2026-04-16 during the pgvector volume recreate, and produce a unified `media_dates` view so the undated gallery reflects stamp dates + EXIF + ffprobe.

## Current Status

**Plan:** [docs/plans/2026-04-20-date-mapping-rebuild.md](docs/plans/2026-04-20-date-mapping-rebuild.md) — 10 tasks with verification steps.

Completed:
- Exhaustive recovery search (Postgres schemas, Kopia, HDD backups, rclone B2/AWS, Loki, git): stamp_* data irrecoverable. Volume wiped 2026-04-16 22:25 UTC for pgvector:pg18 upgrade; old container `67955414ff9a` in Loki shows queries up to 13:33 UTC the same day.
- **Backup wired up** (Homelab commit `918b83d`):
  - `dedup-db-backup` sidecar in `Homelab/infra/backup/docker-compose.yml` on `dedup_default` network, inline creds (local-dev, not secret)
  - `pg-backup-driver.sh` has a no-sops branch for services with inline creds
  - `homelab-volumes.sh` adds `backup_dedup()` tarring both `dedup_db_backup` (logical dump) and `dedup_postgres_data` (raw PGDATA) into `/mnt/823.../backups/dedup/`
  - `justfile pgdump` recipe includes dedup-db-backup
  - **Verified:** first dump 329 MB gz, tarball 345 MB at `/mnt/823.../backups/dedup/20260419_195800/volumes/dedup_db_backup.tar.gz`, gzip integrity OK
  - Next scheduled run: 01:30 UTC pg-dump + 02:00 UTC tar (systemd timers); Kopia snapshots tarball on its next cycle
- Undated-media discovery artifacts committed (photo_project commit `29ecbfc`):
  - `scripts/find_undated_media.py` — 12-way PIL EXIF scan, 17s wall for 41,991 photos + 5,296 videos; output `state/undated_media.json` (gitignored)
  - `scripts/build_undated_gallery.py` — generates 240px thumbnails (WebP for photos, JPG for videos) + paginated HTML
  - Gallery served at `http://ares:8890/output/undated_gallery/` (39 pages of 300 items)
- **T1 done**: `schema/stamp_tables.sql` applied. 5 tables created in dedup-postgres, all empty:
  - stamp_predictions, stamp_no_stamp, stamp_prediction_drift
  - stamp_ocr (added `parsed_date DATE` + `parse_error TEXT` beyond pre-wipe schema)
  - scanmyphotos_manifest (stem PK, disc, source_path, sha256, size_bytes, mtime)
- **T2 started:** `scripts/data/build_scanmyphotos_manifest.py` written, smoke test on 10 files passed (10/10 cross-check against `originals/media/{sha256}.jpg`). Uncommitted.
- **T2 full run (all 7,455) rejected by user.**

## Next Steps

1. **Figure out why the full manifest run was rejected.** Likely suspects: HDD I/O concern, disk spin-up during quiet hours, or a different priority shifted. Ask the user before re-running. The job is ~7,455 files × ~3 MB avg = ~22 GB sequential HDD read, 12-way parallel.
2. **Commit the uncommitted work** once T2 full run completes:
   - `schema/stamp_tables.sql`
   - `scripts/data/build_scanmyphotos_manifest.py`
3. **T3 YOLO batch inference** (`scripts/infer/infer_all.py`): uses `runs/detect/train/weights/best.pt` (60 MB, survived the wipe). 2 hr CPU, writes to `stamp_predictions`.
4. **T4 Haiku stage-1 OCR** — blocked on user confirming budget (previous run was 6,458 photos; plan flags this as needing sign-off).
5. **T5–T10** per plan: stage-2 verify → human review → `stamp_no_stamp` population → `exif_dates` table + `media_dates` view → rebuild undated gallery → final commits + CLAUDE.md update.

## Key Context

- **Design decision:** keep the existing **stem-keyed** schema (`d1_00000549`) for stamp_* tables, don't refactor to sha256. Join to organized media via the new `scanmyphotos_manifest` table (one table beats rewriting every script + the corrections dashboard).
- **Design decision:** skip Gemma first-pass — `~/.claude/memory/project_local_vlm_benchmark.md` says Sonnet/Haiku decisively outperform local VLMs on date stamps. Go Haiku straight.
- **New schema columns beyond pre-wipe:** `stamp_ocr.parsed_date` (DATE) and `stamp_ocr.parse_error` (TEXT) — store the parsed date once at merge time so the unified view doesn't have to reparse `raw_text`.
- **Dedup-postgres creds** are local-dev literals (`dedup/dedup_local_dev/dedup`) hardcoded in `photo_project/dedup/docker-compose.yml` — not secret. Backup sidecar uses them inline.
- **Open questions still unanswered** (in plan, section "Open questions"):
  1. EXIF-dates storage: persistent table (A, recommended) vs on-the-fly (B)?
  2. Training data recovery — `dataset/labels/` has 3 txt files vs the original 3,124. Not blocking; only matters if YOLO accuracy on re-run is visibly worse.
  3. Haiku budget ceiling — confirm same ~$5–15 range as last time, or set a hard cap.
  4. Gemma sanity-check sample before skipping it entirely?
- **Gotcha:** `dataset/images/` is empty (0 files). `scripts/infer/infer_all.py` reads from `scanmyphotos/` symlinks, not `dataset/images/`, so this doesn't block T3. But if retraining comes up, this is another gap.
- **CLAUDE.md drift:** project CLAUDE.md still describes `stamp_predictions`, `stamp_ocr`, etc. as existing. T10 reconciles.
- **HTTP gallery server** is running in this session's background (task id `bp80901c8`, port 8890 from `/home/will/photo_project`). Will die with the session; restart with `cd /home/will/photo_project && python -m http.server 8890 --bind 0.0.0.0` if needed.

## Files Touched

**Committed (photo_project `29ecbfc`):**
- [docs/plans/2026-04-20-date-mapping-rebuild.md](docs/plans/2026-04-20-date-mapping-rebuild.md)
- [scripts/find_undated_media.py](scripts/find_undated_media.py)
- [scripts/build_undated_gallery.py](scripts/build_undated_gallery.py)

**Committed (Homelab `918b83d`):**
- `infra/backup/docker-compose.yml`
- `infra/backup/justfile`
- `infra/backup/scripts/homelab-volumes.sh`
- `infra/backup/scripts/pg-backup-driver.sh`

**Uncommitted in photo_project (will commit after T2 full run):**
- [schema/stamp_tables.sql](schema/stamp_tables.sql) — applied to DB; idempotent re-runs safe
- [scripts/data/build_scanmyphotos_manifest.py](scripts/data/build_scanmyphotos_manifest.py) — 10-file smoke passed, full run not yet done

**Runtime outputs (gitignored, regenerable):**
- `state/undated_media.json` — 10,378 undated photos + 1,090 undated videos + 421 photo errors
- `output/undated_gallery/` — 39 paginated HTML pages + `thumbs/` + `failures.json`

## Blockers

- **T2 full manifest run was rejected** — need to understand why before proceeding. Possibly concerned about HDD thrash, scheduling, or a different priority.
- **T4 Haiku stage-1** is gated on explicit budget approval per the plan's open question #3.
- **Open questions #1, #2, #4** from the plan are non-blocking for T2/T3 but need answers before T4 and T8.
