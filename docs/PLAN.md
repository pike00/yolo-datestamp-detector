# Photo Project Plan

## Overview

Consolidate ~77K media files (467 GB) from `/mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/Photos/img/` into a deduplicated, organized, metadata-enriched collection on the SSD at `/home/will/photo_project/`.

**Source:** 8 overlapping folders from various exports (iCloud, desktop backups, phone backups, scanned photos)
**Estimated duplicates:** ~40% (32K+ files share sizes; true dedup by hash will be higher)
**Prior work:** 72-image Gemini batch pilot (Dec 2023) for OCR date detection, location, quality scoring

---

## Phase 1: Lock Down Backup ✅ COMPLETE

**Completed:** 2026-04-03

### 1.1 — SHA-256 Manifest ✅
- **95,519 files hashed**, zero errors
- Saved at `/mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/Photos/SHA256SUMS.txt`
- **Reuse in Phase 2**: parse this file directly instead of re-hashing — saves hours

### 1.2 — Sealed Archive ✅
- `/mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/Photos_BACKUP_DO_NOT_TOUCH.tar` — **467 GB**
- Contains: `Photos/img/` + `Photos/output/` + `SHA256SUMS.txt`
- **Never modify or delete this file**

### 1.3 — Verified ✅
- Tar entry count: **95,549** = original file count (95,519 img/ + 29 output/ + 1 SHA256SUMS.txt)
- Exact match confirmed

### 1.4 — (Optional) Offsite Backup
- Push the .tar to `backblaze:pike-file-archive/Photos_BACKUP/` via rclone
- Cost: ~$2.50/mo for 500 GB on B2
- Deferred — do after dedup once the archive is smaller

---

## Phase 2: Deduplicate and Move to SSD

**Goal:** Produce a single canonical copy of every unique media file on the SSD.
**Status:** COMPLETE (2026-04-04). Pipeline ran all 4 stages successfully.

### Results
- **95,519 source files** ingested from HDD to `staging/` on SSD
- **62,821 unique files** identified (32,698 duplicates eliminated = 34.2% dedup rate)
- **305 GB** on SSD after dedup (down from 466 GB source)
- **35,357** files have EXIF dates, **27,464** do not
- **4,424** files have GPS coordinates
- **31,751** files have camera make/model
- Canonical selection: 44,124 "only_copy" + 18,697 "folder_priority"
- All 62,821 unique files exported and hash-verified in `originals/`
- 94 junk directories (.dist-info, .venv, .git) from Python files in source -- to clean up
- Pipeline: PostgreSQL-backed, 4-stage (Ingest/Enrich/Deduplicate/Export), Dockerized in `dedup/`

### 2.1 — Hash-Based Dedup Analysis
- **Reuse `SHA256SUMS.txt`** — don't re-hash. Parse it to build the dedup map directly.
  ```
  format: <sha256hash>  <filepath>
  ```
- Group all 95,519 entries by hash
- Only media files are relevant for dedup: JPG, JPEG, HEIC, PNG, MOV, MP4, M4V, GIF, AAE
- Non-media (JSON sidecars, CSV, py, pyc, txt, ini, pdf) are kept as-is alongside their media
- Output: `metadata/dedup_manifest.json` — `{hash: {canonical: path, duplicates: [path, ...]}, ...}`

### 2.2 — Canonical File Selection
For each group of duplicate hashes, pick one canonical copy using this priority:
1. Prefer files with richer EXIF metadata (check with exiftool — more tags = better)
2. Prefer source folder priority: `iCloudPhotos/` > `Photos/` > `20230513 ios Photos/` > `Pictures/` > `Desktop/`
3. Prefer shorter file path as tiebreaker

### 2.3 — Handle JSON Sidecar Files
- Google Photos exports: `.json` sidecars sit alongside media files (same name, `.json` extension)
- When choosing a canonical media file, check if any duplicate has a `.json` sidecar the canonical lacks
- If so, copy the sidecar alongside the canonical at destination

### 2.4 — Handle CSV Album/Memory Files
- `CSV/` folder has Apple/Google "Memories" album definitions (album name → list of filenames)
- Copy all CSVs to `metadata/albums/` as reference — do not try to rewrite paths yet
- Phase 3 will use these to assign album tags

### 2.5 — Copy Deduplicated Set to SSD
- Target: `/home/will/photo_project/originals/`
- Use a flat copy per source subfolder: `originals/<source_folder>/<filename>`
  - Preserves provenance without creating a new organization scheme yet
- Copy (not move) — HDD source and tar backup remain untouched
- Estimated size after dedup: ~280-350 GB
- Verify: spot-check 100 random files by comparing SHA-256 of copied files against manifest

### 2.6 — Dedup Report (output to `metadata/dedup_report.json`)
- Total files before dedup
- Unique files (canonical count)
- Duplicates eliminated
- Space saved (GB)
- Files missing EXIF metadata
- Files identified as scanned photos (from ScanMyPhotos paths)

---

## Phase 3: Organize and Enrich Metadata

**Goal:** Sort into a date-based structure, fill in missing dates via OCR/ML, write metadata back to files.
**Status:** In progress. YOLO stamp detector trained, OCR pipeline built and tested on 100 samples.

### Progress (2026-04-05)
- YOLO stamp detector trained on 86 labeled + 14 negative samples (mAP50=0.995)
- Detection: Precision=1.000, Recall=0.570 (48/100 stamps detected, 0 false positives)
- OCR pipeline (`ocr_stamps.py`): YOLO detect -> crop -> TrOCR + Tesseract -> date parse
- Date extraction: 32/48 detected stamps parsed to dates (67% parse rate)
- Overall yield: 32/100 sample photos got extracted dates
- Remaining: improve recall (more training data), scale to all 7,500 scanned photos

### 3.1 — Initial Sort by EXIF Date
- Read EXIF `DateTimeOriginal` (or `CreateDate`, `ModifyDate` as fallbacks) using exiftool
- Move files with valid dates into `YYYY/YYYY-MM-DD/` structure under `/home/will/photo_project/organized/`
- Files without any EXIF date go to `/home/will/photo_project/needs_date/`
- Maintain a mapping file: `organization_log.json` (original path -> organized path, date source)

### 3.2 — Identify and Classify Scanned Photos
- Scanned photos from: `ScanMyPhotos/` (4 discs), `Scanning Photo Project/`, `Scanned Images - to be batch sorted/`
- These typically lack EXIF dates entirely
- Flag all scanned photos in the manifest
- Group by scan batch (disc number, scan session)

### 3.3 — Scale Up Gemini Batch ML Pipeline
- Extend the pilot approach (already tested on 72 images) to all files in `needs_date/`
- For each image, extract via Gemini:
  - `date_final`: Best date from visual stamps, printed dates, contextual clues
  - `location_guess`: City/neighborhood
  - `quality_score`: 1-10
  - `tags`: 5-7 keywords
  - `text_content`: Any legible text (useful for documents, cards, etc.)
  - `rotation_needed`: Degrees clockwise
- Batch via Google Cloud Batch Prediction API to keep costs low
- Store results in `/home/will/photo_project/metadata/ml_predictions/`

### 3.4 — Date Inference for Remaining Undated Photos
- For photos where Gemini can't determine a date:
  - Check if they were in an album/memory CSV with a date in the name (e.g., "Barcelona Sep 4, 2018")
  - Check file modification time as a last resort (unreliable but better than nothing)
  - Check if they cluster with dated photos from the same batch/folder
- Assign confidence levels: `exif` (high), `ocr` (medium), `album` (medium), `mtime` (low), `cluster` (low)

### 3.5 — Write Metadata Back to Files
- Use `exiftool` to write dates, GPS coordinates, and tags back into file EXIF/XMP
- Only write to files in the SSD working copy, never to the HDD backup
- For HEIC files: ensure EXIF writes are compatible
- For scanned photos: write the ML-determined date, tag as `Scanned`, add quality score to XMP rating
- Log all metadata changes in `metadata_changes.json`

### 3.6 — Final Organization Pass
- Re-sort `needs_date/` files that now have dates into the `YYYY/YYYY-MM-DD/` structure
- Generate a summary report:
  - Date coverage: X% of files have dates
  - Location coverage: X% have GPS/location
  - Quality distribution: histogram of scores
  - Remaining undated files: count and list
- Create a browsable index (HTML or JSON) of the full collection

---

## File Locations Reference

| Path | Purpose | Mutable? |
|---|---|---|
| `/mnt/823c.../Photos/img/` | Original files on HDD | **NO — do not touch** |
| `/mnt/823c.../Photos/SHA256SUMS.txt` | Hash manifest of originals | **NO** |
| `/mnt/823c.../Photos_BACKUP_DO_NOT_TOUCH.tar` | Sealed compressed archive | **NO** |
| `/mnt/823c.../PersonalPhotos.tar` | Older personal archive (not just photos) | **NO** |
| `/home/will/photo_project/PLAN.md` | This plan | Yes |
| `/home/will/photo_project/originals/` | Deduplicated copies from Phase 2 | Yes |
| `/home/will/photo_project/organized/` | Date-sorted structure from Phase 3 | Yes |
| `/home/will/photo_project/needs_date/` | Files awaiting date identification | Yes |
| `/home/will/photo_project/metadata/` | Albums, ML predictions, logs | Yes |
