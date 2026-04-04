# Phase 2: Hash-Based Dedup with Docker Pipeline

**Date**: 2026-04-03  
**Status**: Design Approved  
**Goal**: Deduplicate 77K media files (467GB) using Phase 1 SHA-256 hashes, produce 45K unique files on SSD with full metadata in DuckDB, resumable multi-threaded pipeline in Docker.

---

## Overview

Five-stage dedup pipeline runs inside a Docker container with:
- **SQLAlchemy schema** for type-safe DB operations
- **Multi-threaded execution** for EXIF reads, copies, verification
- **DuckDB** as checkpoint store (resumable if interrupted)
- **Apprise+Telegram** notifications on completion
- **Stage 0** (preflight): Copy full 467GB from HDD to SSD staging area for speed

All code lives in `/home/will/photo_project/dedup/` (new folder in repo).

---

## Architecture

### File Structure

```
/home/will/photo_project/dedup/
├── Dockerfile                 # Container definition
├── docker-compose.yml         # Volumes, env vars, Telegram token
├── requirements.txt           # Python dependencies
├── .env.example              # Template for Telegram config
├── main.py                   # Entry point, stage orchestration
├── stages/
│   ├── __init__.py
│   ├── stage0_copy.py        # HDD → SSD staging (preflight)
│   ├── stage1_load.py        # SHA256SUMS.txt → file_paths table
│   ├── stage2_enrich.py      # EXIF reads → files table (threaded)
│   ├── stage3_select.py      # Canonical selection → canonicals table
│   ├── stage4_copy.py        # Copy to originals/ (threaded, resumable)
│   └── stage5_verify.py      # Re-hash verification (threaded)
├── models/
│   ├── __init__.py
│   └── schema.py             # SQLAlchemy schema definitions
├── utils/
│   ├── __init__.py
│   ├── exif.py              # EXIF reading with Pillow
│   ├── threading.py         # Thread pool executor wrapper
│   ├── notifications.py     # Apprise integration
│   └── db.py                # DuckDB session management
└── tests/
    └── test_schema.py       # Basic schema validation
```

---

## DuckDB Schema (SQLAlchemy)

### Tables

#### `file_paths`
```python
hash: String (SHA-256, primary key)
path: String (original full path on HDD)
size: Integer (bytes)
source_folder: String (e.g., "iCloudPhotos", "Desktop", "Photos")
filename: String (original filename)
extension: String (jpg, heic, mov, etc)
created_at: DateTime (when this record was inserted)
```

#### `files`
```python
hash: String (FK to file_paths, primary key)
canonical_path: String (path chosen as canonical from duplicates)
exif_score: Float (0-1, richness of EXIF metadata)
exif_datetime: DateTime (DateTimeOriginal if present, else null)
exif_gps: String (lat,lon if present, else null)
exif_fields_count: Integer (# of non-null EXIF fields)
folder_source: String (source folder of canonical choice)
selected_reason: String ("best_exif" / "folder_cohesion_tiebreak")
created_at: DateTime
```

#### `canonicals`
```python
hash: String (PK, FK to files)
canonical_path: String
duplicate_count: Integer (how many other files had this hash)
total_size_saved_bytes: Integer (size * (duplicate_count - 1))
verified: Boolean (set to True after re-hash in Stage 5)
verification_hash: String (re-computed hash in Stage 5, should == hash)
created_at: DateTime
```

#### `copy_progress`
```python
hash: String (FK, PK)
status: String (enum: pending / copying / done / error)
copied_path: String (e.g., /home/will/photo_project/originals/b12803ce.jpg)
bytes_copied: Integer (progress tracking for large files)
error_msg: String (if status == error)
retry_count: Integer (0-2 retries before giving up)
updated_at: DateTime
```

#### `staging_progress`
```python
source_path: String (PK, HDD path)
staging_path: String (SSD path under staging/)
status: String (enum: pending / copying / done / error)
bytes_copied: Integer
error_msg: String
updated_at: DateTime
```

#### `final_report`
```python
stage_completed: String (enum: stage0/1/2/3/4/5)
total_files_analyzed: Integer
unique_files: Integer
duplicate_files_removed: Integer
total_space_saved_gb: Float
files_by_source: JSON (count per source folder)
duration_seconds: Integer
errors_encountered: Integer
verified_copies: Integer
failed_verifications: Integer
timestamp: DateTime
```

---

## Stages

### Stage 0: Preflight Copy (HDD → SSD Staging)

**Input**: All 95,519 entries from `/mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/Photos/img/`

**Process**:
1. Create `/home/will/photo_project/staging/` directory
2. For each file in SHA256SUMS.txt:
   - Copy from HDD to `staging/<relative_path>` (preserve subfolder structure for this stage only)
   - Mark progress in `staging_progress` table
   - On error: log and continue, mark for retry
3. Verify total bytes match source size

**Output**: All 95,519 files on SSD under `staging/`, `staging_progress` table complete.

**Resume**: Check `staging_progress`; skip files with status='done'.

**Duration**: ~2-3 hours (467GB at ~40 MB/s sustained HDD→SSD).

---

### Stage 1: Load (SHA256SUMS.txt → `file_paths` table)

**Input**: 
- SHA256SUMS.txt from HDD
- staging/ directory (source of truth for paths now)

**Process**:
1. Parse SHA256SUMS.txt (hash, path pairs)
2. For each hash:
   - Extract filename, extension, source_folder from path
   - Insert into `file_paths` (skip if hash already in table)
3. Count total vs unique hashes

**Output**: `file_paths` table with 95,519 rows, 62,821 unique hashes.

**Resume**: Check table row count; if already at 95,519, skip.

**Duration**: ~5-10 minutes.

---

### Stage 2: Enrich (EXIF Reads → `files` table)

**Input**: `file_paths` table, staging/ directory

**Process** (multi-threaded, default 4 workers):
1. For each unique hash:
   - Collect all candidate paths from `file_paths`
   - Read EXIF from one file per candidate (from staging/)
   - Score by:
     - # of non-null EXIF fields (0-1 score)
     - Presence of DateTimeOriginal
     - Presence of GPS
   - Insert row into `files` table
   - Mark in DB as completed
2. Skip hashes already in `files` table (resume).

**Output**: `files` table with 62,821 unique hashes, EXIF metadata and scores.

**Resume**: Check table row count; process only missing hashes.

**Duration**: ~1-2 hours (random SSD reads, 4 threads).

---

### Stage 3: Select Canonicals

**Input**: `files` table, `file_paths` table

**Process**:
1. For each unique hash:
   - Get all candidate paths from `file_paths` where hash = X
   - Get EXIF score from `files`
   - **Canonical selection** (in order):
     - Highest EXIF score
     - If tie: prefer source folder that contributes most files to this hash group
     - If still tie: prefer shorter path (cleaner)
   - Insert into `canonicals` table with reason
2. Populate `copy_progress` with status='pending' for all canonicals.

**Output**: `canonicals` table (one row per unique hash, ~45K rows), `copy_progress` initialized.

**Resume**: If `canonicals` already has rows, skip.

**Duration**: ~1-2 minutes.

---

### Stage 4: Copy Files

**Input**: `canonicals` table, `copy_progress` table, staging/ directory

**Process** (multi-threaded, default 4 workers):
1. For each pending row in `copy_progress`:
   - Get canonical_path from `canonicals`
   - Read from `staging/<canonical_path>`
   - Write to `/home/will/photo_project/originals/<hash>.<extension>`
   - Update `copy_progress` status='done'
   - On error (after 2 retries): mark status='error', log message
2. Create `/home/will/photo_project/originals/` if not exists.

**Output**: 45,838 files in `originals/` named by content hash, `copy_progress` complete.

**Resume**: Check `copy_progress` status; skip done/error, retry pending.

**Duration**: ~2-4 hours (45K files × ~50-100 KB avg = sequential copy at SSD speeds).

---

### Stage 5: Verify Integrity

**Input**: `canonicals` table, `copy_progress` table (completed), originals/ directory

**Process** (multi-threaded, default 4 workers):
1. For each file in `originals/`:
   - Re-compute SHA-256 hash
   - Compare to original hash (from filename)
   - Update `canonicals` with verification_hash and verified=True
   - On mismatch: log error, set verified=False
2. Generate `final_report`.

**Output**: `canonicals` table fully verified, `final_report` populated.

**Resume**: Skip files already in `canonicals` with verified=True.

**Duration**: ~1-2 hours.

---

## Configuration

### Environment Variables (in `.env`)

```bash
# Telegram notification
TELEGRAM_BOT_TOKEN=<bot_token>
TELEGRAM_CHAT_ID=<chat_id>

# Paths (can override defaults)
HDD_SOURCE_PATH=/mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/Photos/img/
SSD_STAGING_PATH=/home/will/photo_project/staging/
SSD_ORIGINALS_PATH=/home/will/photo_project/originals/
SSD_DB_PATH=/home/will/photo_project/dedup.duckdb

# Threading
THREAD_WORKERS=4
RETRY_LIMIT=2

# Stages to run (default: all)
RUN_STAGES=0,1,2,3,4,5
```

### Docker Compose

```yaml
version: '3.9'
services:
  dedup:
    build: .
    volumes:
      - /mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/Photos/img/:/mnt/hdd:ro
      - /home/will/photo_project/:/workspace:rw
    environment:
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}
    working_dir: /workspace/dedup
    command: python main.py
```

---

## Error Handling

### Stage 0-1: Loading
- **File not found**: Log warning, continue (file may have been deleted)
- **Hash mismatch**: Log error (file may have been modified on HDD)

### Stage 2: EXIF Reads
- **EXIF read fails**: Score = 0 for that file, continue
- **File unreadable**: Mark candidate as invalid, try next

### Stage 4: Copy
- **Source file missing**: Log error, mark copy_progress='error'
- **Disk full**: Detect, notify, halt
- **Permission error**: Retry once, then mark error
- **I/O error**: Retry up to RETRY_LIMIT times

### Stage 5: Verify
- **Hash mismatch**: Log to final_report, keep file (manual review later)
- **File missing in originals**: Log, mark verified=False

### Container Level
- **Out of memory**: OOM killer may terminate; DuckDB checkpoint ensures no data loss
- **Interrupted (Ctrl+C)**: Graceful shutdown, DB remains consistent

---

## Notifications

### Apprise Integration

On completion (successful or error), send Telegram message:

```
🎯 Phase 2 Dedup Complete

✅ Total files: 95,519
🔄 Unique files: 45,838
♻️  Duplicates removed: 49,681
💾 Space saved: ~280 GB

📊 By source:
  - iCloudPhotos: 9,373 files
  - Desktop: 41,797 files
  - ...

⏱️  Duration: 6 hours 23 minutes
⚠️  Errors: 3 (see DB for details)

✔️ All 45,838 copies verified.
```

If errors during run:
```
⚠️ Stage 4 Copy - Partial Failure

✅ Copied: 45,200 / 45,838
❌ Errors: 638

Resume with: docker-compose up
```

---

## Resumability

**Every stage is idempotent**:
- Stage 0: Checks `staging_progress`; skips files already copied
- Stage 1: Skips if `file_paths` already has 95,519 rows
- Stage 2: Processes only hashes not in `files` table
- Stage 3: Skips if `canonicals` already populated
- Stage 4: Resumes from next pending entry in `copy_progress`
- Stage 5: Skips files already verified

**Container restart behavior**:
1. `docker-compose up` reads `.env`, connects to existing DuckDB
2. `main.py` checks which stages are done
3. Resumes from next incomplete stage
4. No data loss, no re-processing of completed work

---

## SQLAlchemy Usage Notes

- **Single DB session per stage** to avoid connection pool exhaustion
- **Batch inserts** for performance (bulk insert 1000 rows at a time)
- **Connection pooling** via SQLAlchemy engine, sized for thread count
- **Schema validation** at startup (tables created if missing)
- **Raw SQL fallback** for complex queries (e.g., folder cohesion tiebreaker)

---

## Testing & Validation

- **Unit tests** for EXIF reading, hash computation
- **Integration test** on `photo_mapping_samples/` (100 files) before full run
- **DuckDB schema tests** validate table structure at startup

---

## Success Criteria

✅ All 95,519 files present in staging or originals  
✅ 45,838 unique files in `originals/` with hash-based names  
✅ DuckDB has complete metadata for all 45,838  
✅ All 45,838 copies verified with correct SHA-256  
✅ Telegram notification sent with summary  
✅ Container completed without data loss even if interrupted mid-stage  

---

## Timeline Estimate

| Stage | Duration | Notes |
|-------|----------|-------|
| Stage 0 (Copy HDD→SSD) | 2-3 hrs | Bottleneck: 467GB ÷ 40 MB/s |
| Stage 1 (Load) | 5-10 min | Sequential parse |
| Stage 2 (Enrich EXIF) | 1-2 hrs | 4 threads, cached SSD reads |
| Stage 3 (Select) | 1-2 min | In-memory scoring |
| Stage 4 (Copy to originals) | 2-4 hrs | 4 threads, 45K files |
| Stage 5 (Verify) | 1-2 hrs | 4 threads, re-hash |
| **Total** | **6-12 hrs** | Typically ~8 hrs overnight |

---
