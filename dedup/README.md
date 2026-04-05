# Dedup Pipeline (PostgreSQL + Docker)

Docker-based multi-threaded deduplication pipeline for 77K media files (467GB → 45K unique).

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Telegram bot token and chat ID (optional, for notifications)
- 1.2TB free SSD space for staging and originals

### Setup

1. **Review/edit environment:**
```bash
cd /home/will/photo_project/dedup
cat .env  # Check PostgreSQL credentials and paths
```

2. **Build Docker image and start services:**
```bash
docker-compose build
docker-compose up -d postgres  # Start database first
docker-compose up dedup        # Run pipeline
```

The pipeline will run all 4 stages (0-3) and send a Telegram notification on completion.

## Architecture

### Four Sequential Stages

| Stage | Name | Duration | Purpose |
|-------|------|----------|---------|
| 0 | **Ingest** | 2-3 hrs | HDD → SSD staging, compute SHA-256 hashes |
| 1 | **Enrich** | 1-2 hrs | Extract EXIF metadata (4 threads) |
| 2 | **Deduplicate** | 1-2 min | Select canonical per hash using priority rules |
| 3 | **Export** | 2-4 hrs | Copy canonicals to `originals/` and verify (4 threads) |

**Total Time:** 5-10 hours (typically ~7 hrs)

### Key Design Decisions

- **Hash-based file names**: `originals/<hash>.<ext>` (collision-proof, content-addressable)
- **PostgreSQL with connection pooling**: Robust relational DB with better concurrency than DuckDB
- **2-table schema**: Simplified schema (SourceFile + UniqueFile) vs 6+ tables
- **Priority-based canonical selection**:
  1. EXIF score (metadata richness)
  2. Folder priority: `iCloudPhotos > Photos > 20230513 ios Photos > Pictures > Desktop > *`
  3. Shortest path (tiebreaker)
- **Resumability**: Every stage checks DB; skips completed work

### Reorganized Code Structure

```
dedup/
├── pipeline/               # New: 4 focused stages
│   ├── ingest.py          # Copy + hash
│   ├── enrich.py          # EXIF extraction only
│   ├── deduplicate.py     # Canonical selection
│   └── export.py          # Copy + verify
├── models/schema.py       # 2 tables: SourceFile, UniqueFile
├── utils/
│   ├── db.py             # PostgreSQL with pooling (was DuckDB)
│   ├── exif.py           # EXIF metadata extraction
│   ├── threading.py      # Multi-threaded executor
│   └── notifications.py  # Telegram alerts
├── config.py             # Environment-based config
└── main.py              # Orchestrator
```

## Database Schema

PostgreSQL at `postgres://dedup:dedup_local_dev@localhost:5432/dedup` contains 2 tables:

### `source_files`
All discovered files with their content hash.
```sql
path              TEXT PRIMARY KEY   -- relative path from HDD root
sha256            TEXT NOT NULL      -- content hash
size              BIGINT             -- file size in bytes
source_folder     TEXT               -- top-level folder (Desktop, iCloudPhotos, etc.)
filename          TEXT               -- basename
extension         TEXT               -- file extension (.jpg, .mov, etc.)
ingested_at       TIMESTAMP          -- creation time
```

### `unique_files`
One row per unique SHA-256 hash with canonical selection and export tracking.
```sql
sha256            TEXT PRIMARY KEY   -- content hash
canonical_path    TEXT NOT NULL      -- selected canonical relative path
selection_reason  TEXT               -- "exif", "folder_priority", "shortest_path", "only_copy"
duplicate_count   INT                -- number of duplicates eliminated
exif_score        FLOAT              -- EXIF metadata richness (0-1)
exif_datetime     TIMESTAMP          -- DateTimeOriginal from EXIF
exif_gps          TEXT               -- GPS coordinates if present
exif_fields_count INT                -- number of EXIF fields
export_status     TEXT               -- "pending" / "copied" / "verified" / "error" / "mismatch"
export_path       TEXT               -- destination path in originals/
verified_hash     TEXT               -- re-computed hash after copy (for verification)
error_msg         TEXT               -- error message if export_status='error'
retry_count       INT                -- retry attempts
created_at        TIMESTAMP          -- entry creation time
updated_at        TIMESTAMP          -- last update time
```

### Monitoring Queries

```sql
-- Total unique files and duplicates
SELECT COUNT(*) as unique_files, SUM(duplicate_count) as duplicates_eliminated
FROM unique_files;

-- Export status summary
SELECT export_status, COUNT(*) FROM unique_files GROUP BY export_status;

-- Space saved by source folder
SELECT 
  sf.source_folder, 
  COUNT(DISTINCT sf.sha256) as copies_eliminated
FROM source_files sf
JOIN unique_files uf ON sf.sha256 = uf.sha256
WHERE sf.path != uf.canonical_path
GROUP BY sf.source_folder
ORDER BY 2 DESC;

-- Files with EXIF metadata
SELECT COUNT(*) FROM unique_files WHERE exif_fields_count > 0;

-- Verification failures
SELECT sha256, error_msg FROM unique_files WHERE export_status IN ('error', 'mismatch');
```

## Configuration

### Environment Variables (.env)

```bash
# Storage paths
HDD_SOURCE_PATH=/mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/Photos/img/
SSD_STAGING_PATH=/home/will/photo_project/staging/
SSD_ORIGINALS_PATH=/home/will/photo_project/originals/

# PostgreSQL database
DB_HOST=postgres                    # Docker service name
DB_PORT=5432                        # PostgreSQL port
DB_NAME=dedup                       # Database name
DB_USER=dedup                       # Database user
DB_PASSWORD=dedup_local_dev         # Database password

# Pipeline control
THREAD_WORKERS=4                    # Threads per stage (increase for faster I/O)
RETRY_LIMIT=2                       # Retries on transient errors

# Stages to run (0=ingest, 1=enrich, 2=deduplicate, 3=export)
RUN_STAGES=0,1,2,3                  # Run all stages (or "2,3" to skip ingestion)

# Telegram notifications (optional)
TELEGRAM_BOT_TOKEN=disabled         # Set to actual token for alerts
TELEGRAM_CHAT_ID=disabled           # Set to actual chat ID for alerts
```

## Monitoring Progress

### While Running

**Docker logs:**
```bash
docker-compose logs -f dedup       # Pipeline logs
docker-compose logs -f postgres    # Database logs
```

**Query database from host:**
```bash
docker exec dedup-postgres psql -U dedup -d dedup -c \
  "SELECT export_status, COUNT(*) FROM unique_files GROUP BY export_status;"
```

### Resume After Interrupt

The pipeline is fully resumable. Each stage checks the DB and skips completed work:
```bash
# Restart where it left off
docker-compose up dedup

# Or skip already-completed stages
sed -i 's/RUN_STAGES=.*/RUN_STAGES=2,3/' .env  # Skip stages 0-1, resume from dedup
docker-compose up dedup
```

## Staging Path Handling

**Important**: The pipeline preserves relative directory structure in staging:
- Source: `/mnt/hdd/Desktop/photo.jpg`
- Staging: `/staging/Desktop/photo.jpg` (preserves Desktop folder)
- Database: `path = "Desktop/photo.jpg"` (relative, no leading slash)

This allows easy resumption and debugging. If staging is deleted, Stage 0 will re-copy from scratch.

## Debugging

### Common Issues

**"Source file not found"**
- Staging directory was deleted or incomplete copy from Stage 0
- Solution: Remove staging and run `RUN_STAGES=0` to re-ingest, or `RUN_STAGES=0,1,2,3` to restart full pipeline

**"Hash mismatch in verification"**
- File corrupted during copy or HDD changed
- Check: `SELECT sha256 FROM unique_files WHERE export_status='mismatch';`
- Solution: Inspect file, re-export, or accept loss

**"Out of disk space"**
- SSD filled during export (need ~500GB staging + ~350GB originals = 850GB working space)
- Solution: Free space or reduce THREAD_WORKERS to slow down I/O

**PostgreSQL won't start**
```bash
# Check logs
docker-compose logs postgres

# Try full reset
docker-compose down -v
docker-compose up postgres
```

## Performance Tuning

- **THREAD_WORKERS**: Default 4. Increase to 8+ for faster I/O on high-speed storage, decrease to 2 if HDD is bottleneck
- **Connection pool**: Edit `utils/db.py` line `pool_size=10` to tune PostgreSQL connection pool
- **Batch insert size**: Edit `pipeline/ingest.py` or other stages if memory usage is high

## Architecture Notes

### Why PostgreSQL?
- Full ACID semantics for data safety
- Connection pooling handles concurrent threads efficiently
- Better query performance than embedded DuckDB for complex dedup logic
- Easier to inspect/debug with standard SQL tools
- Can be backed up/restored separately from Docker

### Why 2 Tables?
- `SourceFile`: immutable record of all discovered files
- `UniqueFile`: dedup results (canonical selection, export status, EXIF)
- Simpler schema = easier queries and less confusion about data flow

### Staging Strategy
- Preserves source folder structure for transparency
- Allows resumption without re-copying all files
- Filesystem-based tracking (not in DB) for staging progress

## Testing

Run tests locally (requires PostgreSQL client libraries):
```bash
cd /home/will/photo_project/dedup
python -m pytest tests/ -v
```

Or test with Docker:
```bash
docker-compose exec dedup python -m pytest tests/ -v
```

## Support

For issues:
1. Check Docker logs: `docker-compose logs dedup`
2. Query database: `docker exec dedup-postgres psql -U dedup -d dedup`
3. Check staging directory exists: `ls -la /home/will/photo_project/staging/`
4. Verify HDD is mounted: `ls /mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/Photos/img/`
