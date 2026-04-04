# Phase 2 Dedup Pipeline

Docker-based multi-threaded deduplication pipeline for 77K media files (467GB → 45K unique).

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Telegram bot token and chat ID (get from BotFather on Telegram)
- 1.2TB free SSD space for staging and dedup

### Setup

1. **Copy environment template:**
```bash
cd /home/will/photo_project/dedup
cp .env.example .env
```

2. **Fill in Telegram credentials in .env:**
```bash
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

3. **Build Docker image:**
```bash
docker-compose build
```

4. **Run the pipeline:**
```bash
docker-compose up
```

The pipeline will run all 6 stages (0-5) overnight and send a Telegram notification on completion.

## Architecture

### Six Sequential Stages

| Stage | Name | Duration | Purpose |
|-------|------|----------|---------|
| 0 | Preflight Copy | 2-3 hrs | HDD → SSD staging (467GB) |
| 1 | Load | 5-10 min | Parse SHA256SUMS.txt → `file_paths` table |
| 2 | Enrich | 1-2 hrs | Read EXIF metadata (4 threads) |
| 3 | Select | 1-2 min | Choose canonical per hash group |
| 4 | Copy | 2-4 hrs | Copy canonicals to `originals/` (4 threads) |
| 5 | Verify | 1-2 hrs | Re-hash verification (4 threads) |

**Total Time:** 6-12 hours (typically ~8 hrs)

### Key Design Decisions

- **Hash-based file names**: `originals/<hash>.<ext>` (collision-proof)
- **DuckDB checkpoint storage**: Each stage reads/writes DB; resumable on interrupt
- **Multi-threaded I/O**: Stages 2, 4, 5 use ThreadPoolExecutor (4 workers default)
- **Folder cohesion tiebreaker**: When EXIF scores tie, prefer files from dominant source folder
- **Resumability**: Every stage checks DB; skips completed work

## Database Schema

DuckDB at `/home/will/photo_project/dedup.duckdb` contains 7 tables:

- **file_paths**: All 95K files from SHA256SUMS.txt (hash, path, size, source_folder, ext)
- **files**: EXIF metadata per unique hash (exif_score, datetime, gps, fields_count)
- **canonicals**: Selected canonical per hash (duplicate_count, verified flag)
- **copy_progress**: Stage 4 progress (status: pending/done/error, copied_path)
- **staging_progress**: Stage 0 progress (source_path, staging_path, status)
- **final_report**: Pipeline summary (total_files, unique_count, space_saved, duration)

Query examples:
```sql
-- How many files copied?
SELECT COUNT(*) FROM copy_progress WHERE status='done';

-- Space saved?
SELECT SUM(total_size_saved_bytes) / 1e9 AS gb_saved FROM canonicals;

-- Errors?
SELECT hash, error_msg FROM copy_progress WHERE status='error' LIMIT 10;
```

## Configuration

Edit `.env` to override defaults:

```bash
# Paths (defaults point to /home/will/photo_project/)
HDD_SOURCE_PATH=/mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/Photos/img/
SSD_STAGING_PATH=/home/will/photo_project/staging/
SSD_ORIGINALS_PATH=/home/will/photo_project/originals/
SSD_DB_PATH=/home/will/photo_project/dedup.duckdb

# Threading
THREAD_WORKERS=4
RETRY_LIMIT=2

# Stages (comma-separated, e.g., skip stage 0 with "1,2,3,4,5")
RUN_STAGES=0,1,2,3,4,5

# Telegram
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
```

## Monitoring Progress

### While Running

**In Docker logs:**
```bash
docker-compose logs -f dedup
```

**Query DB from host:**
```bash
duckdb /home/will/photo_project/dedup.duckdb
SELECT stage_completed, COUNT(*) FROM final_report;
SELECT status, COUNT(*) FROM copy_progress GROUP BY status;
```

### Resume After Interrupt

The pipeline is fully resumable:
```bash
# Just restart it
docker-compose up

# Or skip completed stages
sed -i 's/RUN_STAGES=.*/RUN_STAGES=3,4,5/' .env  # Skip 0-2, resume from stage 3
docker-compose up
```

## Debugging

### Common Issues

**"Source file not found"**
- Stage 0 copy didn't complete or staging/ was deleted
- Solution: Delete `staging/` and `dedup.duckdb`, restart from Stage 0

**"Hash mismatch in verification"**
- File was corrupted during copy or HDD changed
- Check: `SELECT hash, verification_hash FROM canonicals WHERE verified=False LIMIT 10`
- Solution: Manually inspect file, re-copy, or accept loss

**"Out of disk space"**
- SSD filled during copy (need ~500GB + 350GB working space = 850GB)
- Solution: Free space, or reduce THREAD_WORKERS in .env to slow down copy rate

### Useful Queries

```bash
duckdb /home/will/photo_project/dedup.duckdb

# How many unique files per source folder?
SELECT source_folder, COUNT(DISTINCT hash) FROM file_paths GROUP BY source_folder;

# Files with EXIF metadata?
SELECT COUNT(*) FROM files WHERE exif_fields_count > 0;

# Duplicates by size saved?
SELECT source_folder, SUM(total_size_saved_bytes) / 1e9 FROM canonicals 
JOIN files ON canonicals.hash = files.hash
GROUP BY source_folder ORDER BY 2 DESC;
```

## Testing

Run the test suite locally:
```bash
cd /home/will/photo_project/dedup
python -m pytest tests/ -v
```

Expected output: 7 tests passing (6 schema tests + 1 integration test).

## Troubleshooting

**Container won't start:**
- Check .env is present: `ls .env`
- Check Telegram token is set: `grep TELEGRAM .env`
- View build logs: `docker-compose build --no-cache`

**Tests fail:**
- Ensure Python dependencies installed: `pip install -r requirements.txt`
- Clear pytest cache: `rm -rf .pytest_cache __pycache__`

**Pipeline hangs:**
- Check HDD is mounted: `ls /mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/Photos/img/`
- Check SSD has space: `df -h /home/will/photo_project/`
- View container logs: `docker-compose logs dedup`

## Performance Tuning

- **THREAD_WORKERS**: Default 4. Increase to 8 for faster HDD→SSD copy (Stage 0), decrease to 2 if HDD is slow
- **Batch size**: Edit `stage1_load.py` line ~50 (`if len(new_entries) >= 1000`) to tune DB batch inserts
- **Skip stages**: Use RUN_STAGES=3,4,5 to skip expensive Stage 0-1 if already complete

## Architecture Notes

### Why DuckDB?
- Embedded (no server required in Docker)
- Fast columnar storage
- Supports SQLAlchemy ORM
- SQL queries for monitoring

### Why SQLAlchemy?
- Type-safe ORM
- Vendor-agnostic (can swap to Postgres if needed)
- Automatic schema creation and migrations

### Why Hash-Based Names?
- Eliminates duplicate filenames (iCloudPhotos and Desktop both have `photo_001.jpg`)
- Content-addressable (can verify integrity by re-hashing)
- No chance of collision (SHA-256)

## Support

For issues or questions, check:
1. Docker container logs: `docker-compose logs dedup`
2. DuckDB error table: `SELECT * FROM final_report ORDER BY timestamp DESC LIMIT 1`
3. Stage-specific logs in container: stored to stderr during execution
