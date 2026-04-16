# Photo Project

Consolidate ~77K media files (467 GB) from HDD into a deduplicated, organized, metadata-enriched collection.

**Status:** Phase 2 (Deduplication) in progress. Phase 1 (backup) complete.

## Quick Navigation

### Active Work
- **[dedup/](dedup/)** — Hash-based deduplication pipeline (PostgreSQL, 4 stages)
- **[yolo_finetune/](yolo_finetune/)** — YOLO fine-tuning for date stamp detection

### Working Directories
- **staging/** — Files being ingested (HDD → SSD copy)
- **originals/** — Deduplicated canonicals (final deduplicated set)
- **needs_date/** — Photos awaiting date extraction

### Reference & Documentation
- **[docs/](docs/)** -- All design documents, plans, and guides
- **[data/](data/)** -- Results, samples, and metadata
- **[scripts/](scripts/)** -- Utility scripts organized by domain
- **[archive/](archive/)** -- Historical debug outputs

## Project Phases

### Phase 1: Lock Down Backup ✅ COMPLETE
- **95,519 files hashed** and backed up
- SHA-256 manifest created
- Sealed archive: `Photos_BACKUP_DO_NOT_TOUCH.tar` (467 GB)

### Phase 2: Deduplicate and Move to SSD 🟡 IN PROGRESS
Four-stage pipeline in [dedup/](dedup/):
1. **Ingest** — Copy HDD→staging, compute hashes
2. **Enrich** — Extract EXIF metadata
3. **Deduplicate** — Select canonical per hash using priority rules
4. **Export** — Copy to originals/ and verify

**Database:** PostgreSQL (2 tables: `SourceFile`, `UniqueFile`)  
**Duration:** ~7 hours on full dataset  
**Status:** Ready to run on full 77K files

### Phase 3: Organize and Enrich Metadata ⏸️ PLANNED
- Sort files into date-based structure (YYYY/YYYY-MM-DD/)
- Extract dates from scanned photo stamps via OCR/ML
- Write metadata back to files (EXIF, tags, ratings)

## Directory Structure

```
photo_project/
├── scripts/                        # Utility scripts organized by domain
│   ├── date_extraction/            # OCR and stamp detection
│   │   ├── ocr_stamps.py          # YOLO + Tesseract/TrOCR pipeline
│   │   ├── stamp_detect.py        # Stamp region detection
│   │   ├── ocr_compare.py         # Compare OCR engines
│   │   └── florence.py            # Florence VLM approach
│   ├── rotation/                   # Orientation detection
│   │   ├── detect_rotation.py     # EfficientNet orientation model
│   │   └── docker-compose.yml     # Rotation service
│   └── dedup/                      # Dedup pipeline utilities
│       └── check_progress.py      # Pipeline status checker
│
├── data/                           # Results, samples, and metadata
│   ├── ocr_results.json           # OCR pipeline output
│   ├── rotation_results.json      # Orientation predictions
│   ├── samples/                   # 100 pre-selected sample photos
│   └── metadata/                  # Albums, ML predictions
│
├── dedup/                          # Deduplication pipeline (self-contained)
│   ├── main.py                    # Pipeline orchestrator
│   ├── models/                    # PostgreSQL schema
│   ├── pipeline/                  # 4 stages (ingest, enrich, deduplicate, export)
│   ├── utils/                     # Shared utilities (db, exif, threading)
│   ├── tests/                     # Unit & integration tests
│   └── docker-compose.yml         # PostgreSQL + pipeline services
│
├── yolo_finetune/                  # YOLO stamp detector (separate repo)
│   ├── train.py                   # Fine-tune YOLO model
│   ├── annotate.py                # HTTP annotation server
│   ├── infer_all.py               # Batch inference
│   ├── justfile                   # Task runner (run `just` to list)
│   └── dataset/                   # YOLO-format training data
│
├── docs/                           # Documentation
│   ├── PLAN.md                    # Master plan (all phases)
│   ├── HANDOFF.md                 # Current handoff state
│   ├── DATE_EXTRACTION_APPROACHES.md
│   ├── IMPLEMENTATION_CHECKLIST.md
│   └── CLAUDE_CODE_WORKFLOW.md
│
├── archive/                        # Historical debug outputs
│   └── debug_stamps{_v2..v10}/    # Iterative stamp detection debug
│
├── models/                         # ML model weights (gitignored)
├── staging/                        # Files being ingested (gitignored)
├── originals/                      # Deduplicated canonicals (gitignored)
├── needs_date/                     # Photos awaiting dates (gitignored)
├── organized/                      # Final organized output (gitignored)
│
├── CLAUDE.md                       # Project instructions
├── pyproject.toml                  # Python dependencies
└── README.md                       # This file
```

## Getting Started

### Run the Dedup Pipeline

```bash
cd dedup
docker-compose build
docker-compose up
```

See [dedup/README.md](dedup/README.md) for detailed setup and monitoring.

### Run YOLO Fine-tuning

```bash
cd yolo_finetune
python train.py
python infer.py
```

See [yolo_finetune/](yolo_finetune/) for details.

### Run Utility Scripts

```bash
# Run from project root
uv run scripts/date_extraction/ocr_stamps.py        # YOLO + OCR pipeline
uv run scripts/date_extraction/stamp_detect.py       # Stamp detection
uv run scripts/rotation/detect_rotation.py           # Orientation detection
```

## Key Files

- **[docs/PLAN.md](docs/PLAN.md)** — Master plan for all 3 phases
- **[docs/HANDOFF.md](docs/HANDOFF.md)** — Current session handoff
- **[docs/DATE_EXTRACTION_APPROACHES.md](docs/DATE_EXTRACTION_APPROACHES.md)** — Analysis of date extraction methods
- **[CLAUDE.md](CLAUDE.md)** — Global project constraints and guidelines

## Database

- **Phase 2:** PostgreSQL in Docker (dedup pipeline)
  - Tables: `source_files`, `unique_files`
  - Connection: `postgres://dedup:dedup_local_dev@localhost:5432/dedup`
  - See [dedup/README.md](dedup/README.md) for queries

## Configuration

Environment variables in `.env` (not tracked):
- `HDD_SOURCE_PATH` — HDD mount point
- `SSD_STAGING_PATH` — Staging directory
- `SSD_ORIGINALS_PATH` — Originals directory
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` — PostgreSQL
- `THREAD_WORKERS` — Threads for pipeline stages
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` — Notifications (optional)

## Storage Locations

| Path | Size | Status | Purpose |
|------|------|--------|---------|
| `/mnt/823c.../Photos/img/` | 467 GB | READ-ONLY | Original files (HDD) |
| `/mnt/823c.../SHA256SUMS.txt` | 6 MB | READ-ONLY | Hash manifest |
| `/mnt/823c.../Photos_BACKUP_DO_NOT_TOUCH.tar` | 467 GB | SEALED | Backup archive |
| `staging/` | ~467 GB | Temporary | Ingestion stage |
| `originals/` | ~280-350 GB | Permanent | Deduplicated canonicals |
| `needs_date/` | Variable | Temporary | Photos awaiting dates |

## Development Notes

### Code Organization Principles
- **dedup/** and **yolo_finetune/** are independent subsystems with their own configs, tests, and docs
- **scripts/** contains standalone utilities organized by domain (no cross-dependencies)
- **data/** holds results, samples, and metadata
- **docs/** has all design decisions and operational guides

### Resumability
All pipelines are resumable via database state tracking:
```bash
# Resume where it left off
docker-compose up dedup
```

### Testing
```bash
# Unit tests
cd dedup && python -m pytest tests/test_schema.py -v

# Integration tests
cd dedup && python -m pytest tests/test_integration.py -v
```

## Support

For issues or questions, check:
1. **[docs/PLAN.md](docs/PLAN.md)** — Architecture and design decisions
2. **[dedup/README.md](dedup/README.md)** — Dedup pipeline specific docs
3. **[yolo_finetune/](yolo_finetune/)** — ML training docs
4. **[docs/IMPLEMENTATION_CHECKLIST.md](docs/IMPLEMENTATION_CHECKLIST.md)** — Progress tracking

---

**Last Updated:** 2026-04-04  
**Current Phase:** 2 (Deduplication)  
**Next Phase:** 3 (Organization & Metadata Enrichment)
