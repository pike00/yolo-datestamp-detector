# Phase 2 Dedup Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dockerized multi-threaded dedup pipeline that copies 467GB from HDD to SSD, deduplicates 77K files down to 45K unique, and stores complete metadata in DuckDB.

**Architecture:** Six stages (preflight copy + 5 dedup stages), each resumable and idempotent, storing checkpoint state in DuckDB. Multi-threaded EXIF reads, copies, and verification. Apprise notifications on completion.

**Tech Stack:** Python 3.14, SQLAlchemy 2.0, DuckDB, Pillow (EXIF), ThreadPoolExecutor, Docker, Apprise

---

## Task 1: Project Setup & Dependencies

**Files:**
- Create: `dedup/requirements.txt`
- Create: `dedup/Dockerfile`
- Create: `dedup/docker-compose.yml`
- Create: `dedup/.env.example`

**Context:** This is the foundation. All other tasks depend on these files existing and being correct. The dedup folder is at `/home/will/photo_project/dedup/`.

- [ ] **Step 1: Write requirements.txt with Python dependencies**

```txt
sqlalchemy==2.0.23
duckdb==0.10.0
duckdb-engine==0.11.1
pillow==10.1.0
piexif==1.1.3
apprise==1.8.0
python-dotenv==1.0.0
pytest==7.4.0
pytest-cov==4.1.0
```

- [ ] **Step 2: Write Dockerfile**

```dockerfile
FROM python:3.14-slim

RUN apt-get update && apt-get install -y \
    libexif12 \
    exiftool \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "main.py"]
```

- [ ] **Step 3: Write docker-compose.yml**

```yaml
version: '3.9'

services:
  dedup:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - /mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/Photos/img:/mnt/hdd:ro
      - /home/will/photo_project:/workspace:rw
    environment:
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}
      - HDD_SOURCE_PATH=/mnt/hdd
      - SSD_STAGING_PATH=/workspace/staging
      - SSD_ORIGINALS_PATH=/workspace/originals
      - SSD_DB_PATH=/workspace/dedup.duckdb
      - THREAD_WORKERS=4
      - RETRY_LIMIT=2
      - RUN_STAGES=0,1,2,3,4,5
    working_dir: /app
    env_file:
      - .env
```

- [ ] **Step 4: Write .env.example**

```bash
# Telegram bot configuration
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Optional path overrides (use defaults if unset)
# HDD_SOURCE_PATH=/mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/Photos/img/
# SSD_STAGING_PATH=/home/will/photo_project/staging/
# SSD_ORIGINALS_PATH=/home/will/photo_project/originals/
# SSD_DB_PATH=/home/will/photo_project/dedup.duckdb

# Threading
# THREAD_WORKERS=4
# RETRY_LIMIT=2

# Stages to run (comma-separated, default: 0,1,2,3,4,5)
# RUN_STAGES=0,1,2,3,4,5
```

- [ ] **Step 5: Commit**

```bash
cd /home/will/photo_project/dedup
git add requirements.txt Dockerfile docker-compose.yml .env.example
git commit -m "infra: Docker setup and dependencies for dedup pipeline"
```

---

## Task 2: SQLAlchemy Schema Definition

**Files:**
- Create: `dedup/models/__init__.py`
- Create: `dedup/models/schema.py`
- Create: `dedup/tests/test_schema.py`

**Context:** Defines 7 DuckDB tables with SQLAlchemy ORM. This is referenced by all other tasks. Must be correct before moving on.

- [ ] **Step 1: Write models/__init__.py**

```python
from models.schema import (
    FilePath,
    File,
    Canonical,
    CopyProgress,
    StagingProgress,
    FinalReport,
    Base,
)

__all__ = [
    "FilePath",
    "File",
    "Canonical",
    "CopyProgress",
    "StagingProgress",
    "FinalReport",
    "Base",
]
```

- [ ] **Step 2: Write models/schema.py**

```python
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    JSON,
    Index,
)
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class FilePath(Base):
    """All file paths from SHA256SUMS.txt, indexed by hash."""
    __tablename__ = "file_paths"

    hash = Column(String(64), primary_key=True, nullable=False)
    path = Column(String(1024), nullable=False, unique=True)
    size = Column(Integer, nullable=False)
    source_folder = Column(String(256), nullable=False, index=True)
    filename = Column(String(256), nullable=False)
    extension = Column(String(32), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_filePath_hash_source", "hash", "source_folder"),
    )


class File(Base):
    """EXIF-enriched metadata per unique hash."""
    __tablename__ = "files"

    hash = Column(String(64), primary_key=True, nullable=False, index=True)
    canonical_path = Column(String(1024), nullable=False)
    exif_score = Column(Float, nullable=False, default=0.0)
    exif_datetime = Column(DateTime, nullable=True)
    exif_gps = Column(String(128), nullable=True)
    exif_fields_count = Column(Integer, nullable=False, default=0)
    folder_source = Column(String(256), nullable=False)
    selected_reason = Column(String(128), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_file_hash", "hash"),
        Index("idx_file_folder_source", "folder_source"),
    )


class Canonical(Base):
    """Canonical file selections, one per unique hash."""
    __tablename__ = "canonicals"

    hash = Column(String(64), primary_key=True, nullable=False, index=True)
    canonical_path = Column(String(1024), nullable=False)
    duplicate_count = Column(Integer, nullable=False, default=0)
    total_size_saved_bytes = Column(Integer, nullable=False, default=0)
    verified = Column(Boolean, nullable=False, default=False)
    verification_hash = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_canonical_hash", "hash"),
        Index("idx_canonical_verified", "verified"),
    )


class CopyProgress(Base):
    """Track copy status for each canonical file (Stage 4)."""
    __tablename__ = "copy_progress"

    hash = Column(String(64), primary_key=True, nullable=False, index=True)
    status = Column(String(32), nullable=False, default="pending")
    copied_path = Column(String(1024), nullable=True)
    bytes_copied = Column(Integer, nullable=False, default=0)
    error_msg = Column(String(512), nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_copyProgress_status", "status"),
        Index("idx_copyProgress_hash", "hash"),
    )


class StagingProgress(Base):
    """Track staging copy status (Stage 0)."""
    __tablename__ = "staging_progress"

    source_path = Column(String(1024), primary_key=True, nullable=False)
    staging_path = Column(String(1024), nullable=False)
    status = Column(String(32), nullable=False, default="pending")
    bytes_copied = Column(Integer, nullable=False, default=0)
    error_msg = Column(String(512), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_stagingProgress_status", "status"),
    )


class FinalReport(Base):
    """Summary statistics after pipeline completion."""
    __tablename__ = "final_report"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stage_completed = Column(String(32), nullable=False)
    total_files_analyzed = Column(Integer, nullable=False, default=0)
    unique_files = Column(Integer, nullable=False, default=0)
    duplicate_files_removed = Column(Integer, nullable=False, default=0)
    total_space_saved_gb = Column(Float, nullable=False, default=0.0)
    files_by_source = Column(JSON, nullable=True)
    duration_seconds = Column(Integer, nullable=False, default=0)
    errors_encountered = Column(Integer, nullable=False, default=0)
    verified_copies = Column(Integer, nullable=False, default=0)
    failed_verifications = Column(Integer, nullable=False, default=0)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
```

- [ ] **Step 3: Write tests/test_schema.py with unit tests for all tables**

See plan for full test code (include all 6 test functions: test_filePath_creation, test_file_creation, test_canonical_creation, test_copyProgress_creation, test_stagingProgress_creation, test_finalReport_creation).

- [ ] **Step 4: Run tests to verify schema is correct**

```bash
cd /home/will/photo_project/dedup
python -m pytest tests/test_schema.py -v
```

Expected: 6/6 tests passing

- [ ] **Step 5: Commit**

```bash
cd /home/will/photo_project/dedup
git add models/ tests/test_schema.py
git commit -m "feat: SQLAlchemy schema for dedup pipeline (7 tables)"
```

---

## Task 3: Utility Modules (DB, EXIF, Threading, Notifications)

**Files:**
- Create: `dedup/utils/__init__.py`
- Create: `dedup/utils/db.py`
- Create: `dedup/utils/exif.py`
- Create: `dedup/utils/threading.py`
- Create: `dedup/utils/notifications.py`

**Context:** Helper modules used by all stages. DB module manages DuckDB sessions. EXIF module reads metadata. Threading module wraps ThreadPoolExecutor. Notifications module sends Telegram alerts.

See plan for complete code for each module.

- [ ] **Step 1-5: Implement all 5 utility modules as specified in plan**

See plan sections for complete code.

- [ ] **Step 6: Commit**

```bash
cd /home/will/photo_project/dedup
git add utils/
git commit -m "feat: utility modules for DB, EXIF, threading, notifications"
```

---

## Task 4: Stage 0 — Preflight Copy (HDD → SSD Staging)

**Files:**
- Create: `dedup/stages/stage0_copy.py`
- Create: `dedup/stages/__init__.py`

**Context:** First stage: copy all 467GB from HDD to SSD. Uses threading for parallelism. DuckDB tracking ensures resumability.

See plan for complete implementation.

- [ ] **Implement Stage 0 as specified in plan with full code and tests**

- [ ] **Commit**

```bash
cd /home/will/photo_project/dedup
git add stages/stage0_copy.py stages/__init__.py
git commit -m "feat: Stage 0 - preflight HDD to SSD copy (multi-threaded, resumable)"
```

---

## Task 5: Stage 1 — Load SHA256SUMS.txt

**Files:**
- Create: `dedup/stages/stage1_load.py`

**Context:** Parse /mnt/.../SHA256SUMS.txt (95K lines) and populate file_paths table. Resumable via DB.

See plan for complete implementation.

- [ ] **Implement Stage 1 as specified in plan with full code and tests**

- [ ] **Commit**

```bash
cd /home/will/photo_project/dedup
git add stages/stage1_load.py
git commit -m "feat: Stage 1 - load SHA256SUMS.txt into file_paths table"
```

---

## Task 6: Stage 2 — Enrich with EXIF (Multi-Threaded)

**Files:**
- Create: `dedup/stages/stage2_enrich.py`

**Context:** Read EXIF metadata for each unique hash. Multi-threaded for speed. Computes EXIF score for canonical selection.

See plan for complete implementation.

- [ ] **Implement Stage 2 as specified in plan with full code and tests**

- [ ] **Commit**

```bash
cd /home/will/photo_project/dedup
git add stages/stage2_enrich.py
git commit -m "feat: Stage 2 - EXIF enrichment with multi-threaded reads"
```

---

## Task 7: Stage 3 — Select Canonicals

**Files:**
- Create: `dedup/stages/stage3_select.py`

**Context:** Select canonical file per hash group using EXIF score + folder cohesion tiebreaker. Creates Canonical and CopyProgress entries.

See plan for complete implementation.

- [ ] **Implement Stage 3 as specified in plan with full code and tests**

- [ ] **Commit**

```bash
cd /home/will/photo_project/dedup
git add stages/stage3_select.py
git commit -m "feat: Stage 3 - canonical selection with folder cohesion tiebreaker"
```

---

## Task 8: Stage 4 — Copy to Originals (Multi-Threaded, Resumable)

**Files:**
- Create: `dedup/stages/stage4_copy.py`

**Context:** Copy canonical files from staging to originals/ with hash-based names. Multi-threaded, resumable, retry logic.

See plan for complete implementation.

- [ ] **Implement Stage 4 as specified in plan with full code and tests**

- [ ] **Commit**

```bash
cd /home/will/photo_project/dedup
git add stages/stage4_copy.py
git commit -m "feat: Stage 4 - copy canonicals to originals with hash-based names (multi-threaded)"
```

---

## Task 9: Stage 5 — Verify Integrity (Multi-Threaded)

**Files:**
- Create: `dedup/stages/stage5_verify.py`

**Context:** Re-hash all copied files and verify against original hashes. Multi-threaded. Generates final report.

See plan for complete implementation.

- [ ] **Implement Stage 5 as specified in plan with full code and tests**

- [ ] **Commit**

```bash
cd /home/will/photo_project/dedup
git add stages/stage5_verify.py
git commit -m "feat: Stage 5 - re-hash verification with multi-threaded checks"
```

---

## Task 10: Main Orchestration & Configuration

**Files:**
- Create: `dedup/main.py`
- Create: `dedup/config.py`

**Context:** Entry point. Reads .env, orchestrates all 6 stages in order, handles errors gracefully, sends Telegram notifications.

See plan for complete implementation.

- [ ] **Implement main.py and config.py as specified in plan**

- [ ] **Commit**

```bash
cd /home/will/photo_project/dedup
git add main.py config.py
git commit -m "feat: main orchestration and config management"
```

---

## Task 11: Integration Tests & Docker Build

**Files:**
- Modify: `dedup/requirements.txt` (pytest already there)
- Create: `dedup/tests/test_integration.py`
- Create: `dedup/.gitignore`

**Context:** Integration smoke test covering full pipeline on minimal data. Docker build validation.

See plan for complete integration test code.

- [ ] **Write integration test covering all 6 stages on sample data**

- [ ] **Run full test suite**

```bash
cd /home/will/photo_project/dedup
python -m pytest tests/ -v
```

Expected: All tests passing

- [ ] **Build Docker image**

```bash
cd /home/will/photo_project/dedup
docker build -t dedup:latest .
```

- [ ] **Write .gitignore**

- [ ] **Commit**

```bash
cd /home/will/photo_project/dedup
git add tests/test_integration.py .gitignore
git commit -m "test: integration tests and Docker build"
```

---

## Task 12: Documentation & Final Verification

**Files:**
- Create: `dedup/README.md`

**Context:** User-facing documentation. Quick start, architecture overview, debugging guide.

- [ ] **Write README.md as specified in plan**

- [ ] **Final verification: all imports work**

```bash
cd /home/will/photo_project/dedup
python -c "import main; import config; import utils.db; print('✓ All imports successful')"
```

- [ ] **Commit**

```bash
cd /home/will/photo_project/dedup
git add README.md
git commit -m "docs: dedup pipeline README"
```

---

**Total Tasks**: 12 task groups across 6 stages + infra + testing + docs

All code snippets are in the full plan document. Each task is independent (after Task 1). Estimated implementation time: 6-8 hours for one engineer following this plan.
