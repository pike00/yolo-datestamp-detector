# Media Embeddings Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute and store SigLIP ViT-SO400M semantic embeddings for all ~42K deduplicated still images and ~5.3K videos in `originals/media/`, with results in a new `photo_embeddings` Postgres table — the foundation for all downstream photo ML work.

**Architecture:** Single-script batch job (`scripts/media_embeddings/embed_all.py`) runs inside Docker, loads SigLIP once at startup, processes images in batches of 64 and video keyframes (3 per file) in batches of 16, checkpoints via Postgres so it's fully resumable, logs progress via tqdm to stdout.

**Tech Stack:** Python 3.12, `transformers` (SigLIP), `torch` (CPU), `pillow`, `pillow-heif` (HEIC), `ffmpeg` + `ffprobe` (video keyframes), `psycopg[binary]`, `pgvector`, `tqdm`, Docker, `pgvector/pgvector:pg18` Postgres image.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `dedup/docker-compose.yml` | Create | Recreate dedup-postgres with pgvector image |
| `scripts/media_embeddings/__init__.py` | Create | Package marker |
| `scripts/media_embeddings/loader.py` | Create | File scanning, image opening, video keyframe extraction |
| `scripts/media_embeddings/db.py` | Create | Postgres helpers: get completed stems, bulk insert |
| `scripts/media_embeddings/requirements.txt` | Create | Docker pip deps |
| `scripts/media_embeddings/embed_all.py` | Create | Main entry point: model loading, batch loop, tqdm |
| `docker/Dockerfile.media-embeddings` | Create | Container definition |
| `docker/docker-compose.media-embeddings.yml` | Create | Compose service definition |
| `tests/test_media_embeddings/__init__.py` | Create | Package marker |
| `tests/test_media_embeddings/conftest.py` | Create | Shared fixtures: tmp media dirs, mock model, test video |
| `tests/test_media_embeddings/test_loader.py` | Create | Tests for loader.py |
| `tests/test_media_embeddings/test_db.py` | Create | Tests for db.py |
| `justfile` | Modify | Add `embed` and `embed-bg` recipes |

---

## Task 1: Migrate dedup-postgres to pgvector image

The running `dedup-postgres` container uses `postgres:18-alpine` which does not include pgvector. Migrate to `pgvector/pgvector:pg18` (drop-in replacement, same data volume).

**Files:**
- Create: `dedup/docker-compose.yml`

- [ ] **Step 1: Verify pgvector is missing**

```bash
docker exec dedup-postgres psql -U dedup -d dedup \
  -c "SELECT name FROM pg_available_extensions WHERE name = 'vector';"
```
Expected output: `(0 rows)` — confirms migration is needed.

- [ ] **Step 2: Create dedup/docker-compose.yml**

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg18
    container_name: dedup-postgres
    environment:
      POSTGRES_USER: dedup
      POSTGRES_PASSWORD: dedup_local_dev
      POSTGRES_DB: dedup
      PGDATA: /var/lib/postgresql/18/docker
    volumes:
      - dedup_postgres_data:/var/lib/postgresql/18/docker
    ports:
      - "5432:5432"
    restart: unless-stopped

volumes:
  dedup_postgres_data:
    external: true
```

- [ ] **Step 3: Stop old container and start with new image**

```bash
docker stop dedup-postgres
docker rm dedup-postgres
cd /home/will/photo_project/dedup
docker compose up -d
```

Wait ~5 seconds for postgres to start, then verify:
```bash
docker exec dedup-postgres psql -U dedup -d dedup \
  -c "SELECT name, default_version FROM pg_available_extensions WHERE name = 'vector';"
```
Expected: one row with `vector | 0.8.x`

- [ ] **Step 4: Run schema migration**

```bash
docker exec dedup-postgres psql -U dedup -d dedup -c "
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS photo_embeddings (
    sha256      text NOT NULL,
    model       text NOT NULL DEFAULT 'siglip-so400m',
    embedding   vector(1152) NOT NULL,
    media_type  text NOT NULL,
    frame_index integer,
    created_at  timestamptz DEFAULT now(),
    PRIMARY KEY (sha256, model, COALESCE(frame_index, -1))
);

CREATE INDEX IF NOT EXISTS photo_embeddings_ivfflat_idx
  ON photo_embeddings USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);
"
```
Expected: `CREATE EXTENSION`, `CREATE TABLE`, `CREATE INDEX`

- [ ] **Step 5: Verify existing tables survived migration**

```bash
docker exec dedup-postgres psql -U dedup -d dedup -c "\dt"
```
Expected: `stamp_predictions`, `stamp_ocr`, `stamp_prediction_drift`, `stamp_no_stamp`, `stamp_rotations`, `rotation_predictions`, `photo_embeddings` all present.

- [ ] **Step 6: Commit**

```bash
cd /home/will/photo_project
git add dedup/docker-compose.yml
git commit -m "feat: add dedup/docker-compose.yml with pgvector/pgvector:pg18"
```

---

## Task 2: Scaffold package structure

**Files:**
- Create: `scripts/media_embeddings/__init__.py`
- Create: `scripts/media_embeddings/loader.py` (skeleton)
- Create: `scripts/media_embeddings/db.py` (skeleton)
- Create: `tests/test_media_embeddings/__init__.py`
- Create: `tests/test_media_embeddings/conftest.py`

- [ ] **Step 1: Create directories and package markers**

```bash
mkdir -p scripts/media_embeddings tests/test_media_embeddings
touch scripts/media_embeddings/__init__.py
touch tests/test_media_embeddings/__init__.py
```

- [ ] **Step 2: Create loader.py skeleton**

```python
# scripts/media_embeddings/loader.py
from __future__ import annotations

import io
import subprocess
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif', '.heic'}
VIDEO_EXTS = {'.mov', '.mp4', '.m4v', '.mpg', '.3gp'}

try:
    import pillow_heif as _pillow_heif
    _HEIC_AVAILABLE = True
except ImportError:
    _pillow_heif = None  # type: ignore
    _HEIC_AVAILABLE = False


def scan_media_dir(media_dir: Path) -> tuple[list[Path], list[Path]]:
    raise NotImplementedError


def open_image(path: Path) -> Image.Image:
    raise NotImplementedError


def extract_keyframes(video_path: Path, n: int = 3) -> list[Image.Image]:
    raise NotImplementedError
```

- [ ] **Step 3: Create db.py skeleton**

```python
# scripts/media_embeddings/db.py
from __future__ import annotations

import psycopg

MODEL_NAME = "siglip-so400m"


def get_completed_stems(
    conn: psycopg.Connection, model: str = MODEL_NAME
) -> tuple[set[str], set[str]]:
    """Return (done_photo_stems, done_video_stems)."""
    raise NotImplementedError


def bulk_insert_embeddings(
    conn: psycopg.Connection, rows: list[tuple]
) -> None:
    """Insert rows of (sha256, model, embedding, media_type, frame_index)."""
    raise NotImplementedError
```

- [ ] **Step 4: Create conftest.py**

```python
# tests/test_media_embeddings/conftest.py
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import psycopg
import pytest
from PIL import Image


@pytest.fixture
def media_dir(tmp_path: Path) -> Path:
    """Temp dir with 2 JPGs, 1 HEIC stub, and 1 MOV stub."""
    img = Image.new("RGB", (64, 64), color=(128, 0, 0))
    img.save(tmp_path / "aaa.jpg")
    img.save(tmp_path / "bbb.jpeg")
    (tmp_path / "ccc.heic").write_bytes(b"fake-heic")
    (tmp_path / "ddd.mov").write_bytes(b"fake-mov")
    (tmp_path / "eee.json").write_bytes(b"{}")  # must be ignored
    return tmp_path


@pytest.fixture
def real_video(tmp_path: Path) -> Path:
    """3-second black 64x64 video created with ffmpeg."""
    out = tmp_path / "test.mov"
    subprocess.run(
        [
            "ffmpeg", "-f", "lavfi", "-i",
            "color=black:size=64x64:duration=3:rate=1",
            "-c:v", "libx264", "-t", "3", str(out),
            "-y", "-loglevel", "error",
        ],
        check=True,
    )
    return out


@pytest.fixture
def mock_conn():
    """Mock psycopg connection + cursor via context manager."""
    conn = MagicMock(spec=psycopg.Connection)
    cursor = MagicMock()
    cursor.__enter__ = lambda s: cursor
    cursor.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = cursor
    return conn, cursor
```

- [ ] **Step 5: Verify pytest collects fixtures**

```bash
uv run pytest tests/test_media_embeddings/ --collect-only 2>&1 | head -20
```
Expected: no import errors, `0 tests collected`.

- [ ] **Step 6: Commit scaffold**

```bash
git add scripts/media_embeddings/ tests/test_media_embeddings/
git commit -m "feat: scaffold media_embeddings package and test structure"
```

---

## Task 3: TDD — file scanner

**Files:**
- Create: `tests/test_media_embeddings/test_loader.py`
- Modify: `scripts/media_embeddings/loader.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_media_embeddings/test_loader.py
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from media_embeddings.loader import IMAGE_EXTS, VIDEO_EXTS, scan_media_dir


def test_scan_splits_images_and_videos(media_dir):
    images, videos = scan_media_dir(media_dir)
    assert len(images) == 3   # aaa.jpg, bbb.jpeg, ccc.heic
    assert len(videos) == 1   # ddd.mov
    assert all(p.suffix.lower() in IMAGE_EXTS for p in images)
    assert all(p.suffix.lower() in VIDEO_EXTS for p in videos)


def test_scan_ignores_non_media(media_dir):
    images, videos = scan_media_dir(media_dir)
    assert not any(p.suffix == ".json" for p in images + videos)


def test_scan_returns_sorted(media_dir):
    images, videos = scan_media_dir(media_dir)
    assert images == sorted(images)
    assert videos == sorted(videos)
```

- [ ] **Step 2: Run — expect NotImplementedError**

```bash
uv run pytest tests/test_media_embeddings/test_loader.py -v 2>&1 | tail -8
```

- [ ] **Step 3: Implement scan_media_dir in loader.py**

Replace `raise NotImplementedError` under `scan_media_dir`:

```python
def scan_media_dir(media_dir: Path) -> tuple[list[Path], list[Path]]:
    images, videos = [], []
    for p in sorted(media_dir.iterdir()):
        ext = p.suffix.lower()
        if ext in IMAGE_EXTS:
            images.append(p)
        elif ext in VIDEO_EXTS:
            videos.append(p)
    return images, videos
```

- [ ] **Step 4: Run — expect 3 passed**

```bash
uv run pytest tests/test_media_embeddings/test_loader.py -v 2>&1 | tail -8
```

- [ ] **Step 5: Commit**

```bash
git add scripts/media_embeddings/loader.py tests/test_media_embeddings/test_loader.py
git commit -m "feat: implement and test scan_media_dir"
```

---

## Task 4: TDD — image opener

**Files:**
- Modify: `tests/test_media_embeddings/test_loader.py`
- Modify: `scripts/media_embeddings/loader.py`

- [ ] **Step 1: Append failing tests to test_loader.py**

```python
from media_embeddings.loader import open_image


def test_open_jpg_returns_rgb(tmp_path):
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    path = tmp_path / "test.jpg"
    img.save(path)
    result = open_image(path)
    assert result.mode == "RGB"
    assert result.size == (100, 100)


def test_open_jpeg_returns_rgb(tmp_path):
    img = Image.new("RGB", (50, 80))
    path = tmp_path / "test.jpeg"
    img.save(path)
    assert open_image(path).mode == "RGB"


def test_open_heic_delegates_to_pillow_heif(tmp_path):
    path = tmp_path / "test.heic"
    path.write_bytes(b"fake")

    mock_heif = MagicMock()
    mock_heif.mode = "RGB"
    mock_heif.size = (200, 150)
    mock_heif.data = b"\x00" * (200 * 150 * 3)

    with patch("media_embeddings.loader._pillow_heif") as mock_ph:
        mock_ph.read_heif.return_value = mock_heif
        result = open_image(path)

    mock_ph.read_heif.assert_called_once_with(path)
    assert result.size == (200, 150)
```

- [ ] **Step 2: Run — expect NotImplementedError on open_image tests**

```bash
uv run pytest tests/test_media_embeddings/test_loader.py -v 2>&1 | tail -10
```

- [ ] **Step 3: Implement open_image in loader.py**

Replace `raise NotImplementedError` under `open_image`:

```python
def open_image(path: Path) -> Image.Image:
    if path.suffix.lower() == ".heic":
        if not _HEIC_AVAILABLE:
            raise ImportError("pillow-heif required for HEIC: pip install pillow-heif")
        heif = _pillow_heif.read_heif(path)
        return Image.frombytes(heif.mode, heif.size, heif.data, "raw")
    return Image.open(path).convert("RGB")
```

- [ ] **Step 4: Run — expect 6 passed**

```bash
uv run pytest tests/test_media_embeddings/test_loader.py -v 2>&1 | tail -10
```

- [ ] **Step 5: Commit**

```bash
git add scripts/media_embeddings/loader.py tests/test_media_embeddings/test_loader.py
git commit -m "feat: implement and test open_image with HEIC support"
```

---

## Task 5: TDD — video keyframe extractor

**Files:**
- Modify: `tests/test_media_embeddings/test_loader.py`
- Modify: `scripts/media_embeddings/loader.py`

- [ ] **Step 1: Append failing tests to test_loader.py**

```python
from media_embeddings.loader import extract_keyframes


def test_extract_keyframes_returns_three_pil_images(real_video):
    frames = extract_keyframes(real_video, n=3)
    assert len(frames) == 3
    assert all(isinstance(f, Image.Image) for f in frames)
    assert all(f.mode == "RGB" for f in frames)


def test_extract_keyframes_timestamps_are_10_50_90_pct(tmp_path):
    """ffmpeg must be called at 10%/50%/90% of duration."""
    fake_frame = Image.new("RGB", (64, 64))

    with patch("media_embeddings.loader.subprocess.run") as mock_run, \
         patch("media_embeddings.loader.Image") as mock_img:

        ffprobe_result = MagicMock(stdout="10.0\n", returncode=0)
        ffmpeg_result = MagicMock(stdout=b"\x89PNG\r\n", returncode=0)
        mock_run.side_effect = [ffprobe_result] + [ffmpeg_result] * 3
        mock_img.open.return_value.convert.return_value = fake_frame

        extract_keyframes(tmp_path / "v.mov", n=3)

    ffmpeg_calls = [c for c in mock_run.call_args_list
                    if any("ffmpeg" in str(a) for a in c[0])]
    assert len(ffmpeg_calls) == 3

    timestamps = []
    for call in ffmpeg_calls:
        args = call[0][0]
        ts_idx = args.index("-ss") + 1
        timestamps.append(float(args[ts_idx]))

    assert timestamps == pytest.approx([1.0, 5.0, 9.0], abs=0.01)
```

- [ ] **Step 2: Run — expect NotImplementedError**

```bash
uv run pytest tests/test_media_embeddings/test_loader.py::test_extract_keyframes_returns_three_pil_images -v 2>&1 | tail -8
```

- [ ] **Step 3: Implement extract_keyframes in loader.py**

Replace `raise NotImplementedError` under `extract_keyframes`:

```python
def extract_keyframes(video_path: Path, n: int = 3) -> list[Image.Image]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    duration = float(result.stdout.strip())

    frames: list[Image.Image] = []
    for pct in [0.1, 0.5, 0.9][:n]:
        t = duration * pct
        result = subprocess.run(
            [
                "ffmpeg", "-ss", str(t), "-i", str(video_path),
                "-vframes", "1", "-f", "image2", "-vcodec", "png",
                "pipe:1", "-loglevel", "error",
            ],
            capture_output=True, check=True,
        )
        img = Image.open(io.BytesIO(result.stdout)).convert("RGB")
        frames.append(img)
    return frames
```

- [ ] **Step 4: Run — expect 8 passed**

```bash
uv run pytest tests/test_media_embeddings/test_loader.py -v 2>&1 | tail -12
```

- [ ] **Step 5: Commit**

```bash
git add scripts/media_embeddings/loader.py tests/test_media_embeddings/test_loader.py
git commit -m "feat: implement and test extract_keyframes via ffmpeg"
```

---

## Task 6: TDD — DB helpers

**Files:**
- Create: `tests/test_media_embeddings/test_db.py`
- Modify: `scripts/media_embeddings/db.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_media_embeddings/test_db.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from media_embeddings.db import MODEL_NAME, bulk_insert_embeddings, get_completed_stems


def test_get_completed_stems_returns_two_sets(mock_conn):
    conn, cursor = mock_conn
    cursor.fetchall.side_effect = [
        [("abc123",), ("def456",)],  # photos
        [("vid789",)],               # videos
    ]

    done_photos, done_videos = get_completed_stems(conn, MODEL_NAME)

    assert done_photos == {"abc123", "def456"}
    assert done_videos == {"vid789"}
    assert cursor.execute.call_count == 2


def test_get_completed_stems_empty_table(mock_conn):
    conn, cursor = mock_conn
    cursor.fetchall.side_effect = [[], []]

    done_photos, done_videos = get_completed_stems(conn, MODEL_NAME)

    assert done_photos == set()
    assert done_videos == set()


def test_bulk_insert_uses_executemany(mock_conn):
    conn, cursor = mock_conn
    vec = np.zeros(1152, dtype=np.float32)
    rows = [
        ("abc123", MODEL_NAME, vec, "photo", None),
        ("def456", MODEL_NAME, vec, "photo", None),
    ]

    bulk_insert_embeddings(conn, rows)

    cursor.executemany.assert_called_once()
    sql, data = cursor.executemany.call_args[0]
    assert "INSERT INTO photo_embeddings" in sql
    assert "ON CONFLICT DO NOTHING" in sql
    assert data == rows
    conn.commit.assert_called_once()


def test_bulk_insert_video_rows(mock_conn):
    conn, cursor = mock_conn
    vec = np.ones(1152, dtype=np.float32)
    rows = [
        ("vid789", MODEL_NAME, vec, "video_keyframe", 0),
        ("vid789", MODEL_NAME, vec, "video_keyframe", 1),
        ("vid789", MODEL_NAME, vec, "video_keyframe", 2),
    ]

    bulk_insert_embeddings(conn, rows)

    cursor.executemany.assert_called_once()
    conn.commit.assert_called_once()
```

- [ ] **Step 2: Run — expect NotImplementedError**

```bash
uv run pytest tests/test_media_embeddings/test_db.py -v 2>&1 | tail -8
```

- [ ] **Step 3: Implement db.py**

Replace full contents of `scripts/media_embeddings/db.py`:

```python
# scripts/media_embeddings/db.py
from __future__ import annotations

import psycopg

MODEL_NAME = "siglip-so400m"


def get_completed_stems(
    conn: psycopg.Connection, model: str = MODEL_NAME
) -> tuple[set[str], set[str]]:
    """Return (done_photo_stems, done_video_stems).

    Video stems are only counted done once all 3 keyframes are present.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT sha256 FROM photo_embeddings "
            "WHERE model = %s AND media_type = 'photo'",
            (model,),
        )
        done_photos = {row[0] for row in cur.fetchall()}

        cur.execute(
            "SELECT sha256 FROM photo_embeddings "
            "WHERE model = %s AND media_type = 'video_keyframe' "
            "GROUP BY sha256 HAVING COUNT(*) >= 3",
            (model,),
        )
        done_videos = {row[0] for row in cur.fetchall()}

    return done_photos, done_videos


def bulk_insert_embeddings(
    conn: psycopg.Connection, rows: list[tuple]
) -> None:
    """Insert rows of (sha256, model, embedding, media_type, frame_index)."""
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO photo_embeddings "
            "(sha256, model, embedding, media_type, frame_index) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT DO NOTHING",
            rows,
        )
    conn.commit()
```

- [ ] **Step 4: Run — expect 4 passed**

```bash
uv run pytest tests/test_media_embeddings/test_db.py -v 2>&1 | tail -10
```

- [ ] **Step 5: Run full suite**

```bash
uv run pytest tests/test_media_embeddings/ -v 2>&1 | tail -15
```
Expected: 12 passed.

- [ ] **Step 6: Commit**

```bash
git add scripts/media_embeddings/db.py tests/test_media_embeddings/test_db.py
git commit -m "feat: implement and test DB helpers"
```

---

## Task 7: Main script — embed_all.py

**Files:**
- Create: `scripts/media_embeddings/embed_all.py`
- Create: `scripts/media_embeddings/requirements.txt`

- [ ] **Step 1: Create requirements.txt**

```
# scripts/media_embeddings/requirements.txt
transformers>=4.40.0
torch>=2.3.0
pillow>=10.0.0
pillow-heif>=0.16.0
psycopg[binary]>=3.1.0
pgvector>=0.3.0
tqdm>=4.66.0
numpy>=1.26.0
```

- [ ] **Step 2: Create embed_all.py**

```python
# scripts/media_embeddings/embed_all.py
"""SigLIP ViT-SO400M embedding pipeline for originals/media/.

Reads all images and videos from MEDIA_DIR, computes 1152-dim embeddings,
and writes results to the photo_embeddings table in Postgres.

Fully resumable: any sha256 already in photo_embeddings is skipped.

Usage (local):
    uv run scripts/media_embeddings/embed_all.py

Usage (Docker):
    docker compose -f docker/docker-compose.media-embeddings.yml up
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import psycopg
import torch
from pgvector.psycopg import register_vector
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from media_embeddings.db import MODEL_NAME, bulk_insert_embeddings, get_completed_stems
from media_embeddings.loader import extract_keyframes, open_image, scan_media_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MEDIA_DIR = Path(os.environ.get("MEDIA_DIR", "/home/will/photo_project/originals/media"))
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://dedup:dedup_local_dev@127.0.0.1:5432/dedup"
)
MODEL_HF_ID = "google/siglip-so400m-patch14-384"
IMAGE_BATCH = 64
VIDEO_BATCH = 16


def embed_batch(
    model: AutoModel,
    processor: AutoProcessor,
    images: list[Image.Image],
) -> np.ndarray:
    inputs = processor(images=images, return_tensors="pt", padding="max_length")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()


def process_images(
    conn: psycopg.Connection,
    model: AutoModel,
    processor: AutoProcessor,
    image_paths: list[Path],
    done_stems: set[str],
) -> int:
    pending = [p for p in image_paths if p.stem not in done_stems]
    log.info(
        "Images: %d total, %d already done, %d to embed",
        len(image_paths), len(image_paths) - len(pending), len(pending),
    )
    processed = 0
    for i in tqdm(range(0, len(pending), IMAGE_BATCH), desc="images", unit="batch"):
        batch_paths = pending[i : i + IMAGE_BATCH]
        images, stems = [], []
        for path in batch_paths:
            try:
                images.append(open_image(path))
                stems.append(path.stem)
            except Exception as exc:
                log.warning("Skipping %s: %s", path.name, exc)
        if not images:
            continue
        vectors = embed_batch(model, processor, images)
        rows = [
            (stem, MODEL_NAME, vec.tolist(), "photo", None)
            for stem, vec in zip(stems, vectors)
        ]
        bulk_insert_embeddings(conn, rows)
        processed += len(rows)
    return processed


def process_videos(
    conn: psycopg.Connection,
    model: AutoModel,
    processor: AutoProcessor,
    video_paths: list[Path],
    done_stems: set[str],
) -> int:
    pending = [p for p in video_paths if p.stem not in done_stems]
    log.info(
        "Videos: %d total, %d already done, %d to embed",
        len(video_paths), len(video_paths) - len(pending), len(pending),
    )
    processed = 0
    for i in tqdm(range(0, len(pending), VIDEO_BATCH), desc="videos", unit="batch"):
        batch_paths = pending[i : i + VIDEO_BATCH]
        for path in batch_paths:
            try:
                frames = extract_keyframes(path, n=3)
            except Exception as exc:
                log.warning("Skipping video %s: %s", path.name, exc)
                continue
            vectors = embed_batch(model, processor, frames)
            rows = [
                (path.stem, MODEL_NAME, vec.tolist(), "video_keyframe", idx)
                for idx, vec in enumerate(vectors)
            ]
            bulk_insert_embeddings(conn, rows)
            processed += len(frames)
    return processed


def main() -> None:
    log.info("Loading SigLIP model: %s", MODEL_HF_ID)
    processor = AutoProcessor.from_pretrained(MODEL_HF_ID)
    model = AutoModel.from_pretrained(MODEL_HF_ID)
    # set to inference mode (no grad tracking)
    model.training = False
    log.info("Model loaded. Scanning %s", MEDIA_DIR)

    image_paths, video_paths = scan_media_dir(MEDIA_DIR)
    log.info("Found %d images, %d videos", len(image_paths), len(video_paths))

    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        done_photos, done_videos = get_completed_stems(conn, MODEL_NAME)
        log.info(
            "Checkpoint: %d photos done, %d videos done",
            len(done_photos), len(done_videos),
        )
        n_images = process_images(conn, model, processor, image_paths, done_photos)
        n_videos = process_videos(conn, model, processor, video_paths, done_videos)

    log.info("Done. Embedded %d images, %d video frames.", n_images, n_videos)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify imports work locally**

```bash
uv add transformers torch pillow pillow-heif psycopg pgvector tqdm numpy 2>/dev/null || true
uv run python -c "
import sys; sys.path.insert(0, 'scripts')
from media_embeddings import embed_all
print('imports ok')
"
```
Expected: `imports ok`

- [ ] **Step 4: Commit**

```bash
git add scripts/media_embeddings/embed_all.py scripts/media_embeddings/requirements.txt
git commit -m "feat: implement embed_all.py main pipeline"
```

---

## Task 8: Dockerfile and docker-compose

**Files:**
- Create: `docker/Dockerfile.media-embeddings`
- Create: `docker/docker-compose.media-embeddings.yml`

- [ ] **Step 1: Create Dockerfile**

```dockerfile
# docker/Dockerfile.media-embeddings
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libheif-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY scripts/media_embeddings/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY scripts/ /app/scripts/

ENV PYTHONUNBUFFERED=1
ENV MEDIA_DIR=/media

CMD ["python", "scripts/media_embeddings/embed_all.py"]
```

- [ ] **Step 2: Create docker-compose.media-embeddings.yml**

```yaml
# docker/docker-compose.media-embeddings.yml
services:
  media_embeddings:
    build:
      context: ..
      dockerfile: docker/Dockerfile.media-embeddings
    network_mode: host
    environment:
      - DATABASE_URL=postgresql://dedup:dedup_local_dev@127.0.0.1:5432/dedup
      - MEDIA_DIR=/media
      - PYTHONUNBUFFERED=1
    volumes:
      - /home/will/photo_project/originals/media:/media:ro
      - hf_cache:/root/.cache/huggingface

volumes:
  hf_cache:
```

- [ ] **Step 3: Build image**

```bash
cd /home/will/photo_project
docker compose -f docker/docker-compose.media-embeddings.yml build
```
Expected: build completes, no errors. Image is ~3-4 GB.

- [ ] **Step 4: Verify container imports resolve**

```bash
docker compose -f docker/docker-compose.media-embeddings.yml run --rm \
  media_embeddings python -c "
import sys
sys.path.insert(0, 'scripts')
from media_embeddings.loader import scan_media_dir
from media_embeddings.db import get_completed_stems
print('container imports ok')
"
```
Expected: `container imports ok`

- [ ] **Step 5: Commit**

```bash
git add docker/Dockerfile.media-embeddings docker/docker-compose.media-embeddings.yml
git commit -m "feat: add media_embeddings Dockerfile and compose"
```

---

## Task 9: justfile recipe + smoke test

**Files:**
- Modify: `justfile`

- [ ] **Step 1: Add embed recipes to justfile**

After the existing `docker-infer` group, add:

```makefile
# Run media embedding pipeline (Docker, CPU — runs overnight for full collection)
embed:
    docker compose -f docker/docker-compose.media-embeddings.yml up --build

# Run embedding pipeline in background
embed-bg:
    docker compose -f docker/docker-compose.media-embeddings.yml up --build -d
    @echo "Logs: docker logs -f photo_project-media_embeddings-1"
```

- [ ] **Step 2: Ensure dedup-postgres is running**

```bash
docker start dedup-postgres
sleep 3
docker exec dedup-postgres psql -U dedup -d dedup \
  -c "SELECT COUNT(*) FROM photo_embeddings;"
```
Expected: `0`

- [ ] **Step 3: Create smoke test directory with 5 images**

```bash
mkdir -p /tmp/embed_smoke
for f in $(ls /home/will/photo_project/originals/media/*.jpg | head -5); do
    cp "$f" /tmp/embed_smoke/
done
ls /tmp/embed_smoke/
```
Expected: 5 `.jpg` files listed.

- [ ] **Step 4: Run smoke test**

```bash
docker compose -f docker/docker-compose.media-embeddings.yml run --rm \
  -e MEDIA_DIR=/smoke \
  -v /tmp/embed_smoke:/smoke:ro \
  media_embeddings
```
Note: first run downloads SigLIP model weights (~1.8 GB) into the `hf_cache` volume. This is a one-time cost.

Expected output (after model download):
```
HH:MM:SS INFO Loading SigLIP model: google/siglip-so400m-patch14-384
HH:MM:SS INFO Model loaded. Scanning /smoke
HH:MM:SS INFO Found 5 images, 0 videos
HH:MM:SS INFO Checkpoint: 0 photos done, 0 videos done
images: 100%|████| 1/1 [00:XX<00:00, ...]
HH:MM:SS INFO Done. Embedded 5 images, 0 video frames.
```

- [ ] **Step 5: Verify rows in Postgres**

```bash
docker exec dedup-postgres psql -U dedup -d dedup -c "
SELECT media_type, COUNT(*),
       (SELECT array_length(embedding::float4[], 1) FROM photo_embeddings LIMIT 1) AS dim
FROM photo_embeddings
GROUP BY media_type;
"
```
Expected: `photo | 5 | 1152`

- [ ] **Step 6: Run full test suite**

```bash
uv run pytest tests/test_media_embeddings/ -v 2>&1 | tail -20
```
Expected: 12 passed, 0 failed.

- [ ] **Step 7: Final commit**

```bash
git add justfile
git commit -m "feat: add embed and embed-bg justfile recipes"
```

---

## Starting the full overnight run

Once smoke test passes:

```bash
docker start dedup-postgres
just embed-bg
docker logs -f photo_project-media_embeddings-1
```

Monitor completion:
```bash
docker exec dedup-postgres psql -U dedup -d dedup \
  -c "SELECT media_type, COUNT(*) FROM photo_embeddings GROUP BY media_type;"
```

Full run estimate: ~3-4 hours on 12-core Ryzen (42K images + 5.3K videos).
