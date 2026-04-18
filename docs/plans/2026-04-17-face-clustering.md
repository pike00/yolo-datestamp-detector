# Face Detection and Clustering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect all faces across 42K photos using insightface ArcFace, cluster by identity with HDBSCAN, and render a labeled review UI.

**Architecture:** Run insightface buffalo_sc (CPU) in Docker for the detection pass, storing 512-dim ArcFace embeddings per face in Postgres with pgvector. Cluster all face embeddings with HDBSCAN in-memory (min_cluster_size=3). Render per-cluster face crop grids for identity labeling.

**Tech Stack:** Python 3.12, `insightface`, `onnxruntime`, `opencv-python-headless`, `hdbscan`, `numpy`, `psycopg[binary]`, `pgvector`, Docker, Postgres pgvector

---

## File Map

| File | Role |
|------|------|
| `scripts/face_clustering/__init__.py` | Package marker |
| `scripts/face_clustering/detector.py` | `FaceDetector` wrapping insightface buffalo_sc |
| `scripts/face_clustering/db.py` | All DB helpers (detection side + cluster side) |
| `scripts/face_clustering/detect_all.py` | Main detection script — scan, skip, detect, insert |
| `scripts/face_clustering/cluster_faces.py` | Load embeddings, run HDBSCAN, write clusters |
| `scripts/face_clustering/build_review_html.py` | Render `output/face_clusters_review.html` |
| `scripts/face_clustering/requirements.txt` | Docker pip deps |
| `docker/Dockerfile.face-detect` | Docker image for detection pass |
| `docker/docker-compose.face-detect.yml` | Compose config (host networking, read-only media) |
| `tests/test_face_clustering/__init__.py` | Package marker |
| `tests/test_face_clustering/conftest.py` | Shared fixtures |
| `tests/test_face_clustering/test_schema.py` | DB schema integration test |
| `tests/test_face_clustering/test_detector.py` | FaceDetector unit tests |
| `tests/test_face_clustering/test_db.py` | DB helper unit tests |
| `tests/test_face_clustering/test_detect_all.py` | detect_all unit tests |
| `tests/test_face_clustering/test_cluster_faces.py` | cluster_faces unit + integration tests |
| `tests/test_face_clustering/test_build_review_html.py` | HTML builder smoke test |
| `tests/test_face_clustering/test_smoke_files.py` | Docker + justfile existence test |

---

## Task 1: DB Schema Migration

Create the `face_detections` and `face_clusters` tables with all indexes including the ivfflat ANN index.

### 1.1 Write the failing test

Create `tests/test_face_clustering/__init__.py` (empty) and `tests/test_face_clustering/test_schema.py`:

```python
# tests/test_face_clustering/test_schema.py
from __future__ import annotations

import os

import psycopg
import pytest
from pgvector.psycopg import register_vector

SKIP_REASON = "set FACE_SMOKE=1 to run schema integration tests"
pytestmark = pytest.mark.skipif(
    not os.environ.get("FACE_SMOKE"), reason=SKIP_REASON
)

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://dedup:dedup_local_dev@127.0.0.1:5432/dedup"
)


@pytest.fixture(scope="module")
def conn():
    with psycopg.connect(DATABASE_URL) as c:
        register_vector(c)
        yield c


def _get_columns(conn, table: str) -> dict[str, str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = %s ORDER BY ordinal_position",
            (table,),
        )
        return {row[0]: row[1] for row in cur.fetchall()}


def _index_exists(conn, index_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM pg_indexes WHERE indexname = %s", (index_name,)
        )
        return cur.fetchone() is not None


def _index_method(conn, index_name: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT am.amname FROM pg_indexes idx "
            "JOIN pg_class c ON c.relname = idx.indexname "
            "JOIN pg_am am ON am.oid = c.relam "
            "WHERE idx.indexname = %s",
            (index_name,),
        )
        row = cur.fetchone()
        return row[0] if row else None


def test_face_detections_columns(conn):
    cols = _get_columns(conn, "face_detections")
    assert "id" in cols
    assert "sha256" in cols
    assert "bbox_x" in cols
    assert "bbox_y" in cols
    assert "bbox_w" in cols
    assert "bbox_h" in cols
    assert "det_score" in cols
    assert "embedding" in cols
    assert "detected_at" in cols


def test_face_detections_embedding_dimension(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT atttypmod FROM pg_attribute "
            "JOIN pg_class ON pg_class.oid = pg_attribute.attrelid "
            "WHERE pg_class.relname = 'face_detections' "
            "AND pg_attribute.attname = 'embedding'"
        )
        row = cur.fetchone()
    # pgvector stores dims as atttypmod; value is dimension
    assert row is not None and row[0] == 512


def test_face_clusters_columns(conn):
    cols = _get_columns(conn, "face_clusters")
    assert "cluster_id" in cols
    assert "face_id" in cols
    assert "label" in cols
    assert "is_representative" in cols
    assert "created_at" in cols


def test_face_clusters_foreign_key(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT tc.constraint_type FROM information_schema.table_constraints tc "
            "JOIN information_schema.key_column_usage kcu "
            "  ON tc.constraint_name = kcu.constraint_name "
            "WHERE tc.table_name = 'face_clusters' "
            "AND kcu.column_name = 'face_id' "
            "AND tc.constraint_type = 'FOREIGN KEY'"
        )
        assert cur.fetchone() is not None, "face_id FK not found"


def test_face_detections_sha256_index(conn):
    assert _index_exists(conn, "face_detections_sha256_idx"), \
        "btree index on face_detections(sha256) not found"


def test_face_detections_ivfflat_index(conn):
    assert _index_exists(conn, "face_detections_embedding_idx"), \
        "ivfflat index on face_detections(embedding) not found"
    method = _index_method(conn, "face_detections_embedding_idx")
    assert method == "ivfflat", f"expected ivfflat, got {method}"


def test_face_clusters_cluster_id_index(conn):
    assert _index_exists(conn, "face_clusters_cluster_id_idx"), \
        "btree index on face_clusters(cluster_id) not found"
```

### 1.2 Run to verify it fails

```bash
FACE_SMOKE=1 uv run pytest tests/test_face_clustering/test_schema.py -q
```

Expected output (tables do not exist yet):

```
ERROR tests/test_face_clustering/test_schema.py - psycopg.errors.UndefinedTable: ...
```

Without `FACE_SMOKE=1`, all tests are skipped:

```bash
uv run pytest tests/test_face_clustering/test_schema.py -q
# 0 passed, 7 skipped
```

### 1.3 Implement — create migration script

Create `scripts/face_clustering/migrate.py`:

```python
#!/usr/bin/env python3
"""One-shot migration: create face_detections and face_clusters tables."""
from __future__ import annotations

import os

import psycopg
from pgvector.psycopg import register_vector

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://dedup:dedup_local_dev@127.0.0.1:5432/dedup"
)

DDL = """
CREATE TABLE IF NOT EXISTS face_detections (
    id          serial PRIMARY KEY,
    sha256      text NOT NULL,
    bbox_x      int NOT NULL,
    bbox_y      int NOT NULL,
    bbox_w      int NOT NULL,
    bbox_h      int NOT NULL,
    det_score   float NOT NULL,
    embedding   vector(512) NOT NULL,
    detected_at timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS face_detections_sha256_idx
    ON face_detections(sha256);

CREATE INDEX IF NOT EXISTS face_detections_embedding_idx
    ON face_detections USING ivfflat (embedding vector_l2_ops)
    WITH (lists = 100);

CREATE TABLE IF NOT EXISTS face_clusters (
    cluster_id      int NOT NULL,
    face_id         int NOT NULL REFERENCES face_detections(id),
    label           text,
    is_representative bool NOT NULL DEFAULT false,
    created_at      timestamptz DEFAULT now(),
    PRIMARY KEY (cluster_id, face_id)
);

CREATE INDEX IF NOT EXISTS face_clusters_cluster_id_idx
    ON face_clusters(cluster_id);
"""


def main() -> None:
    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(DDL)
        conn.commit()
    print("Migration complete.")


if __name__ == "__main__":
    main()
```

Run the migration:

```bash
uv run scripts/face_clustering/migrate.py
```

### 1.4 Run to verify it passes

```bash
FACE_SMOKE=1 uv run pytest tests/test_face_clustering/test_schema.py -q
```

Expected:

```
7 passed in 0.XXs
```

### 1.5 Commit

```bash
git add scripts/face_clustering/migrate.py \
        tests/test_face_clustering/__init__.py \
        tests/test_face_clustering/test_schema.py
git commit -m "feat(face): db schema migration — face_detections + face_clusters + ivfflat index"
```

---

## Task 2: Package Scaffold + conftest

- [ ] Create empty `scripts/face_clustering/__init__.py`
- [ ] Create `scripts/face_clustering/requirements.txt`
- [ ] Create `tests/test_face_clustering/conftest.py` with three shared fixtures

### 2.1 Write the failing test

Create `tests/test_face_clustering/test_conftest_fixtures.py`:

```python
# tests/test_face_clustering/test_conftest_fixtures.py
from __future__ import annotations

import numpy as np
import psycopg
from unittest.mock import MagicMock


def test_fake_face_dict_shape(fake_face_dict):
    assert fake_face_dict["bbox"] == [10, 20, 50, 60]
    assert abs(fake_face_dict["det_score"] - 0.95) < 1e-6
    assert isinstance(fake_face_dict["embedding"], np.ndarray)
    assert fake_face_dict["embedding"].shape == (512,)
    assert fake_face_dict["embedding"].dtype == np.float32


def test_bgr_image_shape(bgr_image):
    assert bgr_image.shape == (64, 64, 3)
    assert bgr_image.dtype == np.uint8


def test_mock_conn_is_mock(mock_conn):
    conn, cursor = mock_conn
    assert isinstance(conn, MagicMock)
    assert isinstance(cursor, MagicMock)
    # context manager protocol
    with conn.cursor() as cur:
        pass
```

Run — fails because `conftest.py` doesn't exist yet:

```bash
uv run pytest tests/test_face_clustering/test_conftest_fixtures.py -q
```

Expected:

```
ERROR tests/test_face_clustering/test_conftest_fixtures.py - fixture 'fake_face_dict' not found
```

### 2.2 Implement

Create `scripts/face_clustering/__init__.py` (empty):

```python
```

Create `scripts/face_clustering/requirements.txt`:

```
insightface>=0.7.3
onnxruntime>=1.18.0
opencv-python-headless>=4.9.0
hdbscan>=0.8.38
numpy>=1.26.0
psycopg[binary]>=3.1.0
pgvector>=0.3.0
tqdm>=4.66.0
```

Create `tests/test_face_clustering/conftest.py`:

```python
# tests/test_face_clustering/conftest.py
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import psycopg
import pytest


@pytest.fixture
def mock_conn():
    """Return (conn_mock, cursor_mock) with context-manager support."""
    conn = MagicMock(spec=psycopg.Connection)
    cursor = MagicMock()
    cursor.__enter__ = lambda s: cursor
    cursor.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = cursor
    return conn, cursor


@pytest.fixture
def fake_face_dict():
    """A single face detection dict as returned by FaceDetector.detect()."""
    return {
        "bbox": [10, 20, 50, 60],   # [x, y, w, h]
        "det_score": 0.95,
        "embedding": np.zeros(512, dtype=np.float32),
    }


@pytest.fixture
def bgr_image():
    """64x64 uint8 BGR image (solid blue)."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :, 0] = 255  # blue channel
    return img
```

### 2.3 Run to verify it passes

```bash
uv run pytest tests/test_face_clustering/test_conftest_fixtures.py -q
```

Expected:

```
3 passed in 0.XXs
```

### 2.4 Commit

```bash
git add scripts/face_clustering/__init__.py \
        scripts/face_clustering/requirements.txt \
        tests/test_face_clustering/conftest.py \
        tests/test_face_clustering/test_conftest_fixtures.py
git commit -m "feat(face): package scaffold, requirements.txt, and conftest fixtures"
```

---

## Task 3: FaceDetector Wrapper

Implement `scripts/face_clustering/detector.py` — a thin wrapper around insightface that:
- initializes buffalo_sc with CPUExecutionProvider
- filters faces below det_score threshold (default 0.5)
- converts bbox from `[x1, y1, x2, y2]` to `[x, y, w, h]`
- returns a list of dicts matching the `fake_face_dict` shape

All tests mock `face_clustering.detector.insightface` — no model download needed.

### 3.1 Write the failing test

Create `tests/test_face_clustering/test_detector.py`:

```python
# tests/test_face_clustering/test_detector.py
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


@pytest.fixture
def mock_insightface():
    """Patch insightface at the module level so no model is loaded."""
    with patch("face_clustering.detector.insightface") as m:
        # app.FaceAnalysis(...) returns a mock app instance
        mock_app = MagicMock()
        m.app.FaceAnalysis.return_value = mock_app
        yield m, mock_app


def _make_insightface_face(x1, y1, x2, y2, score, emb=None):
    face = MagicMock()
    face.bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
    face.det_score = score
    face.embedding = emb if emb is not None else np.ones(512, dtype=np.float32)
    return face


def test_prepare_called_in_init(mock_insightface):
    from face_clustering.detector import FaceDetector

    _, mock_app = mock_insightface
    FaceDetector()
    mock_app.prepare.assert_called_once_with(ctx_id=-1, det_size=(640, 640))


def test_detect_empty_on_no_faces(mock_insightface, bgr_image):
    from face_clustering.detector import FaceDetector

    _, mock_app = mock_insightface
    mock_app.get.return_value = []

    detector = FaceDetector()
    result = detector.detect(bgr_image)

    assert result == []
    mock_app.get.assert_called_once_with(bgr_image)


def test_low_score_faces_filtered(mock_insightface, bgr_image):
    from face_clustering.detector import FaceDetector

    _, mock_app = mock_insightface
    low = _make_insightface_face(0, 0, 50, 50, score=0.3)
    high = _make_insightface_face(10, 20, 60, 80, score=0.9)
    mock_app.get.return_value = [low, high]

    detector = FaceDetector(min_det_score=0.5)
    result = detector.detect(bgr_image)

    assert len(result) == 1
    assert abs(result[0]["det_score"] - 0.9) < 1e-5


def test_bbox_converted_x1y1x2y2_to_xywh(mock_insightface, bgr_image):
    from face_clustering.detector import FaceDetector

    _, mock_app = mock_insightface
    face = _make_insightface_face(x1=10, y1=20, x2=60, y2=80, score=0.8)
    mock_app.get.return_value = [face]

    detector = FaceDetector()
    result = detector.detect(bgr_image)

    assert len(result) == 1
    assert result[0]["bbox"] == [10, 20, 50, 60]  # w=50, h=60


def test_embedding_preserved(mock_insightface, bgr_image):
    from face_clustering.detector import FaceDetector

    _, mock_app = mock_insightface
    emb = np.arange(512, dtype=np.float32)
    face = _make_insightface_face(0, 0, 50, 50, score=0.7, emb=emb)
    mock_app.get.return_value = [face]

    detector = FaceDetector()
    result = detector.detect(bgr_image)

    np.testing.assert_array_equal(result[0]["embedding"], emb)
```

Run — fails because `detector.py` doesn't exist:

```bash
uv run pytest tests/test_face_clustering/test_detector.py -q
```

Expected:

```
ERROR ... ModuleNotFoundError: No module named 'face_clustering.detector'
```

### 3.2 Implement

Create `scripts/face_clustering/detector.py`:

```python
# scripts/face_clustering/detector.py
from __future__ import annotations

import numpy as np

import insightface


class FaceDetector:
    """Thin wrapper around insightface buffalo_sc for CPU inference.

    detect() converts insightface bbox (x1,y1,x2,y2) to (x,y,w,h) and
    filters detections below min_det_score.
    """

    def __init__(
        self,
        model_name: str = "buffalo_sc",
        min_det_score: float = 0.5,
    ) -> None:
        self.min_det_score = min_det_score
        self._app = insightface.app.FaceAnalysis(
            name=model_name,
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=-1, det_size=(640, 640))

    def detect(self, img_bgr: np.ndarray) -> list[dict]:
        """Run detection on a BGR uint8 numpy array.

        Returns a list of dicts:
            {
                "bbox": [x, y, w, h],   # ints
                "det_score": float,
                "embedding": np.ndarray shape (512,) float32,
            }
        """
        raw = self._app.get(img_bgr)
        results = []
        for face in raw:
            score = float(face.det_score)
            if score < self.min_det_score:
                continue
            x1, y1, x2, y2 = face.bbox.tolist()
            results.append(
                {
                    "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    "det_score": score,
                    "embedding": np.array(face.embedding, dtype=np.float32),
                }
            )
        return results
```

### 3.3 Run to verify it passes

```bash
uv run pytest tests/test_face_clustering/test_detector.py -q
```

Expected:

```
5 passed in 0.XXs
```

### 3.4 Commit

```bash
git add scripts/face_clustering/detector.py \
        tests/test_face_clustering/test_detector.py
git commit -m "feat(face): FaceDetector wrapper with score filtering and bbox conversion"
```

---

## Task 4: DB Helpers (Detect Side)

Implement `get_processed_sha256s` and `insert_faces` in `scripts/face_clustering/db.py`.

### 4.1 Write the failing test

Create `tests/test_face_clustering/test_db.py`:

```python
# tests/test_face_clustering/test_db.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


def test_get_processed_sha256s_returns_set(mock_conn):
    from face_clustering.db import get_processed_sha256s

    conn, cursor = mock_conn
    cursor.fetchall.return_value = [("abc",), ("def",), ("ghi",)]

    result = get_processed_sha256s(conn)

    assert result == {"abc", "def", "ghi"}
    cursor.execute.assert_called_once()
    sql = cursor.execute.call_args[0][0]
    assert "face_detections" in sql
    assert "sha256" in sql


def test_get_processed_sha256s_empty(mock_conn):
    from face_clustering.db import get_processed_sha256s

    conn, cursor = mock_conn
    cursor.fetchall.return_value = []

    result = get_processed_sha256s(conn)

    assert result == set()


def test_insert_faces_uses_executemany(mock_conn, fake_face_dict):
    from face_clustering.db import insert_faces

    conn, cursor = mock_conn
    # executemany doesn't return rows; fetchall gives inserted IDs via RETURNING
    cursor.fetchall.return_value = [(1,), (2,)]

    sha256 = "deadbeef"
    ids = insert_faces(conn, sha256, [fake_face_dict, fake_face_dict])

    cursor.executemany.assert_called_once()
    sql, rows = cursor.executemany.call_args[0]
    assert "INSERT INTO face_detections" in sql
    # all 7 data columns present
    for col in ("sha256", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "det_score", "embedding"):
        assert col in sql
    assert len(rows) == 2
    assert rows[0][0] == sha256        # sha256 first
    assert rows[0][1] == 10            # bbox_x
    assert rows[0][2] == 20            # bbox_y
    assert rows[0][3] == 50            # bbox_w
    assert rows[0][4] == 60            # bbox_h
    assert abs(rows[0][5] - 0.95) < 1e-5  # det_score
    conn.commit.assert_called_once()


def test_insert_faces_returns_ids(mock_conn, fake_face_dict):
    from face_clustering.db import insert_faces

    conn, cursor = mock_conn
    cursor.fetchall.return_value = [(7,), (8,)]

    ids = insert_faces(conn, "sha", [fake_face_dict, fake_face_dict])

    assert ids == [7, 8]


def test_insert_faces_empty_list_noop(mock_conn):
    from face_clustering.db import insert_faces

    conn, cursor = mock_conn

    ids = insert_faces(conn, "sha", [])

    cursor.executemany.assert_not_called()
    assert ids == []
```

Run — fails:

```bash
uv run pytest tests/test_face_clustering/test_db.py -q
```

Expected:

```
ERROR ... ModuleNotFoundError: No module named 'face_clustering.db'
```

### 4.2 Implement

Create `scripts/face_clustering/db.py`:

```python
# scripts/face_clustering/db.py
from __future__ import annotations

import numpy as np
import psycopg


def get_processed_sha256s(conn: psycopg.Connection) -> set[str]:
    """Return the set of sha256 values already present in face_detections."""
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT sha256 FROM face_detections")
        return {row[0] for row in cur.fetchall()}


def insert_faces(
    conn: psycopg.Connection,
    sha256: str,
    faces: list[dict],
) -> list[int]:
    """Insert face dicts for a single image; return list of serial IDs.

    Each dict must have keys: bbox ([x,y,w,h]), det_score, embedding (ndarray 512).
    Uses executemany + RETURNING to get the generated IDs.
    """
    if not faces:
        return []

    rows = [
        (
            sha256,
            face["bbox"][0],    # bbox_x
            face["bbox"][1],    # bbox_y
            face["bbox"][2],    # bbox_w
            face["bbox"][3],    # bbox_h
            face["det_score"],
            face["embedding"].tolist(),
        )
        for face in faces
    ]

    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO face_detections "
            "(sha256, bbox_x, bbox_y, bbox_w, bbox_h, det_score, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) "
            "RETURNING id",
            rows,
        )
        ids = [row[0] for row in cur.fetchall()]

    conn.commit()
    return ids


def load_all_face_embeddings(
    conn: psycopg.Connection,
) -> tuple[list[int], np.ndarray]:
    """Return (face_ids, embeddings_matrix) for all rows in face_detections.

    embeddings_matrix shape: (N, 512), dtype float32.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT id, embedding FROM face_detections ORDER BY id")
        rows = cur.fetchall()

    if not rows:
        return [], np.empty((0, 512), dtype=np.float32)

    ids = [row[0] for row in rows]
    embeddings = np.array([row[1] for row in rows], dtype=np.float32)
    return ids, embeddings


def write_clusters(
    conn: psycopg.Connection,
    assignments: list[tuple[int, int]],
) -> None:
    """Insert (face_id, cluster_id) pairs into face_clusters.

    Skips noise points (cluster_id == -1).
    Clears existing rows first for idempotency.
    """
    with conn.cursor() as cur:
        cur.execute("DELETE FROM face_clusters")
        rows = [
            (cluster_id, face_id)
            for face_id, cluster_id in assignments
            if cluster_id >= 0
        ]
        if rows:
            cur.executemany(
                "INSERT INTO face_clusters (cluster_id, face_id) "
                "VALUES (%s, %s) "
                "ON CONFLICT DO NOTHING",
                rows,
            )
    conn.commit()


def mark_representatives(conn: psycopg.Connection) -> None:
    """Set is_representative=true for the face with the highest det_score per cluster."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE face_clusters fc
            SET is_representative = true
            FROM (
                SELECT DISTINCT ON (fc2.cluster_id)
                    fc2.cluster_id,
                    fc2.face_id
                FROM face_clusters fc2
                JOIN face_detections fd ON fd.id = fc2.face_id
                ORDER BY fc2.cluster_id, fd.det_score DESC
            ) best
            WHERE fc.cluster_id = best.cluster_id
              AND fc.face_id = best.face_id
            """
        )
    conn.commit()
```

### 4.3 Run to verify it passes

```bash
uv run pytest tests/test_face_clustering/test_db.py -q
```

Expected:

```
5 passed in 0.XXs
```

### 4.4 Commit

```bash
git add scripts/face_clustering/db.py \
        tests/test_face_clustering/test_db.py
git commit -m "feat(face): DB helpers — get_processed_sha256s, insert_faces, load_all_face_embeddings, write_clusters, mark_representatives"
```

---

## Task 5: Main Detection Script

Implement `scripts/face_clustering/detect_all.py` — scans `originals/media/`, skips already-processed sha256s, loads BGR images, runs detector, inserts results.

### 5.1 Write the failing test

Create `tests/test_face_clustering/test_detect_all.py`:

```python
# tests/test_face_clustering/test_detect_all.py
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".tiff", ".webp"}


@pytest.fixture
def fake_media_dir(tmp_path: Path) -> Path:
    """Create a small fake media directory with 3 image files."""
    (tmp_path / "aaabbb.jpg").write_bytes(b"fake")
    (tmp_path / "cccddd.jpg").write_bytes(b"fake")
    (tmp_path / "eeefff.jpg").write_bytes(b"fake")
    (tmp_path / "notanimage.json").write_bytes(b"{}")
    return tmp_path


def test_skip_already_processed(fake_media_dir, mock_conn):
    """Images whose sha256 (stem) is in get_processed_sha256s are skipped."""
    from face_clustering import detect_all

    conn, cursor = mock_conn

    with patch("face_clustering.detect_all.get_processed_sha256s") as mock_get, \
         patch("face_clustering.detect_all.FaceDetector") as MockDetector, \
         patch("face_clustering.detect_all.cv2") as mock_cv2, \
         patch("face_clustering.detect_all.insert_faces") as mock_insert, \
         patch("face_clustering.detect_all.psycopg") as mock_psycopg, \
         patch("face_clustering.detect_all.register_vector"), \
         patch("face_clustering.detect_all.MEDIA_DIR", fake_media_dir):

        mock_psycopg.connect.return_value.__enter__ = lambda s: conn
        mock_psycopg.connect.return_value.__exit__ = MagicMock(return_value=False)

        # mark all stems as already processed
        stems = {"aaabbb", "cccddd", "eeefff"}
        mock_get.return_value = stems

        mock_detector = MagicMock()
        MockDetector.return_value = mock_detector

        detect_all.main()

        # detect should never be called
        mock_detector.detect.assert_not_called()
        mock_insert.assert_not_called()


def test_no_face_image_handled_gracefully(fake_media_dir, mock_conn):
    """Images where detector returns [] produce no insert call."""
    from face_clustering import detect_all

    conn, cursor = mock_conn

    with patch("face_clustering.detect_all.get_processed_sha256s") as mock_get, \
         patch("face_clustering.detect_all.FaceDetector") as MockDetector, \
         patch("face_clustering.detect_all.cv2") as mock_cv2, \
         patch("face_clustering.detect_all.insert_faces") as mock_insert, \
         patch("face_clustering.detect_all.psycopg") as mock_psycopg, \
         patch("face_clustering.detect_all.register_vector"), \
         patch("face_clustering.detect_all.MEDIA_DIR", fake_media_dir):

        mock_psycopg.connect.return_value.__enter__ = lambda s: conn
        mock_psycopg.connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_get.return_value = set()  # nothing processed yet

        mock_detector = MagicMock()
        mock_detector.detect.return_value = []    # no faces in any image
        MockDetector.return_value = mock_detector

        img = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_cv2.imread.return_value = img
        mock_cv2.IMREAD_COLOR = 1

        detect_all.main()

        # insert_faces should not be called if no faces detected
        mock_insert.assert_not_called()


def test_insert_called_for_each_face(fake_media_dir, mock_conn, fake_face_dict):
    """insert_faces is called once per image that has detected faces."""
    from face_clustering import detect_all

    conn, cursor = mock_conn

    with patch("face_clustering.detect_all.get_processed_sha256s") as mock_get, \
         patch("face_clustering.detect_all.FaceDetector") as MockDetector, \
         patch("face_clustering.detect_all.cv2") as mock_cv2, \
         patch("face_clustering.detect_all.insert_faces") as mock_insert, \
         patch("face_clustering.detect_all.psycopg") as mock_psycopg, \
         patch("face_clustering.detect_all.register_vector"), \
         patch("face_clustering.detect_all.MEDIA_DIR", fake_media_dir):

        mock_psycopg.connect.return_value.__enter__ = lambda s: conn
        mock_psycopg.connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_get.return_value = {"cccddd", "eeefff"}  # only aaabbb unprocessed

        mock_detector = MagicMock()
        mock_detector.detect.return_value = [fake_face_dict]
        MockDetector.return_value = mock_detector

        img = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_cv2.imread.return_value = img
        mock_cv2.IMREAD_COLOR = 1

        detect_all.main()

        # exactly one insert call for the one unprocessed file (aaabbb)
        mock_insert.assert_called_once()
        call_args = mock_insert.call_args[0]
        assert call_args[0] is conn
        assert call_args[1] == "aaabbb"
        assert call_args[2] == [fake_face_dict]
```

Run — fails:

```bash
uv run pytest tests/test_face_clustering/test_detect_all.py -q
```

Expected:

```
ERROR ... ModuleNotFoundError: No module named 'face_clustering.detect_all'
```

### 5.2 Implement

Create `scripts/face_clustering/detect_all.py`:

```python
# scripts/face_clustering/detect_all.py
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import cv2
import psycopg
from pgvector.psycopg import register_vector
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from face_clustering.db import get_processed_sha256s, insert_faces
from face_clustering.detector import FaceDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MEDIA_DIR = Path(
    os.environ.get("MEDIA_DIR", "/home/will/photo_project/originals/media")
)
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://dedup:dedup_local_dev@127.0.0.1:5432/dedup"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".tiff", ".webp"}


def scan_images(media_dir: Path) -> list[Path]:
    return sorted(
        p for p in media_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> None:
    log.info("Scanning %s", MEDIA_DIR)
    all_images = scan_images(MEDIA_DIR)
    log.info("Found %d image files", len(all_images))

    detector = FaceDetector()

    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        processed = get_processed_sha256s(conn)
        log.info("%d sha256s already processed", len(processed))

        pending = [p for p in all_images if p.stem not in processed]
        log.info("%d images to process", len(pending))

        total_faces = 0
        for path in tqdm(pending, desc="detecting", unit="img"):
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                log.warning("Could not read %s, skipping", path.name)
                continue

            try:
                faces = detector.detect(img)
            except Exception as exc:
                log.warning("Detection failed for %s: %s", path.name, exc)
                continue

            if faces:
                insert_faces(conn, path.stem, faces)
                total_faces += len(faces)

    log.info("Done. Detected %d faces across %d images.", total_faces, len(pending))


if __name__ == "__main__":
    main()
```

### 5.3 Run to verify it passes

```bash
uv run pytest tests/test_face_clustering/test_detect_all.py -q
```

Expected:

```
3 passed in 0.XXs
```

### 5.4 Commit

```bash
git add scripts/face_clustering/detect_all.py \
        tests/test_face_clustering/test_detect_all.py
git commit -m "feat(face): detect_all.py — scan media dir, skip processed, detect, insert"
```

---

## Task 6: DB Helpers (Cluster Side) + Clustering Script

Implement `load_all_face_embeddings` and `write_clusters` (already scaffolded in `db.py` in Task 4), then implement `scripts/face_clustering/cluster_faces.py`.

### 6.1 Write the failing tests

Append to `tests/test_face_clustering/test_db.py` (or create separately as `test_db_cluster.py`):

Create `tests/test_face_clustering/test_db_cluster.py`:

```python
# tests/test_face_clustering/test_db_cluster.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


def test_load_all_face_embeddings_returns_ids_and_matrix(mock_conn):
    from face_clustering.db import load_all_face_embeddings

    conn, cursor = mock_conn
    emb1 = np.zeros(512, dtype=np.float32)
    emb2 = np.ones(512, dtype=np.float32)
    cursor.fetchall.return_value = [(1, emb1), (2, emb2)]

    ids, matrix = load_all_face_embeddings(conn)

    assert ids == [1, 2]
    assert matrix.shape == (2, 512)
    assert matrix.dtype == np.float32
    np.testing.assert_array_equal(matrix[0], emb1)
    np.testing.assert_array_equal(matrix[1], emb2)


def test_load_all_face_embeddings_empty(mock_conn):
    from face_clustering.db import load_all_face_embeddings

    conn, cursor = mock_conn
    cursor.fetchall.return_value = []

    ids, matrix = load_all_face_embeddings(conn)

    assert ids == []
    assert matrix.shape == (0, 512)


def test_write_clusters_skips_noise(mock_conn):
    from face_clustering.db import write_clusters

    conn, cursor = mock_conn

    # face_id 1 → cluster 0, face_id 2 → noise (-1)
    write_clusters(conn, [(1, 0), (2, -1)])

    cursor.executemany.assert_called_once()
    sql, rows = cursor.executemany.call_args[0]
    assert "INSERT INTO face_clusters" in sql
    assert len(rows) == 1         # noise point excluded
    assert rows[0] == (0, 1)      # (cluster_id, face_id)


def test_write_clusters_clears_existing(mock_conn):
    from face_clustering.db import write_clusters

    conn, cursor = mock_conn
    write_clusters(conn, [])

    # DELETE should always fire for idempotency
    delete_calls = [
        c for c in cursor.execute.call_args_list
        if "DELETE" in str(c)
    ]
    assert len(delete_calls) == 1
    conn.commit.assert_called_once()


def test_cluster_faces_two_cluster_synthetic():
    """Integration: HDBSCAN separates two tight clouds in 512-dim space."""
    from face_clustering.cluster_faces import run_clustering

    rng = np.random.default_rng(42)
    # cluster A: 10 points near zeros
    clusterA = rng.normal(loc=0.0, scale=0.01, size=(10, 512)).astype(np.float32)
    # cluster B: 10 points near ones
    clusterB = rng.normal(loc=1.0, scale=0.01, size=(10, 512)).astype(np.float32)
    embeddings = np.vstack([clusterA, clusterB])
    ids = list(range(20))

    assignments = run_clustering(ids, embeddings)

    # assignments is list of (face_id, cluster_id)
    labels_A = {cluster_id for face_id, cluster_id in assignments if face_id < 10}
    labels_B = {cluster_id for face_id, cluster_id in assignments if face_id >= 10}

    # both groups should form distinct clusters (not noise)
    assert -1 not in labels_A, f"cluster A points marked as noise: {assignments}"
    assert -1 not in labels_B, f"cluster B points marked as noise: {assignments}"
    # the two groups must have different cluster IDs
    assert labels_A.isdisjoint(labels_B), \
        f"A and B share cluster IDs: {labels_A & labels_B}"
```

Run — `cluster_faces.run_clustering` doesn't exist yet:

```bash
uv run pytest tests/test_face_clustering/test_db_cluster.py -q
```

Expected:

```
ERROR ... ModuleNotFoundError: No module named 'face_clustering.cluster_faces'
```

### 6.2 Implement

`db.py` already contains `load_all_face_embeddings`, `write_clusters`, and `mark_representatives` from Task 4.

Create `scripts/face_clustering/cluster_faces.py`:

```python
# scripts/face_clustering/cluster_faces.py
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

import hdbscan

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from face_clustering.db import (
    load_all_face_embeddings,
    mark_representatives,
    write_clusters,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://dedup:dedup_local_dev@127.0.0.1:5432/dedup"
)


def run_clustering(
    face_ids: list[int],
    embeddings: np.ndarray,
    min_cluster_size: int = 3,
    min_samples: int = 1,
) -> list[tuple[int, int]]:
    """Run HDBSCAN and return list of (face_id, cluster_id) assignments.

    Noise points get cluster_id == -1.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(embeddings)
    return list(zip(face_ids, labels.tolist()))


def main() -> None:
    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        log.info("Loading face embeddings...")
        face_ids, embeddings = load_all_face_embeddings(conn)

        if len(face_ids) == 0:
            log.info("No face embeddings found. Run face-detect first.")
            return

        log.info("Clustering %d faces with HDBSCAN...", len(face_ids))
        assignments = run_clustering(face_ids, embeddings)

        n_clusters = len({c for _, c in assignments if c >= 0})
        n_noise = sum(1 for _, c in assignments if c < 0)
        log.info(
            "Found %d clusters, %d noise points (%.1f%%)",
            n_clusters, n_noise, 100 * n_noise / len(assignments),
        )

        log.info("Writing cluster assignments...")
        write_clusters(conn, assignments)

        log.info("Marking cluster representatives...")
        mark_representatives(conn)

    log.info("Done.")


if __name__ == "__main__":
    main()
```

### 6.3 Run to verify it passes

```bash
uv run pytest tests/test_face_clustering/test_db_cluster.py -q
```

Expected:

```
5 passed in 0.XXs
```

### 6.4 Commit

```bash
git add scripts/face_clustering/cluster_faces.py \
        tests/test_face_clustering/test_db_cluster.py
git commit -m "feat(face): cluster_faces.py with HDBSCAN + DB cluster helpers and tests"
```

---

## Task 7: HTML Review Builder

Implement `scripts/face_clustering/build_review_html.py` — queries clusters, crops faces from originals, base64-encodes, renders HTML grid sorted by cluster size.

### 7.1 Write the failing test

Create `tests/test_face_clustering/test_build_review_html.py`:

```python
# tests/test_face_clustering/test_build_review_html.py
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


def test_build_review_html_importable():
    """Smoke test: module imports without error."""
    import face_clustering.build_review_html  # noqa: F401


def test_crop_and_encode_returns_base64_string(bgr_image):
    """crop_and_encode returns a non-empty base64 string."""
    from face_clustering.build_review_html import crop_and_encode

    # bbox covers the entire 64x64 image
    result = crop_and_encode(bgr_image, bbox_x=0, bbox_y=0, bbox_w=64, bbox_h=64)

    assert isinstance(result, str)
    assert len(result) > 0
    # base64 chars only
    import base64
    base64.b64decode(result)  # raises if invalid


def test_crop_and_encode_clamps_out_of_bounds(bgr_image):
    """Bbox larger than image is clamped gracefully."""
    from face_clustering.build_review_html import crop_and_encode

    result = crop_and_encode(bgr_image, bbox_x=0, bbox_y=0, bbox_w=200, bbox_h=200)
    assert isinstance(result, str)
    assert len(result) > 0


def test_render_html_contains_clusters():
    """render_html groups faces by cluster and includes cluster header."""
    from face_clustering.build_review_html import render_html

    clusters = {
        0: [
            {"b64": "AAAA", "sha256": "abc", "det_score": 0.9, "label": "Mom"},
            {"b64": "BBBB", "sha256": "def", "det_score": 0.8, "label": "Mom"},
        ],
        1: [
            {"b64": "CCCC", "sha256": "ghi", "det_score": 0.7, "label": None},
        ],
    }
    html = render_html(clusters)

    assert "Cluster 0" in html
    assert "Cluster 1" in html
    assert "Mom" in html
    # sorted by cluster size desc: cluster 0 (2 faces) before cluster 1 (1 face)
    assert html.index("Cluster 0") < html.index("Cluster 1")
    assert "data:image/jpeg;base64,AAAA" in html
```

Run — fails:

```bash
uv run pytest tests/test_face_clustering/test_build_review_html.py -q
```

Expected:

```
ERROR ... ModuleNotFoundError: No module named 'face_clustering.build_review_html'
```

### 7.2 Implement

Create `scripts/face_clustering/build_review_html.py`:

```python
# scripts/face_clustering/build_review_html.py
from __future__ import annotations

import base64
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import psycopg
from pgvector.psycopg import register_vector

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://dedup:dedup_local_dev@127.0.0.1:5432/dedup"
)
MEDIA_DIR = Path(
    os.environ.get("MEDIA_DIR", "/home/will/photo_project/originals/media")
)
OUTPUT_PATH = Path(
    os.environ.get("OUTPUT_PATH", "/home/will/photo_project/output/face_clusters_review.html")
)
THUMB_SIZE = 120
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".tiff", ".webp"]


def crop_and_encode(
    img_bgr: np.ndarray,
    bbox_x: int,
    bbox_y: int,
    bbox_w: int,
    bbox_h: int,
) -> str:
    """Crop a face region, resize to THUMB_SIZE, return base64-encoded JPEG."""
    h, w = img_bgr.shape[:2]
    x1 = max(0, bbox_x)
    y1 = max(0, bbox_y)
    x2 = min(w, bbox_x + bbox_w)
    y2 = min(h, bbox_y + bbox_h)
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.zeros((THUMB_SIZE, THUMB_SIZE, 3), dtype=np.uint8)
    thumb = cv2.resize(crop, (THUMB_SIZE, THUMB_SIZE), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def render_html(clusters: dict[int, list[dict]]) -> str:
    """Render an HTML page from a clusters dict.

    clusters: {cluster_id: [{"b64": str, "sha256": str, "det_score": float, "label": str|None}]}
    Sorted by cluster size descending.
    """
    sorted_ids = sorted(clusters.keys(), key=lambda cid: -len(clusters[cid]))

    parts = [
        "<!DOCTYPE html><html><head>",
        "<meta charset='utf-8'>",
        "<title>Face Clusters Review</title>",
        "<style>",
        "body { font-family: sans-serif; background: #111; color: #eee; margin: 16px; }",
        ".cluster { margin-bottom: 32px; }",
        ".cluster h2 { margin: 0 0 8px; font-size: 16px; }",
        ".faces { display: flex; flex-wrap: wrap; gap: 6px; }",
        "img { width: 120px; height: 120px; object-fit: cover; border: 1px solid #444; }",
        ".face-label { font-size: 11px; color: #aaa; text-align: center; width: 120px; overflow: hidden; white-space: nowrap; }",
        "</style></head><body>",
        f"<h1>Face Clusters ({sum(len(v) for v in clusters.values())} faces, {len(clusters)} clusters)</h1>",
    ]

    for cid in sorted_ids:
        faces = clusters[cid]
        label_str = faces[0]["label"] if faces[0].get("label") else ""
        header = f"Cluster {cid}" + (f" — {label_str}" if label_str else "")
        parts.append(f'<div class="cluster"><h2>{header} ({len(faces)} faces)</h2><div class="faces">')
        for face in faces:
            b64 = face["b64"]
            sha = face["sha256"][:12]
            score = face["det_score"]
            parts.append(
                f'<div>'
                f'<img src="data:image/jpeg;base64,{b64}" title="{sha} score={score:.2f}">'
                f'<div class="face-label">{sha}</div>'
                f'</div>'
            )
        parts.append("</div></div>")

    parts.append("</body></html>")
    return "\n".join(parts)


def _load_image(sha256: str) -> np.ndarray | None:
    for ext in IMAGE_EXTENSIONS:
        p = MEDIA_DIR / f"{sha256}{ext}"
        if p.exists():
            return cv2.imread(str(p), cv2.IMREAD_COLOR)
    return None


def main() -> None:
    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    fc.cluster_id,
                    fd.sha256,
                    fd.bbox_x,
                    fd.bbox_y,
                    fd.bbox_w,
                    fd.bbox_h,
                    fd.det_score,
                    fc.label
                FROM face_clusters fc
                JOIN face_detections fd ON fd.id = fc.face_id
                ORDER BY fc.cluster_id, fd.det_score DESC
                """
            )
            rows = cur.fetchall()

    log.info("Loaded %d face-cluster rows", len(rows))

    clusters: dict[int, list[dict]] = {}
    for cluster_id, sha256, bx, by, bw, bh, score, label in rows:
        img = _load_image(sha256)
        if img is None:
            log.warning("Could not load image for sha256=%s", sha256)
            b64 = ""
        else:
            b64 = crop_and_encode(img, bx, by, bw, bh)
        clusters.setdefault(cluster_id, []).append(
            {"b64": b64, "sha256": sha256, "det_score": score, "label": label}
        )

    html = render_html(clusters)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    log.info("Wrote %s (%d bytes)", OUTPUT_PATH, len(html))


if __name__ == "__main__":
    main()
```

### 7.3 Run to verify it passes

```bash
uv run pytest tests/test_face_clustering/test_build_review_html.py -q
```

Expected:

```
4 passed in 0.XXs
```

### 7.4 Commit

```bash
git add scripts/face_clustering/build_review_html.py \
        tests/test_face_clustering/test_build_review_html.py
git commit -m "feat(face): build_review_html.py — face crop grid sorted by cluster size"
```

---

## Task 8: Docker + justfile + Smoke Test

Wire up Docker, compose config, justfile recipes, and a file-existence smoke test.

### 8.1 Write the failing test

Create `tests/test_face_clustering/test_smoke_files.py`:

```python
# tests/test_face_clustering/test_smoke_files.py
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent


def test_dockerfile_exists():
    p = PROJECT_ROOT / "docker" / "Dockerfile.face-detect"
    assert p.exists(), f"Missing: {p}"


def test_docker_compose_exists():
    p = PROJECT_ROOT / "docker" / "docker-compose.face-detect.yml"
    assert p.exists(), f"Missing: {p}"


def test_justfile_has_face_detect_recipe():
    justfile = (PROJECT_ROOT / "justfile").read_text()
    assert "face-detect:" in justfile, "justfile missing 'face-detect:' recipe"


def test_justfile_has_face_detect_bg_recipe():
    justfile = (PROJECT_ROOT / "justfile").read_text()
    assert "face-detect-bg:" in justfile


def test_justfile_has_face_cluster_recipe():
    justfile = (PROJECT_ROOT / "justfile").read_text()
    assert "face-cluster:" in justfile


def test_justfile_has_face_review_recipe():
    justfile = (PROJECT_ROOT / "justfile").read_text()
    assert "face-review:" in justfile


def test_requirements_txt_has_insightface():
    req = (PROJECT_ROOT / "scripts" / "face_clustering" / "requirements.txt").read_text()
    assert "insightface" in req


def test_requirements_txt_has_hdbscan():
    req = (PROJECT_ROOT / "scripts" / "face_clustering" / "requirements.txt").read_text()
    assert "hdbscan" in req
```

Run — fails:

```bash
uv run pytest tests/test_face_clustering/test_smoke_files.py -q
```

Expected:

```
FAILED ... AssertionError: Missing: .../docker/Dockerfile.face-detect
```

### 8.2 Implement

Create `docker/Dockerfile.face-detect`:

```dockerfile
FROM python:3.12-slim

# insightface native deps: libgomp (OpenMP), libglib2.0 (glib/gio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY scripts/face_clustering/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY scripts/ /app/scripts/

ENV PYTHONUNBUFFERED=1
ENV MEDIA_DIR=/media

CMD ["python", "scripts/face_clustering/detect_all.py"]
```

Create `docker/docker-compose.face-detect.yml`:

```yaml
services:
  face_detect:
    build:
      context: ..
      dockerfile: docker/Dockerfile.face-detect
    network_mode: host
    environment:
      - DATABASE_URL=postgresql://dedup:dedup_local_dev@127.0.0.1:5432/dedup
      - MEDIA_DIR=/media
      - PYTHONUNBUFFERED=1
    volumes:
      - /home/will/photo_project/originals/media:/media:ro
      - insightface_models:/root/.insightface

volumes:
  insightface_models:
```

Add the following recipes to the bottom of `justfile`:

```just
# Run face detection pipeline in Docker (downloads buffalo_sc model on first run)
face-detect:
    docker compose -f docker/docker-compose.face-detect.yml up --build

# Run face detection in background
face-detect-bg:
    docker compose -f docker/docker-compose.face-detect.yml up --build -d
    @echo "Logs: docker logs -f photo_project-face_detect-1"

# Run HDBSCAN clustering on detected face embeddings (local uv)
face-cluster:
    uv run scripts/face_clustering/cluster_faces.py

# Build and open face cluster review HTML
face-review:
    uv run scripts/face_clustering/build_review_html.py
    @echo "Open: output/face_clusters_review.html"
```

### 8.3 Run to verify it passes

```bash
uv run pytest tests/test_face_clustering/test_smoke_files.py -q
```

Expected:

```
8 passed in 0.XXs
```

### 8.4 Full test suite

```bash
uv run pytest tests/test_face_clustering/ -q
```

Expected (all non-FACE_SMOKE tests):

```
XX passed, 7 skipped in X.XXs
```

### 8.5 Manual Docker build verification

```bash
docker compose -f docker/docker-compose.face-detect.yml build
```

First run downloads buffalo_sc (~60 MB). Subsequent runs use the `insightface_models` volume. Verify with:

```bash
docker compose -f docker/docker-compose.face-detect.yml run --rm face_detect python -c "
import insightface
app = insightface.app.FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640,640))
print('OK: buffalo_sc loaded')
"
```

### 8.6 Commit

```bash
git add docker/Dockerfile.face-detect \
        docker/docker-compose.face-detect.yml \
        justfile \
        tests/test_face_clustering/test_smoke_files.py
git commit -m "feat(face): Docker image, compose config, and justfile recipes for face detection pipeline"
```

---

## Self-Review Checklist

- [x] `FaceDetector` is referenced consistently as `from face_clustering.detector import FaceDetector` in `detect_all.py` and all tests patch `face_clustering.detector.insightface`
- [x] `insert_faces(conn, sha256, faces)` — same signature in `db.py`, tests, and `detect_all.py`
- [x] `load_all_face_embeddings(conn)` — same signature in `db.py`, `cluster_faces.py`, and tests
- [x] `write_clusters(conn, assignments)` — `assignments` is `list[tuple[int, int]]` (face_id, cluster_id) consistently
- [x] `run_clustering(ids, embeddings)` — exported from `cluster_faces.py`, imported in `test_db_cluster.py`
- [x] `crop_and_encode` and `render_html` — both exported from `build_review_html.py`, tested directly
- [x] `mark_representatives` — called in `cluster_faces.main()`, implemented in `db.py`
- [x] All imports use `sys.path.insert(0, .../scripts)` pattern matching existing codebase
- [x] No placeholders — every file path, SQL column, and function call is concrete
- [x] Schema test is gated on `FACE_SMOKE=1` — safe to run in CI without a live DB
- [x] `write_clusters` issues `DELETE FROM face_clusters` for idempotency before inserting
- [x] Docker compose uses `insightface_models` volume (not `hf_cache`) to cache buffalo_sc model files
