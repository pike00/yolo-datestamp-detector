# Zero-Shot Scene/Theme Tagging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Assign multi-label scene/theme tags to every photo by computing cosine similarity between pre-existing SigLIP embeddings and text prompt embeddings, storing results in a new `photo_tags` Postgres table.
**Architecture:** SigLIP's text tower encodes 40 label strings into the same 1152-dim space as the already-stored image vectors; a batched matrix multiply scores all 42K photos against all labels in seconds; any label scoring above 0.2 cosine similarity is written to `photo_tags`. No image re-processing is needed.
**Tech Stack:** Python 3.12+, `transformers` (SigLIP text encoder), `numpy`, `psycopg[binary]`, `pgvector`, `pytest` (mock-based unit tests).

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `scripts/scene_tagging/__init__.py` | Create | Package marker |
| `scripts/scene_tagging/labels.py` | Create | LABELS list (40 strings) |
| `scripts/scene_tagging/tagger.py` | Create | `encode_labels`, `score_photo` |
| `scripts/scene_tagging/db.py` | Create | `get_untagged_stems`, `bulk_insert_tags` |
| `scripts/scene_tagging/tag_all.py` | Create | Orchestration entrypoint |
| `scripts/scene_tagging/migrate.sql` | Create | `photo_tags` DDL |
| `tests/test_scene_tagging/__init__.py` | Create | Package marker |
| `tests/test_scene_tagging/conftest.py` | Create | Shared fixtures (`mock_conn`, `mock_cursor`) |
| `tests/test_scene_tagging/test_tagger.py` | Create | Tests for encode_labels + score_photo |
| `tests/test_scene_tagging/test_db.py` | Create | Tests for get_untagged_stems + bulk_insert_tags |
| `tests/test_scene_tagging/test_tag_all.py` | Create | Unit tests for orchestration logic |
| `tests/test_scene_tagging/test_smoke.py` | Create | Integration smoke test (10-photo subset) |
| `justfile` | Modify | Add `tag-scenes` recipe |

---

## Task 1: DB Schema Migration (photo_tags table)

**Files:**
- Create: `scripts/scene_tagging/migrate.sql`
- Create: `tests/test_scene_tagging/test_migration.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_scene_tagging/test_migration.py`:

```python
from __future__ import annotations

import os
import subprocess

import pytest

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://dedup:dedup_local_dev@127.0.0.1:5432/dedup"
)


@pytest.mark.integration
def test_photo_tags_table_exists():
    """Verify photo_tags table has the expected columns after migration."""
    result = subprocess.run(
        [
            "psql", DATABASE_URL,
            "-c",
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = 'photo_tags' ORDER BY ordinal_position;",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    output = result.stdout
    assert "sha256" in output
    assert "label" in output
    assert "score" in output
    assert "model" in output
    assert "tagged_at" in output


@pytest.mark.integration
def test_photo_tags_primary_key_deduplication():
    """Verify ON CONFLICT on (sha256, label, model) is enforced."""
    result = subprocess.run(
        [
            "psql", DATABASE_URL,
            "-c",
            "INSERT INTO photo_tags (sha256, label, score, model) "
            "VALUES ('test-sha', 'beach', 0.5, 'siglip-so400m') "
            "ON CONFLICT (sha256, label, model) DO NOTHING; "
            "INSERT INTO photo_tags (sha256, label, score, model) "
            "VALUES ('test-sha', 'beach', 0.6, 'siglip-so400m') "
            "ON CONFLICT (sha256, label, model) DO NOTHING; "
            "SELECT count(*) FROM photo_tags WHERE sha256 = 'test-sha'; "
            "DELETE FROM photo_tags WHERE sha256 = 'test-sha';",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert " 1 " in result.stdout  # only one row inserted
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/will/photo_project
uv run pytest tests/test_scene_tagging/test_migration.py -m integration -v 2>&1 | head -30
```

Expected: `FAILED` or `ERROR` — table `photo_tags` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/scene_tagging/migrate.sql`:

```sql
CREATE TABLE IF NOT EXISTS photo_tags (
    sha256    text NOT NULL,
    label     text NOT NULL,
    score     float NOT NULL,
    model     text NOT NULL DEFAULT 'siglip-so400m',
    tagged_at timestamptz DEFAULT now(),
    PRIMARY KEY (sha256, label, model)
);

CREATE INDEX IF NOT EXISTS photo_tags_sha256_idx ON photo_tags (sha256);
CREATE INDEX IF NOT EXISTS photo_tags_label_idx  ON photo_tags (label);
```

Apply it:

```bash
psql postgresql://dedup:dedup_local_dev@127.0.0.1:5432/dedup \
    -f scripts/scene_tagging/migrate.sql
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_scene_tagging/test_migration.py -m integration -v
```

Expected:
```
PASSED tests/test_scene_tagging/test_migration.py::test_photo_tags_table_exists
PASSED tests/test_scene_tagging/test_migration.py::test_photo_tags_primary_key_deduplication
2 passed
```

- [ ] **Step 5: Commit**

```bash
git add scripts/scene_tagging/migrate.sql tests/test_scene_tagging/test_migration.py
git commit -m "feat: add photo_tags schema migration and integration test"
```

---

## Task 2: Package Scaffold + conftest

**Files:**
- Create: `scripts/scene_tagging/__init__.py`
- Create: `scripts/scene_tagging/labels.py`
- Create: `tests/test_scene_tagging/__init__.py`
- Create: `tests/test_scene_tagging/conftest.py`
- Create: `tests/test_scene_tagging/test_scaffold.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_scene_tagging/test_scaffold.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


def test_scene_tagging_package_importable():
    import scene_tagging  # noqa: F401


def test_labels_importable():
    from scene_tagging.labels import LABELS
    assert isinstance(LABELS, list)
    assert len(LABELS) == 40


def test_no_duplicate_labels():
    from scene_tagging.labels import LABELS
    assert len(LABELS) == len(set(LABELS)), "Duplicate label found"


def test_conftest_mock_conn_fixture(mock_conn):
    conn, cursor = mock_conn
    assert conn is not None
    assert cursor is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_scene_tagging/test_scaffold.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'scene_tagging'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/scene_tagging/__init__.py` (empty):
```python
```

Create `scripts/scene_tagging/labels.py`:
```python
from __future__ import annotations

LABELS: list[str] = [
    "birthday party", "christmas", "thanksgiving", "halloween",
    "beach", "swimming pool", "hiking", "camping", "skiing",
    "graduation", "wedding", "baby", "toddler", "school",
    "sports game", "concert", "restaurant", "travel abroad",
    "family portrait", "pets", "garden", "snow", "sunset",
    "indoor portrait", "outdoor portrait", "road trip",
    "amusement park", "holiday gathering", "first day of school",
    "new year", "easter", "valentine", "fourth of july",
    "backyard", "playground", "zoo", "museum", "church",
    "neighborhood", "home interior",
]
```

Create `tests/test_scene_tagging/__init__.py` (empty):
```python
```

Create `tests/test_scene_tagging/conftest.py`:
```python
from __future__ import annotations

from unittest.mock import MagicMock

import psycopg
import pytest


@pytest.fixture
def mock_conn():
    """Return (conn, cursor) mocks matching the psycopg context-manager protocol."""
    conn = MagicMock(spec=psycopg.Connection)
    cursor = MagicMock()
    cursor.__enter__ = lambda s: cursor
    cursor.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = cursor
    return conn, cursor
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_scene_tagging/test_scaffold.py -v
```

Expected:
```
PASSED tests/test_scene_tagging/test_scaffold.py::test_scene_tagging_package_importable
PASSED tests/test_scene_tagging/test_scaffold.py::test_labels_importable
PASSED tests/test_scene_tagging/test_scaffold.py::test_no_duplicate_labels
PASSED tests/test_scene_tagging/test_scaffold.py::test_conftest_mock_conn_fixture
4 passed
```

- [ ] **Step 5: Commit**

```bash
git add scripts/scene_tagging/__init__.py scripts/scene_tagging/labels.py \
        tests/test_scene_tagging/__init__.py tests/test_scene_tagging/conftest.py \
        tests/test_scene_tagging/test_scaffold.py
git commit -m "feat: add scene_tagging package scaffold and labels taxonomy"
```

---

## Task 3: Label Encoder (encode_labels + score_photo)

**Files:**
- Create: `scripts/scene_tagging/tagger.py`
- Create: `tests/test_scene_tagging/test_tagger.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_scene_tagging/test_tagger.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from scene_tagging.labels import LABELS
from scene_tagging.tagger import THRESHOLD, encode_labels, score_photo


def _make_mock_model_processor(dim: int = 1152):
    """Return a (model, processor) mock pair that emits unit vectors."""
    processor = MagicMock()
    processor.return_value = {"input_ids": torch.zeros(1, 8, dtype=torch.long)}

    model = MagicMock()

    def fake_text_features(**kwargs):
        n = kwargs["input_ids"].shape[0]
        raw = torch.randn(n, dim)
        return raw / raw.norm(dim=-1, keepdim=True)

    model.get_text_features.side_effect = fake_text_features
    return model, processor


def test_encode_labels_returns_dict_of_arrays():
    model, processor = _make_mock_model_processor()
    label_embeddings = encode_labels(model, processor, LABELS)

    assert isinstance(label_embeddings, dict)
    assert set(label_embeddings.keys()) == set(LABELS)
    for label, vec in label_embeddings.items():
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (1152,), f"Wrong shape for '{label}': {vec.shape}"


def test_encode_labels_vectors_are_unit_normalized():
    model, processor = _make_mock_model_processor()
    label_embeddings = encode_labels(model, processor, LABELS)

    for label, vec in label_embeddings.items():
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5, f"Not unit-normalized: '{label}' norm={norm:.4f}"


def test_score_photo_returns_dict_with_all_labels():
    model, processor = _make_mock_model_processor()
    label_embeddings = encode_labels(model, processor, LABELS)
    photo_vec = np.random.randn(1152).astype(np.float32)
    photo_vec /= np.linalg.norm(photo_vec)

    scores = score_photo(photo_vec, label_embeddings)

    assert set(scores.keys()) == set(LABELS)
    for label, score in scores.items():
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0, f"Score out of range for '{label}': {score}"


def test_score_photo_perfect_match():
    """A photo embedding identical to a label embedding should score ~1.0."""
    dim = 1152
    label_vec = np.random.randn(dim).astype(np.float32)
    label_vec /= np.linalg.norm(label_vec)

    label_embeddings = {"beach": label_vec, "christmas": -label_vec}
    scores = score_photo(label_vec, label_embeddings)

    assert abs(scores["beach"] - 1.0) < 1e-5
    assert abs(scores["christmas"] + 1.0) < 1e-5


def test_threshold_constant_is_in_expected_range():
    assert 0.1 <= THRESHOLD <= 0.5, f"THRESHOLD {THRESHOLD} outside expected range"


def test_score_photo_perfect_match_above_threshold():
    """A photo identical to a label vector should always exceed THRESHOLD."""
    dim = 1152
    beach_vec = np.random.randn(dim).astype(np.float32)
    beach_vec /= np.linalg.norm(beach_vec)
    noise_vec = np.random.randn(dim).astype(np.float32)
    noise_vec /= np.linalg.norm(noise_vec)

    label_embeddings = {"beach": beach_vec, "christmas": noise_vec}
    scores = score_photo(beach_vec, label_embeddings)
    hits = {k: v for k, v in scores.items() if v >= THRESHOLD}

    assert "beach" in hits, f"Perfect match for 'beach' not above threshold {THRESHOLD}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_scene_tagging/test_tagger.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'encode_labels' from 'scene_tagging.tagger'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/scene_tagging/tagger.py`:

```python
from __future__ import annotations

import numpy as np
import torch

THRESHOLD: float = 0.2


def encode_labels(
    model,
    processor,
    labels: list[str],
) -> dict[str, np.ndarray]:
    """Encode label strings into unit-normalized 1152-dim vectors.

    Uses SigLIP's text tower. All 40 labels fit in a single forward pass on CPU.
    Returns a dict mapping label -> np.ndarray of shape (1152,).
    """
    inputs = processor(
        text=labels,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )
    with torch.no_grad():
        features = model.get_text_features(**inputs)

    features = features / features.norm(dim=-1, keepdim=True)
    vecs = features.cpu().numpy()  # shape (n_labels, 1152)
    return {label: vecs[i] for i, label in enumerate(labels)}


def score_photo(
    photo_embedding: np.ndarray,
    label_embeddings: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute cosine similarity between one photo vector and all label vectors.

    photo_embedding must be unit-normalized (as stored in photo_embeddings).
    Returns a dict mapping label -> similarity score in [-1.0, 1.0].
    """
    labels = list(label_embeddings.keys())
    matrix = np.stack([label_embeddings[lbl] for lbl in labels])  # (n_labels, dim)
    scores = matrix @ photo_embedding  # dot product == cosine sim for unit vectors
    return {label: float(scores[i]) for i, label in enumerate(labels)}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_scene_tagging/test_tagger.py -v
```

Expected:
```
PASSED tests/test_scene_tagging/test_tagger.py::test_encode_labels_returns_dict_of_arrays
PASSED tests/test_scene_tagging/test_tagger.py::test_encode_labels_vectors_are_unit_normalized
PASSED tests/test_scene_tagging/test_tagger.py::test_score_photo_returns_dict_with_all_labels
PASSED tests/test_scene_tagging/test_tagger.py::test_score_photo_perfect_match
PASSED tests/test_scene_tagging/test_tagger.py::test_threshold_constant_is_in_expected_range
PASSED tests/test_scene_tagging/test_tagger.py::test_score_photo_perfect_match_above_threshold
6 passed
```

- [ ] **Step 5: Commit**

```bash
git add scripts/scene_tagging/tagger.py tests/test_scene_tagging/test_tagger.py
git commit -m "feat: add encode_labels and score_photo to scene_tagging.tagger"
```

---

## Task 4: DB Helpers (get_untagged_stems + bulk_insert_tags)

**Files:**
- Create: `scripts/scene_tagging/db.py`
- Create: `tests/test_scene_tagging/test_db.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_scene_tagging/test_db.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from scene_tagging.db import MODEL_NAME, bulk_insert_tags, get_untagged_stems


def test_get_untagged_stems_queries_both_tables(mock_conn):
    conn, cursor = mock_conn
    vec = np.zeros(1152, dtype=np.float32)
    cursor.fetchall.return_value = [("abc123", vec), ("def456", vec)]

    results = get_untagged_stems(conn, MODEL_NAME)

    assert cursor.execute.called
    sql = cursor.execute.call_args[0][0]
    assert "photo_embeddings" in sql
    assert "photo_tags" in sql
    assert len(results) == 2
    sha256s = [r[0] for r in results]
    assert "abc123" in sha256s
    assert "def456" in sha256s


def test_get_untagged_stems_passes_model_param(mock_conn):
    conn, cursor = mock_conn
    cursor.fetchall.return_value = []

    get_untagged_stems(conn, MODEL_NAME)

    _, params = cursor.execute.call_args[0]
    assert MODEL_NAME in params


def test_get_untagged_stems_returns_numpy_arrays(mock_conn):
    conn, cursor = mock_conn
    vec = np.random.randn(1152).astype(np.float32)
    cursor.fetchall.return_value = [("abc123", vec)]

    results = get_untagged_stems(conn, MODEL_NAME)

    sha256, embedding = results[0]
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (1152,)


def test_bulk_insert_tags_uses_executemany(mock_conn):
    conn, cursor = mock_conn
    rows = [
        ("abc123", "beach", 0.45, MODEL_NAME),
        ("abc123", "sunset", 0.31, MODEL_NAME),
        ("def456", "christmas", 0.55, MODEL_NAME),
    ]

    bulk_insert_tags(conn, rows)

    cursor.executemany.assert_called_once()
    sql, data = cursor.executemany.call_args[0]
    assert "INSERT INTO photo_tags" in sql
    assert "ON CONFLICT" in sql
    assert data == rows
    conn.commit.assert_called_once()


def test_bulk_insert_tags_empty_rows_is_noop(mock_conn):
    conn, cursor = mock_conn

    bulk_insert_tags(conn, [])

    cursor.executemany.assert_not_called()
    conn.commit.assert_not_called()


def test_model_name_constant():
    assert MODEL_NAME == "siglip-so400m"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_scene_tagging/test_db.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'get_untagged_stems' from 'scene_tagging.db'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/scene_tagging/db.py`:

```python
from __future__ import annotations

import numpy as np
import psycopg

MODEL_NAME = "siglip-so400m"


def get_untagged_stems(
    conn: psycopg.Connection,
    model: str = MODEL_NAME,
) -> list[tuple[str, np.ndarray]]:
    """Fetch (sha256, embedding) for photos with no existing tags for this model.

    The NOT EXISTS subquery makes reruns idempotent — already-tagged photos
    are skipped automatically.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pe.sha256, pe.embedding "
            "FROM photo_embeddings pe "
            "WHERE pe.model = %s "
            "  AND pe.media_type = 'photo' "
            "  AND NOT EXISTS ("
            "      SELECT 1 FROM photo_tags pt "
            "      WHERE pt.sha256 = pe.sha256 AND pt.model = %s"
            "  )",
            (model, model),
        )
        rows = cur.fetchall()

    return [
        (sha256, np.array(embedding, dtype=np.float32))
        for sha256, embedding in rows
    ]


def bulk_insert_tags(
    conn: psycopg.Connection,
    rows: list[tuple[str, str, float, str]],
) -> None:
    """Insert rows of (sha256, label, score, model) into photo_tags.

    Silently skips duplicates via ON CONFLICT DO NOTHING.
    No-ops on empty input without touching the DB.
    """
    if not rows:
        return
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO photo_tags (sha256, label, score, model) "
            "VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (sha256, label, model) DO NOTHING",
            rows,
        )
    conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_scene_tagging/test_db.py -v
```

Expected:
```
PASSED tests/test_scene_tagging/test_db.py::test_get_untagged_stems_queries_both_tables
PASSED tests/test_scene_tagging/test_db.py::test_get_untagged_stems_passes_model_param
PASSED tests/test_scene_tagging/test_db.py::test_get_untagged_stems_returns_numpy_arrays
PASSED tests/test_scene_tagging/test_db.py::test_bulk_insert_tags_uses_executemany
PASSED tests/test_scene_tagging/test_db.py::test_bulk_insert_tags_empty_rows_is_noop
PASSED tests/test_scene_tagging/test_db.py::test_model_name_constant
6 passed
```

- [ ] **Step 5: Commit**

```bash
git add scripts/scene_tagging/db.py tests/test_scene_tagging/test_db.py
git commit -m "feat: add get_untagged_stems and bulk_insert_tags to scene_tagging.db"
```

---

## Task 5: Main Orchestration Script (tag_all.py)

**Files:**
- Create: `scripts/scene_tagging/tag_all.py`
- Create: `tests/test_scene_tagging/test_tag_all.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_scene_tagging/test_tag_all.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


def _make_label_embeddings(labels, dim=1152):
    rng = np.random.default_rng(42)
    result = {}
    for label in labels:
        v = rng.standard_normal(dim).astype(np.float32)
        result[label] = v / np.linalg.norm(v)
    return result


def test_tag_all_runs_without_error():
    """tag_all.main() completes without raising given mocked deps."""
    from scene_tagging.labels import LABELS

    rng = np.random.default_rng(0)
    fake_stems = [
        (f"sha{i:04d}", rng.standard_normal(1152).astype(np.float32))
        for i in range(5)
    ]
    fake_label_embs = _make_label_embeddings(LABELS)

    with (
        patch("scene_tagging.tag_all.psycopg") as mock_psycopg,
        patch("scene_tagging.tag_all.register_vector"),
        patch("scene_tagging.tag_all.AutoProcessor"),
        patch("scene_tagging.tag_all.AutoModel"),
        patch("scene_tagging.tag_all.get_untagged_stems", return_value=fake_stems),
        patch("scene_tagging.tag_all.encode_labels", return_value=fake_label_embs),
        patch("scene_tagging.tag_all.bulk_insert_tags") as mock_insert,
    ):
        mock_conn = MagicMock()
        mock_psycopg.connect.return_value.__enter__.return_value = mock_conn

        from scene_tagging import tag_all
        tag_all.main()

    assert mock_insert.called
    for call_args in mock_insert.call_args_list:
        rows = call_args[0][1]
        for row in rows:
            sha256, label, score, model = row
            assert model == "siglip-so400m"
            assert label in LABELS
            assert isinstance(score, float)


def test_tag_all_filters_below_threshold():
    """Only labels scoring >= THRESHOLD should be inserted."""
    from scene_tagging.labels import LABELS
    from scene_tagging.tagger import THRESHOLD

    dim = 1152
    rng = np.random.default_rng(1)
    photo_vec = rng.standard_normal(dim).astype(np.float32)
    photo_vec /= np.linalg.norm(photo_vec)

    label_embs = {}
    for label in LABELS:
        v = rng.standard_normal(dim).astype(np.float32)
        label_embs[label] = v / np.linalg.norm(v)
    # Force beach to be a perfect match
    label_embs["beach"] = photo_vec.copy()

    with (
        patch("scene_tagging.tag_all.psycopg") as mock_psycopg,
        patch("scene_tagging.tag_all.register_vector"),
        patch("scene_tagging.tag_all.AutoProcessor"),
        patch("scene_tagging.tag_all.AutoModel"),
        patch("scene_tagging.tag_all.get_untagged_stems", return_value=[("sha0001", photo_vec)]),
        patch("scene_tagging.tag_all.encode_labels", return_value=label_embs),
        patch("scene_tagging.tag_all.bulk_insert_tags") as mock_insert,
    ):
        mock_conn = MagicMock()
        mock_psycopg.connect.return_value.__enter__.return_value = mock_conn

        from scene_tagging import tag_all
        tag_all.main()

    all_rows = []
    for call_args in mock_insert.call_args_list:
        all_rows.extend(call_args[0][1])

    scores_map = {row[1]: row[2] for row in all_rows}
    assert "beach" in scores_map, "Perfect-match label 'beach' was not inserted"
    assert scores_map["beach"] > 0.99
    for label, score in scores_map.items():
        assert score >= THRESHOLD, f"Label '{label}' score {score:.4f} below threshold {THRESHOLD}"


def test_tag_all_no_photos_is_noop():
    """If get_untagged_stems returns empty list, no inserts occur."""
    with (
        patch("scene_tagging.tag_all.psycopg") as mock_psycopg,
        patch("scene_tagging.tag_all.register_vector"),
        patch("scene_tagging.tag_all.AutoProcessor"),
        patch("scene_tagging.tag_all.AutoModel"),
        patch("scene_tagging.tag_all.get_untagged_stems", return_value=[]),
        patch("scene_tagging.tag_all.encode_labels"),
        patch("scene_tagging.tag_all.bulk_insert_tags") as mock_insert,
    ):
        mock_conn = MagicMock()
        mock_psycopg.connect.return_value.__enter__.return_value = mock_conn

        from scene_tagging import tag_all
        tag_all.main()

    mock_insert.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_scene_tagging/test_tag_all.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'main' from 'scene_tagging.tag_all'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/scene_tagging/tag_all.py`:

```python
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scene_tagging.db import MODEL_NAME, bulk_insert_tags, get_untagged_stems
from scene_tagging.labels import LABELS
from scene_tagging.tagger import THRESHOLD, encode_labels, score_photo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://dedup:dedup_local_dev@127.0.0.1:5432/dedup"
)
MODEL_HF_ID = "google/siglip-so400m-patch14-384"
BATCH_SIZE = 1000


def main() -> None:
    log.info("Loading SigLIP text tower: %s", MODEL_HF_ID)
    processor = AutoProcessor.from_pretrained(MODEL_HF_ID)
    model = AutoModel.from_pretrained(MODEL_HF_ID)
    model.eval()

    log.info("Encoding %d labels ...", len(LABELS))
    label_embeddings = encode_labels(model, processor, LABELS)
    log.info("Labels encoded.")

    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)

        log.info("Fetching untagged photo embeddings ...")
        stems = get_untagged_stems(conn, MODEL_NAME)
        log.info("%d photos to tag", len(stems))

        if not stems:
            log.info("Nothing to do.")
            return

        total_tags = 0
        for batch_start in range(0, len(stems), BATCH_SIZE):
            batch = stems[batch_start : batch_start + BATCH_SIZE]
            rows: list[tuple[str, str, float, str]] = []

            for sha256, embedding in batch:
                scores = score_photo(embedding, label_embeddings)
                for label, score in scores.items():
                    if score >= THRESHOLD:
                        rows.append((sha256, label, score, MODEL_NAME))

            bulk_insert_tags(conn, rows)
            total_tags += len(rows)
            log.info(
                "Batch %d-%d: inserted %d tags (running total: %d)",
                batch_start,
                batch_start + len(batch) - 1,
                len(rows),
                total_tags,
            )

    log.info("Done. Tagged %d photos, inserted %d label rows.", len(stems), total_tags)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_scene_tagging/test_tag_all.py -v
```

Expected:
```
PASSED tests/test_scene_tagging/test_tag_all.py::test_tag_all_runs_without_error
PASSED tests/test_scene_tagging/test_tag_all.py::test_tag_all_filters_below_threshold
PASSED tests/test_scene_tagging/test_tag_all.py::test_tag_all_no_photos_is_noop
3 passed
```

- [ ] **Step 5: Commit**

```bash
git add scripts/scene_tagging/tag_all.py tests/test_scene_tagging/test_tag_all.py
git commit -m "feat: add tag_all.py orchestration script for scene tagging"
```

---

## Task 6: justfile Recipe + End-to-End Smoke Test

**Files:**
- Modify: `justfile`
- Create: `tests/test_scene_tagging/test_smoke.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_scene_tagging/test_smoke.py`:

```python
from __future__ import annotations

"""Integration smoke test: tag 10 real photos and verify rows land in photo_tags.

Requires:
  - DATABASE_URL pointing at the live dedup Postgres instance
  - photo_embeddings table populated with >= 10 rows
  - photo_tags table created (Task 1 migration applied)

Skipped automatically when DATABASE_URL is not reachable or embeddings are absent.

Run with: uv run pytest tests/test_scene_tagging/test_smoke.py -m integration -v
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://dedup:dedup_local_dev@127.0.0.1:5432/dedup"
)


def _psql(sql: str) -> str:
    result = subprocess.run(
        ["psql", DATABASE_URL, "-c", sql],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"DB not reachable: {result.stderr[:200]}")
    return result.stdout


def _parse_count(psql_output: str) -> int:
    for line in psql_output.strip().split("\n"):
        line = line.strip()
        if line and "---" not in line and not line.startswith("count") and line.isdigit():
            return int(line)
    raise ValueError(f"Could not parse count from: {psql_output!r}")


@pytest.mark.integration
def test_smoke_tag_ten_photos():
    """Run tagging logic on 10 real photos and verify rows appear in photo_tags."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

    # Check embeddings table has data
    out = _psql("SELECT count(*) FROM photo_embeddings WHERE media_type = 'photo';")
    total = _parse_count(out)
    if total < 10:
        pytest.skip(f"Not enough embeddings ({total}) for smoke test")

    # Pull 10 real sha256s
    out = _psql(
        "SELECT sha256 FROM photo_embeddings "
        "WHERE media_type = 'photo' LIMIT 10;"
    )
    sample_sha256s = [
        line.strip() for line in out.strip().split("\n")
        if line.strip() and "---" not in line and line.strip() != "sha256"
    ][:10]
    assert len(sample_sha256s) == 10, f"Could not parse 10 sha256s from output"

    # Clean any pre-existing tags for these 10
    sha_list = ", ".join(f"'{s}'" for s in sample_sha256s)
    _psql(
        f"DELETE FROM photo_tags "
        f"WHERE sha256 IN ({sha_list}) AND model = 'siglip-so400m';"
    )

    # Import and run tagging directly (in-process, faster than subprocess)
    import psycopg
    from pgvector.psycopg import register_vector
    from scene_tagging.db import MODEL_NAME, bulk_insert_tags
    from scene_tagging.labels import LABELS
    from scene_tagging.tagger import THRESHOLD, encode_labels, score_photo
    from transformers import AutoModel, AutoProcessor

    MODEL_HF_ID = "google/siglip-so400m-patch14-384"
    processor = AutoProcessor.from_pretrained(MODEL_HF_ID)
    model = AutoModel.from_pretrained(MODEL_HF_ID)
    model.eval()

    label_embeddings = encode_labels(model, processor, LABELS)

    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT sha256, embedding FROM photo_embeddings "
                "WHERE sha256 = ANY(%s) AND media_type = 'photo' AND model = %s",
                (sample_sha256s, MODEL_NAME),
            )
            rows_raw = cur.fetchall()

        stems = [
            (sha256, np.array(embedding, dtype=np.float32))
            for sha256, embedding in rows_raw
        ]

        insert_rows = []
        for sha256, embedding in stems:
            scores = score_photo(embedding, label_embeddings)
            for label, score in scores.items():
                if score >= THRESHOLD:
                    insert_rows.append((sha256, label, score, MODEL_NAME))

        bulk_insert_tags(conn, insert_rows)

    # Verify rows landed in DB
    out2 = _psql(
        f"SELECT count(*) FROM photo_tags "
        f"WHERE sha256 IN ({sha_list}) AND model = 'siglip-so400m';"
    )
    inserted_count = _parse_count(out2)
    assert inserted_count > 0, "Expected at least some tags for 10 photos"

    out3 = _psql(
        f"SELECT count(DISTINCT sha256) FROM photo_tags "
        f"WHERE sha256 IN ({sha_list}) AND model = 'siglip-so400m';"
    )
    distinct_tagged = _parse_count(out3)
    # At least 8 of 10 real photos should match at least one label
    assert distinct_tagged >= 8, (
        f"Only {distinct_tagged}/10 photos got tags — "
        f"check threshold ({THRESHOLD}) or embedding quality"
    )

    print(f"\nSmoke: {inserted_count} tags across {distinct_tagged}/10 photos")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_scene_tagging/test_smoke.py -m integration -v -s 2>&1 | head -30
```

Expected: `SKIPPED` if DB unreachable, or `FAILED` because `photo_tags` table does not exist yet. Confirm a clear failure or skip before proceeding.

- [ ] **Step 3: Write minimal implementation**

Add to `justfile` (after the last existing recipe):

```just
# Zero-shot scene/theme tagging — scores all untagged photos against 40 labels
tag-scenes:
    uv run scripts/scene_tagging/tag_all.py
```

Verify it parses:

```bash
just --list | grep tag-scenes
```

- [ ] **Step 4: Run tests to verify they pass**

Run all unit tests (no integration flag) to confirm no regressions:

```bash
uv run pytest tests/test_scene_tagging/ \
    --ignore=tests/test_scene_tagging/test_smoke.py \
    --ignore=tests/test_scene_tagging/test_migration.py \
    -v
```

Then run the smoke test against the live DB:

```bash
uv run pytest tests/test_scene_tagging/test_smoke.py -m integration -v -s
```

Expected smoke output:
```
PASSED tests/test_scene_tagging/test_smoke.py::test_smoke_tag_ten_photos

Smoke: 47 tags across 10/10 photos
```

Verify the justfile recipe runs end-to-end (idempotent — skips already-tagged photos):

```bash
just tag-scenes 2>&1 | tail -5
```

Expected tail:
```
HH:MM:SS INFO Done. Tagged NNNNN photos, inserted NNNNNN label rows.
```

Spot-check the DB to confirm labels make sense:

```bash
psql postgresql://dedup:dedup_local_dev@127.0.0.1:5432/dedup \
  -c "SELECT label, count(*), round(avg(score)::numeric, 3) AS avg_score
      FROM photo_tags
      WHERE model = 'siglip-so400m'
      GROUP BY label ORDER BY count DESC LIMIT 10;"
```

- [ ] **Step 5: Commit**

```bash
git add justfile tests/test_scene_tagging/test_smoke.py
git commit -m "feat: add tag-scenes justfile recipe and integration smoke test"
```

---

## Post-Implementation Checklist

- [ ] All 6 tasks committed; `git log --oneline -6` shows one commit per task.
- [ ] `uv run pytest tests/test_scene_tagging/ --ignore=tests/test_scene_tagging/test_smoke.py --ignore=tests/test_scene_tagging/test_migration.py -v` — all unit tests pass.
- [ ] `just tag-scenes` runs to completion without error.
- [ ] `SELECT count(*) FROM photo_tags;` returns > 0.
- [ ] `SELECT count(DISTINCT label) FROM photo_tags;` >= 10.
- [ ] `just --list` shows `tag-scenes` recipe.
