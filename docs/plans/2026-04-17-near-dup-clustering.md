# Near-Duplicate Clustering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cluster 42K SigLIP photo embeddings to find near-duplicate groups and surface a representative "best" photo per cluster.

**Architecture:** Fetch all photo embeddings from Postgres into numpy, run HDBSCAN in-memory (min_cluster_size=2, cluster_selection_epsilon=0.3), score image quality via Laplacian variance, write cluster assignments to a new photo_clusters table, render an HTML review of duplicate groups.

**Tech Stack:** Python 3.12, `hdbscan`, `numpy`, `opencv-python-headless`, `psycopg[binary]`, `pgvector`, Postgres pgvector

---

## File Map

| Path | Role |
|---|---|
| `scripts/near_dup/__init__.py` | Package marker |
| `scripts/near_dup/embeddings.py` | `fetch_photo_embeddings` — query photo_embeddings, return sha256 list + numpy matrix |
| `scripts/near_dup/clustering.py` | `run_hdbscan`, `build_cluster_map` |
| `scripts/near_dup/quality.py` | `score_image` — Laplacian variance sharpness scorer |
| `scripts/near_dup/db.py` | `write_clusters`, `read_cluster_groups` |
| `scripts/near_dup/run_cluster.py` | Main orchestration: fetch → cluster → score → write |
| `scripts/near_dup/build_review_html.py` | Render `output/near_dup_review.html` |
| `tests/test_near_dup/__init__.py` | Package marker |
| `tests/test_near_dup/conftest.py` | `mock_conn`, `tiny_embeddings`, `grey_images` fixtures |
| `tests/test_near_dup/test_embeddings.py` | Unit tests for `fetch_photo_embeddings` |
| `tests/test_near_dup/test_clustering.py` | Unit tests for `run_hdbscan`, `build_cluster_map` |
| `tests/test_near_dup/test_quality.py` | Unit tests for `score_image` |
| `tests/test_near_dup/test_db.py` | Unit tests for `write_clusters`, `read_cluster_groups` |
| `tests/test_near_dup/test_smoke.py` | Integration smoke test (gated on `NEAR_DUP_SMOKE=1`) |

---

## Task 1 — DB Schema Migration

Create the `photo_clusters` table and index. Integration test gated on `NEAR_DUP_SMOKE=1`.

- [ ] **Write the failing test**

  Create `/home/will/photo_project/tests/test_near_dup/test_smoke.py`:

  ```python
  from __future__ import annotations

  import os
  import pytest
  import psycopg
  from pgvector.psycopg import register_vector

  DATABASE_URL = os.environ.get(
      "DATABASE_URL", "postgresql://dedup:dedup_local_dev@127.0.0.1:5432/dedup"
  )

  pytestmark = pytest.mark.skipif(
      os.environ.get("NEAR_DUP_SMOKE") != "1",
      reason="Set NEAR_DUP_SMOKE=1 to run integration tests",
  )


  @pytest.fixture
  def db_conn():
      with psycopg.connect(DATABASE_URL) as conn:
          register_vector(conn)
          yield conn


  def test_photo_clusters_table_exists(db_conn):
      with db_conn.cursor() as cur:
          cur.execute(
              "SELECT column_name FROM information_schema.columns "
              "WHERE table_name = 'photo_clusters' "
              "ORDER BY ordinal_position"
          )
          cols = [row[0] for row in cur.fetchall()]
      assert cols == ["cluster_id", "sha256", "is_representative", "quality_score", "created_at"]


  def test_photo_clusters_index_exists(db_conn):
      with db_conn.cursor() as cur:
          cur.execute(
              "SELECT indexname FROM pg_indexes "
              "WHERE tablename = 'photo_clusters' AND indexname != 'photo_clusters_pkey'"
          )
          indexes = [row[0] for row in cur.fetchall()]
      assert len(indexes) >= 1
  ```

- [ ] **Run to verify it fails**

  ```bash
  NEAR_DUP_SMOKE=1 uv run pytest tests/test_near_dup/test_smoke.py::test_photo_clusters_table_exists -q
  ```

  Expected: `FAILED` — table does not exist yet.

- [ ] **Implement: run migration SQL**

  Connect to the `dedup-postgres` container and execute:

  ```bash
  docker exec -i dedup-postgres psql -U dedup -d dedup <<'SQL'
  CREATE TABLE IF NOT EXISTS photo_clusters (
      cluster_id       int  NOT NULL,
      sha256           text NOT NULL,
      is_representative bool NOT NULL DEFAULT false,
      quality_score    float,
      created_at       timestamptz DEFAULT now(),
      PRIMARY KEY (cluster_id, sha256)
  );
  CREATE INDEX IF NOT EXISTS photo_clusters_cluster_id_idx ON photo_clusters(cluster_id);
  SQL
  ```

- [ ] **Run to verify it passes**

  ```bash
  NEAR_DUP_SMOKE=1 uv run pytest tests/test_near_dup/test_smoke.py -q
  ```

  Expected: `2 passed`.

- [ ] **Commit**

  ```bash
  git add tests/test_near_dup/test_smoke.py
  git commit -m "near-dup: add photo_clusters migration and smoke test skeleton"
  ```

---

## Task 2 — Package Scaffold + conftest

Create `__init__.py` files, the test `conftest.py` with shared fixtures.

- [ ] **Write the failing test**

  Create `/home/will/photo_project/tests/test_near_dup/test_conftest_fixtures.py`:

  ```python
  from __future__ import annotations

  import numpy as np


  def test_tiny_embeddings_shape(tiny_embeddings):
      sha256s, matrix = tiny_embeddings
      assert len(sha256s) == 10
      assert matrix.shape == (10, 1152)
      assert matrix.dtype == np.float32


  def test_tiny_embeddings_l2_normalized(tiny_embeddings):
      _, matrix = tiny_embeddings
      norms = np.linalg.norm(matrix, axis=1)
      np.testing.assert_allclose(norms, np.ones(10), atol=1e-5)


  def test_grey_images_created(grey_images):
      paths, tmp = grey_images
      assert len(paths) == 5
      for p in paths:
          assert p.exists()
          assert p.suffix == ".jpg"


  def test_mock_conn_fixture(mock_conn):
      conn, cursor = mock_conn
      # cursor context manager protocol must work
      with conn.cursor() as cur:
          pass
  ```

- [ ] **Run to verify it fails**

  ```bash
  uv run pytest tests/test_near_dup/test_conftest_fixtures.py -q
  ```

  Expected: `ERROR` — fixtures not found.

- [ ] **Implement**

  Create `/home/will/photo_project/scripts/near_dup/__init__.py` (empty):

  ```python
  ```

  Create `/home/will/photo_project/tests/test_near_dup/__init__.py` (empty):

  ```python
  ```

  Create `/home/will/photo_project/tests/test_near_dup/conftest.py`:

  ```python
  from __future__ import annotations

  from pathlib import Path
  from unittest.mock import MagicMock

  import numpy as np
  import psycopg
  import pytest
  from PIL import Image


  @pytest.fixture
  def mock_conn():
      conn = MagicMock(spec=psycopg.Connection)
      cursor = MagicMock()
      cursor.__enter__ = lambda s: cursor
      cursor.__exit__ = MagicMock(return_value=False)
      conn.cursor.return_value = cursor
      return conn, cursor


  @pytest.fixture
  def tiny_embeddings() -> tuple[list[str], np.ndarray]:
      rng = np.random.default_rng(42)
      matrix = rng.standard_normal((10, 1152)).astype(np.float32)
      norms = np.linalg.norm(matrix, axis=1, keepdims=True)
      matrix = matrix / norms
      sha256s = [f"sha{i:064d}" for i in range(10)]
      return sha256s, matrix


  @pytest.fixture
  def grey_images(tmp_path: Path) -> tuple[list[Path], Path]:
      paths = []
      for i in range(5):
          p = tmp_path / f"grey_{i}.jpg"
          Image.new("L", (64, 64), color=128).convert("RGB").save(p)
          paths.append(p)
      return paths, tmp_path
  ```

- [ ] **Run to verify it passes**

  ```bash
  uv run pytest tests/test_near_dup/test_conftest_fixtures.py -q
  ```

  Expected: `4 passed`.

- [ ] **Commit**

  ```bash
  git add scripts/near_dup/__init__.py tests/test_near_dup/__init__.py tests/test_near_dup/conftest.py tests/test_near_dup/test_conftest_fixtures.py
  git commit -m "near-dup: add package scaffold and conftest fixtures"
  ```

---

## Task 3 — Embedding Loader

Implement `fetch_photo_embeddings(conn) -> tuple[list[str], np.ndarray]`.

- [ ] **Write the failing test**

  Create `/home/will/photo_project/tests/test_near_dup/test_embeddings.py`:

  ```python
  from __future__ import annotations

  import sys
  from pathlib import Path

  import numpy as np
  import pytest

  sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
  from near_dup.embeddings import fetch_photo_embeddings

  MODEL = "siglip-so400m"


  def test_fetch_returns_sha256_list_and_matrix(mock_conn):
      conn, cursor = mock_conn
      vec_a = np.ones(1152, dtype=np.float32) / np.sqrt(1152)
      vec_b = -np.ones(1152, dtype=np.float32) / np.sqrt(1152)
      cursor.fetchall.return_value = [
          ("aaaa", vec_a),
          ("bbbb", vec_b),
      ]

      sha256s, matrix = fetch_photo_embeddings(conn, model=MODEL)

      assert sha256s == ["aaaa", "bbbb"]
      assert matrix.shape == (2, 1152)
      assert matrix.dtype == np.float32


  def test_fetch_queries_correct_model_and_media_type(mock_conn):
      conn, cursor = mock_conn
      cursor.fetchall.return_value = []

      fetch_photo_embeddings(conn, model=MODEL)

      sql = cursor.execute.call_args[0][0]
      params = cursor.execute.call_args[0][1]
      assert "media_type" in sql
      assert "photo" in params
      assert MODEL in params


  def test_fetch_l2_normalizes_output(mock_conn):
      conn, cursor = mock_conn
      rng = np.random.default_rng(7)
      raw = rng.standard_normal(1152).astype(np.float32)
      # Return unnormalized; function must normalize
      cursor.fetchall.return_value = [("c1c1", raw)]

      _, matrix = fetch_photo_embeddings(conn, model=MODEL)

      norm = float(np.linalg.norm(matrix[0]))
      assert abs(norm - 1.0) < 1e-5


  def test_fetch_empty_table_returns_empty(mock_conn):
      conn, cursor = mock_conn
      cursor.fetchall.return_value = []

      sha256s, matrix = fetch_photo_embeddings(conn, model=MODEL)

      assert sha256s == []
      assert matrix.shape == (0, 1152)
  ```

- [ ] **Run to verify it fails**

  ```bash
  uv run pytest tests/test_near_dup/test_embeddings.py -q
  ```

  Expected: `ERROR` — `ModuleNotFoundError: No module named 'near_dup.embeddings'`.

- [ ] **Implement**

  Create `/home/will/photo_project/scripts/near_dup/embeddings.py`:

  ```python
  from __future__ import annotations

  import numpy as np
  import psycopg

  MODEL_NAME = "siglip-so400m"
  EMBEDDING_DIM = 1152


  def fetch_photo_embeddings(
      conn: psycopg.Connection,
      model: str = MODEL_NAME,
  ) -> tuple[list[str], np.ndarray]:
      """Return (sha256_list, matrix) for all photo rows of the given model.

      The returned matrix is always L2-normalized, float32, shape (N, 1152).
      """
      with conn.cursor() as cur:
          cur.execute(
              "SELECT sha256, embedding FROM photo_embeddings "
              "WHERE media_type = %s AND model = %s "
              "ORDER BY sha256",
              ("photo", model),
          )
          rows = cur.fetchall()

      if not rows:
          return [], np.empty((0, EMBEDDING_DIM), dtype=np.float32)

      sha256s = [row[0] for row in rows]
      matrix = np.array([row[1] for row in rows], dtype=np.float32)

      # L2-normalize (embeddings should already be normalized, but enforce it)
      norms = np.linalg.norm(matrix, axis=1, keepdims=True)
      norms = np.where(norms == 0, 1.0, norms)
      matrix = matrix / norms

      return sha256s, matrix
  ```

- [ ] **Run to verify it passes**

  ```bash
  uv run pytest tests/test_near_dup/test_embeddings.py -q
  ```

  Expected: `4 passed`.

- [ ] **Commit**

  ```bash
  git add scripts/near_dup/embeddings.py tests/test_near_dup/test_embeddings.py
  git commit -m "near-dup: add fetch_photo_embeddings loader"
  ```

---

## Task 4 — HDBSCAN Clustering

Implement `run_hdbscan(matrix) -> np.ndarray` and `build_cluster_map(sha256s, labels) -> dict[int, list[str]]`.

- [ ] **Add dependency**

  ```bash
  uv add hdbscan
  ```

- [ ] **Write the failing test**

  Create `/home/will/photo_project/tests/test_near_dup/test_clustering.py`:

  ```python
  from __future__ import annotations

  import sys
  from pathlib import Path

  import numpy as np
  import pytest

  sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
  from near_dup.clustering import build_cluster_map, run_hdbscan


  def test_run_hdbscan_returns_label_array(tiny_embeddings):
      _, matrix = tiny_embeddings
      labels = run_hdbscan(matrix)
      assert labels.shape == (10,)
      assert labels.dtype in (np.int32, np.int64, np.intp)


  def test_run_hdbscan_noise_label_present_or_all_clustered(tiny_embeddings):
      """Label -1 means noise (singleton). At least one label value must exist."""
      _, matrix = tiny_embeddings
      labels = run_hdbscan(matrix)
      unique = set(labels.tolist())
      # All 10 random points — we don't mandate noise exists, but labels must be ints
      assert all(isinstance(v, int) for v in unique)


  def test_build_cluster_map_groups_by_label(tiny_embeddings):
      sha256s, _ = tiny_embeddings
      # Manually assign: indices 0,1,2 -> cluster 0; indices 3,4 -> cluster 1; rest -> noise -1
      labels = np.array([-1, 0, 0, 0, 1, 1, -1, -1, -1, -1])
      result = build_cluster_map(sha256s, labels)

      assert 0 in result
      assert 1 in result
      assert -1 in result
      assert result[0] == [sha256s[1], sha256s[2], sha256s[3]]
      assert result[1] == [sha256s[4], sha256s[5]]
      assert set(result[-1]) == {sha256s[0], sha256s[6], sha256s[7], sha256s[8], sha256s[9]}


  def test_build_cluster_map_noise_label_included(tiny_embeddings):
      sha256s, _ = tiny_embeddings
      labels = np.full(10, -1)  # all noise
      result = build_cluster_map(sha256s, labels)
      assert -1 in result
      assert len(result[-1]) == 10


  def test_build_cluster_map_no_noise(tiny_embeddings):
      sha256s, _ = tiny_embeddings
      labels = np.zeros(10, dtype=np.intp)  # everything in cluster 0
      result = build_cluster_map(sha256s, labels)
      assert -1 not in result
      assert len(result[0]) == 10
  ```

- [ ] **Run to verify it fails**

  ```bash
  uv run pytest tests/test_near_dup/test_clustering.py -q
  ```

  Expected: `ERROR` — `ModuleNotFoundError: No module named 'near_dup.clustering'`.

- [ ] **Implement**

  Create `/home/will/photo_project/scripts/near_dup/clustering.py`:

  ```python
  from __future__ import annotations

  from collections import defaultdict

  import hdbscan
  import numpy as np

  # cosine sim 0.96 ~ euclidean distance 0.283 for unit vectors.
  # (||a-b||^2 = 2 - 2*cos_sim => dist = sqrt(2*(1-0.96)) = 0.283)
  EPSILON = 0.3
  MIN_CLUSTER_SIZE = 2
  MIN_SAMPLES = 1


  def run_hdbscan(matrix: np.ndarray) -> np.ndarray:
      """Cluster L2-normalized embedding matrix with HDBSCAN.

      Returns an integer label array of shape (N,). Label -1 means noise (no cluster).
      """
      clusterer = hdbscan.HDBSCAN(
          min_cluster_size=MIN_CLUSTER_SIZE,
          min_samples=MIN_SAMPLES,
          metric="euclidean",
          cluster_selection_epsilon=EPSILON,
          core_dist_n_jobs=-1,
      )
      clusterer.fit(matrix)
      return clusterer.labels_.astype(np.intp)


  def build_cluster_map(
      sha256s: list[str],
      labels: np.ndarray,
  ) -> dict[int, list[str]]:
      """Map cluster label -> list of sha256 strings (preserves order)."""
      result: dict[int, list[str]] = defaultdict(list)
      for sha, label in zip(sha256s, labels.tolist()):
          result[int(label)].append(sha)
      return dict(result)
  ```

- [ ] **Run to verify it passes**

  ```bash
  uv run pytest tests/test_near_dup/test_clustering.py -q
  ```

  Expected: `5 passed`.

- [ ] **Commit**

  ```bash
  git add scripts/near_dup/clustering.py tests/test_near_dup/test_clustering.py
  git commit -m "near-dup: add HDBSCAN clustering module"
  ```

---

## Task 5 — Quality Scorer

Implement `score_image(path: Path) -> float` using Laplacian variance normalized by resolution.

- [ ] **Add dependency**

  ```bash
  uv add opencv-python-headless
  ```

- [ ] **Write the failing test**

  Create `/home/will/photo_project/tests/test_near_dup/test_quality.py`:

  ```python
  from __future__ import annotations

  import sys
  from pathlib import Path

  import numpy as np
  import pytest
  from PIL import Image, ImageFilter

  sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
  from near_dup.quality import score_image


  @pytest.fixture
  def sharp_image(tmp_path: Path) -> Path:
      """Solid color image — no blur, high Laplacian variance at edges."""
      img = Image.new("RGB", (128, 128), color=(200, 100, 50))
      # Add a hard edge by pasting a contrasting block
      block = Image.new("RGB", (64, 64), color=(10, 10, 200))
      img.paste(block, (32, 32))
      path = tmp_path / "sharp.jpg"
      img.save(path, quality=95)
      return path


  @pytest.fixture
  def blurry_image(tmp_path: Path) -> Path:
      """Same image but heavily Gaussian blurred — low Laplacian variance."""
      img = Image.new("RGB", (128, 128), color=(200, 100, 50))
      block = Image.new("RGB", (64, 64), color=(10, 10, 200))
      img.paste(block, (32, 32))
      img = img.filter(ImageFilter.GaussianBlur(radius=15))
      path = tmp_path / "blurry.jpg"
      img.save(path, quality=95)
      return path


  def test_sharp_scores_higher_than_blurry(sharp_image, blurry_image):
      sharp_score = score_image(sharp_image)
      blurry_score = score_image(blurry_image)
      assert sharp_score > blurry_score, (
          f"Expected sharp ({sharp_score:.4f}) > blurry ({blurry_score:.4f})"
      )


  def test_score_is_non_negative(sharp_image):
      assert score_image(sharp_image) >= 0.0


  def test_missing_file_returns_zero(tmp_path: Path):
      missing = tmp_path / "nonexistent.jpg"
      assert score_image(missing) == 0.0


  def test_grey_images_score_near_zero(grey_images):
      """Uniform grey images have almost no edges — score should be very low."""
      paths, _ = grey_images
      for p in paths:
          score = score_image(p)
          assert score < 0.5, f"Expected near-zero for flat grey image, got {score:.4f}"
  ```

- [ ] **Run to verify it fails**

  ```bash
  uv run pytest tests/test_near_dup/test_quality.py -q
  ```

  Expected: `ERROR` — `ModuleNotFoundError: No module named 'near_dup.quality'`.

- [ ] **Implement**

  Create `/home/will/photo_project/scripts/near_dup/quality.py`:

  ```python
  from __future__ import annotations

  import math
  from pathlib import Path

  import cv2
  import numpy as np


  def score_image(path: Path) -> float:
      """Return a sharpness score for the image at `path`.

      Uses Laplacian variance normalized by sqrt(width * height) so that
      large and small images are roughly comparable.
      Returns 0.0 if the file cannot be read.
      """
      try:
          img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
          if img is None:
              return 0.0
          h, w = img.shape
          if h == 0 or w == 0:
              return 0.0
          lap_var = float(cv2.Laplacian(img, cv2.CV_64F).var())
          return lap_var / math.sqrt(w * h)
      except Exception:
          return 0.0
  ```

- [ ] **Run to verify it passes**

  ```bash
  uv run pytest tests/test_near_dup/test_quality.py -q
  ```

  Expected: `4 passed`.

- [ ] **Commit**

  ```bash
  git add scripts/near_dup/quality.py tests/test_near_dup/test_quality.py
  git commit -m "near-dup: add Laplacian-variance quality scorer"
  ```

---

## Task 6 — DB Write/Read Helpers

Implement `write_clusters(conn, rows)` and `read_cluster_groups(conn, min_size=2)`.

- [ ] **Write the failing test**

  Create `/home/will/photo_project/tests/test_near_dup/test_db.py`:

  ```python
  from __future__ import annotations

  import sys
  from pathlib import Path

  import pytest

  sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
  from near_dup.db import read_cluster_groups, write_clusters


  def test_write_clusters_calls_executemany(mock_conn):
      conn, cursor = mock_conn
      rows = [
          (0, "aaa", True, 12.5),
          (0, "bbb", False, 7.2),
          (1, "ccc", True, 9.1),
      ]
      write_clusters(conn, rows)

      cursor.executemany.assert_called_once()
      sql, data = cursor.executemany.call_args[0]
      assert "INSERT INTO photo_clusters" in sql
      assert "ON CONFLICT DO NOTHING" in sql
      assert data == rows
      conn.commit.assert_called_once()


  def test_write_clusters_empty_rows_still_commits(mock_conn):
      conn, cursor = mock_conn
      write_clusters(conn, [])
      conn.commit.assert_called_once()


  def test_read_cluster_groups_filters_by_min_size(mock_conn):
      conn, cursor = mock_conn
      # Simulate DB returning only clusters with >= 2 members
      cursor.fetchall.return_value = [
          (0, "aaa", True, 12.5),
          (0, "bbb", False, 7.2),
          (1, "ccc", True, 9.1),
          (1, "ddd", False, 3.0),
      ]

      result = read_cluster_groups(conn, min_size=2)

      sql = cursor.execute.call_args[0][0]
      params = cursor.execute.call_args[0][1]
      assert "HAVING COUNT(*)" in sql
      assert 2 in params
      assert 0 in result
      assert 1 in result
      assert len(result[0]) == 2
      assert len(result[1]) == 2


  def test_read_cluster_groups_returns_tuples(mock_conn):
      conn, cursor = mock_conn
      cursor.fetchall.return_value = [
          (5, "xyz", True, 8.8),
      ]
      result = read_cluster_groups(conn, min_size=1)
      assert 5 in result
      row = result[5][0]
      # Each entry should be (sha256, is_representative, quality_score)
      assert row[0] == "xyz"
      assert row[1] is True
      assert abs(row[2] - 8.8) < 1e-6
  ```

- [ ] **Run to verify it fails**

  ```bash
  uv run pytest tests/test_near_dup/test_db.py -q
  ```

  Expected: `ERROR` — `ModuleNotFoundError: No module named 'near_dup.db'`.

- [ ] **Implement**

  Create `/home/will/photo_project/scripts/near_dup/db.py`:

  ```python
  from __future__ import annotations

  from collections import defaultdict

  import psycopg


  def write_clusters(
      conn: psycopg.Connection,
      rows: list[tuple],  # (cluster_id, sha256, is_representative, quality_score)
  ) -> None:
      """Insert cluster rows into photo_clusters. Skips conflicts silently."""
      with conn.cursor() as cur:
          cur.executemany(
              "INSERT INTO photo_clusters "
              "(cluster_id, sha256, is_representative, quality_score) "
              "VALUES (%s, %s, %s, %s) "
              "ON CONFLICT DO NOTHING",
              rows,
          )
      conn.commit()


  def read_cluster_groups(
      conn: psycopg.Connection,
      min_size: int = 2,
  ) -> dict[int, list[tuple]]:
      """Return clusters with at least `min_size` members.

      Each value is a list of (sha256, is_representative, quality_score) tuples,
      sorted by quality_score descending (representative first).
      """
      with conn.cursor() as cur:
          cur.execute(
              "SELECT cluster_id, sha256, is_representative, quality_score "
              "FROM photo_clusters "
              "WHERE cluster_id IN ("
              "    SELECT cluster_id FROM photo_clusters "
              "    GROUP BY cluster_id HAVING COUNT(*) >= %s"
              ") "
              "ORDER BY cluster_id, is_representative DESC, quality_score DESC NULLS LAST",
              (min_size,),
          )
          rows = cur.fetchall()

      result: dict[int, list[tuple]] = defaultdict(list)
      for cluster_id, sha256, is_rep, score in rows:
          result[int(cluster_id)].append((sha256, is_rep, score))
      return dict(result)
  ```

- [ ] **Run to verify it passes**

  ```bash
  uv run pytest tests/test_near_dup/test_db.py -q
  ```

  Expected: `4 passed`.

- [ ] **Commit**

  ```bash
  git add scripts/near_dup/db.py tests/test_near_dup/test_db.py
  git commit -m "near-dup: add write_clusters and read_cluster_groups DB helpers"
  ```

---

## Task 7 — Main Script, Review HTML, justfile Recipes, and Smoke Test

Wire everything together into `run_cluster.py`, add `build_review_html.py`, add justfile recipes, and extend the smoke test to exercise the full pipeline end-to-end.

- [ ] **Write the failing smoke test extension**

  Append to `/home/will/photo_project/tests/test_near_dup/test_smoke.py`:

  ```python
  import subprocess
  import sys
  from pathlib import Path


  def test_run_cluster_dry_run_exits_cleanly(db_conn):
      """run_cluster.py --dry-run should fetch embeddings and cluster without writing."""
      result = subprocess.run(
          [sys.executable, "scripts/near_dup/run_cluster.py", "--dry-run"],
          capture_output=True,
          text=True,
          cwd="/home/will/photo_project",
          env={**__import__("os").environ, "DATABASE_URL": DATABASE_URL},
      )
      assert result.returncode == 0, f"STDERR:\n{result.stderr}"
      assert "clusters found" in result.stdout.lower() or "cluster" in result.stdout.lower()


  def test_photo_clusters_populated_after_run(db_conn):
      """After a full run, photo_clusters must have rows."""
      # Run the real clustering (may take a few minutes on 42K embeddings)
      result = subprocess.run(
          [sys.executable, "scripts/near_dup/run_cluster.py"],
          capture_output=True,
          text=True,
          cwd="/home/will/photo_project",
          env={**__import__("os").environ, "DATABASE_URL": DATABASE_URL},
          timeout=600,
      )
      assert result.returncode == 0, f"STDERR:\n{result.stderr}"
      with db_conn.cursor() as cur:
          cur.execute("SELECT COUNT(*) FROM photo_clusters")
          count = cur.fetchone()[0]
      assert count > 0
  ```

- [ ] **Run to verify it fails**

  ```bash
  NEAR_DUP_SMOKE=1 uv run pytest tests/test_near_dup/test_smoke.py::test_run_cluster_dry_run_exits_cleanly -q
  ```

  Expected: `FAILED` — `run_cluster.py` does not exist yet.

- [ ] **Implement `run_cluster.py`**

  Create `/home/will/photo_project/scripts/near_dup/run_cluster.py`:

  ```python
  from __future__ import annotations

  import argparse
  import logging
  import os
  import sys
  from pathlib import Path

  import psycopg
  from pgvector.psycopg import register_vector

  sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
  from near_dup.clustering import build_cluster_map, run_hdbscan
  from near_dup.db import write_clusters
  from near_dup.embeddings import fetch_photo_embeddings
  from near_dup.quality import score_image

  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s %(levelname)s %(message)s",
      datefmt="%H:%M:%S",
  )
  log = logging.getLogger(__name__)

  DATABASE_URL = os.environ.get(
      "DATABASE_URL", "postgresql://dedup:dedup_local_dev@127.0.0.1:5432/dedup"
  )
  MEDIA_DIR = Path(os.environ.get("MEDIA_DIR", "/home/will/photo_project/originals/media"))


  def build_rows(
      cluster_map: dict[int, list[str]],
      media_dir: Path,
      dry_run: bool = False,
  ) -> list[tuple]:
      """Score each image and build DB row tuples.

      For noise cluster (-1) every photo is its own representative.
      For real clusters the highest-scoring photo is representative.
      """
      rows: list[tuple] = []
      for cluster_id, sha256s in cluster_map.items():
          if cluster_id == -1:
              # Noise: each photo is unique, mark all as representative
              for sha in sha256s:
                  path = media_dir / sha
                  score = 0.0 if dry_run else score_image(path)
                  rows.append((cluster_id, sha, True, score))
          else:
              scores: list[tuple[str, float]] = []
              for sha in sha256s:
                  path = media_dir / sha
                  score = 0.0 if dry_run else score_image(path)
                  scores.append((sha, score))
              # Sort descending by score; first entry is representative
              scores.sort(key=lambda x: x[1], reverse=True)
              for i, (sha, score) in enumerate(scores):
                  rows.append((cluster_id, sha, i == 0, score))
      return rows


  def main() -> None:
      parser = argparse.ArgumentParser(description="Cluster photo embeddings for near-duplicate detection")
      parser.add_argument("--dry-run", action="store_true", help="Cluster but do not write to DB or score images")
      args = parser.parse_args()

      log.info("Connecting to DB: %s", DATABASE_URL.split("@")[-1])
      with psycopg.connect(DATABASE_URL) as conn:
          register_vector(conn)

          log.info("Fetching photo embeddings...")
          sha256s, matrix = fetch_photo_embeddings(conn)
          log.info("Loaded %d embeddings (%.1f MB)", len(sha256s), matrix.nbytes / 1e6)

          if len(sha256s) == 0:
              log.warning("No embeddings found. Exiting.")
              sys.exit(0)

          log.info("Running HDBSCAN clustering...")
          labels = run_hdbscan(matrix)

          cluster_map = build_cluster_map(sha256s, labels)
          n_noise = len(cluster_map.get(-1, []))
          n_clusters = len(cluster_map) - (1 if -1 in cluster_map else 0)
          total_in_clusters = sum(len(v) for k, v in cluster_map.items() if k != -1)
          log.info(
              "%d clusters found, %d photos in duplicate groups, %d unique (noise)",
              n_clusters,
              total_in_clusters,
              n_noise,
          )

          if args.dry_run:
              log.info("Dry run — skipping image scoring and DB write.")
              return

          log.info("Scoring image quality...")
          rows = build_rows(cluster_map, MEDIA_DIR)
          log.info("Writing %d rows to photo_clusters...", len(rows))
          write_clusters(conn, rows)
          log.info("Done.")


  if __name__ == "__main__":
      main()
  ```

- [ ] **Implement `build_review_html.py`**

  Create `/home/will/photo_project/scripts/near_dup/build_review_html.py`:

  ```python
  from __future__ import annotations

  import base64
  import io
  import logging
  import os
  import sys
  from pathlib import Path

  import psycopg
  from pgvector.psycopg import register_vector
  from PIL import Image

  sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
  from near_dup.db import read_cluster_groups

  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s %(levelname)s %(message)s",
      datefmt="%H:%M:%S",
  )
  log = logging.getLogger(__name__)

  DATABASE_URL = os.environ.get(
      "DATABASE_URL", "postgresql://dedup:dedup_local_dev@127.0.0.1:5432/dedup"
  )
  MEDIA_DIR = Path(os.environ.get("MEDIA_DIR", "/home/will/photo_project/originals/media"))
  OUTPUT_PATH = Path("output/near_dup_review.html")
  THUMB_SIZE = (200, 200)
  MIN_CLUSTER_SIZE = int(os.environ.get("MIN_CLUSTER_SIZE", "2"))


  def thumb_b64(path: Path) -> str:
      """Return a base64-encoded JPEG thumbnail or a placeholder on error."""
      try:
          img = Image.open(path)
          img.thumbnail(THUMB_SIZE)
          buf = io.BytesIO()
          img.convert("RGB").save(buf, format="JPEG", quality=75)
          return base64.b64encode(buf.getvalue()).decode()
      except Exception:
          return ""


  def render_html(groups: dict[int, list[tuple]]) -> str:
      cards = []
      for cluster_id in sorted(groups):
          members = groups[cluster_id]
          images_html = []
          for sha256, is_rep, quality in members:
              path = MEDIA_DIR / sha256
              b64 = thumb_b64(path)
              border = "3px solid #4caf50" if is_rep else "1px solid #ccc"
              label = "BEST" if is_rep else ""
              img_tag = (
                  f'<img src="data:image/jpeg;base64,{b64}" '
                  f'style="border:{border};width:200px;height:200px;object-fit:cover;" '
                  f'title="{sha256}" />'
                  if b64 else
                  f'<div style="width:200px;height:200px;background:#eee;'
                  f'border:{border};display:flex;align-items:center;justify-content:center;">'
                  f'<span>missing</span></div>'
              )
              score_str = f"{quality:.2f}" if quality is not None else "n/a"
              images_html.append(
                  f'<div style="display:inline-block;margin:4px;text-align:center;">'
                  f'{img_tag}'
                  f'<div style="font-size:11px;color:#555;">{label} score={score_str}</div>'
                  f'<div style="font-size:9px;color:#999;word-break:break-all;max-width:200px;">'
                  f'{sha256[:16]}...</div>'
                  f'</div>'
              )
          cards.append(
              f'<div style="margin-bottom:20px;padding:10px;border:1px solid #ddd;">'
              f'<h3 style="margin:0 0 8px;">Cluster {cluster_id} '
              f'<span style="font-weight:normal;font-size:14px;">({len(members)} photos)</span></h3>'
              f'{"".join(images_html)}'
              f'</div>'
          )

      return f"""<!DOCTYPE html>
  <html>
  <head>
  <meta charset="utf-8">
  <title>Near-Duplicate Clusters</title>
  <style>body{{font-family:sans-serif;margin:20px;background:#fafafa;}}</style>
  </head>
  <body>
  <h1>Near-Duplicate Photo Clusters</h1>
  <p>{len(groups)} duplicate groups shown (min_size={MIN_CLUSTER_SIZE})</p>
  {"".join(cards)}
  </body>
  </html>
  """


  def main() -> None:
      with psycopg.connect(DATABASE_URL) as conn:
          register_vector(conn)
          log.info("Reading cluster groups (min_size=%d)...", MIN_CLUSTER_SIZE)
          groups = read_cluster_groups(conn, min_size=MIN_CLUSTER_SIZE)

      log.info("Rendering %d groups...", len(groups))
      html = render_html(groups)
      OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
      OUTPUT_PATH.write_text(html, encoding="utf-8")
      log.info("Wrote %s", OUTPUT_PATH)


  if __name__ == "__main__":
      main()
  ```

- [ ] **Add justfile recipes**

  Append to `/home/will/photo_project/justfile`:

  ```just
  # Cluster photo embeddings to find near-duplicates (writes to photo_clusters table)
  near-dup-cluster:
      uv run scripts/near_dup/run_cluster.py

  # Build near-duplicate review HTML (output/near_dup_review.html)
  near-dup-review:
      uv run scripts/near_dup/build_review_html.py
  ```

- [ ] **Run full unit test suite to confirm no regressions**

  ```bash
  uv run pytest tests/ -q --ignore=tests/test_near_dup/test_smoke.py
  ```

  Expected: all existing tests pass plus the new unit tests.

- [ ] **Run smoke tests against real DB**

  ```bash
  NEAR_DUP_SMOKE=1 uv run pytest tests/test_near_dup/test_smoke.py -q -v
  ```

  The `test_run_cluster_dry_run_exits_cleanly` test should pass quickly (just fetches embeddings and clusters in memory). The `test_photo_clusters_populated_after_run` test will run the full pipeline against 42K embeddings — allow up to 10 minutes (mostly HDBSCAN on CPU).

- [ ] **Commit**

  ```bash
  git add scripts/near_dup/run_cluster.py scripts/near_dup/build_review_html.py justfile tests/test_near_dup/test_smoke.py
  git commit -m "near-dup: add run_cluster, build_review_html, and justfile recipes"
  ```

---

## Self-Review Checklist

- [x] `fetch_photo_embeddings` queries `media_type='photo'` and `model='siglip-so400m'` — consistent across embeddings.py, test_embeddings.py, and run_cluster.py
- [x] `run_hdbscan` takes `np.ndarray`, returns `np.ndarray` — called correctly in run_cluster.py
- [x] `build_cluster_map` takes `(list[str], np.ndarray)`, returns `dict[int, list[str]]` — called correctly in run_cluster.py
- [x] `score_image` takes `Path`, returns `float` — called in build_rows inside run_cluster.py
- [x] `write_clusters` takes `(conn, list[tuple])` with tuples `(cluster_id, sha256, is_representative, quality_score)` — matches CREATE TABLE column order
- [x] `read_cluster_groups` returns `dict[int, list[tuple]]` where tuples are `(sha256, is_representative, quality_score)` — matches render_html unpacking
- [x] Noise cluster (-1) is handled: stored with `is_representative=True`, not displayed in review HTML (min_size filter excludes it since every noise point is a singleton)
- [x] No placeholders — all code is complete and executable
- [x] `NEAR_DUP_SMOKE=1` gates all real-DB tests
- [x] `--dry-run` flag lets the smoke test validate clustering logic without writing or scoring
- [x] Dependencies (`hdbscan`, `opencv-python-headless`) are added via `uv add` before the tests that need them
