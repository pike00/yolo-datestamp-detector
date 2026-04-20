---
summary: "Face clustering pipeline (insightface + HDBSCAN) -- Tasks 1-4 done, Tasks 5-8 pending"
---

# Handoff: Face Clustering Pipeline

**Date:** 2026-04-17
**Goal:** Implement face detection and clustering across 42K photos using insightface ArcFace + HDBSCAN, with a labeled review UI.

## Current Status

Working on branch `feature/face-clustering` in worktree `.worktrees/face-clustering`.

**Completed (committed):**
- Task 1: DB schema migration — `face_detections` + `face_clusters` tables, ivfflat index on embedding vector(512)
- Task 2: Package scaffold — `scripts/face_clustering/__init__.py`, `requirements.txt`, conftest fixtures (`mock_conn`, `fake_face_dict`, `bgr_image`)
- Task 3: `FaceDetector` wrapper — wraps insightface buffalo_sc, filters by det_score, converts bbox x1y1x2y2 → xywh
- Task 4: `db.py` — all DB helpers: `get_processed_sha256s`, `insert_faces`, `load_all_face_embeddings`, `write_clusters`, `mark_representatives`

**In progress at interruption:**
- Task 4 spec/quality review was skipped (interrupted before dispatching reviewers)

## Next Steps

1. **Dispatch spec + quality review for Task 4** (or just proceed if you trust the output — db.py matches the plan exactly)
2. **Task 5:** Implement `scripts/face_clustering/detect_all.py` — scans `originals/media/`, skips processed sha256s, runs FaceDetector, inserts
3. **Task 6:** Implement `scripts/face_clustering/cluster_faces.py` — HDBSCAN on all embeddings, write clusters, mark representatives; also finishes `test_db_cluster.py` (the last test imports `cluster_faces.run_clustering` and will pass once Task 6 is done)
4. **Task 7:** Implement `scripts/face_clustering/build_review_html.py` — face crop grid sorted by cluster size
5. **Task 8:** Docker (`docker/Dockerfile.face-detect`, `docker/docker-compose.face-detect.yml`), justfile recipes (`face-detect`, `face-detect-bg`, `face-cluster`, `face-review`), smoke test file (`tests/test_face_clustering/test_smoke_files.py`)
6. After Task 8: run full test suite, then use `superpowers:finishing-a-development-branch` to merge

## Key Context

- **Worktree:** `/home/will/photo_project/.worktrees/face-clustering` (branch `feature/face-clustering`)
- **Plan file:** `docs/plans/2026-04-17-face-clustering.md` — has complete TDD specs for every task including exact test and implementation code
- **insightface NOT installed locally** — tests mock it; it only runs inside Docker
- **`MEDIA_DIR`** defaults to `/home/will/photo_project/originals/media` — that directory must exist for detect_all.py to run
- **pgvector already enabled** in the `dedup` Postgres database
- **DB tables already created** — migration ran successfully in this session
- **test_db_cluster.py** has 5 tests; 4 pass, 1 fails (expects `cluster_faces.run_clustering` — fixed in Task 6)
- Each task uses TDD: write failing test → implement → verify passes → commit

## Files Touched

- `scripts/face_clustering/__init__.py` — empty package marker
- `scripts/face_clustering/migrate.py` — one-shot DDL migration (already ran)
- `scripts/face_clustering/requirements.txt` — Docker pip deps
- `scripts/face_clustering/detector.py` — FaceDetector class
- `scripts/face_clustering/db.py` — all DB helpers (full file, both detect + cluster side)
- `tests/test_face_clustering/__init__.py`
- `tests/test_face_clustering/conftest.py`
- `tests/test_face_clustering/test_schema.py`
- `tests/test_face_clustering/test_conftest_fixtures.py`
- `tests/test_face_clustering/test_detector.py`
- `tests/test_face_clustering/test_db.py`
- `tests/test_face_clustering/test_db_cluster.py`

## Blockers

None. Clean path forward — follow the plan file task by task.
