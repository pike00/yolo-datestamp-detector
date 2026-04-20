---
summary: "Face clustering pipeline E2E — all tasks done, full 33K detection running in bg Docker"
---

# Handoff: Face Clustering Pipeline End-to-End

**Date:** 2026-04-19
**Goal:** Finish face clustering implementation (Tasks 5-8), run a 1K sample end-to-end, kick off full 33K-photo detection as a background Docker container, and serve a review UI on ares.

## Current Status

Branch `feature/face-clustering` in worktree `.worktrees/face-clustering` — 10 commits ahead of main, working tree clean.

**Done this session:**
- Task 5 (`detect_all.py`) — commit `acedf3c`
- Task 6 (`cluster_faces.py`) — commit `f2f89cc` (fixed the last failing test from the prior handoff)
- Task 7 (`build_review_html.py`) — commit `c0eeed5`
- Task 8 (Docker + compose + justfile + smoke tests) — commit `afd5ac8`
- Runtime fixes — commit `ec60fdb`:
  - `db.insert_faces` now uses `executemany(returning=True)` + `cur.results()`/`fetchone()`; the previous `fetchall()` pattern is a psycopg3 footgun that silently returns only the first result set (surfaced as `ProgrammingError: no result available`)
  - `Dockerfile.face-detect` now installs `build-essential`; insightface ships no Python 3.12 wheel
  - New `build_bbox_review_html.py` renders original photos with bboxes + det_score + cluster id labels (green = clustered, orange = noise)
- 1,000-photo sample E2E: 1,610 faces across 554 images, 115 HDBSCAN clusters, rendered both HTML views
- HTTP server serving `/home/will/photo_project/output/` on `http://ares:8893/` (Python `http.server`, backgrounded as task `b1ama0333`)

**In progress at handoff time:**
- Full detection running in detached container `docker-face_detect-1` via `docker compose -f docker/docker-compose.face-detect.yml up -d`. Skipped the 554 already-processed sha256s from the sample run. 41,412 images queued (includes `.heic` files which cv2 logs as "can't open/read" and skips). At handoff time: **2,217 faces across 762 images**. Rate ~5 img/s → ETA ~2.3 hours from 2026-04-19 19:41 local.
- Persistent Monitor task `bysl21ii1` is watching `docker logs -f docker-face_detect-1` for "Done. Detected" / Traceback / Error — will notify on completion or failure.

## Next Steps

1. **Wait for the container to finish** (~2 hours). Confirm with `docker ps --filter name=face_detect` (should be gone or Exited (0)) and `docker logs docker-face_detect-1 | tail -5` (look for `Done. Detected N faces across M images.`).
2. **Re-cluster the full set**: `uv run scripts/face_clustering/cluster_faces.py` from the worktree. min_cluster_size=3, expect many more clusters than the 115 from the sample.
3. **Re-render both HTML views** against the full set:
   - `uv run scripts/face_clustering/build_review_html.py` → `output/face_clusters_review.html`
   - `uv run scripts/face_clustering/build_bbox_review_html.py` → `output/face_bboxes_review.html` (warning: with ~40K images at 800px JPEG ~80KB each this will be ~3 GB of HTML; may need to shard the page or down-sample images)
4. **Update the `index.html` stats block** at `output/index.html` so the landing page reflects the full numbers.
5. **Decide what to do with the HTTP server** (task `b1ama0333`). It's still running on ares:8893 serving the 1K sample output. Either replace files in place or restart after regen.
6. **Finish the branch**: all tests pass (33 passed, 7 FACE_SMOKE-skipped). Use `superpowers:finishing-a-development-branch` to pick merge / PR / keep — paused at that step when the user asked to run the sample end-to-end.

## Key Context

- **Worktree:** `/home/will/photo_project/.worktrees/face-clustering` (branch `feature/face-clustering`)
- **Plan file:** `docs/plans/2026-04-17-face-clustering.md` (Tasks 1-8 all implemented; self-review checklist at the bottom is fully satisfied)
- **Docker compose:** `docker/docker-compose.face-detect.yml` — mounts `originals/media:/media:ro` and a named `insightface_models` volume so buffalo_sc is only downloaded once
- **Symlink gotcha:** the sample run used symlinks from `/home/will/photo_project/face-sample/` to `originals/media/`. Symlinks don't resolve inside Docker unless the target path is also mounted. The one-off mount `-v /home/will/photo_project/originals/media:/home/will/photo_project/originals/media:ro` handled it. The full run uses the real `originals/media` mount directly, so this is not an issue now.
- **hdbscan was installed into `/home/will/photo_project/.venv`** via `uv pip install hdbscan` — it's a C-extension build (pulled scikit-learn + joblib + threadpoolctl). No pyproject.toml in this project (uses PEP 723 inline headers), so this is a per-venv install not pinned anywhere. If the venv is rebuilt, reinstall.
- **Prior handoff archived:** `docs/handoff/2026-04-17 Handoff - Face Clustering Pipeline.md` is still in `docs/handoff/` (not yet archived — user hadn't confirmed archival before pivoting to running the pipeline).
- **Detection quality on sample:** det_score 0.50–0.92, avg 0.74; buffalo_sc is the small model so some soft detections. The 65.6% noise rate on 1K is expected for random sampling with min_cluster_size=3 — clusters only form where faces recur.

## Files Touched

New files this session:
- `scripts/face_clustering/detect_all.py`
- `scripts/face_clustering/cluster_faces.py`
- `scripts/face_clustering/build_review_html.py`
- `scripts/face_clustering/build_bbox_review_html.py`
- `docker/Dockerfile.face-detect`
- `docker/docker-compose.face-detect.yml`
- `tests/test_face_clustering/test_detect_all.py`
- `tests/test_face_clustering/test_build_review_html.py`
- `tests/test_face_clustering/test_smoke_files.py`
- `output/index.html`, `output/face_clusters_review.html`, `output/face_bboxes_review.html` (regenerate after full detection)

Modified:
- `scripts/face_clustering/db.py` — executemany(returning=True) fix
- `tests/test_face_clustering/test_db.py` — mock `results()`/`fetchone()`
- `justfile` — four face-* recipes appended

Uncommitted: none.

## Blockers

None. Only wait time for the background detection.
