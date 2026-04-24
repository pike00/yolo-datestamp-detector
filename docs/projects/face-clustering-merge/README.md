---
title: Face Clustering — Finalize and Merge
status: active
repos: [photo_project]
started: 2026-04-23
last_updated: 2026-04-24
next_step: Wait for cluster_faces.py (PID 1481393, started 07:27) to finish, then render face grid + sharded bbox HTML
---

# Face Clustering — Finalize and Merge

## Goal

Complete the post-detection steps after the full 33K-photo insightface detection run, then merge the `feature/face-clustering` branch to main.

## Tasks

- [x] Verify detection container finished: `docker ps --filter name=face_detect` + `docker logs docker-face_detect-1 | tail -5` (exited 0; 54,796 detections logged → 56,406 face_detections rows over 18,392 distinct images)
- [ ] Re-cluster the full detected set — **in flight** as PID 1481393, launched via main `.venv/bin/python` with `PYTHONPATH=worktree/scripts` (no pyproject in worktree, `uv run` not viable)
- [ ] Re-render face grid HTML: `PYTHONPATH=…/scripts python scripts/face_clustering/build_review_html.py`
- [ ] Re-render bbox overlay HTML: shard or down-sample (18,392 images-with-faces × ~100 KB base64 ≈ 1.8 GB unsharded)
- [ ] Update `output/index.html` stats block with full numbers (currently shows the 1K-sample numbers from 2026-04-19)
- [x] Decide HTTP server fate for ares:8893 — port doesn't exist; actual review server is **:8894** (PID 2954314, `python3 -m http.server --bind 0.0.0.0 8894 --directory output`). Keep it; renderers overwrite the same paths.
- [ ] Merge or PR: run `superpowers:finishing-a-development-branch` for the branch decision (all 33 tests pass, 7 FACE_SMOKE-skipped)

## Session Log

### 2026-04-24

- Detection container confirmed done — `docker-face_detect-1` exited 0 four days ago after 1h 38m, logged "Detected 54796 faces across 41412 images"; DB shows 56,406 detections across 18,392 distinct sha256s (delta vs log = prior partial runs that re-ran).
- Old `face_clusters` carried 554 rows / 115 clusters from the 1K-sample run — being replaced by the in-flight re-cluster.
- Kicked off `cluster_faces.py` against all 56,406 embeddings (HDBSCAN, min_cluster_size=3); running as PID 1481393 from `/home/will/photo_project` using main repo's `.venv/bin/python` + `PYTHONPATH=worktree/scripts`. Logs at `/tmp/face-cluster-logs/cluster.log`.
- Resolved port confusion: README's "ares:8893" doesn't exist; live face-review server is on **:8894** serving `output/`.
- HTML re-renders are blocked on the cluster job; once it finishes the grid HTML can run unmodified, but bbox HTML needs sharding/down-sampling before render.

### 2026-04-23

- Project created.
- At creation: worktree at `.worktrees/face-clustering`, branch `feature/face-clustering`, 10 commits ahead of main. Full 33K detection was running in `docker-face_detect-1` as of 2026-04-19 handoff; container status unknown now.

## Notes

### 2026-04-24

- **Decisions:** Run clustering with main `.venv/bin/python` + `PYTHONPATH` rather than create a worktree-local venv — repo has no `pyproject.toml`, only per-script `requirements.txt`, and main `.venv` already has hdbscan/psycopg/pgvector.
- **Gotchas:** `face_detections.sha256` is the distinct image set with ≥1 face (18,392), not the processed image count (41,412); ~55% of processed photos had no face. Bbox renderer uses inline base64 — sharding alone still yields ~50 MB shards if page size 500. SVG-overlay-on-served-image would shrink HTML 100× but needs a renderer rewrite.
- **Issues:** Cluster job still running at save time (PID 1481393, ~3 min in, single-core ~99%). Background watcher `bq0vee32r` will fire on completion. Until then, no decisions can be made on cluster count or noise rate.

- **Handoff:** [docs/handoff/2026-04-19 Handoff - Face Clustering Pipeline E2E.md](../../handoff/2026-04-19%20Handoff%20-%20Face%20Clustering%20Pipeline%20E2E.md)
- **Plan:** [docs/plans/2026-04-17-face-clustering.md](../../plans/2026-04-17-face-clustering.md)
- **Worktree:** `/home/will/photo_project/.worktrees/face-clustering` (branch `feature/face-clustering`)
- Detection rate was ~5 img/s with ETA ~2.3h from 2026-04-19 19:41; container has almost certainly finished.
- `hdbscan` must be installed in `.venv` — C-extension build, not in pyproject.toml. Reinstall if venv rebuilt: `uv pip install hdbscan`.
- bbox HTML with 40K full-size photos will be huge — plan is to shard or down-sample the bbox overlay view.
- insightface only runs inside Docker; buffalo_sc model weights cached in `insightface_models` named volume.
