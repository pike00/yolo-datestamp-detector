---
title: Face Clustering — Finalize and Merge
status: active
repos: [photo_project]
started: 2026-04-23
last_updated: 2026-04-23
next_step: Check docker-face_detect-1 container status; if done, re-cluster full set and re-render HTML
---

# Face Clustering — Finalize and Merge

## Goal

Complete the post-detection steps after the full 33K-photo insightface detection run, then merge the `feature/face-clustering` branch to main.

## Tasks

- [ ] Verify detection container finished: `docker ps --filter name=face_detect` + `docker logs docker-face_detect-1 | tail -5`
- [ ] Re-cluster the full detected set: `uv run scripts/face_clustering/cluster_faces.py` (from worktree)
- [ ] Re-render face grid HTML: `uv run scripts/face_clustering/build_review_html.py`
- [ ] Re-render bbox overlay HTML: consider sharding or down-sampling (40K images at 80KB = ~3 GB if unsharded)
- [ ] Update `output/index.html` stats block with full numbers
- [ ] Decide HTTP server fate for ares:8893 (kill or replace in place)
- [ ] Merge or PR: run `superpowers:finishing-a-development-branch` for the branch decision (all 33 tests pass, 7 FACE_SMOKE-skipped)

## Session Log

### 2026-04-23

- Project created.
- At creation: worktree at `.worktrees/face-clustering`, branch `feature/face-clustering`, 10 commits ahead of main. Full 33K detection was running in `docker-face_detect-1` as of 2026-04-19 handoff; container status unknown now.

## Notes

- **Handoff:** [docs/handoff/2026-04-19 Handoff - Face Clustering Pipeline E2E.md](../../handoff/2026-04-19%20Handoff%20-%20Face%20Clustering%20Pipeline%20E2E.md)
- **Plan:** [docs/plans/2026-04-17-face-clustering.md](../../plans/2026-04-17-face-clustering.md)
- **Worktree:** `/home/will/photo_project/.worktrees/face-clustering` (branch `feature/face-clustering`)
- Detection rate was ~5 img/s with ETA ~2.3h from 2026-04-19 19:41; container has almost certainly finished.
- `hdbscan` must be installed in `.venv` — C-extension build, not in pyproject.toml. Reinstall if venv rebuilt: `uv pip install hdbscan`.
- bbox HTML with 40K full-size photos will be huge — plan is to shard or down-sample the bbox overlay view.
- insightface only runs inside Docker; buffalo_sc model weights cached in `insightface_models` named volume.
