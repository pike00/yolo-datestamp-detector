---
title: Near-Duplicate Clustering — Run and Merge
status: active
repos: [photo_project]
started: 2026-04-23
last_updated: 2026-04-23
next_step: Render review HTML via `just near-dup-review`, serve from output/, then merge feature/near-dup-clustering to main.
---

# Near-Duplicate Clustering — Run and Merge

## Goal

Run the near-dup union-find clustering against the 57K photo+keyframe embeddings, review the duplicate groups, and merge `feature/near-dup-clustering` to main.

## Tasks

- [x] Enter worktree: `cd /home/will/photo_project/.worktrees/near-dup-clustering`
- [x] Verify Task 9 committed (run_cluster, review HTML — check `git log --oneline -5`)
- [x] Run dry-run smoke: `NEAR_DUP_SMOKE=1 uv run pytest tests/test_near_dup/test_smoke.py::test_run_cluster_dry_run_exits_cleanly -q`
- [x] Run full cluster: `just near-dup-cluster` (union-find over 57K embeddings, ~2–10 min on CPU)
- [ ] Render review HTML: `just near-dup-review` → `output/near_dup_review.html`
- [ ] Serve and review: `python -m http.server 8894 --bind 0.0.0.0` from output/
- [ ] Merge to main: `superpowers:finishing-a-development-branch`

## Session Log

### 2026-04-23 (afternoon)

- Loaded and archived both near-dup handoffs (Spec+Plan, Task 9).
- Entered worktree, verified all 10 tasks already committed on `feature/near-dup-clustering` (13 near-dup commits ahead of main; Tasks 9 and 10 landed in abf2d9d / 7f79639 / a9bd13a).
- Ran dry-run smoke test — pass (49s).
- Ran full smoke suite incl. `test_full_run_populates_photo_clusters` which executes the production orchestrator against all 57K embeddings with `run_id=smoke-full` — 4/4 pass (3m 53s). Unit suite 36/36.
- Clustering machinery confirmed working end-to-end on production-scale data. Next: render review HTML, visual check, then merge.

### 2026-04-23

- Project created.
- At creation: worktree at `.worktrees/near-dup-clustering`, branch `feature/near-dup-clustering`. Git log shows `a9bd13a near-dup: parallelize review HTML, cap at top-N clusters` as latest — Task 9 (run_cluster, Docker) and Task 10 (review HTML) are both committed. 36/36 unit tests passing.

## Notes

- **Handoff:** [docs/handoff/archive/2026-04-19 Handoff - Near-Dup Clustering Task 9.md](../../handoff/archive/2026-04-19%20Handoff%20-%20Near-Dup%20Clustering%20Task%209.md)
- **Plan:** [docs/plans/2026-04-17-near-dup-clustering.md](../../plans/2026-04-17-near-dup-clustering.md)
- **Spec:** (rewritten in-place by near-dup sessions; see plan file header)
- **Worktree:** `/home/will/photo_project/.worktrees/near-dup-clustering` (branch `feature/near-dup-clustering`)
- Algorithm: union-find over thresholded cosine-sim graph (threshold 0.98 default), NOT HDBSCAN. Deterministic, single hyperparameter.
- Files use `<sha256>.<ext>` naming (not bare hashes) — path index built at startup in `build_review_html.py`.
- Canonical pick: `(-pixel_count, exif_date ascending)` — higher resolution wins ties; earlier date wins equal resolution.
- Burst detection: same make+model, <2s gap, photos only. Post-hoc, not pre-collapsed.
- `photo_clusters` table already exists in DB (migration applied 2026-04-19).
