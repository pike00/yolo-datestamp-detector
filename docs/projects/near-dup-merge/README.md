---
title: Near-Duplicate Clustering — Run and Merge
status: active
repos: [photo_project]
started: 2026-04-23
last_updated: 2026-04-23
next_step: Enter worktree, commit Task 9 (run_cluster + review HTML already on disk), run dry-run smoke, then full cluster run
---

# Near-Duplicate Clustering — Run and Merge

## Goal

Run the near-dup union-find clustering against the 57K photo+keyframe embeddings, review the duplicate groups, and merge `feature/near-dup-clustering` to main.

## Tasks

- [ ] Enter worktree: `cd /home/will/photo_project/.worktrees/near-dup-clustering`
- [ ] Verify Task 9 committed (run_cluster, review HTML — check `git log --oneline -5`)
- [ ] Run dry-run smoke: `NEAR_DUP_SMOKE=1 uv run pytest tests/test_near_dup/test_smoke.py::test_run_cluster_dry_run_exits_cleanly -q`
- [ ] Run full cluster: `just near-dup-cluster` (union-find over 57K embeddings, ~2–10 min on CPU)
- [ ] Render review HTML: `just near-dup-review` → `output/near_dup_review.html`
- [ ] Serve and review: `python -m http.server 8894 --bind 0.0.0.0` from output/
- [ ] Merge to main: `superpowers:finishing-a-development-branch`

## Session Log

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
