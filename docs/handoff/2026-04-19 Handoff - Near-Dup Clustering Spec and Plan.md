---
summary: "Near-dup clustering spec + plan authored -- 0/10 tasks implemented, face-detect container blocks full smoke"
---

# Handoff: Near-Duplicate Clustering Spec and Plan

**Date:** 2026-04-19
**Goal:** Author Spec 2 of 5 (near-duplicate clustering over the 57K SigLIP embeddings) and a TDD-structured implementation plan, ready for a future session to execute task-by-task.

## Current Status

Spec and plan authored and committed on a new worktree. Zero implementation code yet — pure docs session.

- **Worktree:** `/home/will/photo_project/.worktrees/near-dup-clustering` (branch `feature/near-dup-clustering`), 1 commit ahead of `main`.
- **Commit:** `fd65518 docs: add near-dup clustering spec + rewritten plan` — adds spec + new plan + deletes stale 2026-04-17 plan in one atomic change.
- **DB state:** `photo_clusters` table does NOT exist yet (Task 1 creates it). Existing `photo_embeddings` has the full 57,875 rows (42K photos + 5.3K videos × 3 keyframes).

The session started from the stale [docs/plans/2026-04-17-near-dup-clustering.md](docs/plans/2026-04-17-near-dup-clustering.md) (HDBSCAN + Laplacian quality + photos only + hardcoded threshold) and replaced it with a fresh spec-driven plan after scoping. Key divergences documented in the commit message.

## Next Steps

1. **Wait for face-detect container to finish** before running Task 10's full smoke test — `docker ps --filter name=face_detect` should be empty or Exited(0). The spec notes this in its Pre-flight Checklist. Tasks 1–9 (schema, modules, dry run) are safe to run concurrently since they don't touch CPU heavily.
2. **Execute the plan.** From the worktree root:
   ```bash
   cd /home/will/photo_project/.worktrees/near-dup-clustering
   # Option A: subagent-driven
   # Invoke superpowers:subagent-driven-development with docs/plans/2026-04-19-near-dup-clustering.md
   # Option B: manual task-by-task, following the TDD rhythm (test → fail → implement → pass → commit)
   ```
3. **Reconcile main branch divergence separately.** `main` is 17 commits ahead of `origin/main` and 1 diverged — not blocking near-dup work, but worth investigating before any push. Likely a rebase/cherry-pick that hasn't been pushed. Check with `git log --oneline origin/main..main` and `git log --oneline main..origin/main`.
4. **After implementation**, use `superpowers:finishing-a-development-branch` to decide merge vs PR vs keep-alive.

## Key Context

### Approved design decisions (from scoping conversation)

| Dimension | Choice |
|---|---|
| Algorithm | Union-find over thresholded cosine-sim graph (not HDBSCAN) — deterministic, transitive, one hyperparameter |
| Scope | Photos + video keyframes (all 57,875 rows, single SigLIP space) |
| Threshold | 0.98 cosine sim default, `--threshold` CLI arg, stored per-row |
| Canonical pick | Sort by `(-pixel_count, exif_date ascending)` — highest res first, earliest date tiebreak, NULL last |
| Burst flagging | Same EXIF Make+Model + <2 s gap + `media_type='photo'` |
| Storage | New `photo_clusters` table with `run_id` so threshold sweeps coexist |
| Dep convention | PEP 723 inline headers — project has no `pyproject.toml` for deps |
| Container pattern | Mirrors face_detect + media_embeddings: pinned requirements.txt, `network_mode: host`, `host.docker.internal` DSN |

### Schema (Task 1 applies this)

```sql
CREATE TABLE photo_clusters (
    cluster_id         int  NOT NULL,
    sha256             text NOT NULL,
    frame_index        int,
    media_type         text NOT NULL,
    is_representative  bool NOT NULL DEFAULT false,
    width              int,
    height             int,
    exif_date          timestamptz,
    exif_make          text,
    exif_model         text,
    burst_id           int,
    cluster_threshold  real NOT NULL,
    run_id             text NOT NULL,
    created_at         timestamptz DEFAULT now(),
    PRIMARY KEY (run_id, cluster_id, sha256, COALESCE(frame_index, -1))
);
```

### Gotchas

- **PEP 723 not `uv add`.** Project has no pyproject.toml for deps — the stale plan's `uv add hdbscan` step would not work. All new scripts declare deps in their inline header (mirror `scripts/media_embeddings/embed_all.py`).
- **hdbscan already installed** into `.venv` from the face-clustering session. Not needed for near-dup (we use union-find) but FYI if anyone reads the old plan expecting the dep to be missing.
- **EXIF DateTimeOriginal lives in the ExifIFD sub-directory** (tag 0x9003 under parent tag 0x8769), not the top-level IFD0. `metadata.py` in Task 5 uses `exif.get_ifd(0x8769)` then falls back to 0x0132 (DateTime) in IFD0. The test fixture `image_with_exif` sets both so tests are robust.
- **Video keyframes get NULL width/height** and sort last in `pick_representative` by construction — no special-casing needed.
- **Handoff-table divergence:** session start showed the 2026-04-17 Face Clustering handoff in the table at top; a newer 2026-04-19 E2E handoff already exists. This new near-dup handoff joins the pair.

## Files Touched

**Added (committed on `feature/near-dup-clustering`):**
- [docs/specs/2026-04-19-near-dup-clustering-spec.md](.worktrees/near-dup-clustering/docs/specs/2026-04-19-near-dup-clustering-spec.md) — 165 lines, context + decisions + architecture + runtime estimate + non-goals
- [docs/plans/2026-04-19-near-dup-clustering.md](.worktrees/near-dup-clustering/docs/plans/2026-04-19-near-dup-clustering.md) — 10 TDD tasks, each with failing-test → verify-fail → implement → verify-pass → commit rhythm

**Deleted (same commit):**
- `docs/plans/2026-04-17-near-dup-clustering.md` — stale draft that predated scoping

**Not yet created** (scheduled for Task 1–10 of the plan):
- `scripts/near_dup/` — `embeddings.py`, `clustering.py`, `metadata.py`, `canonical.py`, `burst.py`, `db.py`, `run_cluster.py`, `build_review_html.py`
- `docker/near_dup/Dockerfile`, `docker/near_dup/requirements.txt`
- `docker/docker-compose.near-dup.yml`
- `tests/test_near_dup/*`
- `justfile` recipes (`near-dup-cluster`, `near-dup-cluster-bg`, `near-dup-review`)

## Blockers

- **CPU contention with face-detect container.** The background `docker-face_detect-1` is CPU-bound and expected to run ~2 hours from 19:41 on 2026-04-19 (≈ through 21:45). Task 10's full smoke test runs a ~2-minute CPU-heavy matmul; running it while face-detect is active will thrash both. Tasks 1–9 are unaffected. Verify container state with `docker ps --filter name=face_detect` before Task 10.
- **`main` branch diverged from `origin/main`** (17 ahead, 1 diverged). Not blocking this feature, but should be reconciled before any push/PR. Investigate out-of-band.

## Handoff-table bookkeeping

After loading this handoff in a future session, the prior 2026-04-17 Face Clustering handoff is safe to archive — it was superseded by the 2026-04-19 E2E handoff on the same feature. The user hadn't confirmed archival, so the `/handoff-load` flow should surface it.
