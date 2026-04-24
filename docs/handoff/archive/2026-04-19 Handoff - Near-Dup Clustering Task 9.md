---
summary: "Near-dup clustering 8/10 committed, Task 9 files written but uncommitted, dry-run smoke unverified"
---

# Handoff: Near-Duplicate Clustering — Task 9 in Progress

**Date:** 2026-04-19
**Goal:** Execute the near-dup clustering implementation plan task by task on the `feature/near-dup-clustering` worktree.

## Current Status

8 of 10 tasks committed, Task 9 implemented on disk but **not committed** and **not verified**.

### Committed commits (on `feature/near-dup-clustering`, 9 ahead of `main`)

```
8e0514b near-dup: add write_cluster_rows and read_cluster_groups
02ee40c near-dup: add burst detection
999af34 near-dup: add canonical picker
019ea92 near-dup: add image metadata reader
49fa26b near-dup: add find_pairs, union_find, build_cluster_map
91e4839 near-dup: add fetch_all_embeddings loader
222f053 near-dup: scaffold package and shared fixtures
465b2df near-dup: add photo_clusters migration and smoke test skeleton
fd65518 docs: add near-dup clustering spec + rewritten plan
```

Unit suite status after Task 8: **36/36 passed** (`uv run pytest tests/test_near_dup -q --ignore=tests/test_near_dup/test_smoke.py`).

### Uncommitted (Task 9 in progress)

```
M  justfile                                  # added near-dup-cluster, -bg, -review recipes
M  tests/test_near_dup/test_smoke.py         # appended dry-run and full-run smoke tests
?? docker/docker-compose.near-dup.yml
?? docker/near_dup/                          # Dockerfile + requirements.txt
?? scripts/near_dup/run_cluster.py           # main orchestrator
```

**Session interrupted immediately before running the dry-run smoke test** to confirm `run_cluster.py` works end-to-end. That test never ran this session.

## Next Steps

1. **Re-enter the worktree:**
   ```bash
   cd /home/will/photo_project/.worktrees/near-dup-clustering
   ```
2. **Run the dry-run smoke test** (should take <30 s — it fetches 57K embeddings, does the matmul, and exits before enrichment):
   ```bash
   NEAR_DUP_SMOKE=1 uv run pytest tests/test_near_dup/test_smoke.py::test_run_cluster_dry_run_exits_cleanly -q
   ```
   If it fails, likely causes: `host.docker.internal` resolution (not relevant for this host-side run since it's going through localhost), or a missing import. The test asserts the word "components" or "cluster" appears somewhere in combined stdout+stderr.
3. **Commit Task 9:**
   ```bash
   git add scripts/near_dup/run_cluster.py docker/near_dup docker/docker-compose.near-dup.yml justfile tests/test_near_dup/test_smoke.py
   git commit -m "near-dup: add run_cluster orchestrator, docker compose, justfile recipes"
   ```
4. **Check face-detect container status before Task 10 full smoke:**
   ```bash
   docker ps --filter name=face_detect
   ```
   As of the last check it was "Up About an hour" competing for CPU. Task 10's full smoke runs a ~2-minute CPU-heavy matmul over 57K × 1152 — don't start while face-detect is churning.
5. **Task 10 — review HTML + full smoke.** The plan for this task is [docs/plans/2026-04-19-near-dup-clustering.md](docs/plans/2026-04-19-near-dup-clustering.md) §Task 10. Write `scripts/near_dup/build_review_html.py` (spec already in the plan), run the full-run smoke test, sanity-check by serving `output/near_dup_review.html` from ares.

## Key Context

### Divergences from the committed plan found during execution

Two came up; both are fixed in the code on disk AND the plan source:

1. **Postgres rejects `COALESCE(...)` inside `PRIMARY KEY`.** The plan's Task 1 SQL failed. Mirrored `photo_embeddings` instead: no PK, a `CREATE UNIQUE INDEX photo_clusters_pk ON photo_clusters (run_id, cluster_id, sha256, COALESCE(frame_index, -1))` expression index. Plan updated in commit `465b2df`.
2. **Clustering test fixture noise too loose.** At `sigma=0.005` two perturbed normalized vectors land around cosine sim 0.972, below the 0.98 threshold, so the `test_find_pairs_detects_known_duplicate_groups` test failed. Lowered to `0.001` (sim lands ≈ 0.998). Plan updated in commit `49fa26b`.
3. **Burst `different_camera_splits_bursts` test was a `None != None` false-positive.** Extended the fixture to 2 photos per camera so both form real bursts with distinct IDs. Fix in commit `02ee40c`.

### Environment facts

- **Worktree:** `/home/will/photo_project/.worktrees/near-dup-clustering` on branch `feature/near-dup-clustering` (exists, 9 commits ahead of main).
- **Main branch state:** still diverged from `origin/main` (17 ahead, 1 diverged) — unresolved, not blocking this feature.
- **Postgres:** `photo_clusters` table exists with correct schema and indexes. `photo_embeddings` has all 57,875 rows.
- **face-detect container:** still running (`docker-face_detect-1`, "Up About an hour" at last check). Blocks Task 10 full-run smoke but nothing else.
- **Dependencies:** all scripts use PEP 723 inline headers. No `pyproject.toml` changes. `hdbscan` is installed in `.venv` from face-clustering but is not used here.

### Design decisions (reference — full detail in the spec)

- Algorithm: union-find over thresholded cosine-sim graph (not HDBSCAN)
- Scope: photos + video keyframes (57,875 rows)
- Threshold: 0.98 default, `--threshold` CLI arg, stored per-row; `run_id` lets sweeps coexist
- Canonical pick: `(-pixel_count, exif_date ascending)`; NULLs sort last
- Burst: same `exif_make`+`exif_model`, `<2s` gap, `media_type='photo'` only

## Files Touched This Session

**Committed (on the feature branch, not on main):**
- `tests/test_near_dup/test_smoke.py` (first version — 2 schema tests only)
- `tests/test_near_dup/conftest.py`
- `tests/test_near_dup/test_conftest_fixtures.py`
- `tests/test_near_dup/test_embeddings.py`
- `tests/test_near_dup/test_clustering.py`
- `tests/test_near_dup/test_metadata.py`
- `tests/test_near_dup/test_canonical.py`
- `tests/test_near_dup/test_burst.py`
- `tests/test_near_dup/test_db.py`
- `scripts/near_dup/__init__.py`
- `scripts/near_dup/embeddings.py`
- `scripts/near_dup/clustering.py`
- `scripts/near_dup/metadata.py`
- `scripts/near_dup/canonical.py`
- `scripts/near_dup/burst.py`
- `scripts/near_dup/db.py`
- Plan file edits (SQL fix, fixture sigma, burst test fix)

**Uncommitted (Task 9 on-disk):**
- `scripts/near_dup/run_cluster.py`
- `docker/near_dup/Dockerfile`
- `docker/near_dup/requirements.txt`
- `docker/docker-compose.near-dup.yml`
- `justfile` (appended 3 recipes)
- `tests/test_near_dup/test_smoke.py` (appended 2 subprocess-based smoke tests)

**Still to write (Task 10):**
- `scripts/near_dup/build_review_html.py`

## Blockers

- **Task 10 full-run smoke needs face-detect container to be idle** to avoid CPU thrash. Verify with `docker ps --filter name=face_detect` before running.
- **No active blockers for Task 9 completion** — just needs the dry-run test to pass and a commit.
