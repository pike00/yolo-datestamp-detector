---
summary: "Media embeddings pipeline (SigLIP) -- Tasks 1-8 done, Task 9 (smoke test) pending"
---

# Handoff: Media Embeddings Pipeline

**Date:** 2026-04-16
**Goal:** Build a SigLIP ViT-SO400M embedding pipeline for ~42K deduplicated photos and ~5.3K videos in `originals/media/`, storing 1152-dim vectors in Postgres as the foundation for all downstream photo ML (clustering, search, face grouping).

## Current Status

**Branch:** `feature/media-embeddings` (worktree at `.worktrees/media-embeddings`)

Tasks complete (reviewed + approved):
- ✅ Task 1: Migrated `dedup-postgres` to `pgvector/pgvector:pg18`, created `photo_embeddings` table + ivfflat index
- ✅ Task 2: Scaffolded `scripts/media_embeddings/` package + `tests/test_media_embeddings/`
- ✅ Task 3: `scan_media_dir()` — splits files by extension, sorted, ignores non-media
- ✅ Task 4: `open_image()` — Pillow for JPEG/PNG, pillow-heif for HEIC, always returns RGB
- ✅ Task 5: `extract_keyframes()` — ffprobe for duration, ffmpeg at 10%/50%/90%, guards empty duration
- ✅ Task 6: `get_completed_stems()`, `bulk_insert_embeddings()` — Postgres checkpoint + idempotent insert
- ✅ Task 7: `embed_all.py` — full pipeline, model.eval(), per-file error handling, tqdm logging
- ✅ Task 8: `docker/Dockerfile.media-embeddings` + `docker/docker-compose.media-embeddings.yml` (Docker build ran, files committed — verify build passed in Task 8 agent output)

Pending:
- ⏳ Task 9: Add `embed` / `embed-bg` justfile recipes, run smoke test (5 images), verify 5 rows at dim=1152 in Postgres

**Test count:** 14 tests passing in `tests/test_media_embeddings/` + 50 pre-existing = 64 total

## Next Steps

1. **Verify Task 8 agent completed** — check `git log` on the feature branch for "feat: add media_embeddings Dockerfile and compose" commit. If missing, create `docker/Dockerfile.media-embeddings` and `docker/docker-compose.media-embeddings.yml` per plan.
2. **Task 9: Add justfile recipes** — append to `justfile` in worktree:
   ```makefile
   embed:
       docker compose -f docker/docker-compose.media-embeddings.yml up --build
   embed-bg:
       docker compose -f docker/docker-compose.media-embeddings.yml up --build -d
       @echo "Logs: docker logs -f photo_project-media_embeddings-1"
   ```
3. **Task 9: Run smoke test** — copy 5 JPGs to `/tmp/embed_smoke`, run `docker compose ... run --rm -e MEDIA_DIR=/smoke -v /tmp/embed_smoke:/smoke:ro media_embeddings`, verify 5 rows at dim=1152 in `photo_embeddings`
4. **Run full test suite** — `uv run pytest tests/ -q` from worktree, expect 64 passed
5. **Merge branch** — use `superpowers:finishing-a-development-branch` skill to decide merge strategy
6. **Start overnight run** — `docker start dedup-postgres && just embed-bg`
7. **Post-run** — `VACUUM ANALYZE photo_embeddings` after bulk insert for IVFFlat index to work

## Key Context

- **Worktree:** `.worktrees/media-embeddings` on `feature/media-embeddings` branch
- **Postgres:** `dedup-postgres` container now uses `pgvector/pgvector:pg18`; compose file at `dedup/docker-compose.yml`; creds `dedup/dedup_local_dev/dedup`; starts with `docker start dedup-postgres`
- **Missing stamp_* tables:** Pre-existing issue unrelated to this work — `stamp_predictions`, `stamp_ocr` etc. were never provisioned in this container. Separate concern.
- **Model:** `google/siglip-so400m-patch14-384`, downloads ~1.8 GB to `hf_cache` Docker volume on first run (one-time)
- **Runtime estimate:** ~3-4 hours on 12-core Ryzen CPU for full 42K images + 5.3K videos
- **VACUUM ANALYZE:** Run after full batch to let query planner use the ivfflat index for similarity search
- **Broader roadmap:** This is Spec 1 of 5 — embeddings → near-dup clustering → face clustering → scene tagging → search UI. See `docs/handoff/2026-04-16-media-embeddings-spec.md` and `docs/handoff/2026-04-16-media-embeddings-plan.md`

## Files Touched

| File | Status |
|---|---|
| `dedup/docker-compose.yml` | Created — pgvector postgres compose |
| `scripts/media_embeddings/__init__.py` | Created |
| `scripts/media_embeddings/loader.py` | Created — file scan, image open, video keyframes |
| `scripts/media_embeddings/db.py` | Created — Postgres checkpoint helpers |
| `scripts/media_embeddings/embed_all.py` | Created — main pipeline script |
| `scripts/media_embeddings/requirements.txt` | Created — Docker pip deps |
| `docker/Dockerfile.media-embeddings` | Created |
| `docker/docker-compose.media-embeddings.yml` | Created |
| `tests/test_media_embeddings/conftest.py` | Created |
| `tests/test_media_embeddings/test_loader.py` | Created — 10 tests |
| `tests/test_media_embeddings/test_db.py` | Created — 4 tests |
| `justfile` | Pending — needs embed/embed-bg recipes |
| `docs/handoff/2026-04-16-media-embeddings-spec.md` | Created — design spec |
| `docs/handoff/2026-04-16-media-embeddings-plan.md` | Created — implementation plan |

## Blockers

- None blocking. Task 9 (smoke test) requires the Docker image to be built first — check Task 8 build succeeded before running smoke test.
