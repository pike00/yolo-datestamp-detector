---
title: Semantic Photo Search UI
status: active
repos: [photo_project]
started: 2026-04-23
last_updated: 2026-04-23
next_step: Implement after scene-tagging is complete (tag data needed for browse sidebar)
---

# Semantic Photo Search UI

## Goal

Browser UI where the user types a text query or clicks a photo to find visually similar photos. FastAPI on port 8890 with SigLIP text encoding, ANN search via pgvector, tag browse sidebar, and lazy-loading photo grid.

## Tasks

- [ ] Task 1: Package scaffold + TestClient conftest (`scripts/search_ui/`, `tests/test_search_ui/`)
- [ ] Task 2: DB helpers — `ann_search`, `fetch_tags`, `fetch_by_tag`, `fetch_embedding` in `db.py`
- [ ] Task 3: SigLIP text encoder wrapper — `SigLIPTextEncoder` in `model.py`
- [ ] Task 4: FastAPI app + photo serving endpoint (`/photo/{sha256}` thumbnail at 800px)
- [ ] Task 5: Text search endpoint + tests (`GET /search?q=`)
- [ ] Task 6: Similar-image endpoint + tests (`GET /similar/{sha256}`)
- [ ] Task 7: Tag browse endpoints + tests (`GET /tags`, `GET /by-tag`)
- [ ] Task 8: Frontend HTML — dark theme, tag sidebar, results grid, lightbox (`ui/search.html`)
- [ ] Task 9: Docker + Compose + justfile recipes (`just search`, `just search-bg`)
- [ ] Smoke test: launch via `just search-bg`, verify tags + search endpoints respond

## Session Log

### 2026-04-23

- Project created.
- Full implementation plan exists at docs/plans/2026-04-17-search-ui.md with complete TDD specs for all 9 tasks.

## Notes

- **Plan:** [docs/plans/2026-04-17-search-ui.md](../../plans/2026-04-17-search-ui.md)
- **Dependency:** scene-tagging project must be complete first (tag data populates the sidebar)
- Port 8890; no conflict with annotate :8888 or corrections :8889
- Tech stack: FastAPI + uvicorn, psycopg[binary], pgvector, Pillow, transformers (SigLIP text tower only), torch CPU, vanilla JS frontend
- Docker image uses host networking + read-only media mount
- SigLIP model loaded from local HuggingFace cache (already downloaded by embedding pipeline)
- No external API calls — fully local inference
