---
title: Zero-Shot Scene Tagging
status: active
repos: [photo_project]
started: 2026-04-23
last_updated: 2026-04-23
next_step: Create photo_tags schema migration and apply it; implement scene_tagging package (Tasks 1–2 in plan)
---

# Zero-Shot Scene Tagging

## Goal

Assign multi-label scene/theme tags (beach, birthday party, christmas, etc.) to all 42K photos by scoring pre-existing SigLIP embeddings against 40 text prompts. No image re-processing needed — it's a batched matrix multiply on stored vectors.

## Tasks

- [ ] Task 1: DB schema — create `photo_tags` table + indexes; write + run migration
- [ ] Task 2: Package scaffold — `scripts/scene_tagging/__init__.py`, `labels.py`, conftest
- [ ] Task 3: Label encoder — `encode_labels`, `score_photo` in `tagger.py`
- [ ] Task 4: DB helpers — `get_untagged_stems`, `bulk_insert_tags` in `db.py`
- [ ] Task 5: Orchestration — `tag_all.py` main script (batch=1000, threshold=0.2)
- [ ] Task 6: Justfile recipe `tag-scenes` + integration smoke test (10 real photos)
- [ ] Run `just tag-scenes` to tag all 42K photos; verify label distribution makes sense
- [ ] Commit all files

## Session Log

### 2026-04-23

- Project created.
- Full implementation plan exists at docs/plans/2026-04-17-scene-tagging.md with complete TDD specs for all 6 tasks including exact test and implementation code.

## Notes

- **Plan:** [docs/plans/2026-04-17-scene-tagging.md](../../plans/2026-04-17-scene-tagging.md)
- Architecture: SigLIP text tower encodes all 40 labels into 1152-dim space; matrix-multiply over 42K stored vectors; labels above 0.2 cosine sim written to `photo_tags`.
- Prerequisite for search-ui tag-browse feature.
- SigLIP model: `google/siglip-so400m-patch14-384` (already cached in HuggingFace local cache from embedding pipeline).
- Threshold 0.2 is intentionally loose — spot-check DB output to validate label quality before fleet tagging.
- Uses same `psycopg[binary]` + pgvector pattern as the rest of the pipeline.
