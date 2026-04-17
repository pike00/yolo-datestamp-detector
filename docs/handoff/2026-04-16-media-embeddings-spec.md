# Spec: Media Embeddings Pipeline (Phase 1 of Photo ML)

**Date:** 2026-04-16  
**Status:** Approved, ready for implementation planning  
**Scope:** Spec 1 of 5 in the photo ML roadmap

---

## Context

The photo collection contains ~42K deduplicated still images + 5.3K videos in `originals/media/` (sha256-named files, already deduped). This spec covers computing and storing semantic embedding vectors for every file — the foundation all downstream ML work (clustering, search, face grouping, auto-tagging) depends on.

### Broader roadmap (future specs, not in scope here)
1. **This spec** — embedding extraction
2. Near-duplicate / best-of clustering
3. Face detection + face clustering
4. Object/scene/theme tagging (zero-shot CLIP labels or captioner)
5. Search + browse UI

---

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Model | SigLIP ViT-SO400M | Best open model for image similarity; better zero-shot than CLIP L/14 |
| Embedding dim | 1152 | SigLIP SO400M output size |
| Compute | Local CPU overnight | Upload bottleneck (~2 hrs) erases GPU wall-time advantage; $0 cost |
| Resume strategy | Postgres checkpoint | Skip any sha256 already in table; safe to re-run |
| Container | Docker (`docker-compose.media-embeddings.yml`) | Consistent with existing OCR pipeline pattern |
| Model cache | Named Docker volume `hf_cache` | ~1.8 GB download once, persists across runs |

---

## Architecture

Single script: `scripts/media_embeddings/embed_all.py`

```
originals/media/ (read-only mount)
        |
        v
  embed_all.py
  ├── startup: load SigLIP, connect Postgres, fetch existing sha256 set
  ├── scan: glob all files → images list + videos list
  ├── image loop (batch=64)
  │   ├── open: Pillow (JPG/PNG/BMP/TIF) or pillow-heif (HEIC)
  │   ├── resize to 384px preserving aspect
  │   ├── SigLIP → 1152-dim vector
  │   └── bulk insert to photo_embeddings
  └── video loop (batch=16)
      ├── ffprobe → duration
      ├── ffmpeg extract 3 keyframes at 10%/50%/90%
      ├── SigLIP → 1152-dim vector per frame
      └── bulk insert with frame_index 0/1/2
```

---

## Data Model

New table in existing `dedup` Postgres database:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE photo_embeddings (
    sha256      text NOT NULL,
    model       text NOT NULL DEFAULT 'siglip-so400m',
    embedding   vector(1152) NOT NULL,
    media_type  text NOT NULL,          -- 'photo' | 'video_keyframe'
    frame_index integer,                -- NULL for photos, 0/1/2 for video
    created_at  timestamptz DEFAULT now(),
    PRIMARY KEY (sha256, model, COALESCE(frame_index, -1))
);

CREATE INDEX ON photo_embeddings USING ivfflat (embedding vector_cosine_ops);
```

`sha256` matches the filename stem in `originals/media/` and links to existing dedup tables.

## Logging

Progress logged to stdout via `tqdm` — visible with `docker logs -f media_embeddings`. Each batch logs:
- Files processed / total
- Images per second
- ETA remaining
- Any per-file errors (logged but non-fatal, script continues)

---

## Container

```
docker/media_embeddings/
└── Dockerfile

docker-compose.media-embeddings.yml
```

**Dockerfile:**
- Base: `python:3.12-slim`
- System deps: `ffmpeg`, `libheif-dev`
- Python deps: `transformers`, `torch` (CPU), `pillow`, `pillow-heif`, `psycopg[binary]`, `pgvector`, `tqdm`
- Entrypoint: `python scripts/media_embeddings/embed_all.py`

**docker-compose.media-embeddings.yml:**
```yaml
services:
  media_embeddings:
    build: docker/media_embeddings
    volumes:
      - /home/will/photo_project/originals/media:/media:ro
      - hf_cache:/root/.cache/huggingface
      - ./scripts:/app/scripts:ro
    environment:
      - DATABASE_URL=postgresql://dedup:dedup_local_dev@host.docker.internal:5432/dedup
    network_mode: host
volumes:
  hf_cache:
```

**Run:**
```bash
docker compose -f docker-compose.media-embeddings.yml up --build
```

Fully re-runnable. Postgres checkpoint ensures resume from last completed batch.

---

## Runtime Estimate

| File type | Count | Batch size | Est. time (CPU) |
|---|---|---|---|
| JPG/JPEG | 35,201 | 64 | ~1.95 hrs |
| HEIC | 6,253 | 64 | ~0.35 hrs |
| PNG/BMP/TIF/GIF | 533 | 64 | ~0.03 hrs |
| Videos (3 frames each) | 5,296 × 3 = 15,888 frames | 16 | ~0.88 hrs |
| **Total** | **~57,875 embeddings** | | **~3.2 hrs** |

---

## Files to Create

| File | Purpose |
|---|---|
| `scripts/media_embeddings/embed_all.py` | Main embedding script |
| `docker/media_embeddings/Dockerfile` | Container definition |
| `docker-compose.media-embeddings.yml` | Compose file |

### Postgres migration
- The existing `dedup-postgres` container uses standard `postgres:15` which does NOT include pgvector
- Switch the `dedup-postgres` image to `pgvector/pgvector:pg15` (drop-in replacement, same data volume)
- Run `CREATE EXTENSION IF NOT EXISTS vector` + table DDL before first run

---

## Pre-flight Checklist

- [ ] Confirm `dedup-postgres` container supports pgvector (`SELECT * FROM pg_available_extensions WHERE name = 'vector'`)
- [ ] Confirm `originals/media/` path is accessible and file count matches (~47K)
- [ ] Confirm disk space for hf_cache volume (~2 GB for model weights)
- [ ] Start `dedup-postgres` before running (`docker start dedup-postgres`)
