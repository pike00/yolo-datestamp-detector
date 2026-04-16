# Photo Project

## Overview
Consolidating ~77K media files (467 GB) into deduplicated, organized, metadata-enriched collection.
Current focus: extracting date stamps from ~7,500 scanned 4x6 photos (ScanMyPhotos Discs 1-4).

## Critical Constraints
- **NEVER modify files on HDD** at `/mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/` — read-only source
- **No external API calls** (Gemini, OpenAI, Anthropic) without explicit user approval
- Local LLM via Ollama is OK if installed

## Source Photo Paths
- Disc 1: `/mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/Photos/img/Photos/ScanMyPhotos/Disc 1/` (1,775 files)
- Disc 2: `/mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/Photos/img/Photos/ScanMyPhotos/Disc 2/` (2,040 files)
- Disc 3: `/mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/Photos/img/Photos/ScanMyPhotos/Disc 3/` (2,076 files)
- Disc 4: `/mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/Photos/img/Photos/ScanMyPhotos/Disc 4/` (1,576 files)
- Samples: `data/samples/` (100 pre-selected JPGs across discs)

## Environment
- Python 3.14 via uv — activate venv: `source .venv/bin/activate`
- CPU-only (AMD Ryzen 12 cores, 27GB RAM, integrated AMD Radeon Vega — no discrete GPU)
- Tesseract 5.3.4 installed system-wide
- Deps managed in `pyproject.toml` with uv

## Key Documentation
- `docs/PLAN.md` -- Master plan for the full 77K photo consolidation (Phases 1-3)
- `docs/HANDOFF.md` -- Detailed handoff for scanned photo date extraction pipeline
- `docs/DATE_EXTRACTION_APPROACHES.md` -- Scored analysis of 6 approaches with costs
- `docs/CLAUDE_CODE_WORKFLOW.md` -- Claude Code patterns, hooks, and optimization techniques

## Date Stamp Characteristics
- Orange/red/amber LED digits imprinted by camera, typically bottom edge of photo
- Format: `M D 'YY` (e.g., "10 3 '99"), spanning ~1986-2010
- Many photos have NO stamp — detector must handle `found: False`
- Some photos are rotated 90° — stamps may appear on side edges

## Project Structure
- `scripts/date_extraction/` -- OCR and stamp detection scripts
- `scripts/rotation/` -- Orientation detection (model + docker)
- `scripts/dedup/` -- Dedup pipeline utilities
- `data/` -- Results (ocr_results.json, rotation_results.json), samples, metadata
- `archive/` -- Historical debug output (debug_stamps v1-v10)
- `dedup/` -- Deduplication sub-project (self-contained, own venv)
- `yolo_finetune/` -- YOLO date stamp detector (separate repo)
- `models/` -- ML model weights (gitignored)

## Claude Code Patterns
See `docs/CLAUDE_CODE_WORKFLOW.md` for detailed guidance on:
- Memory persistence across sessions
- Token optimization strategies
- Subagent architecture and model selection
- Verification and evaluation patterns
- Hook automation and continuous learning
