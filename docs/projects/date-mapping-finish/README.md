---
title: Date Mapping — Finish the Run
status: active
repos: [photo_project]
started: 2026-04-23
last_updated: 2026-04-23
next_step: Run extract_exif_dates.py (T1 — biggest single win, ~30K photos gain a date in one pass)
---

# Date Mapping — Finish the Run

## Goal

Complete the ScanMyPhotos stamp OCR pipeline and EXIF extraction so the undated gallery shrinks from ~10K photos to <3K. Schema, manifest, and nightly backups are all in place; what remains is running the data passes at scale.

## Tasks

- [ ] T1: Populate `exif_dates` — run `scripts/data/extract_exif_dates.py` (~30K hits, ~20 min)
- [ ] T2: Finish YOLO inference on remaining 4,923 stems — `just infer-bg`
- [ ] T3: Auto-populate `stamp_no_stamp` for stems with confidence < 0.05
- [ ] T4: Stage-1 OCR at fleet scale (~7,250 stems via Sonnet/production model subagents, 4–6 hr)
- [ ] T5: Stage-2 reconciliation on flagged rows (digit confusion, low-conf)
- [ ] T6: Human review of `needs_review` queue via corrections dashboard
- [ ] T7: Regenerate undated gallery — `python scripts/find_undated_media.py && python scripts/build_undated_gallery.py`
- [ ] T8: Commit all changes + update CLAUDE.md data stores section + handoff

## Session Log

### 2026-04-23

- Project created.
- State at creation: `exif_dates` empty (biggest gap), `stamp_predictions` at 2,531/7,454 (34%), `stamp_ocr` has 200 bench rows only, `stamp_no_stamp` empty, `video_dates` 4,206 rows.

## Notes

- **Plan:** [docs/plans/2026-04-21-date-mapping-continue.md](../../plans/2026-04-21-date-mapping-continue.md)
- **Skip ScanMyPhotos sha256s in EXIF pass** — their authoritative date comes from stamp_ocr, not scanner EXIF. Risk: scanner-stamped EXIF (2014–2024) would poison dates of 1990s photos.
- **Production OCR model decision (D1)** gates T4. Default: Sonnet via subagents (matches proven parallel-Haiku pattern). Confirm after VLM bench report is available.
- **Budget cap (D3):** ~$60 fleet cost at Sonnet pricing; soft cap $150, hard cap $250.
- `stamp_predictions` float-column fix already applied (2026-04-20 session). YOLO model is `gpu-40ep/best.pt` (env-driven via `YOLO_WEIGHTS`).
- T1 must skip `scanmyphotos_manifest` sha256s to avoid scanner-EXIF pollution.
