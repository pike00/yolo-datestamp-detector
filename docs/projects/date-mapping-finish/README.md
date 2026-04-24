---
title: Date Mapping — Finish the Run
status: active
repos: [photo_project]
started: 2026-04-23
last_updated: 2026-04-24
next_step: Let T4 (gemma4-31b-cloud@litellm fleet OCR) complete in background; return for T5 (stage-2 reconciliation) once stage-1 coverage > ~6,500 stems
ocr_model: gemma4-31b-cloud@litellm
ocr_bench_report: output/vlm_bench_report.html
---

# Date Mapping — Finish the Run

## Goal

Complete the ScanMyPhotos stamp OCR pipeline and EXIF extraction so the undated gallery shrinks from ~10K photos to <3K. Schema, manifest, and nightly backups are all in place; what remains is running the data passes at scale.

## Tasks

- [x] T1: Populate `exif_dates` — 31,027 rows (DateTimeOriginal 30,392, DateTime 633, Digitized 2). Manifest-skip + keyframe-skip added to `extract_exif_dates.py`.
- [x] T2: Finish YOLO inference — 7,164 / 7,454 rows (290 residual stems have no detection even at conf=0.01). `infer_all.py` now skips already-predicted stems on resume.
- [x] T3: Auto-populate `stamp_no_stamp` — 477 stems (187 conf<0.05 + 290 missing predictions).
- [ ] T4: Stage-1 OCR at fleet scale — **D1 resolved: `gemma4-31b-cloud@litellm`** (Pareto-frontier top, 68.5% vs frozen gemma4:31b-cloud GT, 1.19 img/s; free tier). Running in background via `scripts/ocr/ocr_fleet.py`. ~6,972 pending at launch, Ollama Cloud free-tier quota ~300 req/hr, ETA 20+ hours. Bench archived: [docs/projects/vlm-ocr-bench/](../vlm-ocr-bench/README.md).
- [ ] T5: Stage-2 reconciliation on flagged rows — blocked on T4
- [ ] T6: Human review of `needs_review` queue via corrections dashboard — blocked on T5
- [x] T7: Regenerate undated gallery — 10,393 undated photos post-T1 (matches plan's 10,378 baseline). Gallery at `output/undated_gallery/`. Will drop to ~2-3K after T4+T5 completes.
- [x] T8: Commit this session's work + update CLAUDE.md — T1-T3 + T7 + fleet runner committed. T4 run proceeds in background; final commit with T4-T6 deliverables deferred to next session.

## Session Log

### 2026-04-24 (evening)

- **D1 finalized.** VLM OCR bench archived; production model locked to `gemma4-31b-cloud@litellm`. Ground truth for the bench frozen to `gemma4:31b-cloud@ollama-cloud` in `state/bench/manifest.json` (manifest-driven default in the report script).
- **Report artifacts (final):** [output/vlm_bench_report.html](../../../output/vlm_bench_report.html) (full metadata), [output/vlm_bench_report.md](../../../output/vlm_bench_report.md), [output/vlm_bench_pareto.png](../../../output/vlm_bench_pareto.png).
- **Report script upgraded:** `scripts/ocr/report_vlm_bench.py` now hydrates latency + token counts from JSONL, accepts `--ground-truth`, emits HTML alongside MD/PNG. Re-run with no args to regenerate.
- **T4 continues** in background against the same model; no fleet change.

### 2026-04-24

- **Discovered README was stale:** YOLO was already at 7,164/7,454 (not 2,531), VLM bench was 14/14 complete (not 1/9 running), retry loop had already exited. Generated the bench report without waiting.
- **T1 `exif_dates` — done.** 31,027 rows (DateTimeOriginal 30,392, DateTime 633, Digitized 2). Patched `scripts/data/extract_exif_dates.py` to skip `scanmyphotos_manifest` sha256s (prevents scanner-EXIF poisoning) AND `photo_embeddings.media_type='video_keyframe'` (covered by `video_dates`). Yearly distribution looks organic (peak 2010s, tail into 80s).
- **T2 YOLO — done.** Patched `scripts/infer/infer_all.py` to skip already-predicted stems on resume (was a wasteful full re-run). 290 residual stems had no detection even at conf=0.01 (true no-stamps). Final credible-bbox count: 6,679 (above plan target).
- **T3 `stamp_no_stamp` — done.** Auto-populated 477 rows via the plan's SQL (187 conf<0.05 + 290 missing predictions).
- **D1 resolved.** Bench report (`output/vlm_bench_report.md`) gave top-3 at ~55-56% agreement with (unreviewed) Sonnet ground truth: `gemma4:31b-cloud@ollama-cloud`, `gemini-3-flash-preview:cloud`, `gpt-oss-120b-cloud`. Selected `gemma4-31b-cloud` via LiteLLM for fleet run (top bench score + free tier).
- **T4 running.** New `scripts/ocr/ocr_fleet.py` — purpose-built stage-1 runner talking to LiteLLM directly (bypasses the shard orchestrator since the subagent pattern isn't needed for a non-Haiku model). Launched in background; 6,972 pending at start. Ollama Cloud free-tier quota ~300 req/hr → ETA ~20 hours continuous, likely longer with 429 backoff. Log: `state/logs/ocr_fleet_gemma4-31b-cloud.log`.
- **T7 — done (interim).** Regenerated undated gallery: 10,393 undated photos remain (unchanged from plan baseline; OCR hasn't closed yet). Gallery at `output/undated_gallery/` (39 pages).
- **T5/T6 deferred:** Must wait for T4 to reach usable stage-1 coverage (>6,500 stems or so). Resume next session.
- **T8 — partial commit.** CLAUDE.md Key Data Stores updated with `scanmyphotos_manifest`, `exif_dates`, `video_dates`, and the `media_dates` / `media_has_date` views. Backup of dedup DB runs nightly via the `dedup-db-backup` sidecar.

### 2026-04-23

- Project created.
- State at creation: `exif_dates` empty (biggest gap), `stamp_predictions` at 2,531/7,454 (34%), `stamp_ocr` has 200 bench rows only, `stamp_no_stamp` empty, `video_dates` 4,206 rows.

## Notes

- **Plan:** [docs/plans/2026-04-21-date-mapping-continue.md](../../plans/2026-04-21-date-mapping-continue.md)
- **Skip ScanMyPhotos sha256s in EXIF pass** — their authoritative date comes from stamp_ocr, not scanner EXIF. Risk: scanner-stamped EXIF (2014–2024) would poison dates of 1990s photos.
- **Production OCR model decision (D1) — resolved 2026-04-24:** `gemma4-31b-cloud@litellm`. Bench frozen + archived under [docs/projects/vlm-ocr-bench/](../vlm-ocr-bench/README.md). Rationale: Pareto-frontier top (68.5% agreement vs frozen gemma4:31b-cloud GT, 1.19 img/s median), free tier on Ollama Cloud via LiteLLM.
- **Budget cap (D3):** ~$60 fleet cost at Sonnet pricing; soft cap $150, hard cap $250.
- `stamp_predictions` float-column fix already applied (2026-04-20 session). YOLO model is `gpu-40ep/best.pt` (env-driven via `YOLO_WEIGHTS`).
- T1 must skip `scanmyphotos_manifest` sha256s to avoid scanner-EXIF pollution.
