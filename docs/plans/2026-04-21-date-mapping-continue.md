---
date: 2026-04-21
status: proposed
owner: will
supersedes: none
continues: 2026-04-20-date-mapping-rebuild.md, 2026-04-20-vlm-ocr-bench.md
---

# Date-Mapping: Finish the Run

## Why

[2026-04-20-date-mapping-rebuild.md](2026-04-20-date-mapping-rebuild.md) rebuilt the stamp_* schema, the `scanmyphotos_manifest`, the `media_dates` / `media_has_date` views, and wired nightly dedup-postgres backups. [2026-04-20-vlm-ocr-bench.md](2026-04-20-vlm-ocr-bench.md) locked down the winning OCR model on a 200-stem corpus. What's left is to run the pipeline at fleet scale and close the loop on the undated gallery.

## Current state (verified 2026-04-21)

| Table / view          | Rows  | Coverage                 |
|-----------------------|-------|--------------------------|
| scanmyphotos_manifest | 7,454 | 100% of disc scans       |
| stamp_predictions     | 2,531 | 34% of manifest          |
| stamp_ocr             | 1,000 | 200 distinct stems (bench, 5 models) |
| stamp_ocr parsed_date | 190   | 2.5% of manifest         |
| stamp_no_stamp        | 0     | none classified          |
| exif_dates            | 0     | **empty** — biggest gap  |
| video_dates           | 4,206 | 4,206/~5,300 videos      |
| media_has_date        | 4,396 | unique sha256 w/ a date  |
| photo_embeddings      | 57,324 | 41K photos + 15K video keyframes |

Undated-gallery implication: of ~46K unique media sha256s, only ~4.4K currently resolve to a date. The bulk of the fix is **exif_dates** (will close ~30K photos in one pass) and **OCR on the remaining ~7,250 stems** (closes the scans).

Bbox schema fixed (real floats 0–1). Bench ground truth present as `model='sonnet'`. 4 cloud-Ollama models already benchmarked against it.

## Decisions still to confirm

- [ ] **D1 — Production OCR model.** Bench rows exist for `sonnet`, `gemma4:31b-cloud@ollama-cloud`, `kimi-k2.5:cloud@ollama-cloud`, `gemini-3-flash-preview:cloud@ollama-cloud`, `qwen3.5:cloud@ollama-cloud`. `report_vlm_bench.py` output (from `2026-04-20-vlm-ocr-bench.md`) should name the winner. **Default if not blocked: `sonnet` via Task-tool subagents** (matches the proven parallel-Haiku pattern from the memory entry `project_parallel_haiku_ocr.md` but with Sonnet for accuracy).
- [ ] **D2 — No-stamp auto-classification threshold.** Combine signals: `confidence < 0.05` in `stamp_predictions` OR `YOLO produced no bbox` → auto-add to `stamp_no_stamp` with `source='auto'`. Alternative: leave for manual review. **Default: auto-classify below 0.05; everything 0.05–0.15 goes to manual queue.**
- [ ] **D3 — Budget cap for Sonnet OCR.** ~7,250 stems × 1 call avg = ~7,250 Sonnet calls, plus whatever fraction needs stage-2 reconciliation. At current Sonnet pricing on a 256×256 crop + short prompt ≈ $0.008/call, fleet cost is ~$60, stage-2 adds maybe another $30. **Default cap: $150 soft, $250 hard.**

## Tasks

Each task is verifiable on its own. Each one updates Postgres (which is now backed up nightly, so mid-run snapshots are automatic).

### T1. Populate `exif_dates` — 20 min

**Why**: Biggest single win. ~30K photos gain a date with one PIL pass.

**What**:
1. Write [scripts/data/extract_exif_dates.py](scripts/data/extract_exif_dates.py), mirroring `extract_video_dates.py`. Reuse the PIL+pillow-heif header-only probe from `scripts/find_undated_media.py` (fast — 17s for 42K files verified on 2026-04-19). Upsert into `exif_dates (sha256, date_taken, source, raw, extracted_at)`.
2. Skip sha256s where `media_type='video_keyframe'` in `photo_embeddings` — those are covered by `video_dates`.
3. Run once; idempotent on re-run via `ON CONFLICT (sha256) DO NOTHING`.

**Verify**:
- `SELECT count(*) FROM exif_dates` in the 25,000–35,000 range.
- `SELECT source, count(*) FROM media_dates GROUP BY 1` now shows `exif: ~30K`, `video: 4,206`, `stamp_ocr:sonnet: ~190` (unchanged at this step).
- Gallery rebuild (T7) will drop the undated photo count from ~10K → ~2–3K on the strength of this alone.

### T2. Finish YOLO inference — ~2 hr CPU background

**Why**: 4,923 stems still lack a bbox. Can't OCR what you haven't detected.

**What**:
- Background: `just infer-bg` (or `python scripts/infer/infer_all.py` under tmux).
- Wait for completion. Log to `state/infer_progress.log`.

**Verify**:
- `SELECT count(*) FROM stamp_predictions` ≈ 7,400 (a handful may skip on image open errors).
- Confidence histogram: `SELECT round(confidence::numeric,1) AS c, count(*) FROM stamp_predictions GROUP BY 1 ORDER BY 1`. Expect roughly bimodal — big cluster ≥ 0.6 (real stamps) + tail near 0 (no stamp / failed).
- Bbox sanity: `SELECT count(*) FROM stamp_predictions WHERE w BETWEEN 0.01 AND 0.5 AND h BETWEEN 0.01 AND 0.2 AND confidence > 0.5` gives the "credible stamp bboxes" count — should be ~5,000–6,500.

### T3. Auto-populate `stamp_no_stamp` (cheap classifier) — 5 min

**Why**: Don't waste Sonnet calls on images with no stamp visible.

**What**: Per decision D2, insert stems where `confidence < 0.05` (or no row in `stamp_predictions`) with `source='auto'`.

```sql
INSERT INTO stamp_no_stamp (stem, source)
SELECT m.stem, 'auto'
FROM scanmyphotos_manifest m
LEFT JOIN stamp_predictions p ON p.stem = m.stem
WHERE p.stem IS NULL OR p.confidence < 0.05
ON CONFLICT (stem) DO NOTHING;
```

**Verify**: count matches `(SELECT count(*) FROM stamp_predictions WHERE confidence < 0.05) + (7454 - 7400)`. Expect 200–600.

### T4. Stage-1 OCR at fleet scale — wall-clock 4–6 hr, cost within D3 cap

**Why**: Extract date text from every credible bbox.

**What**:
1. `orchestrate_ocr.py crop-stage1 --limit 8000 --exclude-no-stamp` — crops every stem that (a) has a `stamp_predictions` row with `confidence ≥ 0.05`, (b) is not in `stamp_no_stamp`, and (c) doesn't already have a `stamp_ocr` row for the production model.
2. Dispatch shards to Sonnet subagents (pattern from `project_parallel_haiku_ocr.md` memory — proven to work at this scale, 126 shards × ~50 stems).
3. `orchestrate_ocr.py merge-stage1` writes rows to `stamp_ocr` with `model=<production>`, `stage=1`, `host_label='production'`, and parses `raw_text` into `parsed_date` in the same pass (parser already in `ocr_stamps.py`). Non-parseable rows get `parse_error` set.

**Verify**:
- Coverage: `SELECT count(DISTINCT stem) FROM stamp_ocr WHERE model=<production> AND stage=1` ≥ 6,500.
- Parse success ≥ 70% (≥ 4,550 stems with a non-null `parsed_date`).
- Cost check vs D3 cap before triggering stage-2.

### T5. Stage-2 reconciliation on flagged rows — ~1 hr

**Why**: 5/9 digit confusion and `?` markers need dual-crop verification. Memory entry `feedback_haiku_digit_confusion.md` documents the pattern.

**What**:
- `orchestrate_ocr.py crop-stage2` selects candidates: `stage=1 AND (parse_error IS NOT NULL OR confidence < 0.7 OR raw_text LIKE '%?%')`.
- Dispatch stage-2 shards (fewer, larger crops).
- `merge-stage2` reconciles: stage-1 and stage-2 agree → `review_status='auto_accepted'`; disagree → `review_status='needs_review'`.

**Verify**:
- `SELECT review_status, count(*) FROM stamp_ocr GROUP BY 1` shows a sensible mix.
- Human queue (`needs_review`) ≤ 500. If larger, loosen the confidence threshold in D2 or T3.

### T6. Human review via corrections dashboard — open-ended

**Why**: Close the `needs_review` tail.

**What**: `just dashboard` (port 8889). Queue ordered by lowest confidence. Corrections write `review_status='manual'` and an authoritative `parsed_date`.

**Verify**: `SELECT count(*) FROM stamp_ocr WHERE review_status='needs_review'` trends to 0 over sessions. (Not a blocker for the gallery rebuild — the auto-accepted rows already show up in the view.)

### T7. Regenerate undated gallery — 10 min

**Why**: Visual proof the pipeline closed the loop.

**What**:
- `find_undated_media.py` already queries `media_has_date` (updated 2026-04-20), no changes needed.
- Re-run: `python scripts/find_undated_media.py && python scripts/build_undated_gallery.py`.
- Confirm HTTP server bind on `0.0.0.0:8890`; open `http://ares:8890/output/undated_gallery/`.

**Verify**:
- Undated photos drop from 10,378 → 2,000–3,000 range (digital photos without EXIF + any scans still flagged `needs_review`).
- Undated videos unchanged at 1,090 (that's an independent problem — no creation_time on those files, nothing to recover from).
- The 2026-04-19 canary photo [originals/media/004e39c9c54fb8f0f7ac9f61193bc703462eeca7908dbc217b022de62f820e21.jpg](originals/media/004e39c9c54fb8f0f7ac9f61193bc703462eeca7908dbc217b022de62f820e21.jpg) no longer appears.

### T8. Commit + handoff — 15 min

**What**:
- Commit: `scripts/data/extract_exif_dates.py`, any updates to `orchestrate_ocr.py` for the production model name, new schema migrations if any.
- Update `CLAUDE.md`:
  - "Key Data Stores" section: note `exif_dates`, `scanmyphotos_manifest`, `media_dates` view, `media_has_date` view.
  - "Critical Constraints": reference `Homelab/infra/backup/` nightly pg_dump (added 2026-04-20; see `dedup-db-backup` sidecar).
- `just handoff-create`.

## Risks

- **Sonnet rate limits at parallel dispatch.** The parallel-Haiku pattern shipped 126 shards; Sonnet has different limits. Mitigation: dispatch one shard first, observe response latency + throttling, then scale. If limits bite, fall back to a cheaper cloud model from the bench (bench report names the second-best model).
- **YOLO finishes T2 but leaves a long low-confidence tail.** If > 1,500 stems land in 0.05–0.5, T5 gets big. Mitigation: tighten D2's lower bound to 0.1 and move the rest to `stamp_no_stamp` with `source='auto_low_conf'`.
- **exif_dates scan picks up dates that actually belong to scan creation (scanner software), not the photographed moment.** Scanner-stamped EXIF from 2014–2024 for photos of 1994 would poison the gallery. Mitigation: in T1, reject obviously-scanner dates (all ScanMyPhotos-discovered sha256s from the manifest — skip them in the exif pass since their authoritative date comes from stamp_ocr anyway).
- **Undated gallery HTTP server death.** Current background server on `8890` has died once this session. If repeated, wrap in systemd unit (follow-up, not in this plan).

## Estimated wall-clock

| Phase                        | Time          |
|------------------------------|---------------|
| T1 (exif_dates)              | 20 min        |
| T2 (YOLO finish)             | ~2 hr CPU bg  |
| T3 (no-stamp auto)           | 5 min         |
| T4 (stage-1 OCR)             | 4–6 hr async  |
| T5 (stage-2)                 | 1 hr          |
| T6 (human review)            | open-ended    |
| T7 (gallery)                 | 10 min        |
| T8 (commit + handoff)        | 15 min        |
| **Active (excluding T6)**    | ~8–10 hr      |
