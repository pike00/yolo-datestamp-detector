---
summary: "VLM OCR bench scaffolding complete -- 11/11 code tasks done, ground truth seeded, ready to run cross-host benches"
---

# Handoff: VLM OCR Bench Scaffolding

**Date:** 2026-04-20
**Goal:** Stand up a portable benchmark harness that compares OSS Ollama vision models (kimi-vl, qwen2.5-vl, minicpm-v, gemma3, llama3.2-vision, internvl3) plus Ollama Cloud free-tier models against a Sonnet ground truth on 200 stratified ScanMyPhotos date-stamp crops, emitting a Pareto-frontier report across (accuracy, throughput, model size, host).

## Current Status

All implementation tasks complete. Pipeline verified end-to-end via a gemma4:e4b dry-run on willbook CPU (report + Pareto PNG generated, rows cleaned up after). No real benchmarks run yet -- next session kicks those off.

**What landed:**

- **Task 0 (schema fix):** `schema/stamp_predictions_float_bbox.sql` -- `x/y/w/h` integer columns silently truncated YOLO's normalized floats to 0/1. Migrated to `real`, truncated the table, re-ran `just infer`. 7,142 healthy bboxes now in `stamp_predictions`. Confidence buckets: 673 / 433 / 2569 / 3467.
- **Task 1 (schema):** `schema/stamp_ocr_host_label.sql` -- added nullable `host_label TEXT` + index.
- **Task 2 (gitignore):** `state/bench/` and `output/vlm_bench_*` ignored.
- **Task 2.5 (backup helper):** `scripts/ocr/bench_backup.sh` + `just bench-backup <label>`. Additive timestamped snapshots of stamp_predictions/stamp_ocr/stamp_no_stamp as zstd-compressed SQL + rsync of `state/bench/` to the HDD. Plugs the 24h gap in the nightly backup that the 2026-04-16 loss fell into.
- **Task 3 (shared utils):** `scripts/ocr/ocr_util.py` with `normalize_date`, `strip_thinking_blocks`, `extract_final_answer`. `ocr_gemma.py` now imports. 14 unit tests.
- **Task 4 (corpus builder):** `scripts/ocr/build_bench_corpus.py` -- stratified sampler (50 stems x 4 confidence buckets) with deterministic seed, pre-cropped to `state/bench/crops/`. Ran: 200 stems sampled, 200 corpus JPGs + 200 crop JPGs written, zero skew.
- **Task 5 (Sonnet IO orchestrator):** `scripts/ocr/seed_bench_ground_truth.py` -- plan / merge / review-html / freeze / status subcommands. Inserts via `(stem, model)` PK. Coerces `YYYY-MM-00` partial dates to NULL on insert (Postgres DATE rejects day=0).
- **Task 6 (Sonnet ground truth):** Dispatched 40 Claude Code subagents in parallel batches to OCR all 200 crops. **200/200 sonnet rows in `stamp_ocr`, 134 with parsed dates.** Backup snapshot at `/mnt/823.../backups/photo_project_bench/20260420T211648Z_post-ground-truth-pre-review/` (104K SQL + 125M state/bench).
- **Task 7 (Ollama runner):** `scripts/ocr/bench_vlm_ocr.py` -- pre-cropped image input, `/api/chat` call, `<think>` stripping, normalize, write to Postgres or JSONL. Picks up `OLLAMA_API_KEY` env for Ollama Cloud. Handles TIMEOUT / OOM_ERROR / RATE_LIMIT / HTTP_xxx as error rows without retries.
- **Task 8 (report):** `scripts/ocr/report_vlm_bench.py` -- joins sonnet truth to non-sonnet rows, computes agree/unparsed/timeout/oom/rate_limit/high_conf_wrong percentages + median/p95 latency, emits `output/vlm_bench_report.md` + `output/vlm_bench_pareto.png`.
- **Task 9 (profile wrappers):** four shells under `scripts/ocr/bench_profiles/`: `ares-cpu.sh`, `m2pro.sh`, `cloud-gpu.sh`, `ollama-cloud.sh`.
- **Task 10 (justfile):** 7 `bench-*` recipes: backup, build, seed-plan, seed-review, seed-freeze, run, report.
- **Task 11 (dry-run):** gemma4:e4b on 10 stems -> pipeline works end-to-end, report markdown + PNG generated, dryrun rows cleaned up.

## Next Steps

**1. Human review of the Sonnet ground truth.** Not frozen yet. Open the review HTML and spot-check, especially the 66 rows with `parsed_date IS NULL`:

```bash
just bench-seed-review
# opens state/bench/ground_truth_review.html
# fix obvious errors directly in DB via UPDATE stamp_ocr SET raw_text=..., parsed_date=... WHERE stem=... AND model='sonnet'
just bench-seed-freeze
```

**2. Pull models on the host where you want to bench.** On ares (CPU, overnight) or the M2 Pro MacBook (faster), run:

```bash
# On ares -- rough order of smallest-first so the fast ones finish before you sleep
for m in gemma3 minicpm-v qwen2.5-vl:7b internvl3:8b kimi-vl llama3.2-vision:11b; do
  just bench-run ares-cpu "$m"
  just bench-backup "post-ares-$(echo $m | tr '/:' '__')"
done
```

**3. Ollama Cloud run (free tier).** Create key at `https://ollama.com/settings/keys`, then:

```bash
export OLLAMA_API_KEY=...
for m in gemini-3-pro:cloud gpt-oss:cloud kimi-vl:cloud qwen2.5-vl:cloud minicpm-v:cloud; do
  bash scripts/ocr/bench_profiles/ollama-cloud.sh "$m" || true  # skip if tag unavailable
done
```

**4. Generate the report.**

```bash
just bench-report
# output/vlm_bench_report.md + output/vlm_bench_pareto.png
```

## Key Context

- **Two data-loss findings this session.** (a) The 2026-04-16 dedup Postgres volume wipe destroyed the original Sonnet OCR rows; ground truth had to be regenerated from scratch. (b) `infer_all.py:69-73` writes normalized floats into integer columns -- the schema bug was inherited from commit 9260004 ("inferred from _db.py column usage" -- inferred wrongly). Both fixed and locked down with `bench_backup.sh` milestones.
- **PK strategy for multi-host rows:** `model` column is suffixed with `@<host-label>` so `(stem, model)` PK stays collision-free (e.g. `kimi-vl:latest@ares-cpu` vs `kimi-vl:latest@m2pro`). `host_label` column is added separately for grouping. `report_vlm_bench.py` splits on `@` at report time.
- **Partial-date handling:** `normalize_date` returns `"YYYY-MM-00"` for month+year-only crops; the inserters in both `seed_bench_ground_truth.py` and `bench_vlm_ocr.py` coerce that to NULL because Postgres DATE rejects day=0. Raw text is preserved.
- **Ollama Cloud support:** If `OLLAMA_API_KEY` is set, `bench_vlm_ocr.py` sends `Authorization: Bearer ...` on every request. HTTP 429 rows record as `RATE_LIMIT` and the run continues.
- **Elapsed time lost on Postgres path:** `bench_vlm_ocr.py` only persists `elapsed_s` in JSONL output -- not in `stamp_ocr` columns. `report_vlm_bench.py` hardcodes 0.0 for rows loaded from Postgres. For hosts that write direct-to-postgres (ares-cpu, cloud-gpu), latency percentiles will show as 0. Workaround: use JSONL output on those profiles too, then ingest via `--ingest-jsonl`. Noted for possible future task.
- **`stamp_predictions` was re-run with integer-to-float schema fix, producing 7,142 rows (vs 2,132 pre-migration).** Bucket counts comfortably exceed the 50 needed per bucket.

## Files Touched

**New code:**
- `scripts/ocr/ocr_util.py`
- `scripts/ocr/build_bench_corpus.py`
- `scripts/ocr/seed_bench_ground_truth.py`
- `scripts/ocr/bench_vlm_ocr.py`
- `scripts/ocr/report_vlm_bench.py`
- `scripts/ocr/bench_backup.sh`
- `scripts/ocr/bench_profiles/{ares-cpu,m2pro,cloud-gpu,ollama-cloud}.sh`
- `scripts/ocr/__init__.py`, `scripts/__init__.py`

**New tests:**
- `tests/conftest.py`
- `tests/test_ocr_util.py` (14)
- `tests/test_build_bench_corpus.py` (4)
- `tests/test_seed_bench_ground_truth.py` (6)
- `tests/test_bench_vlm_ocr.py` (6)
- `tests/test_report_vlm_bench.py` (6)

**Schema migrations:**
- `schema/stamp_predictions_float_bbox.sql`
- `schema/stamp_ocr_host_label.sql`

**Modified:**
- `scripts/ocr/ocr_gemma.py` -- imports `normalize_date` from `ocr_util`
- `justfile` -- 7 new `bench-*` recipes
- `.gitignore` -- `state/bench/`, `output/vlm_bench_*`

**Docs:**
- `docs/specs/2026-04-20-vlm-ocr-bench-design.md`
- `docs/plans/2026-04-20-vlm-ocr-bench.md`

**Data (gitignored, backed up to HDD):**
- `state/bench/corpus/` -- 200 source JPGs
- `state/bench/crops/` -- 200 padded stamp crops
- `state/bench/manifest.json` -- stratification + stem list
- `state/bench/shards_sonnet/sonnet_[0000-0039].{json,result.json}` -- 40 shards + subagent results
- `stamp_ocr` -- 200 rows under `model='sonnet'`, `host_label='claude-code-subagent'`

## Blockers

None. Ground truth is not frozen (`ground_truth_frozen: false` in manifest) but that's a user-action gate, not a blocker for benchmarking -- reports will include an "unreviewed" banner until `just bench-seed-freeze` runs.
