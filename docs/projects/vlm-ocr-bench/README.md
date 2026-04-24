---
title: VLM OCR Benchmark — Finish and Report
status: active
repos: [photo_project]
started: 2026-04-23
last_updated: 2026-04-23
next_step: Check 5h retry loop progress (tail state/bench/results_litellm/loop.log); fix LiteLLM ANTHROPIC/GOOGLE key regression
---

# VLM OCR Benchmark — Finish and Report

## Goal

Complete the 9-model Ollama Cloud VLM OCR benchmark, generate the Pareto-frontier report comparing accuracy/latency/cost, and use the result to pick the production OCR model for the date-mapping fleet run.

## Tasks

- [ ] Wait for `run_litellm_bench_loop.sh` to complete remaining 8 models (PID 3386226, 5h retry cadence)
- [ ] Fix LiteLLM key regression: add ANTHROPIC/GOOGLE/DEEPSEEK keys back to `ai/litellm/.env.sops`
- [ ] Generate bench report: `uv run scripts/ocr/report_vlm_bench.py --ingest-jsonl state/bench/results_litellm`
- [ ] Review Pareto chart; confirm production OCR model (default: Sonnet)
- [ ] Commit uncommitted changes in photo_project and Homelab (listed in handoff)
- [ ] Update date-mapping-finish project with confirmed D1 model decision

## Session Log

### 2026-04-23

- Project created.
- At creation: `gemma4-31b-cloud` 200/200 done; `gemini-3-flash-preview-cloud` 116/200; 7 other models at 0/200.
- Retry loop running detached (nohup, PID 3386226). Each round: 5h sleep + bench attempt. Log: `state/bench/results_litellm/loop.log`.

## Notes

- **Handoff:** [docs/handoff/2026-04-21 Handoff - Ollama Cloud Bench via LiteLLM.md](../../handoff/2026-04-21%20Handoff%20-%20Ollama%20Cloud%20Bench%20via%20LiteLLM.md)
- **Spec:** [docs/specs/2026-04-20-vlm-ocr-bench-design.md](../../specs/2026-04-20-vlm-ocr-bench-design.md)
- **Plan:** [docs/plans/2026-04-20-vlm-ocr-bench.md](../../plans/2026-04-20-vlm-ocr-bench.md)
- Ollama Cloud free tier has undocumented hourly quotas — all 9 models share a per-account limit. 5h spacing chosen empirically.
- `bench_vlm_litellm.py --resume` skips already-clean rows; transient `rate_limit`/`timeout` rows are NOT persisted (fixed 2026-04-21).
- Ground truth: 200 Sonnet rows in `stamp_ocr` (not yet frozen — can still spot-check via `just bench-seed-review`).
- LiteLLM virtual key `photo-bench` scoped to 9 Ollama Cloud models, $5 budget cap. Stored in gitignored `.env`.
