---
title: VLM OCR Benchmark — Finish and Report
status: archived
repos: [photo_project]
started: 2026-04-23
last_updated: 2026-04-24
archived: 2026-04-24
outcome: "Production OCR model = gemma4-31b-cloud@litellm (Pareto-frontier, 68.5% agreement vs frozen gemma4:31b-cloud GT, 1.19 img/s)"
---

# VLM OCR Benchmark — Finish and Report

## Goal

Complete the 9-model Ollama Cloud VLM OCR benchmark, generate the Pareto-frontier report comparing accuracy/latency/cost, and use the result to pick the production OCR model for the date-mapping fleet run.

## Outcome

Production OCR model for the date-mapping fleet: **`gemma4-31b-cloud@litellm`**
(Pareto-frontier top; 68.5% agreement vs frozen gemma4:31b-cloud GT; 1.19 img/s
median throughput; free tier on Ollama Cloud via LiteLLM).

Ground truth for the bench is frozen to `gemma4:31b-cloud@ollama-cloud` in
[state/bench/manifest.json](../../../state/bench/manifest.json). Final artifacts:

- [output/vlm_bench_report.html](../../../output/vlm_bench_report.html) — full metadata report
- [output/vlm_bench_report.md](../../../output/vlm_bench_report.md) — markdown summary
- [output/vlm_bench_pareto.png](../../../output/vlm_bench_pareto.png) — Pareto chart

## Tasks

- [x] Wait for `run_litellm_bench_loop.sh` to complete remaining 8 models — 8/9 at 200/200; `gpt-oss-120b-cloud` at 195/200 accepted (5 persistent failures, not worth re-run)
- [x] Generate bench report — ran `uv run scripts/ocr/report_vlm_bench.py`; HTML/MD/PNG written to `output/`
- [x] Review Pareto chart; confirm production OCR model — **`gemma4-31b-cloud@litellm` selected**
- [x] Update date-mapping-finish project with confirmed D1 model decision — already reflected in T4
- [ ] Fix LiteLLM key regression: add ANTHROPIC/GOOGLE/DEEPSEEK keys back to `ai/litellm/.env.sops` — **carried forward** (LiteLLM infra, not bench scope)
- [ ] Commit uncommitted changes in photo_project and Homelab — **carried forward**

## Session Log

### 2026-04-24

- **Archived.** Production model confirmed: `gemma4-31b-cloud@litellm`.
- **Bench finished.** 8/9 models at 200/200 clean; `gpt-oss-120b-cloud` at 195/200 (5 persistent failures accepted; not worth another retry round).
- **Ground truth frozen** to `gemma4:31b-cloud@ollama-cloud` in `state/bench/manifest.json`. Rationale: top agreement vs (noisy, unreviewed) Sonnet baseline + lowest unparsed rate. Header note documents that "Agree%" is relative, not absolute correctness.
- **Report script hardened:** `scripts/ocr/report_vlm_bench.py` now (a) hydrates `elapsed_s` + token counts from JSONL (stamp_ocr doesn't persist them), (b) takes `--ground-truth` from manifest, (c) excludes GT model from candidates, (d) emits a single-file HTML with per-model min/median/P95/max latency, avg in/out tokens, first/last-run timestamps, stratification buckets, and Pareto frontier badges.
- **Pareto frontier (final):** `gemma4-31b-cloud@litellm` (68.5%, 1.19 img/s) = picked; `gemma3-27b-cloud@litellm` (44.5%, 1.22 img/s) and `gemma3-12b-cloud@litellm` (38.5%, 1.32 img/s) on frontier for raw speed only.
- **Carried-forward items** (not bench scope): LiteLLM key regression for ANTHROPIC/GOOGLE/DEEPSEEK (blocks non-OCR flows), and uncommitted work in photo_project/Homelab.

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
