---
summary: "YOLO weights fix + re-infer done; Ollama Cloud bench via LiteLLM -- 1/9 done, 5h retry loop running until complete"
---

# Handoff: Ollama Cloud Bench via LiteLLM

**Date:** 2026-04-21
**Goal:** Fix wrong YOLO weights being used for stamp prediction, re-infer, re-crop bench corpus, then run VLM OCR bench against Ollama Cloud models routed through the local LiteLLM proxy.

## Current Status

### Completed

- **Identified root cause of "much worse than before" feeling.** `scripts/infer/infer_all.py` was loading `runs/detect/train/weights/best.pt` — a yolo26s CPU-trained checkpoint stopped at epoch 2 (P 0.960 / R 0.951 / mAP50 0.958). The gpu-40ep yolo26m model (P 0.971 / R 0.959 / mAP50 0.963), which is byte-identical to the HF-published `pike00/yolo-date-stamp-detector/best.pt` (sha256 `5dd873f0…`), was sitting unused.
- **MODEL_PATH made env-driven.** `infer_all.py:32` now reads `YOLO_WEIGHTS` env var, defaults to `runs/detect/gpu-40ep/weights/best.pt`.
- **Truncated + re-inferred.** `stamp_predictions` had 7,142 suspect rows labeled `yolo26m-best` but produced by the wrong checkpoint. Truncated and re-ran: **7,164 new rows, mean conf 0.82** (vs 0.54 in the suspect run). User confirmed the new bboxes are "much better" via the dashboard.
- **Built `scripts/annotate/render_bbox_dashboard.py`** — static HTML grid of current `stamp_predictions` (full-photo thumbnails with bbox overlay + zoomed crop). Served via the pre-existing :8772 http.server (now killed); can relaunch against `output/bbox_dashboard/`.
- **Re-cropped bench corpus** with the new bboxes via `scripts/ocr/recrop_bench_from_db.py`: 194/200 refreshed, 6 stems kept old crops (those stems don't have YOLO predictions at all; no stamp visible).
- **LiteLLM wired to Ollama Cloud.** Added 9 `*-cloud` models to `ai/litellm/config.yaml` under `openai/<model>` prefix with `api_base: https://ollama.com/v1` + `api_key: os.environ/OLLAMA_API_KEY`. Added `OLLAMA_API_KEY` to docker-compose env block. User added `OLLAMA_API_KEY=...` to `ai/litellm/.env.sops`.
- **Wrote `bench_vlm_litellm.py`** — OpenAI-compat runner (hits `/v1/chat/completions` with `image_url` data URIs). Same JSONL row schema as `bench_vlm_ocr.py` so `report_vlm_bench.py` ingests the results unchanged.
- **Minted `photo-bench` virtual key** via LiteLLM `/key/generate`. Scoped to the 9 Ollama Cloud models, $5 budget. Stored in gitignored `photo_project/.env` as `LITELLM_API_KEY=sk-em09opCGk0_Za09VDa6oyQ` + `LITELLM_BASE_URL=http://localhost:4000`.
- **Reloaded LiteLLM** via `hl up litellm -d`. Container healthy.
- **Smoke-tested end-to-end.** `gemma4-31b-cloud` via LiteLLM returned `"5 21 '01"` on `d4_00000501.jpg` — route works.
- **First full bench attempt** completed 200/200 `gemma4-31b-cloud` (clean, avg 1.2s/img) and 116/200 `gemini-3-flash-preview-cloud` before Ollama Cloud free-tier hourly quota triggered HTTP 429 rejections.
- **Hardened runner vs rate limits.** `bench_vlm_litellm.py` no longer persists `rate_limit` / `timeout` rows (they were breaking `--resume`). Now uses exponential backoff 60s → 120s → 300s → 600s on RL responses, resets on success.
- **Stripped poisoned RL rows** from existing JSONLs so `--resume` re-attempts them: gemini 200→116 rows, qwen3-vl 4→0, qwen3-vl-instruct 2→0.

### In Progress

- **Detached 5h-wait retry loop running.** `scripts/ocr/run_litellm_bench_loop.sh` launched via `setsid nohup`, PID **3386226** (survives Claude / terminal exit). Sleeps 5h, runs any model with clean<200, repeats until all 9 hit 200 or 30 rounds (~6 days). Log: `state/bench/results_litellm/loop.log`. First attempt fires at **2026-04-22 02:55 local**.

### Bench state at handoff

| Model | clean / target |
|---|---|
| `gemma4-31b-cloud` | 200 / 200 ✓ |
| `gemini-3-flash-preview-cloud` | 116 / 200 |
| `qwen3-vl-235b-cloud` | 0 / 200 |
| `qwen3-vl-235b-instruct-cloud` | 0 / 200 |
| `gemma3-27b-cloud` | 0 / 200 |
| `gemma3-12b-cloud` | 0 / 200 |
| `kimi-k2.5-cloud` | 0 / 200 |
| `qwen3.5-cloud` | 0 / 200 |
| `gpt-oss-120b-cloud` | 0 / 200 |

## Next Steps

1. **Wait for the loop.** Each round is 5h of sleep + however long the bench takes. If Ollama Cloud's quota is generous enough per 5h window, this could finish in 2-3 rounds (10-15h); if stricter, longer. Check progress any time with:
   ```bash
   tail -f /home/will/photo_project/state/bench/results_litellm/loop.log
   for f in /home/will/photo_project/state/bench/results_litellm/*.jsonl; do
     printf "%-40s clean=%d\n" "$(basename $f)" "$(grep -c '"error": null' $f)"
   done
   ```

2. **Fix ANTHROPIC/GOOGLE/DEEPSEEK key regression.** Those keys were never in `ai/litellm/.env.sops` — they were injected at the shell level when litellm was first brought up 4 days ago. The recreate for our OLLAMA_API_KEY addition stripped them, so `haiku`, `sonnet`, `claude-*`, `flash`, `gemini-2.5-flash`, `deepseek-chat`, `deepseek-reasoner` LiteLLM routes now return blank-auth errors. Same sops workflow as the Ollama key: paste values → decrypt/append/re-encrypt → `hl up litellm -d`.

3. **Generate the bench report** once all 9 models hit 200:
   ```bash
   cd /home/will/photo_project
   uv run scripts/ocr/report_vlm_bench.py --ingest-jsonl state/bench/results_litellm
   ```

4. **Commit the changes.** Both repos have uncommitted work:
   - `photo_project`: `scripts/infer/infer_all.py` (env-driven MODEL_PATH), new files `scripts/annotate/render_bbox_dashboard.py`, `scripts/ocr/recrop_bench_from_db.py`, `scripts/ocr/bench_vlm_litellm.py`, `scripts/ocr/run_litellm_bench_loop.sh`, `state/bench/manifest.json` (updated bboxes)
   - `~/Documents/Homelab`: `ai/litellm/docker-compose.yml`, `ai/litellm/config.yaml`, `ai/litellm/.env.sops`

## Key Context

- **HF model ≡ local gpu-40ep.** `huggingface.co/pike00/yolo-date-stamp-detector/best.pt` is **byte-identical** to `runs/detect/gpu-40ep/weights/best.pt` (sha256 `5dd873f0119…`). Pulling from HF is redundant unless you want a clean copy.
- **The other local `best.pt` is a different, unfinished model.** `runs/detect/train/weights/best.pt` (60 MB, sha256 `eb71c6ea…`) is yolo26s CPU-stopped at epoch 2 out of 100 with patience 10. Don't delete it if you want reference, but it should never be the inference default.
- **Ollama Cloud free tier has undocumented hourly quotas.** Ran hot for ~316 requests (200 + 116) then got hard-walled at HTTP 429 across all `:cloud` models simultaneously. The rate limit is per-account, not per-model. Hence the 5h spacing in the loop.
- **Why OpenAI-compat over LiteLLM's native `ollama/` provider.** Ollama exposes `/v1/chat/completions`, and using `openai/<model>` with `api_base: https://ollama.com/v1` is a one-line config change that handles auth via Bearer token. The native `ollama/` prefix has quirks with cloud auth.
- **Virtual key scoping.** `photo-bench` can ONLY call the 9 listed models. Good defense: even if the key leaks, blast radius is the $5 budget cap on those models.
- **`bench_vlm_litellm.py` `--resume` contract.** It loads existing stems from the JSONL and skips them. Persisted transient errors (rate_limit, timeout) break this. The runner now avoids persisting those. Any old JSONL shards produced before this session's fix need their transient-error rows stripped before resume (already done for current files via a one-liner in the session).

## Files Touched

### photo_project (modified/created)
- [scripts/infer/infer_all.py](../../scripts/infer/infer_all.py) — MODEL_PATH now reads `YOLO_WEIGHTS` env var
- [scripts/annotate/render_bbox_dashboard.py](../../scripts/annotate/render_bbox_dashboard.py) — **new** static HTML bbox dashboard
- [scripts/ocr/recrop_bench_from_db.py](../../scripts/ocr/recrop_bench_from_db.py) — **new** refresh bench crops from fresh stamp_predictions
- [scripts/ocr/bench_vlm_litellm.py](../../scripts/ocr/bench_vlm_litellm.py) — **new** OpenAI-compat bench runner
- [scripts/ocr/run_litellm_bench_loop.sh](../../scripts/ocr/run_litellm_bench_loop.sh) — **new** 5h-wait retry loop
- `.env` — **new**, gitignored, holds `LITELLM_API_KEY` + `LITELLM_BASE_URL`
- `state/bench/manifest.json` — refreshed bboxes + added `recropped_at` / `recropped_source` fields
- `state/bench/crops/*.jpg` — 194 crops regenerated with new bboxes
- `state/bench/results_litellm/*.jsonl` — bench output (gemma4 200/200, gemini-3-flash-preview 116/200)

### Homelab (modified)
- `~/Documents/Homelab/ai/litellm/docker-compose.yml` — added `OLLAMA_API_KEY: ${OLLAMA_API_KEY}` to litellm env
- `~/Documents/Homelab/ai/litellm/config.yaml` — added 9 `*-cloud` model entries under "Ollama Cloud" block
- `~/Documents/Homelab/ai/litellm/.env.sops` — user added `OLLAMA_API_KEY` (len 57)

### Postgres
- `stamp_predictions` table: TRUNCATE'd and re-populated with 7,164 rows from the correct gpu-40ep yolo26m model

## Blockers

- **Rate limit.** Ollama Cloud free-tier quotas are opaque and low. The 5h-wait loop is the user-specified mitigation; it will eventually complete but wall-clock time is hours-to-days. If timing matters, upgrading to paid Ollama ([pricing](https://ollama.com/pricing)) would let the full slate finish in <30 min.
- **Broken LiteLLM routes.** `haiku`, `sonnet`, `flash`, `deepseek-*` via LiteLLM return blank-auth errors until the missing provider keys are added back to `ai/litellm/.env.sops`. Unrelated to the bench but would bite anything else using LiteLLM in the meantime.
- **Uncommitted changes in two repos.** Covered in Next Steps #4.
