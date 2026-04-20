# VLM OCR Bench — Design

**Date:** 2026-04-20
**Status:** Design approved, pending implementation plan
**Goal:** Identify one or more local Ollama vision models that can replace Sonnet for date-stamp OCR on the 6,458-photo ScanMyPhotos corpus and future scans.

## Background

In 2026-04-14 a prior bench compared Gemma4:e4b (74% agreement with Sonnet on a 50-image shard, ~18s/img CPU) and Qwen3-VL:4b (32% agreement due to unsuppressed thinking tokens, ~67s/img CPU). Both fell short of the 94% OK rate Sonnet produced on the full 6,458-photo run, so Sonnet was used for production.

Since then the OSS VLM landscape has moved — notably Moonshot's Kimi-VL-A3B-Thinking (MoE, ~3B active), qwen2.5-vl, MiniCPM-V v2.6/v4, Gemma3 multimodal, Llama3.2-Vision, and InternVL3 are all on Ollama. This bench decides whether any of them are production-ready replacements for Sonnet.

## Non-goals

- Re-OCR of the 6,458 photos already covered by Sonnet. The existing `stamp_ocr` rows for `model=sonnet` are the ground truth and will not be rewritten.
- Prompt engineering across many variants per model. One prompt, same wording as the current `ocr_gemma.py`.
- Fine-tuning. Pure off-the-shelf eval.
- Stage-2 (low-confidence review) benchmarking. Stage-1 only.

## Success criteria

**No single "winner."** The output is a Pareto-frontier report across (normalized-date agreement with Sonnet, images/sec throughput, model size in parameters). The user then chooses a model based on the host they intend to deploy on and the accuracy/throughput tradeoff they're willing to accept.

A model is included in the final recommendation shortlist if it is on the Pareto frontier AND its high-confidence-wrong rate (confidently-emitted, parseable, but disagreeing-with-Sonnet dates) is below 3%. High-confidence-wrong dates silently poison downstream date-mapping and are worse than unparseable output.

## Models under test

1. **kimi-vl** (Moonshot Kimi-VL-A3B-Thinking, MoE ~3B active / ~16B total)
2. **qwen2.5-vl:7b**
3. **minicpm-v** (latest Ollama tag — v2.6 or v4 when pulled)
4. **gemma3** (multimodal, pulling default vision tag)
5. **llama3.2-vision:11b**
6. **internvl3:8b**

Exact Ollama model tags are pinned at bench start and recorded in the report so re-runs are reproducible.

## Corpus

200 stems, stratified by:
- YOLO confidence bucket: `[0.0, 0.3), [0.3, 0.6), [0.6, 0.85), [0.85, 1.0]` — 4 buckets
- Era bucket (based on Sonnet's normalized_date): 19xx vs 20xx — 2 buckets

Target: ~25 stems per (confidence × era) cell, for 4 × 2 × 25 = 200 total. If a cell is underfilled in the available data, redistribute from over-filled cells and note the skew in the report.

Every corpus stem must have a Sonnet ground-truth row in `stamp_ocr`. Stems without Sonnet coverage are ineligible.

## Hosts

The bench harness is portable and runs on any host with Ollama + Python + access to the corpus:

1. **`--profile ares-cpu`** — Docker container on ares, pairs with the existing ollama container, writes to Postgres directly over the internal network.
2. **`--profile m2pro`** — Native Ollama on the user's M2 Pro MacBook; corpus copied over Tailscale, results written to ares Postgres over Tailscale or emitted as JSONL for later import.
3. **`--profile cloud-gpu`** — Ephemeral g5.xlarge spot (or similar) with Tailscale join; results streamed to ares Postgres. Launched via existing AWS spot pattern when willing to spend ~$2-5 for fast turnaround.

The same model can be benched on multiple hosts — each (model, host) pair is a distinct row in the report.

## Components

### 1. `scripts/ocr/build_bench_corpus.py`

One-time packager. Run on ares.

- Reads `stamp_predictions` (for bboxes) and `stamp_ocr` where `model = 'sonnet'` (for ground truth) from Postgres.
- Applies stratified sampling as above.
- Copies source JPGs from `scanmyphotos/` to `state/bench/corpus/<stem>.jpg`.
- Writes `state/bench/manifest.json`:
  ```json
  {
    "generated_at": "2026-04-20T...",
    "stratification": {"confidence_buckets": [...], "era_buckets": [...]},
    "stems": [
      {
        "stem": "d1_00000123",
        "bbox": {"x": 0.5, "y": 0.95, "w": 0.2, "h": 0.05},
        "bbox_source": "yolo",
        "yolo_confidence": 0.87,
        "sonnet_raw_text": "10 3'99",
        "sonnet_normalized_date": "1999-10-03"
      },
      ...
    ]
  }
  ```
- The `state/bench/` directory (corpus + manifest) is the portable unit: rsync to ares, scp to M2 Pro, s3 sync to a cloud GPU box. It sits alongside the existing `state/shards/` pattern used by the parallel OCR orchestrator.

### 2. `scripts/ocr/bench_vlm_ocr.py`

The portable runner. Refactored from `scripts/ocr/ocr_gemma.py` — most of the OCR plumbing (crop, b64, normalize_date, upsert) carries over.

CLI:
```
bench_vlm_ocr.py --model <ollama-tag>
                 --host <ollama-url>          # default http://localhost:11434
                 --corpus state/bench/manifest.json
                 --corpus-images state/bench/corpus
                 --output postgres://... | jsonl://state/bench/results/
                 --host-label <ares-cpu|m2pro|cloud-gpu-g5xlarge>
                 [--limit N] [--resume]
```

Per-image flow:
1. Load image from `--corpus-images/<stem>.jpg`.
2. Crop using the bbox from manifest (same `PAD_FACTOR = 0.5` as `ocr_gemma.py`).
3. Base64-encode with `max_side=512`.
4. Call Ollama `/api/chat` with the same prompt as `ocr_gemma.py`, `temperature=0`, `num_predict=2048` (raised from 512 to absorb thinking tokens), `"think": false`.
5. Receive `message.content`. Strip `<think>…</think>` and `<thinking>…</thinking>` with a multiline regex. If stripped content is empty, fall back to the last non-empty line of the raw response.
6. Apply `normalize_date()` from `ocr_gemma.py` to produce `normalized_date`.
7. Write a row: `(stem, model, host_label, raw_text, normalized_date, elapsed_s, eval_count, prompt_eval_count, bbox_source)`.

Error handling:
- **Per-image timeout 180s** — record `raw_text="TIMEOUT"`, `normalized_date=NULL`, `elapsed_s=180`, continue.
- **Ollama OOM (HTTP 500 with "out of memory")** — record `raw_text="OOM_ERROR"`, `normalized_date=NULL`, continue. Surfaces in the report as a per-model OOM rate.
- **Postgres unreachable** — fall back to JSONL shard next to the corpus. `report_vlm_bench.py` imports orphan shards on its next run.
- **Model pull failure** — abort before processing any images; no partial results under a wrong model label.
- **No retries.** Flakiness on a given host is signal, not noise.

### 3. `scripts/ocr/report_vlm_bench.py`

Read-only analysis. Run on any host with DB access.

- Pulls all `stamp_ocr` rows where `stem` is in the manifest and `model` is in the bench slate, plus Sonnet rows for the same stems.
- Ingests any orphan JSONL shards in `state/bench/results/` into `stamp_ocr` first.
- For each (model, host_label) pair, computes:
  - `agree_pct` — normalized_date equality with Sonnet, across 200 stems
  - `unparsed_pct` — rows where `normalized_date IS NULL` and `raw_text` is not a known error code
  - `timeout_pct` — rows with `raw_text='TIMEOUT'`
  - `oom_pct` — rows with `raw_text='OOM_ERROR'`
  - `high_conf_wrong_pct` — rows with non-NULL `normalized_date` that disagrees with Sonnet
  - `median_s`, `p95_s` — latency distribution
  - `imgs_per_sec` — 1 / median_s (rough throughput proxy; the bench is sequential)
- Emits:
  - `output/vlm_bench_report.md` — markdown table, Pareto shortlist, notable failure modes
  - `output/vlm_bench_pareto.png` — scatter of agree_pct vs imgs_per_sec, bubble size = model params, color = host_label

### 4. Profile wrappers

Three shell wrappers that set env vars and invoke the runner, so host-specific glue is visible and the Python stays clean:

- `scripts/ocr/bench_profiles/ares-cpu.sh`
- `scripts/ocr/bench_profiles/m2pro.sh`
- `scripts/ocr/bench_profiles/cloud-gpu.sh`

Each one handles: Ollama endpoint URL, corpus path resolution, output destination (Postgres DSN vs local JSONL), `--host-label`, and any host-specific Ollama env (e.g., `OLLAMA_NUM_PARALLEL=1` on ares-cpu to avoid RAM pressure).

## Schema delta

Add one column to `stamp_ocr`:

```sql
ALTER TABLE stamp_ocr ADD COLUMN host_label TEXT;
```

Nullable, defaults NULL for the existing ~13K rows (Sonnet, Haiku, Gemma baseline). New bench rows populate it. The composite PK stays `(stem, model)`; if a model is benched on multiple hosts, we decide at rollout time whether to extend the PK to `(stem, model, host_label)` or to accept that each (model, host) pair writes under a distinct `model` value like `kimi-vl@ares-cpu`.

**Decision deferred to the plan:** pick one of those two at the start of implementation. The model-tag-suffix approach is simpler and avoids a schema migration on a live table; the composite-PK approach is cleaner. Either works for the report.

## Testing

- `tests/test_bench_normalize.py` — direct tests for the thinking-block stripper: `<think>...</think>` removal, multi-line answers, empty-after-strip fallback, nested tags (shouldn't happen but test anyway).
- `tests/test_bench_corpus.py` — stratified sampling produces 200 stems, coverage across all 8 cells (4 conf × 2 era) ≥ 15 per cell (allow skew), every stem has a Sonnet ground-truth row.
- **Smoke test:** `bench_vlm_ocr.py --corpus <manifest> --limit 3 --model gemma3 --output jsonl://...` run end-to-end; verifies plumbing, thinking-block stripping path, manifest-driven bbox application.
- No mocks of Ollama. The smoke test calls a real model.

## Estimated runtime

200 images × 6 models = 1,200 inferences per host.

- **ares-cpu** — ~30s/img average across the slate → ~10h total, runs overnight in a Docker container.
- **m2pro** — Apple Silicon Metal is 5-10× faster than CPU for 7B models in Q4 → rough estimate 1-2h total.
- **cloud-gpu (g5.xlarge)** — ~1-3s/img → <1h total, ~$1-3 for the spot instance lifetime.

The Pareto report is only meaningful once at least two hosts have run each model, so plan to run at minimum `ares-cpu + m2pro` or `ares-cpu + cloud-gpu`.

## Open questions deferred to the plan

1. Exact Ollama tag for `minicpm-v` and `gemma3` — pin at plan-start by querying `ollama.com/search` or `ollama list` on a host that has them.
2. Whether `ares` Ollama container can hold llama3.2-vision:11b in 27GB RAM alongside other workloads, or whether that model must be tested only on m2pro / cloud-gpu.
3. PK strategy for multi-host rows (see Schema delta).

## Risks

- **Thinking-model token explosion** — kimi-vl is explicitly a thinking model. Despite `num_predict=2048` and the strip regex, it may still truncate answers. Mitigation: monitor `eval_count` vs budget; if consistently maxing out, bump to 4096 and note in report.
- **Ollama OOM on ares** — 27GB host RAM is tight for llama3.2-vision:11b + internvl3:8b concurrently. Run one model at a time; the `OLLAMA_KEEP_ALIVE=0` env var forces model unload between requests if needed.
- **Corpus drift** — if `stamp_ocr` gets re-run against Sonnet between corpus build and report, ground truth shifts. Freeze the corpus's Sonnet rows into the manifest at build time (already in the spec above) and report against the frozen values, not live DB.
- **Prompt bias** — the current prompt was tuned for Haiku/Sonnet/Gemma. Models behaving worse may be prompt-sensitive rather than genuinely worse. Out of scope for this bench; flag in the report if a model produces consistent structural errors (e.g., always English-month-name instead of digits).

## Deliverables

- 3 new scripts in `scripts/ocr/` + 3 profile wrappers in `scripts/ocr/bench_profiles/`.
- 2 new test modules in `tests/`.
- Schema migration (adds `host_label` column).
- `state/bench/` directory with 200 JPGs + manifest (gitignored; archived separately if needed).
- `output/vlm_bench_report.md` + `output/vlm_bench_pareto.png`.
- A follow-up decision memo (outside this spec) recommending which model(s) to deploy based on the report.
