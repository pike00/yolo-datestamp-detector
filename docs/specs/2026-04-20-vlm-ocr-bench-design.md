# VLM OCR Bench — Design

**Date:** 2026-04-20
**Status:** Design approved, pending implementation plan
**Goal:** Identify one or more local Ollama vision models that can replace Sonnet for date-stamp OCR on the 6,458-photo ScanMyPhotos corpus and future scans.

## Background

In 2026-04-14 a prior bench compared Gemma4:e4b (74% agreement with Sonnet on a 50-image shard, ~18s/img CPU) and Qwen3-VL:4b (32% agreement due to unsuppressed thinking tokens, ~67s/img CPU). Both fell short of the 94% OK rate Sonnet produced on the full 6,458-photo run, so Sonnet was used for production.

Since then the OSS VLM landscape has moved — notably Moonshot's Kimi-VL-A3B-Thinking (MoE, ~3B active), qwen2.5-vl, MiniCPM-V v2.6/v4, Gemma3 multimodal, Llama3.2-Vision, and InternVL3 are all on Ollama. This bench decides whether any of them are production-ready replacements for Sonnet.

**Ground-truth note:** The original Sonnet OCR rows for the 6,458-photo run were lost in the 2026-04-16 dedup Postgres volume overwrite, along with `corrections_queue.json`. `stamp_predictions` has since been rebuilt by re-running YOLO (2,132 rows as of 2026-04-20). Ground truth for this bench is therefore regenerated from scratch using the Claude Code subagent OCR pattern (same approach as the prior parallel-haiku-ocr project: subagents Read each crop file and transcribe via Sonnet under the user's Claude subscription — no direct API billing).

## Non-goals

- Re-OCR of all 6,458 ScanMyPhotos photos. Only the 200 bench stems get fresh Sonnet OCR.
- Prompt engineering across many variants per model. One prompt, same wording as the current `ocr_gemma.py`.
- Fine-tuning. Pure off-the-shelf eval.
- Stage-2 (low-confidence review) benchmarking. Stage-1 only.
- Direct Sonnet API calls. All Sonnet invocations go through the Claude Code subagent pattern to stay inside the subscription.

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

200 stems, stratified at selection time by **YOLO confidence bucket only** (since Sonnet ground truth no longer exists to key off of at selection time):

- Confidence buckets: `[0.0, 0.3), [0.3, 0.6), [0.6, 0.85), [0.85, 1.0]` — 4 buckets, 50 stems each. If a bucket is underfilled in the 2,132 available predictions, redistribute from over-filled buckets and note the skew in the manifest.

**Post-hoc era coverage:** once ground truth is generated (next step), the report surfaces era-bucket coverage (19xx vs 20xx) as a diagnostic. If heavily skewed (e.g., >80% 19xx), the user can optionally add a second round of sampling to balance — this is a manual call, not automatic.

## Hosts

The bench harness is portable and runs on any host with Ollama + Python + access to the corpus:

1. **`--profile ares-cpu`** — Docker container on ares, pairs with the existing ollama container, writes to Postgres directly over the internal network.
2. **`--profile m2pro`** — Native Ollama on the user's M2 Pro MacBook; corpus copied over Tailscale, results written to ares Postgres over Tailscale or emitted as JSONL for later import.
3. **`--profile cloud-gpu`** — Ephemeral g5.xlarge spot (or similar) with Tailscale join; results streamed to ares Postgres. Launched via existing AWS spot pattern when willing to spend ~$2-5 for fast turnaround.

The same model can be benched on multiple hosts — each (model, host) pair is a distinct row in the report.

## Components

### 1. `scripts/ocr/build_bench_corpus.py`

One-time packager. Run on the host that has read access to the ScanMyPhotos source photos.

- Reads `stamp_predictions` from Postgres for bbox coordinates.
- Applies Phase A stratified sampling: 50 stems per YOLO confidence bucket (4 buckets × 50 = 200).
- Copies source JPGs from `scanmyphotos/` to `state/bench/corpus/<stem>.jpg`. Also writes `state/bench/crops/<stem>.jpg` — pre-cropped + padded stamp regions — so Sonnet ground-truthing and downstream Ollama runs use byte-identical inputs.
- Writes `state/bench/manifest.json`:
  ```json
  {
    "generated_at": "2026-04-20T...",
    "stratification": {"confidence_buckets": [...]},
    "stems": [
      {
        "stem": "d1_00000123",
        "bbox": {"x": 0.5, "y": 0.95, "w": 0.2, "h": 0.05},
        "bbox_source": "yolo",
        "yolo_confidence": 0.87,
        "crop_path": "state/bench/crops/d1_00000123.jpg"
      },
      ...
    ]
  }
  ```
  Ground-truth fields (`sonnet_raw_text`, `sonnet_normalized_date`) are filled in by the next step, not this one.
- The `state/bench/` directory (corpus + crops + manifest) is the portable unit: rsync to ares, scp to M2 Pro, s3 sync to a cloud GPU box. It sits alongside the existing `state/shards/` pattern used by the parallel OCR orchestrator.

### 2. `scripts/ocr/seed_bench_ground_truth.py`

Regenerates the Sonnet ground truth for the 200 bench stems, using the Claude Code subagent OCR pattern — no direct Anthropic API calls. Modeled on the existing `scripts/ocr/orchestrate_ocr.py` parallel-haiku orchestrator pattern.

- Reads the manifest, iterates the 200 stems.
- For each stem, dispatches a Claude Code Task (subagent) with a minimal prompt: "Read `state/bench/crops/<stem>.jpg` and output ONLY the date-stamp text you see — orange LED digits. Example: `10 3'99`. Output exactly one line."
- Collects the subagent's last-line output as `raw_text`; applies `normalize_date()` from `ocr_gemma.py` to produce `normalized_date`.
- Writes rows to `stamp_ocr` with `model='sonnet'`, `host_label='claude-code-subagent'`. Idempotent via the `(stem, model)` PK: re-running skips stems already present (subject to the `--force` flag).
- Dispatches in small parallel waves (default 5 concurrent subagents) to match the prior `parallel-haiku` pattern and avoid overwhelming the local scheduler.
- **Human spot-check gate:** after the 200 rows are written, the script emits `state/bench/ground_truth_review.html` — a simple gallery of `<crop thumbnail, raw_text>` pairs sorted by parse failure first. The user skims and corrects obvious errors before flipping a `ground_truth_frozen=true` flag in the manifest. Until that flag is set, the report explicitly labels its accuracy numbers as "vs unverified Sonnet ground truth."

### 3. `scripts/ocr/bench_vlm_ocr.py`

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
1. Load the pre-cropped image from `state/bench/crops/<stem>.jpg` (byte-identical to what Sonnet saw during ground-truth generation, eliminating crop-vs-bbox drift between models).
2. No re-cropping. The full pre-cropped file is sent.
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

### 4. `scripts/ocr/report_vlm_bench.py`

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

### 5. Profile wrappers

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

Nullable. `stamp_ocr` is currently empty (post data-loss); the only rows going in will be the new bench rows. The composite PK stays `(stem, model)`; if a model is benched on multiple hosts, we decide at rollout time whether to extend the PK to `(stem, model, host_label)` or to accept that each (model, host) pair writes under a distinct `model` value like `kimi-vl@ares-cpu`.

**Decision deferred to the plan:** pick one of those two at the start of implementation. The model-tag-suffix approach is simpler and avoids a schema migration on a live table; the composite-PK approach is cleaner. Either works for the report.

## Testing

- `tests/test_bench_normalize.py` — direct tests for the thinking-block stripper: `<think>...</think>` removal, multi-line answers, empty-after-strip fallback, nested tags (shouldn't happen but test anyway).
- `tests/test_bench_corpus.py` — stratified sampling produces 200 stems with 50 per confidence bucket (allow ±5 skew if a bucket is underfilled), every stem has a YOLO `stamp_predictions` row and a readable source JPG.
- `tests/test_bench_ground_truth.py` — subagent-result parsing: last-non-empty-line extraction, dispatch wrapper doesn't crash on empty subagent output, idempotency skip when row already exists.
- **Smoke test:** `bench_vlm_ocr.py --corpus <manifest> --limit 3 --model gemma3 --output jsonl://...` run end-to-end; verifies plumbing, thinking-block stripping path, pre-cropped-image loading.
- No mocks of Ollama. The smoke test calls a real model.

## Estimated runtime

**One-time setup (ground truth):**
- `build_bench_corpus.py` — seconds.
- `seed_bench_ground_truth.py` — 200 subagent dispatches in waves of 5. Claude Code subagent startup overhead dominates per-image latency; budget ~10-20s per stem wall-clock → ~30-70 minutes total. Plus the human spot-check skim, call it ~1-1.5 hours end-to-end.

**Per-host bench run:** 200 images × 6 models = 1,200 inferences per host.

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
- **Ground-truth correctness** — Sonnet ground truth is regenerated via Claude Code subagents, not a direct API. Subagent answers are Sonnet-quality by construction but variance between runs (model cache changes, prompt interpretation) is unknown. Mitigation: the spot-check HTML gate forces a human to review all 200 before the bench flips `ground_truth_frozen=true`.
- **Prompt bias** — the current prompt was tuned for Haiku/Sonnet/Gemma. Models behaving worse may be prompt-sensitive rather than genuinely worse. Out of scope for this bench; flag in the report if a model produces consistent structural errors (e.g., always English-month-name instead of digits).

## Deliverables

- 4 new scripts in `scripts/ocr/`: `build_bench_corpus.py`, `seed_bench_ground_truth.py`, `bench_vlm_ocr.py`, `report_vlm_bench.py`.
- 3 profile wrappers in `scripts/ocr/bench_profiles/`.
- 3 new test modules in `tests/`: `test_bench_normalize.py`, `test_bench_corpus.py`, `test_bench_ground_truth.py`.
- Schema migration (adds `host_label` column).
- `state/bench/` directory with corpus/, crops/, manifest.json, ground_truth_review.html (gitignored).
- `output/vlm_bench_report.md` + `output/vlm_bench_pareto.png`.
- A follow-up decision memo (outside this spec) recommending which model(s) to deploy based on the report.
