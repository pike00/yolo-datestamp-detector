# Parallel Haiku OCR with Low-Confidence Review

**Date:** 2026-04-14
**Status:** Design approved, pending implementation plan

## Goal

Transcribe camera date stamps for ~6,458 scanned photos (ScanMyPhotos Discs 1-4) by dispatching Claude Haiku subagents in parallel via the Claude Code Task tool, with a second review pass for any result that looks uncertain. Run on the user's Claude Code subscription — no Anthropic API key, no per-token billing.

Begin with a 200-photo pilot, inspect the results, then commit to the full run.

## Context

The existing sequential pipeline (`yolo_finetune/scripts/ocr_stamps.py`) already handles YOLO-bbox-guided cropping, Haiku prompting, cost tracking, and resume — but it is sequential, uses `ANTHROPIC_API_KEY`, and has no review step for uncertain outputs. This spec does not modify that script. It adds a parallel orchestrator that reuses the same crop logic and prompt while routing inference through Claude Code subagents.

Inputs that already exist:
- `yolo_finetune/state/scanmyphotos_predictions.json` — 6,458 YOLO bboxes
- `yolo_finetune/state/corrections_queue.json` — human-confirmed bboxes (override YOLO where present)
- `yolo_finetune/scanmyphotos/*.jpg` — source images
- `yolo_finetune/state/ocr_results.json` — results from prior sequential runs (resume target)

## Architecture

Two actors:

- **Main orchestrator** (Opus session driving the run) — owns all deterministic work: pre-cropping, sharding, dispatch, aggregation, reconciliation. Does no OCR itself.
- **Haiku subagents** (dispatched via Task tool with `run_in_background: true`) — each worker receives a shard manifest path, reads the cropped image files via the Read tool, transcribes the date stamp using a fixed prompt, and writes a shard result JSON via the Write tool. No Python code inside the subagent. The subagent's own multimodal inference is the OCR call.

Five subagents stay in flight at a time. As each completes, the orchestrator merges its shard result and dispatches the next pending shard.

### Data flow

```
scanmyphotos_predictions.json + corrections_queue.json
    │
    ▼
orchestrator: build pending set (skip stems already in ocr_results.json)
    │
    ▼
orchestrator: pre-crop each pending stem → output/ocr_crops_stage1/<stem>.jpg
    │
    ▼
orchestrator: write shard manifests → state/shards/stage1/shard_NNN.json (50 stems each)
    │
    ▼
orchestrator: dispatch up to 5 background Haiku subagents, one shard each
    │         subagents read crops, transcribe, write shard result JSONs
    ▼
orchestrator: merge shard results → ocr_results.json (as each shard completes)
    │
    ▼
orchestrator: scan ocr_results.json for low-confidence triggers
    │
    ▼
orchestrator: pre-crop stage-2 views → ocr_crops_stage2_crop/ + ocr_crops_stage2_full/
    │
    ▼
orchestrator: dispatch stage-2 review subagents (25 stems per shard, 2 views each)
    │
    ▼
orchestrator: reconcile review results
    ├─→ agreement → overwrite ocr_results.json entry, mark review_status: confirmed
    └─→ disagreement → ocr_manual_queue.json (both answers preserved, stage-1 unchanged)
```

## Components

### `scripts/orchestrate_ocr.py`

Single uv inline-script-header Python file. All subcommands are pure deterministic IO — no LLM calls. Dispatch itself (calling the Task tool) happens from the orchestrator Claude session, not from Python, because Python has no way to invoke Claude Code subagents. The script exists to do the IO-heavy deterministic work that the Claude session would otherwise clutter its context with.

Subcommands:

- `crop-stage1 [--limit N]` — build pending set and write stage-1 crops + shard manifests. `--limit` caps the pending set for the pilot run.
- `list-shards stage1|stage2` — print the paths of shard manifests that have no matching result file yet. The orchestrator reads this to decide what to dispatch next.
- `merge-stage1 <shard_result_path>` — validate and merge a single stage-1 shard result JSON into `ocr_results.json`.
- `crop-stage2` — scan `ocr_results.json` for triggers, pre-crop both views per triggered stem, write stage-2 shard manifests.
- `merge-stage2 <shard_result_path>` — validate stage-2 shard result, reconcile into `ocr_results.json` and `ocr_manual_queue.json`.
- `requeue <shard_path>` — mark a shard as pending again (used by the orchestrator when a subagent fails its first attempt).
- `status` — print summary: pending/done shard counts per stage, review queue size, manual queue size, failed shards.

### `scripts/subagent_prompts/stage1_prompt.md`

Exact prompt given to stage-1 subagents. Must contain:
- The OCR rules copied verbatim from `ocr_stamps.py` (preserve spacing, `?` for uncertain, `NONE` if absent, no reformatting)
- Instructions to Read each image in the shard manifest in order
- Instructions to Write results to the shard result path as a specific JSON schema
- A reminder that the subagent must not modify any other files

### `scripts/subagent_prompts/stage2_prompt.md`

Same rules as stage 1, but instructs the subagent to read two images per stem (the larger crop and the full image) and emit both transcriptions per stem. No reconciliation logic in the subagent — that stays in Python.

### Shard manifests

Stage-1 shard (`state/shards/stage1/shard_NNN.json`):
```json
{
  "shard_id": "0042",
  "result_path": "state/shards/stage1/shard_0042_result.json",
  "stems": [
    {"stem": "d1_00000133", "crop_path": "output/ocr_crops_stage1/d1_00000133.jpg", "bbox_source": "yolo", "confidence": 0.87},
    ...
  ]
}
```

Stage-1 shard result (written by subagent):
```json
{
  "shard_id": "0042",
  "results": {
    "d1_00000133": {"text": "10 3 '99", "bbox_source": "yolo", "confidence": 0.87}
  }
}
```

Stage-2 shard manifest adds a second crop path per stem:
```json
{"stem": "d1_00000133", "crop_path": "...stage2_crop/d1_00000133.jpg", "full_path": "...stage2_full/d1_00000133.jpg", "stage1_text": "1? 3 '99", "confidence": 0.22}
```

Stage-2 shard result:
```json
{"d1_00000133": {"view_crop": "10 3 '99", "view_full": "10 3 '99"}}
```

Agreement/disagreement is computed by `merge-stage2`, not by the subagent.

## Pre-cropping rules

### Stage 1
- Load source image, compute padded bbox from the YOLO prediction (or human correction if available) with `PAD_FACTOR=0.5` (unchanged from `ocr_stamps.py`)
- Downscale to max 512px on longest side
- Save as JPEG quality 90 to `output/ocr_crops_stage1/<stem>.jpg`

### Stage 2 — triggered when any of:
- stage-1 text contains `?`
- stage-1 text is neither `NONE` nor matches regex `^\d{1,2} \d{1,2} '\d{2}$`
- YOLO confidence is below 0.3

For each triggered stem, write two crops:
- **Large crop** (`ocr_crops_stage2_crop/<stem>.jpg`) — same bbox as stage 1, but `PAD_FACTOR=1.5` and **no downscale** (full resolution of the cropped region, capped at 1536px on the longest side to keep image tokens reasonable)
- **Full image** (`ocr_crops_stage2_full/<stem>.jpg`) — the whole source photo, downscaled only if max side > 1536px

The 1536px cap prevents a pathological image from blowing up a subagent's context.

## Reconciliation rules

For each stem in a stage-2 shard result:

1. Normalize both `view_crop` and `view_full` by trimming whitespace and collapsing internal spaces
2. If both are `NONE` → final text = `NONE`, `review_status: "no_stamp"`, overwrite in `ocr_results.json`
3. If both are identical (post-normalization) → final text = that string, `review_status: "confirmed"`, overwrite in `ocr_results.json`
4. Otherwise → append to `ocr_manual_queue.json` with both answers, stage-1 text, YOLO confidence, and both view paths. Leave `ocr_results.json` entry unchanged (preserves stage-1 result so no data is lost).

## Parallelism and reliability

- **Stage 1:** 50 stems per shard, up to 5 in-flight subagents
- **Stage 2:** 25 stems per shard (each stem costs two Read+transcribe calls, so effective 50 calls per shard), up to 5 in-flight
- Each shard merge saves `ocr_results.json` on disk so a crashed run loses at most the in-flight work
- A shard whose result file is missing, malformed, or missing stems is requeued once. If it fails a second time, the full list of stems is written to `state/failed_shards.json` and the run continues.
- `orchestrate_ocr.py status` can be run at any time to see progress

## Pilot gate

The 200-photo pilot is the first and only gate before committing to the full run. After the pilot:

- Report: stamps found vs `NONE` vs review-triggered, agreement rate in stage 2, manual-queue count, any failed shards
- Spot-check 10-20 results against source photos for correctness
- User approves go/no-go on the remaining 6,258

## File layout

New files:
- `yolo_finetune/scripts/orchestrate_ocr.py`
- `yolo_finetune/scripts/subagent_prompts/stage1_prompt.md`
- `yolo_finetune/scripts/subagent_prompts/stage2_prompt.md`
- `yolo_finetune/state/ocr_manual_queue.json` (created at first disagreement)
- `yolo_finetune/state/failed_shards.json` (created at first failure)

New gitignored directories:
- `yolo_finetune/output/ocr_crops_stage1/`
- `yolo_finetune/output/ocr_crops_stage2_crop/`
- `yolo_finetune/output/ocr_crops_stage2_full/`
- `yolo_finetune/state/shards/stage1/`
- `yolo_finetune/state/shards/stage2/`

Existing files reused unchanged:
- `yolo_finetune/state/scanmyphotos_predictions.json`
- `yolo_finetune/state/corrections_queue.json`
- `yolo_finetune/state/ocr_results.json` (appended to, not rewritten)
- `yolo_finetune/scripts/ocr_stamps.py` (kept for ad-hoc single-image runs)

## Explicit non-goals

- No changes to `ocr_stamps.py`
- No changes to YOLO inference, training, or bbox predictions
- No manual-queue review UI — disagreements land in a JSON file; building a reviewer is a follow-up
- No cost tracking — subscription only, no per-token accounting
- No retry of successful stage-1 results even if their text looks suspicious (the trigger rules determine review scope; ad-hoc re-reviews are a separate run)
- No fallback to the sequential `ocr_stamps.py` path if subagent dispatch is unreliable — if the pilot reveals parallelism problems, fix them before the full run rather than quietly switching paths

## Risks

- **Subscription usage:** ~6,458 stage-1 calls plus an expected 2,000–4,000 stage-2 calls (two per triggered stem) is meaningful 5-hour-window capacity. Start the full run when no other heavy session is active.
- **Background subagent parallelism:** 5 concurrent `run_in_background` subagents has not been exercised in this project. The pilot is the test.
- **Prompt drift:** prompts live in dedicated files under `subagent_prompts/` so they cannot silently diverge between runs. Any change to the prompts is a visible diff.
- **Subagent output validation:** subagents may return malformed JSON, extra prose, or miss stems. The merge step must validate shape and flag any shard that does not match the expected schema, rather than silently dropping stems.
