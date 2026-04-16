# Handoff: Parallel Sonnet OCR Pipeline

**Date:** 2026-04-14
**Goal:** OCR all 6,458 ScanMyPhotos date stamps using Sonnet stage-1 + Opus stage-2 review, dispatched as Claude Code background subagents.

## Current Status

### ✅ Done

- **Orchestrator built**: [yolo_finetune/scripts/orchestrate_ocr.py](../../yolo_finetune/scripts/orchestrate_ocr.py) — pure-IO pipeline with subcommands (status, crop-stage1, merge-stage1, list-shards, requeue, crop-stage2, merge-stage2). 42 pytest tests pass.
- **Stage-1 OCR complete**: All **6,458 / 6,458** photos transcribed by Sonnet (state/ocr_results.json).
  - **OK**: 5,636 (87.3%) — clean dates matching `M D 'YY`, `M D'YY`, or `'YY M D` formats
  - **NONE**: 213 (3.3%) — no stamp visible
  - **REVIEW**: 767 (11.9%) — auto-flagged by trigger rules (`?` in text, regex mismatch, low YOLO conf)
- **Pilot review HTML** at [yolo_finetune/pilot_review.html](../../yolo_finetune/pilot_review.html) with:
  - Per-card photo + crop + OCR text + metadata
  - Filter buttons (All / OK / Review / NONE)
  - **Manual Opus flag checkboxes** (localStorage-persisted, downloads `manual_opus_flags.json`)
  - **Manual rotation override buttons** (0/90/180/270, localStorage-persisted)
  - CSS rotation transforms applied based on EfficientNetV2 predictions
- **Local VLM benchmark**: ran `gemma4:e4b` and `qwen3-vl:4b` on shard 0000 for comparison. **Decision: Sonnet wins.** Gemma4 is ~74% match with Sonnet (mostly formatting noise) and runs ~32 hours CPU; Qwen3-VL has reasoning-loop issues and produces empty outputs.
- **Sonnet ≫ Haiku** on accuracy: Haiku misread `'95` as `'99` consistently; Sonnet was correct. We re-ran the first 200 stems after switching from Haiku → Sonnet.
- **Plan + spec** committed to `/home/will/photo_project/docs/superpowers/{specs,plans}/2026-04-14-parallel-haiku-ocr-review*.md`.

### 🔄 In progress (background)

- **Rotation detection** running on all 6,458 photos via `scripts/detect_rotation_batch.py` using the existing fine-tuned EfficientNetV2-S model at `/home/will/photo_project/models/orientation_model_v2_0.9882.pth` (98.82% accuracy). At ~3,600 / 6,458 (56%) when handoff written. Output: [yolo_finetune/state/rotation_predictions.json](../../yolo_finetune/state/rotation_predictions.json). ETA ~3 min from handoff time.
  - Distribution so far: ~71% upright, ~22% need 90° CW, ~5% need 270° CW, ~1% need 180°
- **HTTP server** serving `pilot_review.html` on `192.168.0.2:8000` (Bash background ID `bkfsmnm8a`). Will need to be restarted in a fresh session.

### ⏸ Pending

- Stage-2 Opus review (not started)

## Next Steps

1. **Wait for rotation detection to finish** (or check `state/rotation_predictions.json` count). Run `tail /tmp/rotation-run.log` if curious — process ID was a Bash background task.
2. **Regenerate the HTML** so the full rotation set displays:
   ```
   cd /home/will/photo_project/yolo_finetune && uv run scripts/build_pilot_review_html.py
   ```
3. **Restart the HTTP server** if it stopped:
   ```
   cd /home/will/photo_project/yolo_finetune && python3 -m http.server 8000 --bind 192.168.0.2
   ```
4. **User reviews** http://192.168.0.2:8000/pilot_review.html and:
   - Adds manual Opus flags (clicks Opus checkbox on cards needing review)
   - Adjusts rotations where the CNN was wrong (clicks 0/90/180/270 buttons)
   - Downloads `manual_opus_flags.json` and saves it to `yolo_finetune/state/manual_opus_flags.json`
5. **Update orchestrator stage-2 trigger to incorporate** (NOT YET IMPLEMENTED):
   - Existing auto-flags from `should_review()` (text-based triggers) ✓ already works
   - **Manual flags** from `state/manual_opus_flags.json` — NEED to wire in
   - **Rotation != 0** stems with CNN confidence > 0.95 — auto-apply, don't review
   - **Rotation != 0** with CNN confidence < 0.85 — flag for stage-2 (cross-check)
   - This is in `cmd_crop_stage2()` and/or `select_review_stems()` — needs editing
6. **Run crop-stage2** to pre-crop two views (large padded crop + full image) per flagged stem:
   ```
   uv run scripts/orchestrate_ocr.py crop-stage2
   ```
   Then update it to rotate the source image first per the final rotation decision before cropping.
7. **Dispatch Opus stage-2 workers** in parallel waves (use `model: "opus"` in Agent calls). Each worker reads `scripts/subagent_prompts/stage2_prompt.md` + the manifest path. Opus reviews two independent views per stem.
8. **Merge stage-2 results** with `merge-stage2 <result.json>` — auto-reconciles agree/disagree, writes manual disagreements to `state/ocr_manual_queue.json`.
9. **Final HTML** with reconciled results, plus a summary report.
10. **Commit + finalize the branch** (currently `feat/parallel-haiku-ocr` in the yolo_finetune repo).

## Key Context

### Architecture decisions

- **Sonnet for stage-1, Opus for stage-2** (cost-tiering on subscription).
- **Rotation is a presentation problem, not OCR**: Sonnet reads rotated text correctly, so we don't re-OCR rotated photos. Rotation matters for the HTML review and the final archive only.
- **Stage-1 trigger rules** (`scripts/orchestrate_ocr.py:should_review`):
  - Text contains `?`
  - Confidence < 0.3
  - Text is neither `NONE` nor matches `^(?:\d{1,2} \d{1,2} ?'\d{2}|'\d{2} \d{1,2} \d{1,2})$` (covers all 3 known stamp formats)
- **Three valid stamp formats discovered**:
  - `M D 'YY` (space before apostrophe) — e.g., `10 3 '99`
  - `M D'YY` (no space) — e.g., `9 6'95`
  - `'YY M D` (year first, older cameras) — e.g., `'94 8 23`
- **Shard size**: 50 stems per stage-1 shard, 25 per stage-2 shard. 126 stage-1 shards total, all done.
- **Parallelism**: I dispatched ~10 background subagents per wave; the harness handled queuing fine.
- **Catch-up merge pattern**: `for f in state/shards/stage1/shard_*_result.json; do uv run scripts/orchestrate_ocr.py merge-stage1 "$f"; done` — idempotent, run any time to absorb completed shards.

### Critical constraints (from CLAUDE.md)

- **NEVER modify files on the HDD** at `/mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/` (read-only source).
- The new orchestrator only modifies files inside `yolo_finetune/`. Source images live in `yolo_finetune/scanmyphotos/` (gitignored, populated from the dedup pipeline).

### Pre-existing modified files — DO NOT TOUCH

These were dirty before this session started and should not be in any of the new commits:
- `yolo_finetune/CLAUDE.md`
- `yolo_finetune/docker/docker-compose.yml`
- `yolo_finetune/justfile`
- `yolo_finetune/scripts/train.py`
- `yolo_finetune/scripts/gpu_bench_one_epoch.py` (untracked)

### Bug fixes worth remembering

- `cmd_crop_stage1` originally restarted `shard_index = 0` on each invocation, **clobbering existing manifests** when called incrementally. Fixed in commit `9433f6d` to scan existing shard files and continue numbering past them.
- The first stamp format regex was too strict (`^\d{1,2} \d{1,2} '\d{2}$`). Loosened twice: first to allow optional space before apostrophe (`b3e7116`), then to also accept year-first format (`dfdf13c`).

## Files Touched

### New files (committed on `feat/parallel-haiku-ocr`)

- `yolo_finetune/scripts/orchestrate_ocr.py` (main orchestrator, ~470 lines)
- `yolo_finetune/scripts/subagent_prompts/stage1_prompt.md`
- `yolo_finetune/scripts/subagent_prompts/stage2_prompt.md`
- `yolo_finetune/tests/__init__.py`
- `yolo_finetune/tests/test_orchestrate_ocr.py` (42 tests)

### New files (uncommitted, in working tree)

- `yolo_finetune/scripts/build_pilot_review_html.py` (renders the review HTML)
- `yolo_finetune/scripts/detect_rotation_batch.py` (rotation detection wrapper)
- `yolo_finetune/scripts/ocr_ollama_bench.py` (one-off comparison script for local VLMs)
- `yolo_finetune/pilot_review.html` (output, gitignored — regenerate from script)

### Runtime data (gitignored)

- `yolo_finetune/state/ocr_results.json` (6,458 stem entries)
- `yolo_finetune/state/rotation_predictions.json` (in progress)
- `yolo_finetune/state/shards/stage1/shard_NNNN.json` + `shard_NNNN_result.json` (126 manifest+result pairs)
- `yolo_finetune/output/ocr_crops_stage1/d1_NNNNNNNN.jpg` (6,458 cropped JPEGs, 512px max side)

### Docs

- `/home/will/photo_project/docs/superpowers/specs/2026-04-14-parallel-haiku-ocr-review-design.md`
- `/home/will/photo_project/docs/superpowers/plans/2026-04-14-parallel-haiku-ocr-review.md`

### Branch state

- Repo: `/home/will/photo_project/yolo_finetune` (its own git repo, NOT the parent `photo_project`)
- Branch: `feat/parallel-haiku-ocr`
- 16 commits ahead of `main`
- Last commit: `dfdf13c fix(ocr): accept year-first 'YY M D stamp format`

## Blockers

- **None hard-blocking**, but the next-session orchestrator needs to:
  1. Decide whether to wire manual flags + rotation triggers into `crop-stage2` BEFORE running it, or do a separate "merge external flags" pre-step.
  2. Update `cmd_crop_stage2` to rotate the source image before cropping when rotation != 0 (otherwise stage-2 crops are still upside-down).
  3. The current stage-2 prompt assumes upright crops — should be fine if we rotate before cropping.
- **Subscription usage**: stage-2 will be ~767 (auto) + manual + ~rotation-triggered = probably 1,000-1,500 Opus calls × 2 views = 2,000-3,000 image reads. Worth considering 5-hour-window capacity.
