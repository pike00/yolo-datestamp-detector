# Handoff: Opus Stage-2 Dispatch Progress

**Date:** 2026-04-15
**Goal:** Run Opus stage-2 verification on the 82 shards of stage-1-flagged date-stamp stems, then wire the results back into the final HTML and archive.

Continuation of [2026-04-14 Handoff - Manual Review UI and Stage-2 Prep.md](./2026-04-14%20Handoff%20-%20Manual%20Review%20UI%20and%20Stage-2%20Prep.md). The user opted to **skip the manual browser review entirely** and push straight through stage-2 on the auto-flagged set.

---

## Current Status

### Stage-2 Opus dispatch — **59 / 82 shards done (72%)**, paused by user

| | Count |
|---|---:|
| Shards merged | 59 / 82 |
| Shards pending (never dispatched) | 23 (shards 0059–0081) |
| Failed shards | 0 |
| Manual review queue | **162 stems** |

All Opus subagents from the last wave completed before the user called "stop." Nothing was mid-flight when we paused — the pipeline is in a clean state. The 23 pending shards were simply never dispatched.

### Rough per-wave stats

| Wave | Shards | Total conf | Total NONE | Total disagree | Notes |
|---|---|---:|---:|---:|---|
| 1 | 0000 | 25 | 0 | 0 | Clean warmup |
| 2 | 0001–0004 | 96 | 1 | 3 | Clean |
| 3 | 0005–0014 | ~240 | ~3 | ~10 | Clean (most tokens tracked pre-compaction) |
| 4 | 0015–0018 | 90 | 0 | 10 | Write-permission denials on 0015/0016 (see Gotchas) |
| 5 | 0019–0028 | 194 | 29 | 27 | Hit year-first `'93 MM DD` format pockets |
| 6 | 0029–0038 | 199 | 9 | 42 | High disagree rate (shards 0030/0036/0038) |
| 7 | 0039–0058 | 368 | 60 | 71 | Big NONE clusters (0042 had 18/25 NONE), high disagree on 0055/0056/0058 |

### What's in `ocr_manual_queue.json`

162 disagreement records at `yolo_finetune/state/ocr_manual_queue.json`. Each entry has:
```json
{
  "stem": "d1_00000405",
  "stage1_text": "9 25'93",
  "view_crop": "9 ??'93",
  "view_full": "9 25'93",
  "confidence": null
}
```
Reconciliation heuristic from a few manual spot-checks:
- If `view_full` matches `stage1_text` and `view_crop` just has extra `?`s → stage-1 is probably right, crop is too tight.
- If `view_full` sees a stamp and `view_crop == "NONE"` → bbox cropped off the stamp entirely.
- Year-first `'93 MM DD` format is a frequent disagreement source — the stage-2 crop often clips a digit.

### Known bad data flagged by a subagent

Shard 0015 agent noted: **`d1_00001280` and `d1_00001281` have stage-2 crops rotated 180° vs the full image.** Crops read `'94 12 11` but full images clearly show `11 21 '94`. This is the known rotation-aware-crop gap from the previous handoff (`cmd_crop_stage2` doesn't rotate the source before cropping). Probably a handful of other stems with `rotation != 0` are similarly broken.

### Usage

As of the pause: **37% of 5-hour session**, **47% weekly**, **17% weekly Sonnet**. User is on Max 20x. Session resets in ~3h from the check. The remaining 23 shards are ~1.4M tokens of Opus work (estimate: ~61k/shard × 23) — should fit in one session window. Weekly has ample runway.

---

## Next Steps

### Immediate — resume stage-2 dispatch

1. **Dispatch the remaining 23 shards** (0059–0081) in a single wave of ~10-20 parallel `Agent` calls with `model: "opus"`, `subagent_type: "general-purpose"`, `run_in_background: true`. Prompt template lives at [yolo_finetune/scripts/ocr/subagent_prompts/stage2_prompt.md](../../yolo_finetune/scripts/ocr/subagent_prompts/stage2_prompt.md). Substitute the shard number.
2. **Merge each** as its notification arrives: `cd /home/will/photo_project/yolo_finetune && uv run scripts/ocr/orchestrate_ocr.py merge-stage2 state/shards/stage2/shard_NNNN_result.json`
3. Or batch-catch-up:
   ```
   cd /home/will/photo_project/yolo_finetune
   for f in state/shards/stage2/shard_*_result.json; do
     uv run scripts/ocr/orchestrate_ocr.py merge-stage2 "$f"
   done
   ```
4. Final target: `uv run scripts/ocr/orchestrate_ocr.py status` showing `Stage-2 shards: 0 pending, 82 done`.

### Then — deal with the manual queue

5. The full queue will likely grow to **~200–220 disagreements**. Options:
   - **Browser review**: extend the existing 3x3 review UI at `scripts/ocr/build_pilot_review_html.py` to show `view_crop` / `view_full` / `stage1_text` side-by-side for disagreement entries. Probably the least painful path.
   - **Scripted heuristic**: for any disagreement where `view_full` matches `DATE_FORMAT_RE` and the crop doesn't, trust `view_full`. Probably resolves 50%+ automatically. Write as a new `orchestrate_ocr.py resolve-queue --auto` subcommand.
   - Hybrid: auto-resolve the easy ones, leave the hard ones for the browser.

### Then — regenerate HTML + commit

6. Regenerate: `uv run scripts/ocr/build_pilot_review_html.py`
7. Bake `review_status` per card into the HTML so the final state shows stage-2 verdicts.
8. **Commit the still-uncommitted `build_pilot_review_html.py` rewrite** from the previous handoff session (+298/−200 lines vs main). This has been sitting uncommitted across two sessions now.
9. Commit stage-2 orchestrator changes if any (there were none this session — only pipeline execution).

### Later — final archive step

- Rotate each source JPG by `rotation_override || cnn_rot` and write to the deduplicated archive. Separate script, out of scope for OCR.

---

## Key Context

### Why we skipped manual review

User explicitly asked to run Opus over the entire auto-flagged set and skip the browser review pass. Reasoning: stage-1 Sonnet is already high-quality; stage-2 Opus verification on the auto-flagged subset gives a bigger confidence bump per dollar than eyeballing thousands of clean cards. Trade-off: the manual queue grew faster (162 vs the ~600 originally anticipated, because disagreements propagate instead of being pre-filtered by a human).

### Subagent Write-permission flakiness

**Two shards (0015, 0016) hit Write-tool permission denials mid-run this session.** Other shards before and after worked fine. It's intermittent, not systemic.

Recovery pattern:
- Shard 0015's agent fell back to Bash (violated hard rules) but succeeded.
- Shard 0016's agent correctly reported the full prepared result JSON in its final reply; the dispatcher wrote it manually via the Write tool in the parent session.

**Mitigation added to prompts from shard 0019 onward:** "If Write is denied, retry once; if still denied, report the full prepared result JSON in your final reply so the dispatcher can write it." No failures since.

### Dispatch pattern

- **Wave size**: started at 1-4, scaled to 10, then 20. All worked. The harness queues cleanly.
- **Prompt**: inline the full stage2_prompt.md contents + SHARD_MANIFEST_PATH + BASE_DIR per agent. Don't point at the file — subagents read it fresh each time and that's one wasted tool call per agent.
- **Model**: always `opus`, always `run_in_background: true`.
- **Merge timing**: as each completion notification arrives, run `merge-stage2` on that one result. Idempotent, safe to re-run.

### Orchestrator still lives at

- Working dir: `/home/will/photo_project/yolo_finetune` (NOT the parent `photo_project` dir — `orchestrate_ocr.py` expects to be run from `yolo_finetune/`)
- Script: `scripts/ocr/orchestrate_ocr.py`
- State: `state/shards/stage2/shard_NNNN.json` (manifests) + `shard_NNNN_result.json` (results, exist for 59/82)
- Manual queue: `state/ocr_manual_queue.json`
- Crops: `output/ocr_crops_stage2_crop/*.jpg` + `output/ocr_crops_stage2_full/*.jpg` (unchanged this session)

---

## Files Touched

### Changes this session

- `yolo_finetune/state/shards/stage2/shard_{0000..0058}_result.json` — 59 new result files written by Opus subagents
- `yolo_finetune/state/ocr_manual_queue.json` — grew from 0 to 162 entries via `merge-stage2`
- `yolo_finetune/state/ocr_results.json` (via the orchestrator's `upsert_ocr_result` calls that modify Postgres, not JSON — Postgres is the source of truth for OCR state)

### NOT touched this session

- `scripts/ocr/orchestrate_ocr.py` — no changes
- `scripts/ocr/build_pilot_review_html.py` — **still uncommitted from the previous session** (+298 / −200 lines vs the merged version on main). Third session without landing this.
- Any rotation-awareness fix in `cmd_crop_stage2` — still deferred. Only a handful of stems (`d1_00001280`, `d1_00001281`, maybe a dozen others) are affected.

---

## Blockers

- **None hard-blocking.** All state is on disk, the orchestrator is idempotent, the remaining 23 shards can be dispatched whenever.
- **Subscription budget**: remaining 23 shards will cost roughly 28% of one 5-hour Opus window on Max 20x. Dispatch when the user's session counter has room.
- **Manual queue growth**: at current rate the final queue will be ~200–220 disagreements. If that's too much for browser review, prioritize the auto-resolve heuristic first.

---

## Quick Resume Checklist

```bash
cd /home/will/photo_project/yolo_finetune
uv run scripts/ocr/orchestrate_ocr.py status           # should show 23 pending, 59 done
uv run scripts/ocr/orchestrate_ocr.py list-shards stage2 | wc -l   # sanity check
```
Then dispatch shards 0059–0081 via the Agent tool following the wave pattern above.
