# Handoff: GPU Training Run and Drift Comparison

**Date:** 2026-04-14
**Goal:** Execute full 40-epoch YOLO training on AWS GPU and compare new predictions against prior model's predictions on 6458 scanned photos.

## Current Status

**Training: COMPLETE**
- 40-epoch run on `g4dn.xlarge` (on-demand) via [scripts/gpu_bench_one_epoch.py](../../scripts/gpu_bench_one_epoch.py)
- Instance `i-0f0b53e53bf3acd82`, launched ~18:11 UTC, terminated ~18:50 UTC
- Wall: 39.75 min, training: 33.55 min, per-epoch: 50.3s
- **Cost: $0.3485** (vs $5 hard cap — 7% utilization)
- Weights at [runs/detect/gpu-40ep/weights/best.pt](../../runs/detect/gpu-40ep/weights/best.pt) (44 MB)
- Also `last.pt` same size

**Validation metrics (post-40-epoch, not "after 1 epoch" — the script's output label is a stale template string):**
- mAP50: **0.9625** (vs 1-epoch bench 0.9389)
- mAP50-95: **0.7567** (vs 1-epoch bench 0.6685)
- precision: 0.9670
- recall: 0.9652

**Drift comparison: IN PROGRESS (background task)**
- Script: [scripts/compare_predictions.py](../../scripts/compare_predictions.py) (new file)
- Task ID: `bblfl4doy`, log at `/tmp/yolo_compare.log`
- Comparing 6458 old predictions in [state/scanmyphotos_predictions.json](../../state/scanmyphotos_predictions.json) against fresh inference with new weights
- Smoke test (32 stems) showed ~1.9 img/s on CPU, all stable
- ETA ~57 min from launch
- Will write report to `state/prediction_drift.json` with {old, new, iou, flag} per stem
- Flag taxonomy: `stable` (IoU ≥ 0.5), `drift` (IoU < 0.5), `gone` (new model finds nothing)

## Next Steps

1. **Check comparison completion**: `tail /tmp/yolo_compare.log` or read `state/prediction_drift.json` — count flagged entries by category
2. **Review drifted photos**: Inspect `state/prediction_drift.json` for entries with `flag ∈ {drift, gone}`. Build a visual review UI or use existing [ui/dashboard.html](../../ui/dashboard.html) / [ui/batch_review.html](../../ui/batch_review.html) wired to the drift set
3. **Decide model promotion**: If the new weights improve (or at least don't regress) on the drift set, copy `runs/detect/gpu-40ep/weights/best.pt` to [runs/detect/train/weights/best.pt](../../runs/detect/train/weights/best.pt) so [scripts/infer_all.py](../../scripts/infer_all.py) picks it up (its `MODEL_PATH` is hardcoded there)
4. **Re-run inference on full 7456 images**: Currently only 6458 have predictions. Running `./scripts/infer_all.py` after model promotion will fill in the remaining ~1000 pending.
5. **Commit the uncommitted work** (see Files Touched) — the `feat/parallel-haiku-ocr` branch now carries unrelated GPU training work; consider splitting into a separate branch before PR

## Key Context

**Why the script is named "gpu_bench_one_epoch" even though it does full training**: This was originally a benchmarking harness. Passing `--epochs 40` makes it do a full training run. The script output labels ("GPU BENCH RESULTS", "validation metrics (after 1 epoch)") are cosmetic/stale and do not reflect reality when `--epochs > 1`.

**Spot capacity failure**: First launch attempt failed — 5 of 6 AZs had no g4dn.xlarge spot capacity and the 6th (us-east-1e) doesn't offer g4dn.xlarge at all, which the error handler didn't skip. Patched in this session to continue past `Unsupported` errors like it already did for `InsufficientInstanceCapacity`. Final run used `--pricing on-demand` for reliability.

**Per-epoch time was way below the 1-epoch bench projection**: Bench measured 117s/epoch on a 1-epoch run; actual 40-epoch run averaged 50.3s/epoch. The bench was inflated by first-epoch compile/warmup cost that didn't amortize over a single epoch. Future projections should discount the first-epoch cost.

**Cost monitor (killed at end of run)**: A persistent background monitor polled EC2 every 3 min and would have terminated the instance if wall-time × $0.5342/hr (on-demand + EBS ceiling) exceeded $4.50. It fired ~13 times across the run and never tripped. Log: `/tmp/yolo_cost_monitor.log`

**No GPU visibility during training**: The script captures no per-epoch progress signal retrievable from outside the instance. CloudWatch CPU metrics (1-min resolution) are the only remote signal — during this run, 38% → 64% sustained indicated training was active. If live epoch tracking matters for future runs, add `while true; do curl --upload-file /var/log/user-data.log "$log_put"; sleep 60; done &` in `build_user_data` before the docker run. Script also has a no-op tensorboard upload patch applied at the tail end of this session — current run didn't benefit because its user-data was already baked.

**Tensorboard artifacts patch**: At the tail of this session, [scripts/gpu_bench_one_epoch.py](../../scripts/gpu_bench_one_epoch.py) was updated so the artifact tar includes the full `runs/bench/` directory (weights + `events.out.tfevents.*` + curves + `results.csv`), not just `weights/`. The renamed vars are `run_key` / `run_put` / `run.tar.zst`. Future runs will land at `runs/detect/gpu-40ep/bench/` (nested). The current in-flight drift comparison doesn't care about this.

**Inference-size choice for drift comparison**: Used `INFERENCE_SIZE=384` and `CONF_THRESHOLD=0.01` to match [scripts/infer_all.py](../../scripts/infer_all.py) so the comparison isolates the model change, not the resize change. Old predictions were also generated at 384.

## Files Touched

**Modified, uncommitted:**
- [scripts/gpu_bench_one_epoch.py](../../scripts/gpu_bench_one_epoch.py) — weights/artifact upload, Unsupported AZ skip, dynamic presigned URL expiry, run.tar.zst artifact key

**Created, uncommitted:**
- [scripts/compare_predictions.py](../../scripts/compare_predictions.py) — drift report generator
- [runs/detect/gpu-40ep/weights/best.pt](../../runs/detect/gpu-40ep/weights/) + `last.pt` (44 MB each, gitignored via `runs/`)
- `state/prediction_drift.json` (in progress — being written by background task)

**Pre-existing uncommitted (not from this session's work):**
- `CLAUDE.md`, `docker/docker-compose.yml`, `justfile`, `scripts/train.py`
- `docs/`, `pilot_review.html`, `scripts/build_pilot_review_html.py`, `scripts/ocr_ollama_bench.py`

## Blockers

- None. Drift comparison is running unattended; results available in `state/prediction_drift.json` when task `bblfl4doy` completes.
- Open question for next session: what's the threshold for "accept new model"? >95% stable? <5% gone? Needs a decision once drift numbers are in.
