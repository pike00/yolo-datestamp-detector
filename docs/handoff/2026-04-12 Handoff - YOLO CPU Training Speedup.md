---
name: YOLO CPU Training Speedup
date: 2026-04-12
---

# Handoff: YOLO CPU Training Speedup

**Date:** 2026-04-12
**Goal:** Diagnose why YOLO fine-tune in this repo is taking ~14h/epoch on CPU and make it finish in a sane amount of time.

## Current Status

- User started a training run: epoch 1 finished at `11.4s/it × 4456 iters ≈ 14h`, meaning ~58 days at 100 epochs.
- Epoch 1 metrics already excellent: `mAP50=0.951`, `mAP50-95=0.715`. Model is converging fast; the runtime cost is the problem, not learning.
- Root-caused the slowdown. No code changes made yet.
- Training run from the transcript is presumably still going — user should **kill it** before restarting with the fixes below.

## Root Causes (ranked by impact)

1. **Augmented hard cases bloat training set 7×.** [scripts/train.py:131-148](../../scripts/train.py#L131-L148) pulls in `15,175` augmented images on top of `2,651` real ones → `17,824` total. Every epoch pays a 7× multiplier.
2. **Model is `yolo26s` (~9.9M params).** [scripts/train.py:229](../../scripts/train.py#L229) hardcodes it. `yolo26n` (~3M) would be 3–4× faster and is almost certainly sufficient for single-class date-stamp detection.
3. **`imgsz=640`.** [scripts/train.py:243](../../scripts/train.py#L243). Cost is ~quadratic in image size. 416 or 512 is plenty — inference recipe in justfile already uses `imgsz=384`.
4. **`epochs=100` with `batch=4`.** [scripts/train.py:241-244](../../scripts/train.py#L241-L244). Model already at mAP50=0.95 after epoch 1; 30–40 epochs with `patience=10` is enough.
5. **Non-issue: `workers=0`.** Forced by ultralytics itself at [trainer.py:160](~/.cache/uv/archive-v0/r8IyZ6-b_1feI5IgNOQ5G/ultralytics/engine/trainer.py#L160):
   ```python
   if self.device.type in {"cpu", "mps"}:
       self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading
   ```
   This is intentional and probably correct on a 5600G. Do not chase this.

## Stacked speedup estimate

Applying fixes 1–4:
- 7× (no aug) × 3× (nano) × 2.4× (416) × 3× (30 epochs) ≈ **~150× faster**
- 58 days → **~9 hours total**, still CPU. Finishable overnight.

**Or just rent a GPU.** Any ~$0.50/hr cloud GPU (T4, A10, RTX 4000) finishes the original config in 20–40 min. ROI is massive vs hand-tuning CPU knobs.

## Next Steps

1. **Kill the current training run** if still alive (`docker compose -f docker/docker-compose.yml stop train` or equivalent).
2. **Drop augmented data**: `just augment-clean`.
3. **Download `yolo26n.pt`** into the project root (user already has `yolo26s.pt` and `yolo26m.pt` there):
   ```
   wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
   ```
4. **Edit [scripts/train.py](../../scripts/train.py)** — user wanted CLI flags so iteration doesn't require script edits:
   - Add argparse flags: `--model`, `--epochs`, `--imgsz`, `--batch`, `--no-aug`
   - Default `--model=yolo26n.pt`, `--epochs=40`, `--imgsz=416`, `--batch=16`
   - `--no-aug` should skip the block at [train.py:131-148](../../scripts/train.py#L131-L148) even if `dataset/augmented/` exists
   - Replace hardcoded values at [train.py:229](../../scripts/train.py#L229) and [train.py:241-244](../../scripts/train.py#L241-L244)
5. **Wire flags through [justfile](../../justfile)** `train` recipe (already uses `*ARGS` so `just train --no-aug --model yolo26n.pt` should just work once argparse is added).
6. **Consider: actually use a GPU.** This is the highest-ROI path but requires rsyncing `scanmyphotos/` and `dataset/` to a rented box.

## Key Context

- User is on AMD Ryzen 5 5600G, no dedicated GPU usable by torch. Explicit `device=cpu` in [scripts/train.py:245](../../scripts/train.py#L245).
- Dockerized training via [docker/docker-compose.yml](../../docker/docker-compose.yml), Python 3.14 base image, uv for deps, `UV_INDEX_URL=https://download.pytorch.org/whl/cpu`.
- Dataset is symlinked from `scanmyphotos/` into `dataset/images/{train,val}/` by `setup_dataset()` in [scripts/train.py:33-169](../../scripts/train.py#L33-L169). Labels are copied (not symlinked).
- Task is detecting date stamps on scanned photos (single class: `target`). Training data uses `dN_NNNNNNNN` disc-prefixed naming.
- Resume logic: trainer auto-resumes from `runs/detect/train/weights/best.pt` if it exists ([train.py:223-229](../../scripts/train.py#L223-L229)). **If you want a clean run with the new config, delete or move that file first**, otherwise you'll resume into the old model architecture and the `--model` flag will be ignored.
- Notifications: apprise → Mattermost every 10 epochs via callbacks. Default URL baked into [train.py:264](../../scripts/train.py#L264). Pass `--no-notify` or override with `APPRISE_URL` env.
- Quote from user that kicked this off: *"why is this taking so long?"*

## Files Touched

None yet — this session was pure diagnosis. Pending edits:
- [scripts/train.py](../../scripts/train.py) — add CLI flags, unhardcode model/epochs/imgsz/batch
- [justfile](../../justfile) — nothing required if argparse is wired correctly (already passes `*ARGS`)

## Blockers

- **Decision needed**: does the user want the CPU-optimization path (edit train.py + retrain locally over ~9h), or the GPU path (rent a box)? Last message in session was me offering to make the edits and asking for the go-ahead; session was interrupted before answering.
- **Open question**: after switching to `yolo26n`, does the existing `runs/detect/train/weights/best.pt` need to be archived? Resume logic will try to load it and architectures won't match.
