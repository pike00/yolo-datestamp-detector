# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "pillow>=12.0",
# ]
# ///
"""Parallel Haiku OCR orchestrator for ScanMyPhotos date stamps.

This script is the deterministic IO half of the pipeline. It pre-crops
images, writes shard manifests, validates and merges shard results from
Haiku subagents, and reconciles stage-2 review outputs. It does NOT call
any LLM — subagent dispatch is driven by a Claude Code orchestrator
session using the Task tool.

Subcommands:
    crop-stage1 [--limit N]  Pre-crop pending stems and write stage-1 shards
    merge-stage1 <result>    Merge a stage-1 shard result into ocr_results.json
    crop-stage2              Compute triggered stems and write stage-2 shards
    merge-stage2 <result>    Reconcile a stage-2 shard result
    list-shards <stage>      List shard manifests that have no result yet
    requeue <shard>          Mark a shard as pending again
    status                   Print progress summary
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
SCANMYPHOTOS_DIR = BASE_DIR / "scanmyphotos"
STATE_DIR = BASE_DIR / "state"
OUTPUT_DIR = BASE_DIR / "output"

PREDICTIONS_FILE = STATE_DIR / "scanmyphotos_predictions.json"
CORRECTIONS_FILE = STATE_DIR / "corrections_queue.json"
RESULTS_FILE = STATE_DIR / "ocr_results.json"
MANUAL_QUEUE_FILE = STATE_DIR / "ocr_manual_queue.json"
FAILED_SHARDS_FILE = STATE_DIR / "failed_shards.json"

SHARDS_DIR = STATE_DIR / "shards"
STAGE1_SHARDS_DIR = SHARDS_DIR / "stage1"
STAGE2_SHARDS_DIR = SHARDS_DIR / "stage2"

STAGE1_CROPS_DIR = OUTPUT_DIR / "ocr_crops_stage1"
STAGE2_CROP_DIR = OUTPUT_DIR / "ocr_crops_stage2_crop"
STAGE2_FULL_DIR = OUTPUT_DIR / "ocr_crops_stage2_full"

# Cropping constants (stage 1 mirrors ocr_stamps.py exactly)
STAGE1_PAD_FACTOR = 0.5
STAGE1_MAX_SIDE = 512
STAGE2_PAD_FACTOR = 1.5
STAGE2_MAX_SIDE = 1536

# Stage-1 shard sizing
STAGE1_SHARD_SIZE = 50
# Stage-2 shard sizing (each stem costs two subagent image reads)
STAGE2_SHARD_SIZE = 25

# Stage-2 trigger rules
DATE_FORMAT_RE = re.compile(r"^\d{1,2} \d{1,2} '\d{2}$")
LOW_CONFIDENCE_THRESHOLD = 0.3


def load_json(path: Path, default):
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def _result_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(manifest_path.stem + "_result.json")


def cmd_status(_args) -> int:
    results = load_json(RESULTS_FILE, {})
    manual_queue = load_json(MANUAL_QUEUE_FILE, [])
    failed = load_json(FAILED_SHARDS_FILE, [])
    predictions = load_json(PREDICTIONS_FILE, {})

    def _count_pending_done(stage_dir: Path) -> tuple[int, int]:
        if not stage_dir.exists():
            return 0, 0
        manifests = [p for p in stage_dir.glob("shard_*.json") if "_result" not in p.stem]
        done = sum(1 for p in manifests if _result_path(p).exists())
        pending = len(manifests) - done
        return pending, done

    stage1_pending, stage1_done = _count_pending_done(STAGE1_SHARDS_DIR)
    stage2_pending, stage2_done = _count_pending_done(STAGE2_SHARDS_DIR)

    print(f"YOLO predictions:    {len(predictions)}")
    print(f"OCR results:         {len(results)}")
    print(f"Stage-1 shards:      {stage1_pending} pending, {stage1_done} done")
    print(f"Stage-2 shards:      {stage2_pending} pending, {stage2_done} done")
    print(f"Manual review queue: {len(manual_queue)}")
    print(f"Failed shards:       {len(failed)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="Print progress summary")

    args = parser.parse_args(argv)
    if args.cmd == "status":
        return cmd_status(args)
    raise AssertionError(f"unhandled cmd: {args.cmd}")


if __name__ == "__main__":
    sys.exit(main())
