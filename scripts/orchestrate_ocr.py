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
DATE_FORMAT_RE = re.compile(r"^\d{1,2} \d{1,2} ?'\d{2}$")
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


def load_predictions() -> dict[str, dict]:
    """YOLO predictions, tagged with source."""
    raw = load_json(PREDICTIONS_FILE, {})
    return {k: {**v, "source": "yolo"} for k, v in raw.items()}


def load_corrections() -> dict[str, dict]:
    """Human-confirmed bboxes from the corrections queue."""
    data = load_json(CORRECTIONS_FILE, {})
    boxes: dict[str, dict] = {}
    for entry in data.get("files", []):
        corr = entry.get("user_correction")
        if not corr or corr.get("x") is None:
            continue
        if corr.get("action") in ("confirmed", "corrected"):
            boxes[entry["stem"]] = {
                "x": corr["x"], "y": corr["y"],
                "w": corr["w"], "h": corr["h"],
                "source": "human",
            }
    return boxes


def load_bbox_map() -> dict[str, dict]:
    """Unified bbox map. Human corrections override YOLO for the same stem."""
    yolo = load_predictions()
    human = load_corrections()
    return {**yolo, **human}


def compute_pending_stems() -> list[str]:
    """Stems that have a YOLO prediction but no entry in ocr_results.json."""
    predictions = load_json(PREDICTIONS_FILE, {})
    results = load_json(RESULTS_FILE, {})
    pending = [s for s in predictions if s not in results]
    return sorted(pending)


def compute_crop_box(img_w: int, img_h: int, bbox: dict, pad_factor: float) -> tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) in pixel coords, clamped to the image."""
    cx = bbox["x"] * img_w
    cy = bbox["y"] * img_h
    bw = bbox["w"] * img_w
    bh = bbox["h"] * img_h

    pad_x = bw * pad_factor
    pad_y = bh * pad_factor

    x1 = max(0, int(cx - bw / 2 - pad_x))
    y1 = max(0, int(cy - bh / 2 - pad_y))
    x2 = min(img_w, int(cx + bw / 2 + pad_x))
    y2 = min(img_h, int(cy + bh / 2 + pad_y))
    return (x1, y1, x2, y2)


def should_review(text: str, confidence: float | None) -> bool:
    """Return True if a stage-1 result should be re-reviewed in stage 2."""
    if confidence is not None and confidence < LOW_CONFIDENCE_THRESHOLD:
        return True
    if text is None:
        return True
    if "?" in text:
        return True
    if text == "NONE":
        return False
    if not DATE_FORMAT_RE.match(text):
        return True
    return False


def select_review_stems() -> list[str]:
    results = load_json(RESULTS_FILE, {})
    flagged: list[str] = []
    for stem, entry in results.items():
        text = entry.get("text", "")
        conf = entry.get("confidence")
        if should_review(text, conf):
            flagged.append(stem)
    return sorted(flagged)


def normalize_text(text: str) -> str:
    """Collapse whitespace for agreement comparison."""
    return " ".join(text.split())


def reconcile_pair(view_crop: str, view_full: str) -> tuple[str, str | None]:
    """Compare two views.

    Returns (status, final_text):
        ("confirmed", text) - both views agree on a non-NONE answer
        ("no_stamp", "NONE") - both views say NONE
        ("disagree", None)  - any disagreement
    """
    nc = normalize_text(view_crop)
    nf = normalize_text(view_full)
    if nc == "NONE" and nf == "NONE":
        return "no_stamp", "NONE"
    if nc == nf:
        return "confirmed", nc
    return "disagree", None


def crop_image_to_file(
    src: Path, dst: Path, bbox: dict, pad_factor: float, max_side: int,
) -> None:
    """Crop one photo to its padded bbox and save as JPEG."""
    from PIL import Image

    img = Image.open(src).convert("RGB")
    box = compute_crop_box(img.width, img.height, bbox, pad_factor)
    crop = img.crop(box)
    if max(crop.size) > max_side:
        crop.thumbnail((max_side, max_side), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    crop.save(dst, "JPEG", quality=90)


def cmd_crop_stage1(args) -> int:
    pending = compute_pending_stems()
    if args.limit is not None:
        pending = pending[: args.limit]

    if not pending:
        print("No pending stems.")
        return 0

    bbox_map = load_bbox_map()
    STAGE1_SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    STAGE1_CROPS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Pre-cropping {len(pending)} stems...")
    shard_index = 0
    shard_entries: list[dict] = []

    def flush_shard() -> None:
        nonlocal shard_entries, shard_index
        if not shard_entries:
            return
        shard_id = f"{shard_index:04d}"
        manifest_path = STAGE1_SHARDS_DIR / f"shard_{shard_id}.json"
        result_path = STAGE1_SHARDS_DIR / f"shard_{shard_id}_result.json"
        # Paths inside the manifest are relative to BASE_DIR for portability
        manifest = {
            "shard_id": shard_id,
            "stage": 1,
            "result_path": str(result_path.relative_to(BASE_DIR)),
            "stems": shard_entries,
        }
        save_json(manifest_path, manifest)
        shard_index += 1
        shard_entries = []

    for stem in pending:
        bbox = bbox_map.get(stem)
        if not bbox:
            print(f"  SKIP {stem}: no bbox")
            continue
        src = SCANMYPHOTOS_DIR / f"{stem}.jpg"
        if not src.exists():
            print(f"  SKIP {stem}: source image missing")
            continue
        dst = STAGE1_CROPS_DIR / f"{stem}.jpg"
        try:
            crop_image_to_file(
                src=src, dst=dst, bbox=bbox,
                pad_factor=STAGE1_PAD_FACTOR, max_side=STAGE1_MAX_SIDE,
            )
        except Exception as e:
            print(f"  SKIP {stem}: crop failed ({e})")
            continue

        shard_entries.append({
            "stem": stem,
            "crop_path": str(dst.relative_to(BASE_DIR)),
            "bbox_source": bbox.get("source", "yolo"),
            "confidence": bbox.get("confidence"),
        })
        if len(shard_entries) >= STAGE1_SHARD_SIZE:
            flush_shard()

    flush_shard()
    print(f"Wrote {shard_index} shard manifest(s) to {STAGE1_SHARDS_DIR.relative_to(BASE_DIR)}/")
    return 0


def _validate_stage1_shard_result(data: dict) -> tuple[bool, str]:
    if not isinstance(data, dict):
        return False, "shard result is not an object"
    if "results" not in data or not isinstance(data["results"], dict):
        return False, "missing or non-object 'results' field"
    for stem, entry in data["results"].items():
        if not isinstance(entry, dict):
            return False, f"entry for {stem} is not an object"
        if "text" not in entry or not isinstance(entry["text"], str):
            return False, f"entry for {stem} missing string 'text'"
    return True, ""


def cmd_merge_stage1(args) -> int:
    shard_path = Path(args.shard_result).resolve()
    if not shard_path.exists():
        print(f"ERROR: shard result not found: {shard_path}", file=sys.stderr)
        return 2

    try:
        data = load_json(shard_path, None)
    except json.JSONDecodeError as e:
        print(f"ERROR: invalid JSON in {shard_path.name}: {e}", file=sys.stderr)
        return 3
    ok, err = _validate_stage1_shard_result(data)
    if not ok:
        print(f"ERROR: invalid shard result {shard_path.name}: {err}", file=sys.stderr)
        return 3

    results = load_json(RESULTS_FILE, {})
    added = 0
    for stem, entry in data["results"].items():
        results[stem] = {**entry, "stage": 1}
        added += 1
    save_json(RESULTS_FILE, results)
    print(f"Merged {added} stems from {shard_path.name}")
    return 0


def cmd_list_shards(args) -> int:
    stage_dir = STAGE1_SHARDS_DIR if args.stage == "stage1" else STAGE2_SHARDS_DIR
    if not stage_dir.exists():
        return 0
    for manifest in sorted(stage_dir.glob("shard_*.json")):
        if "_result" in manifest.stem:
            continue
        if _result_path(manifest).exists():
            continue
        print(manifest)
    return 0


def cmd_requeue(args) -> int:
    manifest = Path(args.shard_path).resolve()
    if not manifest.exists():
        print(f"ERROR: manifest not found: {manifest}", file=sys.stderr)
        return 2
    result = _result_path(manifest)
    if result.exists():
        result.unlink()
        print(f"Removed {result.name}; shard is now pending again")
    else:
        print(f"No result file to remove; shard was already pending")
    return 0


def _save_full_image(src: Path, dst: Path, max_side: int) -> None:
    from PIL import Image
    img = Image.open(src).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, "JPEG", quality=90)


def cmd_crop_stage2(_args) -> int:
    flagged = select_review_stems()
    if not flagged:
        print("No stems trigger stage-2 review.")
        return 0

    bbox_map = load_bbox_map()
    results = load_json(RESULTS_FILE, {})

    STAGE2_SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    STAGE2_CROP_DIR.mkdir(parents=True, exist_ok=True)
    STAGE2_FULL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Preparing stage-2 review for {len(flagged)} stems...")
    shard_index = 0
    shard_entries: list[dict] = []

    def flush_shard() -> None:
        nonlocal shard_entries, shard_index
        if not shard_entries:
            return
        shard_id = f"{shard_index:04d}"
        manifest_path = STAGE2_SHARDS_DIR / f"shard_{shard_id}.json"
        result_path = STAGE2_SHARDS_DIR / f"shard_{shard_id}_result.json"
        save_json(manifest_path, {
            "shard_id": shard_id,
            "stage": 2,
            "result_path": str(result_path.relative_to(BASE_DIR)),
            "stems": shard_entries,
        })
        shard_index += 1
        shard_entries = []

    for stem in flagged:
        src = SCANMYPHOTOS_DIR / f"{stem}.jpg"
        if not src.exists():
            print(f"  SKIP {stem}: source missing")
            continue
        bbox = bbox_map.get(stem)
        if not bbox:
            print(f"  SKIP {stem}: no bbox")
            continue

        crop_dst = STAGE2_CROP_DIR / f"{stem}.jpg"
        full_dst = STAGE2_FULL_DIR / f"{stem}.jpg"
        try:
            crop_image_to_file(
                src=src, dst=crop_dst, bbox=bbox,
                pad_factor=STAGE2_PAD_FACTOR, max_side=STAGE2_MAX_SIDE,
            )
            _save_full_image(src=src, dst=full_dst, max_side=STAGE2_MAX_SIDE)
        except Exception as e:
            print(f"  SKIP {stem}: crop failed ({e})")
            continue

        shard_entries.append({
            "stem": stem,
            "crop_path": str(crop_dst.relative_to(BASE_DIR)),
            "full_path": str(full_dst.relative_to(BASE_DIR)),
            "stage1_text": results.get(stem, {}).get("text", ""),
            "confidence": bbox.get("confidence"),
        })
        if len(shard_entries) >= STAGE2_SHARD_SIZE:
            flush_shard()

    flush_shard()
    print(f"Wrote {shard_index} stage-2 shard manifest(s)")
    return 0


def _validate_stage2_shard_result(data: dict) -> tuple[bool, str]:
    if not isinstance(data, dict) or "results" not in data:
        return False, "missing 'results' field"
    if not isinstance(data["results"], dict):
        return False, "'results' is not an object"
    for stem, entry in data["results"].items():
        if not isinstance(entry, dict):
            return False, f"entry for {stem} is not an object"
        for field in ("view_crop", "view_full"):
            if field not in entry or not isinstance(entry[field], str):
                return False, f"entry for {stem} missing string '{field}'"
    return True, ""


def cmd_merge_stage2(args) -> int:
    shard_path = Path(args.shard_result).resolve()
    if not shard_path.exists():
        print(f"ERROR: shard result not found: {shard_path}", file=sys.stderr)
        return 2

    try:
        data = load_json(shard_path, None)
    except json.JSONDecodeError as e:
        print(f"ERROR: invalid JSON in {shard_path.name}: {e}", file=sys.stderr)
        return 3

    ok, err = _validate_stage2_shard_result(data)
    if not ok:
        print(f"ERROR: invalid shard result {shard_path.name}: {err}", file=sys.stderr)
        return 3

    results = load_json(RESULTS_FILE, {})
    manual_queue = load_json(MANUAL_QUEUE_FILE, [])

    confirmed = no_stamp = disagree = 0
    for stem, entry in data["results"].items():
        status, final_text = reconcile_pair(entry["view_crop"], entry["view_full"])
        if status == "confirmed":
            results[stem] = {
                **results.get(stem, {}),
                "text": final_text,
                "stage": 2,
                "review_status": "confirmed",
            }
            confirmed += 1
        elif status == "no_stamp":
            results[stem] = {
                **results.get(stem, {}),
                "text": "NONE",
                "stage": 2,
                "review_status": "no_stamp",
            }
            no_stamp += 1
        else:  # disagree
            stage1_entry = results.get(stem, {})
            stage1_text = stage1_entry.get("text", "")
            results[stem] = {
                **stage1_entry,
                "review_status": "disagreement",
            }
            manual_queue.append({
                "stem": stem,
                "stage1_text": stage1_text,
                "view_crop": entry["view_crop"],
                "view_full": entry["view_full"],
                "confidence": stage1_entry.get("confidence"),
            })
            disagree += 1

    save_json(RESULTS_FILE, results)
    save_json(MANUAL_QUEUE_FILE, manual_queue)
    print(f"Merged {shard_path.name}: {confirmed} confirmed, {no_stamp} no-stamp, {disagree} disagreements")
    return 0


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

    p_crop1 = sub.add_parser("crop-stage1", help="Pre-crop pending stems")
    p_crop1.add_argument("--limit", type=int, help="Cap pending set size (pilot runs)")

    p_merge1 = sub.add_parser("merge-stage1", help="Merge a stage-1 shard result into ocr_results.json")
    p_merge1.add_argument("shard_result", help="Path to shard_NNNN_result.json")

    p_list = sub.add_parser("list-shards", help="List shard manifests with no result yet")
    p_list.add_argument("stage", choices=["stage1", "stage2"])

    p_requeue = sub.add_parser("requeue", help="Mark a shard as pending again")
    p_requeue.add_argument("shard_path", help="Path to shard_NNNN.json")

    sub.add_parser("crop-stage2", help="Prepare stage-2 review shards")

    p_merge2 = sub.add_parser("merge-stage2", help="Merge and reconcile a stage-2 shard result")
    p_merge2.add_argument("shard_result")

    args = parser.parse_args(argv)
    if args.cmd == "crop-stage1":
        return cmd_crop_stage1(args)
    if args.cmd == "merge-stage1":
        return cmd_merge_stage1(args)
    if args.cmd == "list-shards":
        return cmd_list_shards(args)
    if args.cmd == "requeue":
        return cmd_requeue(args)
    if args.cmd == "crop-stage2":
        return cmd_crop_stage2(args)
    if args.cmd == "merge-stage2":
        return cmd_merge_stage2(args)
    if args.cmd == "status":
        return cmd_status(args)
    raise AssertionError(f"unhandled cmd: {args.cmd}")


if __name__ == "__main__":
    sys.exit(main())
