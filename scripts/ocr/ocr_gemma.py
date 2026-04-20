"""OCR date stamps from scanned photos using local Gemma4 via Ollama.

Processes all images with confirmed bounding boxes (human-reviewed first,
then YOLO predictions). Saves results to PostgreSQL with resume support.

Designed to run inside Docker alongside an Ollama container, but works
standalone too (just point OLLAMA_HOST and DB env vars).

Usage:
    python ocr_gemma.py                          # process all confirmed
    python ocr_gemma.py --limit 50               # first 50 only
    python ocr_gemma.py --resume                  # skip already-processed
    python ocr_gemma.py --source confirmed-only   # human-confirmed bboxes only
    python ocr_gemma.py --image d1_00000133       # single image
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import psycopg
import requests
from PIL import Image

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
from scripts.ocr.ocr_util import normalize_date

BASE_DIR = Path(os.environ.get("DATA_DIR", "/data"))
SCANMYPHOTOS_DIR = BASE_DIR / "scanmyphotos"
CORRECTIONS_FILE = BASE_DIR / "corrections_queue.json"

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e4b")

DB_CONN_STRING = os.environ.get(
    "DATABASE_URL",
    "postgresql://dedup:dedup_local_dev@localhost:5432/dedup",
)

PAD_FACTOR = 0.5
SAVE_EVERY = 25
REQUEST_TIMEOUT = 180

PROMPT = """This is a cropped photo showing a camera date stamp -- orange LED digits.
Transcribe EXACTLY what you see, preserving spaces and apostrophes.
Example formats: "10 3 '99" or "'94 6 22" or "8 24'95"
Output ONLY the stamp text, nothing else."""


def wait_for_ollama(max_wait: int = 300) -> bool:
    """Block until Ollama is reachable and model is loaded."""
    print(f"Waiting for Ollama at {OLLAMA_URL} ...")
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                # Accept exact match or match without tag
                model_base = MODEL.split(":")[0]
                if any(MODEL in m or model_base in m for m in models):
                    print(f"Ollama ready, model {MODEL} available.")
                    return True
                print(f"Model {MODEL} not found. Available: {models}")
                print(f"Pulling {MODEL} ...")
                requests.post(
                    f"{OLLAMA_URL}/api/pull",
                    json={"name": MODEL, "stream": False},
                    timeout=3600,
                )
                return True
        except requests.ConnectionError:
            pass
        time.sleep(2)
    print("ERROR: Ollama not reachable within timeout.")
    return False


def load_confirmed_boxes() -> dict[str, dict]:
    """Load human-confirmed bounding boxes from corrections queue."""
    if not CORRECTIONS_FILE.exists():
        return {}
    with open(CORRECTIONS_FILE) as f:
        data = json.load(f)
    boxes = {}
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


def load_predictions() -> dict[str, dict]:
    """Load YOLO predictions from stamp_predictions."""
    with psycopg.connect(DB_CONN_STRING) as conn:
        rows = conn.execute(
            "SELECT stem, x, y, w, h, confidence FROM stamp_predictions"
        ).fetchall()
    return {
        r[0]: {
            "x": r[1],
            "y": r[2],
            "w": r[3],
            "h": r[4],
            "confidence": r[5],
            "source": "yolo",
        }
        for r in rows
    }


def load_bboxes(source: str) -> dict[str, dict]:
    """Build bbox map. Human corrections take priority."""
    if source == "confirmed-only":
        return load_confirmed_boxes()
    yolo = load_predictions()
    human = load_confirmed_boxes()
    return {**yolo, **human}


def get_db():
    """Get a psycopg connection to the dedup database."""
    return psycopg.connect(DB_CONN_STRING)


def ensure_ocr_table(conn):
    """No-op kept for backward compatibility -- table is managed centrally."""
    return None


def load_processed_stems(conn) -> set[str]:
    """Stems already processed by this model in the database."""
    rows = conn.execute(
        "SELECT stem FROM stamp_ocr WHERE model = %s", (MODEL,)
    ).fetchall()
    return {r[0] for r in rows}


def upsert_result(conn, stem: str, result: dict):
    """Insert or update a single OCR result for this model."""
    normalized = result["normalized_date"]
    conn.execute(
        """INSERT INTO stamp_ocr (stem, raw_text, normalized_date, bbox_source,
                                   model, elapsed_s, eval_count, prompt_eval_count)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (stem, model) DO UPDATE SET
               raw_text = EXCLUDED.raw_text,
               normalized_date = EXCLUDED.normalized_date,
               bbox_source = EXCLUDED.bbox_source,
               elapsed_s = EXCLUDED.elapsed_s,
               eval_count = EXCLUDED.eval_count,
               prompt_eval_count = EXCLUDED.prompt_eval_count,
               updated_at = NOW()""",
        (
            stem,
            result["raw_text"],
            normalized if normalized and not normalized.endswith("-00") else None,
            result["bbox_source"],
            MODEL,
            result["elapsed_s"],
            result["eval_count"],
            result["prompt_eval_count"],
        ),
    )


def crop_stamp(img: Image.Image, pred: dict) -> Image.Image:
    """Crop image to stamp region with padding."""
    w_img, h_img = img.size
    cx, cy = pred["x"] * w_img, pred["y"] * h_img
    bw, bh = pred["w"] * w_img, pred["h"] * h_img
    pad_x = bw * PAD_FACTOR
    pad_y = bh * PAD_FACTOR
    x1 = max(0, int(cx - bw / 2 - pad_x))
    y1 = max(0, int(cy - bh / 2 - pad_y))
    x2 = min(w_img, int(cx + bw / 2 + pad_x))
    y2 = min(h_img, int(cy + bh / 2 + pad_y))
    return img.crop((x1, y1, x2, y2))


def img_to_b64(img: Image.Image, max_side: int = 512) -> str:
    """Resize and encode to base64 JPEG."""
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def ocr_single(stem: str, img_path: Path, bbox: dict) -> dict:
    """Send one cropped image to Gemma4, return result dict."""
    img = Image.open(img_path).convert("RGB")
    crop = crop_stamp(img, bbox)
    b64 = img_to_b64(crop)

    t0 = time.time()
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT, "images": [b64]}],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 512},
            "think": False,
        },
        timeout=REQUEST_TIMEOUT,
    )
    elapsed = time.time() - t0

    data = resp.json()
    raw_text = data.get("message", {}).get("content", "").strip()
    normalized = normalize_date(raw_text)

    return {
        "raw_text": raw_text,
        "normalized_date": normalized,
        "bbox_source": bbox.get("source"),
        "elapsed_s": round(elapsed, 1),
        "eval_count": data.get("eval_count", 0),
        "prompt_eval_count": data.get("prompt_eval_count", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="OCR date stamps via local Gemma4")
    parser.add_argument("--limit", type=int, help="Max images to process")
    parser.add_argument("--image", type=str, help="Single image stem")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed")
    parser.add_argument(
        "--source",
        choices=["auto", "confirmed-only"],
        default="auto",
        help="Bbox source: 'auto' merges human+YOLO, 'confirmed-only' uses human only",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Min YOLO confidence to include (ignored for human boxes)",
    )
    args = parser.parse_args()

    if not wait_for_ollama():
        sys.exit(1)

    conn = get_db()
    ensure_ocr_table(conn)

    bboxes = load_bboxes(args.source)

    if args.image:
        stems = [args.image]
    else:
        stems = sorted(bboxes.keys())
        if args.min_confidence > 0:
            stems = [
                s
                for s in stems
                if bboxes[s].get("source") == "human"
                or bboxes[s].get("confidence", 1.0) >= args.min_confidence
            ]

    if args.resume:
        done = load_processed_stems(conn)
        stems = [s for s in stems if s not in done]
        print(f"Resuming: {len(done)} already in DB, {len(stems)} remaining")
    if args.limit:
        stems = stems[: args.limit]

    if not stems:
        print("Nothing to process.")
        conn.close()
        return

    human_count = sum(1 for s in stems if bboxes.get(s, {}).get("source") == "human")
    yolo_count = len(stems) - human_count
    print(f"Processing {len(stems)} images with {MODEL}")
    print(f"Bbox sources: {human_count} human-confirmed, {yolo_count} YOLO-only")
    print()

    stats = {"processed": 0, "parsed": 0, "failed_parse": 0, "errors": 0}
    total_elapsed = 0.0
    t_start = time.time()

    for i, stem in enumerate(stems):
        img_path = SCANMYPHOTOS_DIR / f"{stem}.jpg"
        if not img_path.exists():
            print(f"  SKIP {stem}: file not found")
            continue

        bbox = bboxes[stem]

        try:
            result = ocr_single(stem, img_path, bbox)
        except requests.exceptions.Timeout:
            print(f"  TIMEOUT {stem}, continuing")
            stats["errors"] += 1
            continue
        except Exception as e:
            print(f"  ERROR {stem}: {e}")
            stats["errors"] += 1
            continue

        upsert_result(conn, stem, result)
        stats["processed"] += 1
        total_elapsed += result["elapsed_s"]

        if result["normalized_date"]:
            stats["parsed"] += 1
        else:
            stats["failed_parse"] += 1

        status = result["raw_text"]
        date_str = f" -> {result['normalized_date']}" if result["normalized_date"] else " [unparsed]"
        print(f"  [{i + 1}/{len(stems)}] {stem}: \"{status}\"{date_str}  ({result['elapsed_s']:.1f}s)")

        if (i + 1) % SAVE_EVERY == 0:
            conn.commit()
            avg = total_elapsed / stats["processed"]
            remaining = (len(stems) - i - 1) * avg
            print(f"    -- committed. avg {avg:.1f}s/img, ~{remaining / 3600:.1f}h remaining --")

    conn.commit()
    conn.close()

    wall = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Processed:     {stats['processed']}")
    print(f"Dates parsed:  {stats['parsed']}")
    print(f"Unparsed:      {stats['failed_parse']}")
    print(f"Errors:        {stats['errors']}")
    if stats["processed"]:
        avg = total_elapsed / stats["processed"]
        print(f"Avg time/img:  {avg:.1f}s")
    print(f"Wall time:     {wall / 3600:.1f}h")
    print(f"Results:       stamp_ocr table in {DB_CONN_STRING.split('@')[1]}")


if __name__ == "__main__":
    main()
