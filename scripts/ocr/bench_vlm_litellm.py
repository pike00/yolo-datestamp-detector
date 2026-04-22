# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pillow>=10.0",
#     "psycopg[binary]>=3.1.0",
#     "requests>=2.31",
# ]
# ///
"""LiteLLM (OpenAI-compat) VLM runner for the date-stamp bench.

Mirrors bench_vlm_ocr.py's row schema so results drop into
scripts/ocr/report_vlm_bench.py unchanged. Talks /v1/chat/completions
with image_url content blocks and a Bearer virtual key.

Auth:
    LITELLM_API_KEY env var (virtual key minted in the LiteLLM admin UI,
    starts with 'sk-...').

Usage:
    LITELLM_API_KEY=sk-... uv run scripts/ocr/bench_vlm_litellm.py \
        --model gemma4-31b-cloud \
        --base-url http://localhost:4000 \
        --manifest state/bench/manifest.json \
        --crops-dir state/bench/crops \
        --output jsonl://state/bench/results/gemma4-31b-cloud_litellm.jsonl \
        --host-label litellm \
        [--limit N] [--resume]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from _db import get_db  # noqa: E402
from ocr.ocr_util import extract_final_answer, normalize_date  # noqa: E402

PROMPT = """This is a cropped photo showing a camera date stamp -- orange LED digits.
Transcribe EXACTLY what you see, preserving spaces and apostrophes.
Example formats: "10 3 '99" or "'94 6 22" or "8 24'95"
Output ONLY the stamp text, nothing else."""

REQUEST_TIMEOUT = 180
MAX_TOKENS = 2048
CROP_MAX_SIDE = 512


def make_model_key(model: str, host_label: str) -> str:
    return f"{model.strip()}@{host_label.strip()}"


def encode_crop(path: Path, max_side: int) -> str:
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def load_manifest_stems(path: Path, limit: int | None) -> list[dict]:
    data = json.loads(path.read_text())
    stems = data["stems"]
    return stems[:limit] if limit else stems


def write_jsonl_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def ocr_one(base_url: str, api_key: str, model: str, b64: str) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
    }
    t0 = time.time()
    try:
        r = requests.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.Timeout:
        return {
            "raw_text": "TIMEOUT",
            "elapsed_s": REQUEST_TIMEOUT,
            "eval_count": 0,
            "prompt_eval_count": 0,
            "error": "timeout",
        }

    elapsed = round(time.time() - t0, 2)
    if r.status_code == 429:
        return {
            "raw_text": "RATE_LIMIT",
            "elapsed_s": elapsed,
            "eval_count": 0,
            "prompt_eval_count": 0,
            "error": "rate_limit",
        }
    if r.status_code != 200:
        snippet = r.text[:200].replace("\n", " ")
        return {
            "raw_text": f"HTTP_{r.status_code}",
            "elapsed_s": elapsed,
            "eval_count": 0,
            "prompt_eval_count": 0,
            "error": f"http_{r.status_code}: {snippet}",
        }
    data = r.json()
    raw = data["choices"][0]["message"].get("content") or ""
    usage = data.get("usage", {}) or {}
    return {
        "raw_text": raw,
        "elapsed_s": elapsed,
        "eval_count": int(usage.get("completion_tokens") or 0),
        "prompt_eval_count": int(usage.get("prompt_tokens") or 0),
        "error": None,
    }


def upsert_pg_row(conn, row: dict) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO stamp_ocr (stem, raw_text, parsed_date, bbox_source, model, host_label)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (stem, model) DO UPDATE SET
                raw_text    = EXCLUDED.raw_text,
                parsed_date = EXCLUDED.parsed_date,
                bbox_source = EXCLUDED.bbox_source,
                host_label  = EXCLUDED.host_label,
                updated_at  = NOW()
            """,
            (
                row["stem"],
                row["raw_text"],
                row["parsed_date"],
                row.get("bbox_source", "yolo"),
                row["model_key"],
                row["host_label"],
            ),
        )


def load_resume_set_pg(conn, model_key: str) -> set[str]:
    return {
        r[0]
        for r in conn.execute(
            "SELECT stem FROM stamp_ocr WHERE model = %s", (model_key,)
        ).fetchall()
    }


def load_resume_set_jsonl(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        if line.strip():
            done.add(json.loads(line)["stem"])
    return done


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        required=True,
        help="LiteLLM model_name (e.g. gemma4-31b-cloud, flash, haiku)",
    )
    p.add_argument("--base-url", default="http://localhost:4000")
    p.add_argument("--manifest", default="state/bench/manifest.json")
    p.add_argument("--crops-dir", default="state/bench/crops")
    p.add_argument(
        "--output",
        required=True,
        help="postgres://... or jsonl://path/to/file.jsonl",
    )
    p.add_argument("--host-label", default="litellm")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    api_key = os.environ.get("LITELLM_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("LITELLM_API_KEY is not set")

    model_key = make_model_key(args.model, args.host_label)
    manifest_path = Path(args.manifest).resolve()
    crops_dir = Path(args.crops_dir).resolve()
    stems = load_manifest_stems(manifest_path, args.limit)
    print(f"Model key: {model_key}")
    print(f"Stems:     {len(stems)}")
    print(f"Endpoint:  {args.base_url}")

    output_scheme, _, output_loc = args.output.partition("://")
    if output_scheme == "postgres":
        conn = get_db()
        done = load_resume_set_pg(conn, model_key) if args.resume else set()
        jsonl_path = None
    elif output_scheme == "jsonl":
        jsonl_path = Path(output_loc).resolve()
        conn = None
        done = load_resume_set_jsonl(jsonl_path) if args.resume else set()
    else:
        raise SystemExit(f"Unknown --output scheme: {output_scheme}")

    if args.resume:
        before = len(stems)
        stems = [s for s in stems if s["stem"] not in done]
        print(f"Resume: skipping {before - len(stems)} stems already present.")

    processed = 0
    total_elapsed = 0.0
    t_start = time.time()
    consecutive_rate_limits = 0
    BACKOFFS = [60, 120, 300, 600, 600]  # seconds; escalates then holds at 10 min

    for i, s in enumerate(stems):
        crop_path = crops_dir / f"{s['stem']}.jpg"
        if not crop_path.exists():
            print(f"  MISSING crop: {crop_path}")
            continue
        b64 = encode_crop(crop_path, CROP_MAX_SIDE)
        res = ocr_one(args.base_url, api_key, args.model, b64)

        # Transient errors (rate limit, timeout): don't persist — let --resume retry.
        if res["error"] in ("rate_limit", "timeout"):
            consecutive_rate_limits += 1
            print(
                f"  [{i+1}/{len(stems)}] {s['stem']}: {res['error'].upper()} "
                f"({res['elapsed_s']:.1f}s)  consecutive={consecutive_rate_limits}"
            )
            if res["error"] == "rate_limit":
                backoff = BACKOFFS[min(consecutive_rate_limits - 1, len(BACKOFFS) - 1)]
                print(f"    backing off {backoff}s")
                time.sleep(backoff)
            continue

        consecutive_rate_limits = 0
        cleaned_text = extract_final_answer(res["raw_text"])
        parsed = normalize_date(cleaned_text)
        if parsed and parsed.endswith("-00"):
            parsed = None
        row = {
            "stem": s["stem"],
            "model_key": model_key,
            "host_label": args.host_label,
            "raw_text": cleaned_text,
            "parsed_date": parsed,
            "bbox_source": s.get("bbox_source", "yolo"),
            "elapsed_s": res["elapsed_s"],
            "eval_count": res["eval_count"],
            "prompt_eval_count": res["prompt_eval_count"],
            "error": res["error"],
            "ran_at": datetime.now(timezone.utc).isoformat(),
        }
        if conn is not None:
            upsert_pg_row(conn, row)
            if (i + 1) % 10 == 0:
                conn.commit()
        else:
            write_jsonl_row(jsonl_path, row)
        processed += 1
        total_elapsed += res["elapsed_s"]
        date_str = f" -> {parsed}" if parsed else " [unparsed]"
        err_str = f"  ERR:{res['error']}" if res["error"] else ""
        print(
            f"  [{i+1}/{len(stems)}] {s['stem']}: {cleaned_text!r}{date_str}  "
            f"({res['elapsed_s']:.1f}s){err_str}"
        )

    if conn is not None:
        conn.commit()
        conn.close()

    wall = time.time() - t_start
    print()
    print(f"Processed:  {processed}")
    if processed:
        print(f"avg/img:    {total_elapsed/processed:.1f}s")
    print(f"Wall time:  {wall/3600:.2f}h")


if __name__ == "__main__":
    main()
