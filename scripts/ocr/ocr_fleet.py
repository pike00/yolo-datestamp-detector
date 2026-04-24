# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pillow>=10.0",
#     "psycopg[binary]>=3.1.0",
#     "requests>=2.31",
# ]
# ///
"""Fleet-scale stage-1 OCR via LiteLLM (gemma4-31b-cloud by default).

Reads pending stems from Postgres: YOLO predictions with confidence >= 0.05
that are not in stamp_no_stamp and don't already have a row for the target
model at stage=1. Crops on the fly, posts to LiteLLM /v1/chat/completions,
writes results (raw_text, parsed_date, parse_error) into stamp_ocr.

Resumable. Exponential backoff on rate-limit (HTTP 429) and transient errors.
Quota-aware — expected to stretch across hours on Ollama Cloud free tier.

Usage:
    LITELLM_API_KEY=sk-... LITELLM_BASE_URL=http://localhost:4000 \\
        uv run scripts/ocr/ocr_fleet.py [--model gemma4-31b-cloud] [--limit N]
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psycopg
import requests
from PIL import Image

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "scripts"))

from ocr.ocr_util import extract_final_answer, normalize_date  # noqa: E402

SCANMYPHOTOS_DIR = BASE / "scanmyphotos"
LOG_DIR = BASE / "state" / "logs"
MILESTONE_DIR = BASE / "output" / "ocr_fleet_milestones"

PROMPT = """This is a cropped photo showing a camera date stamp -- orange LED digits.
Transcribe EXACTLY what you see, preserving spaces and apostrophes.
Example formats: "10 3 '99" or "'94 6 22" or "8 24'95"
Output ONLY the stamp text, nothing else."""

PAD_FACTOR = 0.5
CROP_MAX_SIDE = 512
SAVE_EVERY = 20
REQUEST_TIMEOUT = 180
HOST_LABEL = "production"
STAGE = 1
DB_URL = os.environ.get("DATABASE_URL", "postgresql://dedup:dedup_local_dev@localhost:5432/dedup")


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def compute_pending(conn, model_key: str, limit: int | None):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.stem, p.x, p.y, p.w, p.h, p.confidence
            FROM stamp_predictions p
            LEFT JOIN stamp_no_stamp ns ON ns.stem = p.stem
            LEFT JOIN stamp_ocr o ON o.stem = p.stem AND o.model = %s AND o.stage = 1
            WHERE p.confidence >= 0.05
              AND ns.stem IS NULL
              AND o.stem IS NULL
            ORDER BY p.stem
            """,
            (model_key,),
        )
        rows = cur.fetchall()
    return rows[:limit] if limit else rows


def crop_to_b64(src: Path, x, y, w, h) -> str | None:
    try:
        img = Image.open(src).convert("RGB")
    except Exception:
        return None
    iw, ih = img.size
    cx, cy = x * iw, y * ih
    bw, bh = w * iw, h * ih
    pad_x = bw * PAD_FACTOR
    pad_y = bh * PAD_FACTOR
    x1 = max(0, int(cx - bw / 2 - pad_x))
    y1 = max(0, int(cy - bh / 2 - pad_y))
    x2 = min(iw, int(cx + bw / 2 + pad_x))
    y2 = min(ih, int(cy + bh / 2 + pad_y))
    crop = img.crop((x1, y1, x2, y2))
    if max(crop.size) > CROP_MAX_SIDE:
        crop.thumbnail((CROP_MAX_SIDE, CROP_MAX_SIDE), Image.LANCZOS)
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def call_litellm(model: str, b64: str, base_url: str, api_key: str) -> str:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }
        ],
        "max_tokens": 2048,
        "temperature": 0.0,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    if r.status_code == 429:
        raise RateLimitError(r.text[:300])
    if r.status_code >= 500:
        raise TransientError(f"{r.status_code}: {r.text[:200]}")
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


class RateLimitError(Exception):
    pass


class TransientError(Exception):
    pass


def count_rows(conn, model_key: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM stamp_ocr WHERE model=%s AND host_label=%s",
            (model_key, HOST_LABEL),
        )
        return cur.fetchone()[0]


def post_milestone(conn, model_key: str, milestone: int, webhook_url: str, public_base: str, total_pending: int, logf) -> None:
    """Post a Mattermost milestone: N/total progress + a few recent parsed examples."""
    out_dir = MILESTONE_DIR / str(milestone)
    out_dir.mkdir(parents=True, exist_ok=True)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT o.stem, o.raw_text, o.parsed_date, p.x, p.y, p.w, p.h
            FROM stamp_ocr o JOIN stamp_predictions p ON p.stem = o.stem
            WHERE o.model=%s AND o.host_label=%s AND o.parsed_date IS NOT NULL
            ORDER BY o.updated_at DESC
            LIMIT 4
            """,
            (model_key, HOST_LABEL),
        )
        examples = cur.fetchall()

        cur.execute(
            """
            SELECT
              count(*) FILTER (WHERE parsed_date IS NOT NULL),
              count(*)
            FROM stamp_ocr WHERE model=%s AND host_label=%s
            """,
            (model_key, HOST_LABEL),
        )
        parsed_count, row_count = cur.fetchone()

    # Crop each example into the milestone dir for webhook image URLs
    example_urls: list[tuple[str, str, str]] = []
    for stem, raw_text, parsed_date, x, y, w, h in examples:
        src = SCANMYPHOTOS_DIR / f"{stem}.jpg"
        if not src.exists():
            continue
        try:
            img = Image.open(src).convert("RGB")
        except Exception:
            continue
        iw, ih = img.size
        cx, cy = x * iw, y * ih
        bw, bh = w * iw, h * ih
        pad_x, pad_y = bw * PAD_FACTOR, bh * PAD_FACTOR
        x1 = max(0, int(cx - bw / 2 - pad_x))
        y1 = max(0, int(cy - bh / 2 - pad_y))
        x2 = min(iw, int(cx + bw / 2 + pad_x))
        y2 = min(ih, int(cy + bh / 2 + pad_y))
        crop = img.crop((x1, y1, x2, y2))
        if max(crop.size) > CROP_MAX_SIDE:
            crop.thumbnail((CROP_MAX_SIDE, CROP_MAX_SIDE), Image.LANCZOS)
        dst = out_dir / f"{stem}.jpg"
        crop.save(dst, format="JPEG", quality=85)
        url = f"{public_base.rstrip('/')}/ocr_fleet_milestones/{milestone}/{stem}.jpg"
        example_urls.append((url, raw_text, parsed_date.isoformat()))

    total = total_pending + row_count - milestone  # rough: remaining = initial pending - rows done so far (offset by pre-run rows)
    # simpler: show row_count / (row_count + pending_still_in_queue). We'll just show "N rows so far".
    pct = (100 * parsed_count / row_count) if row_count else 0
    lines = [
        f"📸 **Date Stamp OCR — milestone {milestone:,}**",
        f"{row_count:,} rows written · parsed {parsed_count:,} ({pct:.0f}%) · model `{model_key}`",
        "",
        "**Latest parsed reads:**",
    ]
    for url, raw, date_iso in example_urls:
        raw_esc = (raw or "").replace("`", "'").strip()
        lines.append(f"![](%s) `{raw_esc}` → **{date_iso}**" % url)
    if not example_urls:
        lines.append("_(no parsed examples available for this milestone)_")

    payload = {"text": "\n".join(lines), "username": "ocr-fleet", "icon_emoji": ":frame_with_picture:"}
    try:
        r = requests.post(webhook_url, json=payload, timeout=15)
        if r.status_code == 200:
            logf(f"milestone {milestone} posted to Mattermost ({len(example_urls)} examples)")
        else:
            logf(f"milestone webhook failed {r.status_code}: {r.text[:200]}")
    except requests.exceptions.RequestException as e:
        logf(f"milestone webhook error: {e}")


def persist(conn, batch, model_key: str) -> int:
    if not batch:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO stamp_ocr (stem, model, raw_text, bbox_source, confidence, stage, host_label, parsed_date, parse_error)
            VALUES (%(stem)s, %(model)s, %(raw_text)s, %(bbox_source)s, %(confidence)s, %(stage)s, %(host_label)s, %(parsed_date)s, %(parse_error)s)
            ON CONFLICT (stem, model) DO UPDATE SET
                raw_text    = EXCLUDED.raw_text,
                bbox_source = EXCLUDED.bbox_source,
                confidence  = EXCLUDED.confidence,
                stage       = EXCLUDED.stage,
                host_label  = EXCLUDED.host_label,
                parsed_date = EXCLUDED.parsed_date,
                parse_error = EXCLUDED.parse_error,
                updated_at  = NOW()
            """,
            batch,
        )
        conn.commit()
    return len(batch)


def main() -> int:
    # Load photo_project/.env first (LITELLM_API_KEY, LITELLM_BASE_URL)
    load_env_file(BASE / ".env")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4-31b-cloud")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--base-url", default=os.environ.get("LITELLM_BASE_URL", "http://localhost:4000"))
    ap.add_argument("--api-key", default=os.environ.get("LITELLM_API_KEY"))
    ap.add_argument("--max-retries", type=int, default=8)
    args = ap.parse_args()

    if not args.api_key:
        print("ERROR: LITELLM_API_KEY not set", file=sys.stderr)
        return 1

    model_key = f"{args.model}@litellm"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG_DIR / f"ocr_fleet_{args.model}.log", "a", buffering=1)

    def logf(msg: str) -> None:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        line = f"{ts} {msg}"
        print(line, file=sys.stderr)
        log.write(line + "\n")

    stop_flag = {"v": False}

    def handle_sig(*_):
        stop_flag["v"] = True
        logf("stop requested — finishing current batch")

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    webhook_url = os.environ.get("MATTERMOST_WEBHOOK_URL", "").strip()
    public_base = os.environ.get("PUBLIC_BASE_URL", "").strip()
    milestone_every = int(os.environ.get("MILESTONE_EVERY", "1000"))

    conn = psycopg.connect(DB_URL)
    pending = compute_pending(conn, model_key, args.limit)
    total = len(pending)
    logf(f"model={model_key} pending={total}")

    if total == 0:
        logf("nothing pending; done")
        return 0

    # Next milestone is the first multiple of milestone_every > current row count
    initial_rows = count_rows(conn, model_key)
    next_milestone = ((initial_rows // milestone_every) + 1) * milestone_every
    milestone_enabled = bool(webhook_url) and bool(public_base)
    logf(
        f"milestones {'enabled' if milestone_enabled else 'disabled'}; "
        f"initial_rows={initial_rows}, next_milestone={next_milestone}, every={milestone_every}"
    )

    batch: list[dict] = []
    done = 0
    ok = 0
    parsed = 0
    errors = 0
    rl_sleep = 30  # seconds — grows exponentially on 429, resets on success

    t_start = time.time()
    for stem, x, y, w, h, conf in pending:
        if stop_flag["v"]:
            break
        src = SCANMYPHOTOS_DIR / f"{stem}.jpg"
        if not src.exists():
            errors += 1
            logf(f"skip {stem}: missing source")
            continue
        b64 = crop_to_b64(src, x, y, w, h)
        if b64 is None:
            errors += 1
            logf(f"skip {stem}: crop failed")
            continue

        retries = 0
        raw = None
        while retries <= args.max_retries:
            try:
                raw = call_litellm(args.model, b64, args.base_url, args.api_key)
                rl_sleep = 30
                break
            except RateLimitError as e:
                logf(f"429 on {stem}: backing off {rl_sleep}s ({e})")
                time.sleep(rl_sleep)
                rl_sleep = min(rl_sleep * 2, 1800)
                retries += 1
            except (TransientError, requests.exceptions.RequestException) as e:
                wait = min(30 * (retries + 1), 300)
                logf(f"transient on {stem}: wait {wait}s ({e})")
                time.sleep(wait)
                retries += 1

        if raw is None:
            errors += 1
            logf(f"give up on {stem} after {retries} retries")
            continue

        cleaned = extract_final_answer(raw)
        date_str = normalize_date(cleaned)
        parsed_date = None
        parse_error = None
        if date_str and not date_str.endswith("-00"):
            parsed_date = date_str  # psycopg handles "YYYY-MM-DD" → DATE
            parsed += 1
        elif cleaned.strip().upper() == "NONE":
            parse_error = "none"
        else:
            parse_error = "unparseable" if date_str is None else f"partial:{date_str}"

        batch.append(
            dict(
                stem=stem,
                model=model_key,
                raw_text=cleaned,
                bbox_source="yolo",
                confidence=float(conf) if conf is not None else None,
                stage=STAGE,
                host_label=HOST_LABEL,
                parsed_date=parsed_date,
                parse_error=parse_error,
            )
        )
        ok += 1
        done += 1

        if len(batch) >= SAVE_EVERY:
            persist(conn, batch, model_key)
            logf(f"flushed {len(batch)} rows ({done}/{total} done, parsed={parsed}, errors={errors})")
            batch = []

            # Milestone check — fires when the DB row count crosses next_milestone
            if milestone_enabled:
                current = count_rows(conn, model_key)
                while current >= next_milestone:
                    post_milestone(conn, model_key, next_milestone, webhook_url, public_base, total, logf)
                    next_milestone += milestone_every

    persist(conn, batch, model_key)
    elapsed = time.time() - t_start
    logf(
        f"finished: processed={done}/{total} parsed={parsed} errors={errors} "
        f"elapsed={elapsed:.0f}s ({done / max(elapsed, 1):.2f} req/s)"
    )
    conn.close()
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
