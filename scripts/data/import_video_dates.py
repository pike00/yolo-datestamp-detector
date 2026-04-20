# /// script
# requires-python = ">=3.12"
# dependencies = ["psycopg[binary]>=3.1.0"]
# ///
"""Import state/video_dates.json into the video_dates Postgres table.

Filenames in video_dates.json are already of the form {sha256}.{ext}, so
we parse out the sha256 on import and store a timestamptz + raw string.
Idempotent via upsert.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import psycopg

BASE = Path(__file__).resolve().parent.parent.parent
VIDEO_DATES_JSON = BASE / "state" / "video_dates.json"
DB_CONN = os.environ.get("DATABASE_URL", "postgresql://dedup:dedup_local_dev@127.0.0.1:5432/dedup")


def main() -> int:
    if not VIDEO_DATES_JSON.exists():
        print(f"ERROR: {VIDEO_DATES_JSON} not found", file=sys.stderr)
        return 1

    raw = json.loads(VIDEO_DATES_JSON.read_text())
    rows = []
    skipped_nodate = 0
    skipped_bad_name = 0
    for name, rec in raw.items():
        date_parts = rec.get("date")
        if not date_parts:
            skipped_nodate += 1
            continue
        stem = name.rsplit(".", 1)[0] if "." in name else name
        if len(stem) != 64 or not all(c in "0123456789abcdef" for c in stem):
            skipped_bad_name += 1
            continue
        # date_parts is [Y, M, D, h, m, s]
        dt = datetime(*date_parts, tzinfo=timezone.utc)
        rows.append(
            {
                "sha256": stem,
                "date_taken": dt,
                "source": rec.get("source", "ffprobe"),
                "raw": rec.get("raw"),
            }
        )

    print(
        f"parsed {len(rows)} video dates from {VIDEO_DATES_JSON.name}  "
        f"(skipped: {skipped_nodate} no-date, {skipped_bad_name} bad names)",
        file=sys.stderr,
    )

    if not rows:
        return 0

    with psycopg.connect(DB_CONN) as conn, conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO video_dates (sha256, date_taken, source, raw)
            VALUES (%(sha256)s, %(date_taken)s, %(source)s, %(raw)s)
            ON CONFLICT (sha256) DO UPDATE SET
                date_taken   = EXCLUDED.date_taken,
                source       = EXCLUDED.source,
                raw          = EXCLUDED.raw,
                extracted_at = now()
            """,
            rows,
        )
        conn.commit()
    print(f"upserted {len(rows)} rows into video_dates", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
