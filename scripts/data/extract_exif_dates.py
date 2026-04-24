# /// script
# requires-python = ">=3.12"
# dependencies = ["pillow>=10", "pillow-heif>=1", "psycopg[binary]>=3.1.0"]
# ///
"""Extract EXIF dates for all photos under originals/media/ and write to exif_dates.

Uses PIL header-only reads in a 12-way ProcessPoolExecutor (same pattern
as find_undated_media.py, but captures hits instead of misses). Upserts
into exif_dates keyed by sha256 (= file stem).

Prefers DateTimeOriginal → DateTimeDigitized → DateTime, stores which tag
was used in `source`.

Usage:
    python scripts/data/extract_exif_dates.py [--force] [--limit N] [--workers 12]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from PIL import ExifTags, Image
import pillow_heif

pillow_heif.register_heif_opener()

BASE = Path(__file__).resolve().parent.parent.parent
ORIGINALS = BASE / "originals" / "media"
DB_CONN = os.environ.get("DATABASE_URL", "postgresql://dedup:dedup_local_dev@127.0.0.1:5432/dedup")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".bmp", ".tif", ".tiff", ".gif"}

# (tag_id, tag_name) — try in priority order
TAG_CANDIDATES = [
    (0x9003, "DateTimeOriginal"),
    (0x9004, "DateTimeDigitized"),
    (0x0132, "DateTime"),
]

EXIF_FORMATS = [
    "%Y:%m:%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y:%m:%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
]


def parse_exif_timestamp(s: str) -> datetime | None:
    s = s.strip().rstrip("\x00").strip()
    if not s:
        return None
    # EXIF zero sentinels
    if s.startswith("0000:") or s.startswith("0000-"):
        return None
    for fmt in EXIF_FORMATS:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def probe(path: Path) -> dict | None:
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return None
            for tag_id, tag_name in TAG_CANDIDATES:
                val = exif.get(tag_id)
                if not val:
                    try:
                        val = exif.get_ifd(ExifTags.IFD.Exif).get(tag_id)
                    except Exception:
                        val = None
                if not val:
                    continue
                dt = parse_exif_timestamp(str(val))
                if dt is not None:
                    return {
                        "sha256": path.stem,
                        "date_taken": dt,
                        "source": tag_name,
                        "raw": str(val),
                    }
            return None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-probe even if sha256 already in exif_dates")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    paths = sorted(p for p in ORIGINALS.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        paths = paths[: args.limit]
    print(f"scanning {len(paths)} photos", file=sys.stderr)

    with psycopg.connect(DB_CONN) as conn:
        existing = {
            r[0] for r in conn.execute("SELECT sha256 FROM exif_dates").fetchall()
        }
        # Skip ScanMyPhotos scans — their authoritative date is stamp_ocr, not scanner EXIF
        # (scanner software writes 2014-2024 timestamps that would poison 1990s photos).
        scan_shas = {
            r[0] for r in conn.execute("SELECT sha256 FROM scanmyphotos_manifest").fetchall()
        }
        # Skip video keyframes — covered by video_dates.
        keyframe_shas = {
            r[0] for r in conn.execute(
                "SELECT sha256 FROM photo_embeddings WHERE media_type='video_keyframe'"
            ).fetchall()
        }
    skip = existing | scan_shas | keyframe_shas
    pending = paths if args.force else [p for p in paths if p.stem not in skip]
    print(
        f"{len(existing)} already probed, {len(scan_shas)} manifest skips, "
        f"{len(keyframe_shas)} keyframe skips, {len(pending)} this run",
        file=sys.stderr,
    )

    hits: list[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(probe, pending, chunksize=64):
            if r is not None:
                hits.append(r)
            done += 1
            if done % 5000 == 0:
                print(f"  probed {done}/{len(pending)} (hits so far: {len(hits)})", file=sys.stderr)

    print(f"hits: {len(hits)} / {len(pending)} probed", file=sys.stderr)

    if hits:
        with psycopg.connect(DB_CONN) as conn, conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO exif_dates (sha256, date_taken, source, raw)
                VALUES (%(sha256)s, %(date_taken)s, %(source)s, %(raw)s)
                ON CONFLICT (sha256) DO UPDATE SET
                    date_taken   = EXCLUDED.date_taken,
                    source       = EXCLUDED.source,
                    raw          = EXCLUDED.raw,
                    extracted_at = now()
                """,
                hits,
            )
            conn.commit()
        print(f"upserted {len(hits)} rows into exif_dates", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
