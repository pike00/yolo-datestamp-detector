# /// script
# requires-python = ">=3.12"
# dependencies = ["psycopg[binary]>=3.1.0"]
# ///
"""Write state/undated_media.json from the media_dates view.

A media file (photo or video) is considered undated if its sha256 does
NOT appear in the media_has_date view. media_has_date is the DISTINCT
union of exif_dates, video_dates, and stamp_ocr-via-scanmyphotos_manifest.

The output JSON shape matches the pre-view version so scripts/build_undated_gallery.py
keeps working unchanged.

Legacy behaviour (pre-view, PIL EXIF scan at runtime) is preserved in git
history (see commits before 2026-04-20).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import psycopg

BASE = Path(__file__).resolve().parent.parent
MEDIA_DIR = BASE / "originals" / "media"
OUT = BASE / "state" / "undated_media.json"
DB_CONN = os.environ.get("DATABASE_URL", "postgresql://dedup:dedup_local_dev@127.0.0.1:5432/dedup")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".bmp", ".tif", ".tiff", ".gif"}
VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".mpg", ".3gp", ".avi"}


def main() -> int:
    files = sorted(p for p in MEDIA_DIR.iterdir() if p.is_file())
    photos = [p for p in files if p.suffix.lower() in IMAGE_EXTS]
    videos = [p for p in files if p.suffix.lower() in VIDEO_EXTS]
    print(f"scanning {len(photos)} photos, {len(videos)} videos", file=sys.stderr)

    with psycopg.connect(DB_CONN) as conn:
        dated = {r[0] for r in conn.execute("SELECT sha256 FROM media_has_date").fetchall()}
    print(f"sha256s with at least one date: {len(dated)}", file=sys.stderr)

    undated_photos = [{"name": p.name} for p in photos if p.stem not in dated]
    undated_videos = [
        {"name": v.name, "reason": "no_date_in_media_dates"}
        for v in videos
        if v.stem not in dated
    ]

    print(f"undated photos: {len(undated_photos)}", file=sys.stderr)
    print(f"undated videos: {len(undated_videos)}", file=sys.stderr)

    OUT.write_text(
        json.dumps(
            {
                "undated_photos": undated_photos,
                "undated_videos": undated_videos,
                "photo_errors": [],
                "counts": {
                    "photos_scanned": len(photos),
                    "videos_scanned": len(videos),
                    "undated_photos": len(undated_photos),
                    "undated_videos": len(undated_videos),
                    "dated_sha256s": len(dated),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"wrote {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
