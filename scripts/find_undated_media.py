# /// script
# requires-python = ">=3.12"
# dependencies = ["pillow>=10", "pillow-heif>=1"]
# ///
"""Scan originals/media for items without a known date.

Photos: no EXIF DateTimeOriginal / DateTimeDigitized / DateTime.
Videos: missing from state/video_dates.json or listed with date=null.

Uses PIL header-only reads (fast) with a process pool.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ExifTags
import pillow_heif

pillow_heif.register_heif_opener()

MEDIA_DIR = Path("/home/will/photo_project/originals/media")
VIDEO_DATES = Path("/home/will/photo_project/state/video_dates.json")
OUT = Path("/home/will/photo_project/state/undated_media.json")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".bmp", ".tif", ".tiff", ".gif"}
VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".mpg", ".3gp", ".avi"}

DATE_TAG_IDS = {
    0x9003,  # DateTimeOriginal
    0x9004,  # DateTimeDigitized
    0x0132,  # DateTime
}


def probe_photo(path: Path) -> tuple[str, bool, str | None]:
    """Return (name, has_date, raw_date_str_or_error)."""
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if exif:
                for tag_id in DATE_TAG_IDS:
                    val = exif.get(tag_id)
                    if val:
                        return path.name, True, str(val)
                # also check IFDs
                try:
                    ifd = exif.get_ifd(ExifTags.IFD.Exif)
                    for tag_id in DATE_TAG_IDS:
                        val = ifd.get(tag_id)
                        if val:
                            return path.name, True, str(val)
                except Exception:
                    pass
            return path.name, False, None
    except Exception as e:
        return path.name, False, f"ERR:{type(e).__name__}"


def main() -> int:
    files = sorted(p for p in MEDIA_DIR.iterdir() if p.is_file())
    photos = [p for p in files if p.suffix.lower() in IMAGE_EXTS]
    videos = [p for p in files if p.suffix.lower() in VIDEO_EXTS]
    print(f"scanning {len(photos)} photos, {len(videos)} videos", file=sys.stderr)

    # --- photos: parallel EXIF probe ---
    undated_photos: list[dict] = []
    errors: list[dict] = []
    with ProcessPoolExecutor(max_workers=12) as pool:
        for name, has_date, info in pool.map(probe_photo, photos, chunksize=64):
            if not has_date:
                entry = {"name": name}
                if info and info.startswith("ERR:"):
                    entry["error"] = info
                    errors.append(entry)
                else:
                    undated_photos.append(entry)

    # --- videos: consult state/video_dates.json ---
    video_dates: dict = {}
    if VIDEO_DATES.exists():
        video_dates = json.loads(VIDEO_DATES.read_text())
    undated_videos: list[dict] = []
    for v in videos:
        rec = video_dates.get(v.name)
        if rec is None:
            undated_videos.append({"name": v.name, "reason": "not_in_manifest"})
        elif rec.get("date") is None:
            undated_videos.append({"name": v.name, "reason": "no_creation_time"})

    print(f"undated photos: {len(undated_photos)}  errors: {len(errors)}", file=sys.stderr)
    print(f"undated videos: {len(undated_videos)}", file=sys.stderr)

    OUT.write_text(
        json.dumps(
            {
                "undated_photos": undated_photos,
                "undated_videos": undated_videos,
                "photo_errors": errors,
                "counts": {
                    "photos_scanned": len(photos),
                    "videos_scanned": len(videos),
                    "undated_photos": len(undated_photos),
                    "undated_videos": len(undated_videos),
                    "photo_errors": len(errors),
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
