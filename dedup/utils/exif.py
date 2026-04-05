import json
import subprocess
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Extensions worth reading EXIF from
MEDIA_EXTENSIONS = {
    # Images
    "jpg", "jpeg", "png", "tiff", "tif", "bmp", "gif", "webp",
    "heic", "heif", "dng", "cr2", "nef", "arw", "orf", "rw2",
    # Video
    "mov", "mp4", "m4v", "avi", "3gp", "mpg", "mpeg", "mkv", "wmv",
}

# Extensions to silently skip (not media, no EXIF expected)
SKIP_EXTENSIONS = {
    "json", "csv", "txt", "py", "pyc", "pyi", "ini", "cfg", "toml",
    "md", "rst", "html", "xml", "yaml", "yml",
    "db", "sqlite", "sql",
    "aae", "ds_store", "gitignore",
    "zip", "tar", "gz", "7z",
    "pdf", "doc", "docx", "pptx", "xls", "xlsx",
    "sh", "bat", "ps1", "fish", "csh", "nu",
    "so", "pem", "pub", "lock", "sample",
    "c", "js", "typed", "pth", "partial",
}


def is_media_file(file_path: str) -> bool:
    """Check if a file has a recognized media extension."""
    ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
    return ext in MEDIA_EXTENSIONS


def read_exif_metadata(file_path: str) -> Dict[str, Any]:
    """
    Read EXIF metadata using exiftool (handles HEIC, MOV, JPEG, etc.).

    Returns dict with:
    - exif_score (0-1): richness of metadata
    - exif_datetime: DateTimeOriginal if present
    - exif_gps: "lat,lon" if present
    - exif_fields_count: # of non-null EXIF fields
    """
    empty = {
        "exif_score": 0.0,
        "exif_datetime": None,
        "exif_gps": None,
        "exif_fields_count": 0,
    }

    # Skip non-media files silently
    ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
    if ext in SKIP_EXTENSIONS or ext not in MEDIA_EXTENSIONS:
        return empty

    try:
        result = subprocess.run(
            [
                "exiftool", "-json", "-q",
                "-DateTimeOriginal", "-CreateDate", "-ModifyDate",
                "-GPSLatitude", "-GPSLongitude",
                "-Make", "-Model", "-ImageSize", "-MIMEType",
                "-n",  # numeric GPS output
                file_path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0 or not result.stdout.strip():
            return empty

        data = json.loads(result.stdout)[0]

        # Count meaningful fields
        fields_count = sum(1 for v in data.values() if v and v != "")

        # Extract datetime (prefer DateTimeOriginal > CreateDate > ModifyDate)
        exif_datetime = None
        for key in ("DateTimeOriginal", "CreateDate", "ModifyDate"):
            val = data.get(key)
            if val and isinstance(val, str) and val != "0000:00:00 00:00:00":
                try:
                    from datetime import datetime as dt
                    exif_datetime = dt.strptime(val[:19], "%Y:%m:%d %H:%M:%S")
                    break
                except ValueError:
                    continue

        # Extract GPS
        exif_gps = None
        lat = data.get("GPSLatitude")
        lon = data.get("GPSLongitude")
        if lat is not None and lon is not None:
            try:
                exif_gps = f"{float(lat)},{float(lon)}"
            except (ValueError, TypeError):
                pass

        # Score: 0-1 based on field density and presence of key fields
        score = 0.0
        if fields_count > 0:
            score = min(0.6, fields_count / 10.0)  # 10 fields = 0.6
            if exif_datetime:
                score += 0.2
            if exif_gps:
                score += 0.2
            score = min(1.0, score)

        return {
            "exif_score": score,
            "exif_datetime": exif_datetime,
            "exif_gps": exif_gps,
            "exif_fields_count": fields_count,
        }

    except subprocess.TimeoutExpired:
        logger.warning(f"exiftool timeout on {file_path}")
        return empty
    except Exception as e:
        logger.warning(f"Failed to read EXIF from {file_path}: {e}")
        return empty
