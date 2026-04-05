import json
import subprocess
import logging
from datetime import datetime as dt
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

EMPTY_RESULT = {
    "exif_score": 0.0,
    "exif_datetime": None,
    "exif_gps": None,
    "exif_fields_count": 0,
    "exif_data": None,
    "camera_make": None,
    "camera_model": None,
    "image_width": None,
    "image_height": None,
    "mime_type": None,
}


def is_media_file(file_path: str) -> bool:
    """Check if a file has a recognized media extension."""
    ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
    return ext in MEDIA_EXTENSIONS


def read_exif_metadata(file_path: str) -> Dict[str, Any]:
    """
    Read all EXIF metadata using exiftool (handles HEIC, MOV, JPEG, etc.).

    Returns dict with structured fields plus full exiftool JSON in exif_data.
    """
    ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
    if ext not in MEDIA_EXTENSIONS:
        return EMPTY_RESULT.copy()

    try:
        # Get ALL metadata as JSON (no field filter)
        result = subprocess.run(
            ["exiftool", "-json", "-q", "-n", file_path],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0 or not result.stdout.strip():
            return EMPTY_RESULT.copy()

        data = json.loads(result.stdout)[0]

        # Remove SourceFile key (it's just the path, not metadata)
        data.pop("SourceFile", None)

        # Count meaningful fields
        fields_count = sum(1 for v in data.values() if v is not None and v != "")

        # Extract datetime (prefer DateTimeOriginal > CreateDate > ModifyDate)
        exif_datetime = None
        for key in ("DateTimeOriginal", "CreateDate", "ModifyDate"):
            val = data.get(key)
            if val and isinstance(val, str) and val != "0000:00:00 00:00:00":
                try:
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

        # Extract structured fields
        camera_make = data.get("Make")
        camera_model = data.get("Model")
        image_width = data.get("ImageWidth") or data.get("ExifImageWidth")
        image_height = data.get("ImageHeight") or data.get("ExifImageHeight")
        mime_type = data.get("MIMEType")

        # Coerce dimensions to int
        try:
            image_width = int(image_width) if image_width else None
        except (ValueError, TypeError):
            image_width = None
        try:
            image_height = int(image_height) if image_height else None
        except (ValueError, TypeError):
            image_height = None

        # Score: 0-1 based on field density and presence of key fields
        score = 0.0
        if fields_count > 0:
            score = min(0.6, fields_count / 10.0)
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
            "exif_data": data,
            "camera_make": camera_make,
            "camera_model": camera_model,
            "image_width": image_width,
            "image_height": image_height,
            "mime_type": mime_type,
        }

    except subprocess.TimeoutExpired:
        logger.warning(f"exiftool timeout on {file_path}")
        return EMPTY_RESULT.copy()
    except Exception as e:
        logger.warning(f"Failed to read EXIF from {file_path}: {e}")
        return EMPTY_RESULT.copy()
