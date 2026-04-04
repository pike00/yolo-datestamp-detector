from PIL import Image
import piexif
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def read_exif_metadata(file_path: str) -> Dict[str, Any]:
    """
    Read EXIF metadata from an image file.

    Returns dict with:
    - exif_score (0-1): richness of metadata
    - exif_datetime: DateTimeOriginal if present
    - exif_gps: "lat,lon" if present
    - exif_fields_count: # of non-null EXIF fields
    """
    try:
        img = Image.open(file_path)
        exif_data = img.getexif()

        # Count populated fields
        fields_count = len(exif_data) if exif_data else 0

        # Extract DateTimeOriginal
        exif_datetime = exif_data.get(306) if exif_data else None  # Tag 306 = DateTime
        if exif_datetime and isinstance(exif_datetime, str):
            try:
                from datetime import datetime as dt
                exif_datetime = dt.strptime(exif_datetime, "%Y:%m:%d %H:%M:%S")
            except ValueError:
                exif_datetime = None

        # Extract GPS if present
        exif_gps = None
        if exif_data and 34853 in exif_data:  # GPS IFD pointer
            try:
                gps_ifd = exif_data.get_ifd(piexif.ImageIFD.GPSTag)
                if gps_ifd:
                    lat = gps_ifd.get(piexif.GPSIFD.GPSLatitude)
                    lon = gps_ifd.get(piexif.GPSIFD.GPSLongitude)
                    if lat and lon:
                        exif_gps = f"{lat},{lon}"
            except Exception as e:
                logger.warning(f"Failed to parse GPS from {file_path}: {e}")

        # Score: 0-1 based on field density and presence of key fields
        score = 0.0
        if fields_count > 0:
            score = min(1.0, fields_count / 30.0)  # 30 fields = max score
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

    except Exception as e:
        logger.warning(f"Failed to read EXIF from {file_path}: {e}")
        return {
            "exif_score": 0.0,
            "exif_datetime": None,
            "exif_gps": None,
            "exif_fields_count": 0,
        }
