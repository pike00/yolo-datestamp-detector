from __future__ import annotations

import io
import subprocess
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif', '.heic'}
VIDEO_EXTS = {'.mov', '.mp4', '.m4v', '.mpg', '.3gp'}

try:
    import pillow_heif as _pillow_heif
    _HEIC_AVAILABLE = True
except ImportError:
    _pillow_heif = None  # type: ignore
    _HEIC_AVAILABLE = False


def scan_media_dir(media_dir: Path) -> tuple[list[Path], list[Path]]:
    images, videos = [], []
    for p in sorted(media_dir.iterdir()):
        ext = p.suffix.lower()
        if ext in IMAGE_EXTS:
            images.append(p)
        elif ext in VIDEO_EXTS:
            videos.append(p)
    return images, videos


def open_image(path: Path) -> Image.Image:
    if path.suffix.lower() == ".heic":
        if _pillow_heif is None:
            raise ImportError("pillow-heif required for HEIC: pip install pillow-heif")
        heif = _pillow_heif.read_heif(path)
        return Image.frombytes(heif.mode, heif.size, heif.data, "raw")
    return Image.open(path).convert("RGB")


def extract_keyframes(video_path: Path, n: int = 3) -> list[Image.Image]:
    raise NotImplementedError
