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
    raise NotImplementedError


def open_image(path: Path) -> Image.Image:
    raise NotImplementedError


def extract_keyframes(video_path: Path, n: int = 3) -> list[Image.Image]:
    raise NotImplementedError
