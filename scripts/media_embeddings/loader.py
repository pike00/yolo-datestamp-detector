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
        return Image.frombytes(heif.mode, heif.size, heif.data, "raw").convert("RGB")
    return Image.open(path).convert("RGB")


def extract_keyframes(video_path: Path, n: int = 3) -> list[Image.Image]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    raw = result.stdout.strip()
    if not raw:
        raise ValueError(f"ffprobe returned no duration for {video_path}")
    duration = float(raw)

    frames: list[Image.Image] = []
    for pct in [0.1, 0.5, 0.9][:n]:
        t = duration * pct
        result = subprocess.run(
            ["ffmpeg", "-noaccurate_seek", "-ss", str(t), "-i", str(video_path),
             "-vframes", "1", "-f", "image2", "-vcodec", "png",
             "pipe:1", "-loglevel", "error"],
            capture_output=True, check=True,
        )
        img = Image.open(io.BytesIO(result.stdout)).convert("RGB")
        frames.append(img)
    return frames
