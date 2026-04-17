from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from media_embeddings.loader import IMAGE_EXTS, VIDEO_EXTS, scan_media_dir


def test_scan_splits_images_and_videos(media_dir):
    images, videos = scan_media_dir(media_dir)
    assert len(images) == 3   # aaa.jpg, bbb.jpeg, ccc.heic
    assert len(videos) == 1   # ddd.mov
    assert all(p.suffix.lower() in IMAGE_EXTS for p in images)
    assert all(p.suffix.lower() in VIDEO_EXTS for p in videos)


def test_scan_ignores_non_media(media_dir):
    images, videos = scan_media_dir(media_dir)
    assert not any(p.suffix == ".json" for p in images + videos)


def test_scan_returns_sorted(media_dir):
    images, videos = scan_media_dir(media_dir)
    assert images == sorted(images)
    assert videos == sorted(videos)
