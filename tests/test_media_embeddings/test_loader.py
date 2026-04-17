from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from media_embeddings.loader import IMAGE_EXTS, VIDEO_EXTS, scan_media_dir, open_image, extract_keyframes


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


def test_open_jpg_returns_rgb(tmp_path):
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    path = tmp_path / "test.jpg"
    img.save(path)
    result = open_image(path)
    assert result.mode == "RGB"
    assert result.size == (100, 100)


def test_open_jpeg_returns_rgb(tmp_path):
    img = Image.new("RGB", (50, 80))
    path = tmp_path / "test.jpeg"
    img.save(path)
    assert open_image(path).mode == "RGB"


def test_open_heic_delegates_to_pillow_heif(tmp_path):
    path = tmp_path / "test.heic"
    path.write_bytes(b"fake")

    mock_heif = MagicMock()
    mock_heif.mode = "RGB"
    mock_heif.size = (200, 150)
    mock_heif.data = b"\x00" * (200 * 150 * 3)

    with patch("media_embeddings.loader._pillow_heif") as mock_ph:
        mock_ph.read_heif.return_value = mock_heif
        result = open_image(path)

    mock_ph.read_heif.assert_called_once_with(path)
    assert result.size == (200, 150)


def test_open_heic_converts_to_rgb(tmp_path):
    path = tmp_path / "test.heic"
    path.write_bytes(b"fake")

    mock_heif = MagicMock()
    mock_heif.mode = "RGBA"  # non-RGB mode — common in HEIC files
    mock_heif.size = (200, 150)
    mock_heif.data = b"\x00" * (200 * 150 * 4)  # 4 bytes per pixel for RGBA

    with patch("media_embeddings.loader._pillow_heif") as mock_ph:
        mock_ph.read_heif.return_value = mock_heif
        result = open_image(path)

    assert result.mode == "RGB"


def test_extract_keyframes_returns_three_pil_images(real_video):
    frames = extract_keyframes(real_video, n=3)
    assert len(frames) == 3
    assert all(isinstance(f, Image.Image) for f in frames)
    assert all(f.mode == "RGB" for f in frames)


def test_extract_keyframes_timestamps_are_10_50_90_pct(tmp_path):
    fake_frame = Image.new("RGB", (64, 64))

    with patch("media_embeddings.loader.subprocess.run") as mock_run, \
         patch("media_embeddings.loader.Image") as mock_img:

        ffprobe_result = MagicMock(stdout="10.0\n", returncode=0)
        ffmpeg_result = MagicMock(stdout=b"\x89PNG\r\n", returncode=0)
        mock_run.side_effect = [ffprobe_result] + [ffmpeg_result] * 3
        mock_img.open.return_value.convert.return_value = fake_frame

        extract_keyframes(tmp_path / "v.mov", n=3)

    ffmpeg_calls = [c for c in mock_run.call_args_list
                    if any("ffmpeg" in str(a) for a in c[0])]
    assert len(ffmpeg_calls) == 3

    timestamps = []
    for call in ffmpeg_calls:
        args = call[0][0]
        ts_idx = args.index("-ss") + 1
        timestamps.append(float(args[ts_idx]))

    assert timestamps == pytest.approx([1.0, 5.0, 9.0], abs=0.01)


def test_extract_keyframes_raises_on_missing_duration(tmp_path):
    with patch("media_embeddings.loader.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        with pytest.raises(ValueError, match="ffprobe returned no duration"):
            extract_keyframes(tmp_path / "bad.mov")
