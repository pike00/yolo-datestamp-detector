from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import psycopg
import pytest
from PIL import Image


@pytest.fixture
def media_dir(tmp_path: Path) -> Path:
    img = Image.new("RGB", (64, 64), color=(128, 0, 0))
    img.save(tmp_path / "aaa.jpg")
    img.save(tmp_path / "bbb.jpeg")
    (tmp_path / "ccc.heic").write_bytes(b"fake-heic")
    (tmp_path / "ddd.mov").write_bytes(b"fake-mov")
    (tmp_path / "eee.json").write_bytes(b"{}")
    return tmp_path


@pytest.fixture
def real_video(tmp_path: Path) -> Path:
    out = tmp_path / "test.mov"
    subprocess.run(
        ["ffmpeg", "-f", "lavfi", "-i", "color=black:size=64x64:duration=3:rate=1",
         "-c:v", "libx264", "-t", "3", str(out), "-y", "-loglevel", "error"],
        check=True,
    )
    return out


@pytest.fixture
def mock_conn():
    conn = MagicMock(spec=psycopg.Connection)
    cursor = MagicMock()
    cursor.__enter__ = lambda s: cursor
    cursor.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = cursor
    return conn, cursor
