"""Tests for bench_vlm_ocr.py helper functions.

Does NOT call Ollama. The smoke test (Step 6) covers the network path.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts.ocr.bench_vlm_ocr import (
    encode_crop,
    load_manifest_stems,
    make_model_key,
    write_jsonl_row,
)


def test_make_model_key_joins_with_at_sign():
    assert make_model_key("kimi-vl:latest", "ares-cpu") == "kimi-vl:latest@ares-cpu"


def test_make_model_key_strips_whitespace():
    assert make_model_key("  qwen2.5-vl:7b ", " m2pro ") == "qwen2.5-vl:7b@m2pro"


def test_encode_crop_produces_base64(tmp_path):
    from PIL import Image

    p = tmp_path / "t.jpg"
    Image.new("RGB", (100, 100), "red").save(p, format="JPEG")
    b64 = encode_crop(p, max_side=512)
    # Round-trip decode to confirm valid image bytes.
    import base64
    import io
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert img.size == (100, 100)


def test_encode_crop_resizes_when_over_max(tmp_path):
    from PIL import Image
    import base64
    import io

    p = tmp_path / "big.jpg"
    Image.new("RGB", (2000, 1000), "red").save(p, format="JPEG")
    b64 = encode_crop(p, max_side=512)
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert max(img.size) == 512


def test_load_manifest_stems_returns_subset_when_limit(tmp_path):
    m = {
        "stems": [
            {"stem": "s1", "crop_path": "crops/s1.jpg"},
            {"stem": "s2", "crop_path": "crops/s2.jpg"},
            {"stem": "s3", "crop_path": "crops/s3.jpg"},
        ]
    }
    p = tmp_path / "m.json"
    p.write_text(json.dumps(m))
    assert len(load_manifest_stems(p, limit=2)) == 2
    assert len(load_manifest_stems(p, limit=None)) == 3


def test_write_jsonl_row_appends_single_line(tmp_path):
    out = tmp_path / "results.jsonl"
    row1 = {"stem": "a", "raw_text": "1 2'99", "parsed_date": "1999-01-02"}
    row2 = {"stem": "b", "raw_text": "TIMEOUT", "parsed_date": None}
    write_jsonl_row(out, row1)
    write_jsonl_row(out, row2)
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == row1
    assert json.loads(lines[1]) == row2
