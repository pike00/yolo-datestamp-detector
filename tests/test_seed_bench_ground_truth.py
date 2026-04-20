"""Tests for the Sonnet ground-truth seeder IO."""

import json
from pathlib import Path

import pytest

from scripts.ocr.seed_bench_ground_truth import (
    build_shard_manifests,
    parse_subagent_output,
    select_pending_stems,
)


def test_select_pending_stems_skips_rows_already_in_db(tmp_path, monkeypatch):
    """If a stem already has a model='sonnet' row, it should be skipped."""

    manifest = {
        "stems": [
            {"stem": "s1", "crop_path": "crops/s1.jpg"},
            {"stem": "s2", "crop_path": "crops/s2.jpg"},
            {"stem": "s3", "crop_path": "crops/s3.jpg"},
        ]
    }

    existing_in_db = {"s2"}

    def fake_load_existing():
        return existing_in_db

    monkeypatch.setattr(
        "scripts.ocr.seed_bench_ground_truth.load_existing_sonnet_stems",
        fake_load_existing,
    )

    pending = select_pending_stems(manifest, force=False)
    assert [s["stem"] for s in pending] == ["s1", "s3"]


def test_select_pending_stems_force_returns_all(monkeypatch):
    manifest = {
        "stems": [{"stem": "s1"}, {"stem": "s2"}, {"stem": "s3"}]
    }
    monkeypatch.setattr(
        "scripts.ocr.seed_bench_ground_truth.load_existing_sonnet_stems",
        lambda: {"s1", "s2", "s3"},
    )
    pending = select_pending_stems(manifest, force=True)
    assert [s["stem"] for s in pending] == ["s1", "s2", "s3"]


def test_build_shard_manifests_splits_into_waves(tmp_path):
    pending = [{"stem": f"s{i}", "crop_path": f"crops/s{i}.jpg"} for i in range(17)]
    shard_dir = tmp_path / "shards"
    shards = build_shard_manifests(pending, wave_size=5, out_dir=shard_dir)
    assert len(shards) == 4  # 5+5+5+2
    assert shards[0]["size"] == 5
    assert shards[-1]["size"] == 2
    # Each manifest file exists and is readable JSON.
    for s in shards:
        p = Path(s["path"])
        assert p.exists()
        data = json.loads(p.read_text())
        assert "stems" in data
        assert len(data["stems"]) == s["size"]


def test_parse_subagent_output_takes_last_nonempty_line():
    raw = "I see the stamp.\nIt reads:\n10 3'99\n"
    assert parse_subagent_output(raw) == "10 3'99"


def test_parse_subagent_output_handles_empty():
    assert parse_subagent_output("") == ""
    assert parse_subagent_output("\n\n  \n") == ""


def test_parse_subagent_output_strips_thinking():
    raw = "<think>some reasoning</think>\n\n10 3'99"
    assert parse_subagent_output(raw) == "10 3'99"
