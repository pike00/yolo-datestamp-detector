from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from media_embeddings.db import MODEL_NAME, bulk_insert_embeddings, get_completed_stems


def test_get_completed_stems_returns_two_sets(mock_conn):
    conn, cursor = mock_conn
    cursor.fetchall.side_effect = [
        [("abc123",), ("def456",)],
        [("vid789",)],
    ]

    done_photos, done_videos = get_completed_stems(conn, MODEL_NAME)

    assert done_photos == {"abc123", "def456"}
    assert done_videos == {"vid789"}
    assert cursor.execute.call_count == 2


def test_get_completed_stems_empty_table(mock_conn):
    conn, cursor = mock_conn
    cursor.fetchall.side_effect = [[], []]

    done_photos, done_videos = get_completed_stems(conn, MODEL_NAME)

    assert done_photos == set()
    assert done_videos == set()


def test_bulk_insert_uses_executemany(mock_conn):
    conn, cursor = mock_conn
    vec = np.zeros(1152, dtype=np.float32)
    rows = [
        ("abc123", MODEL_NAME, vec, "photo", None),
        ("def456", MODEL_NAME, vec, "photo", None),
    ]

    bulk_insert_embeddings(conn, rows)

    cursor.executemany.assert_called_once()
    sql, data = cursor.executemany.call_args[0]
    assert "INSERT INTO photo_embeddings" in sql
    assert "ON CONFLICT DO NOTHING" in sql
    assert data == rows
    conn.commit.assert_called_once()


def test_bulk_insert_video_rows(mock_conn):
    conn, cursor = mock_conn
    vec = np.ones(1152, dtype=np.float32)
    rows = [
        ("vid789", MODEL_NAME, vec, "video_keyframe", 0),
        ("vid789", MODEL_NAME, vec, "video_keyframe", 1),
        ("vid789", MODEL_NAME, vec, "video_keyframe", 2),
    ]

    bulk_insert_embeddings(conn, rows)

    cursor.executemany.assert_called_once()
    conn.commit.assert_called_once()
