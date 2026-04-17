from __future__ import annotations

import psycopg

MODEL_NAME = "siglip-so400m"


def get_completed_stems(
    conn: psycopg.Connection, model: str = MODEL_NAME
) -> tuple[set[str], set[str]]:
    """Return (done_photo_stems, done_video_stems)."""
    raise NotImplementedError


def bulk_insert_embeddings(
    conn: psycopg.Connection, rows: list[tuple]
) -> None:
    """Insert rows of (sha256, model, embedding, media_type, frame_index)."""
    raise NotImplementedError
