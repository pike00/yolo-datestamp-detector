from __future__ import annotations

import psycopg

MODEL_NAME = "siglip-so400m"


def get_completed_stems(
    conn: psycopg.Connection, model: str = MODEL_NAME
) -> tuple[set[str], set[str]]:
    """Return (done_photo_stems, done_video_stems).

    Video stems are only counted done once all 3 keyframes are present.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT sha256 FROM photo_embeddings "
            "WHERE model = %s AND media_type = 'photo'",
            (model,),
        )
        done_photos = {row[0] for row in cur.fetchall()}

        cur.execute(
            "SELECT sha256 FROM photo_embeddings "
            "WHERE model = %s AND media_type = 'video_keyframe' "
            "GROUP BY sha256 HAVING COUNT(*) >= 3",
            (model,),
        )
        done_videos = {row[0] for row in cur.fetchall()}

    return done_photos, done_videos


def bulk_insert_embeddings(
    conn: psycopg.Connection, rows: list[tuple]
) -> None:
    """Insert rows of (sha256, model, embedding, media_type, frame_index)."""
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO photo_embeddings "
            "(sha256, model, embedding, media_type, frame_index) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT DO NOTHING",
            rows,
        )
    conn.commit()
