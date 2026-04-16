"""Shared Postgres helpers for YOLO finetune scripts.

Imported by sibling scripts via a sys.path hack:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from _db import get_db, load_predictions, ...

Importing scripts must declare `psycopg[binary]>=3.1.0` in their PEP 723
dependencies (or have it installed in the active environment).

Tables in the dedup database used by these helpers:
    stamp_predictions       - YOLO bbox predictions
    stamp_prediction_drift  - old vs new prediction comparison
    stamp_ocr               - OCR results, composite PK on (stem, model)
    stamp_no_stamp          - stems with no date stamp
"""

from __future__ import annotations

import os

import psycopg

DB_CONN_STRING = os.environ.get(
    "DATABASE_URL",
    "postgresql://dedup:dedup_local_dev@localhost:5432/dedup",
)
PREDICTION_MODEL_LABEL = os.environ.get("YOLO_MODEL_LABEL", "yolo26m-best")
OCR_MODEL_HAIKU = "haiku"


def get_db():
    """Open a psycopg connection (caller closes)."""
    return psycopg.connect(DB_CONN_STRING)


# ---------- stamp_predictions ----------


def load_predictions() -> dict[str, dict]:
    """All YOLO predictions as {stem: {x, y, w, h, confidence}}."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT stem, x, y, w, h, confidence FROM stamp_predictions"
        ).fetchall()
    return {
        r[0]: {"x": r[1], "y": r[2], "w": r[3], "h": r[4], "confidence": r[5]}
        for r in rows
    }


def get_predicted_stems() -> set[str]:
    with get_db() as conn:
        return {
            r[0] for r in conn.execute("SELECT stem FROM stamp_predictions").fetchall()
        }


def upsert_predictions(items, model_label: str | None = None) -> int:
    """Upsert (stem, x, y, w, h, confidence) tuples.

    items: iterable of 6-tuples.
    """
    label = model_label or PREDICTION_MODEL_LABEL
    rows = [(s, x, y, w, h, c, label) for (s, x, y, w, h, c) in items]
    if not rows:
        return 0
    with get_db() as conn, conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO stamp_predictions (stem, x, y, w, h, confidence, model)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (stem) DO UPDATE SET
                x = EXCLUDED.x, y = EXCLUDED.y,
                w = EXCLUDED.w, h = EXCLUDED.h,
                confidence = EXCLUDED.confidence,
                model = EXCLUDED.model,
                updated_at = NOW()
            """,
            rows,
        )
        conn.commit()
    return len(rows)


# ---------- stamp_no_stamp ----------


def load_skipped_stems() -> set[str]:
    with get_db() as conn:
        return {
            r[0] for r in conn.execute("SELECT stem FROM stamp_no_stamp").fetchall()
        }


def add_skipped(stem: str, source: str = "user") -> None:
    with get_db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO stamp_no_stamp (stem, source) VALUES (%s, %s)
            ON CONFLICT (stem) DO NOTHING
            """,
            (stem, source),
        )
        conn.commit()


def remove_skipped(stem: str) -> None:
    with get_db() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM stamp_no_stamp WHERE stem = %s", (stem,))
        conn.commit()


# ---------- stamp_prediction_drift ----------


def load_drift() -> dict[str, dict]:
    """Return drift rows as {stem: {old, new, iou, flag}} matching the legacy JSON shape."""
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT stem,
                   old_x, old_y, old_w, old_h, old_confidence,
                   new_x, new_y, new_w, new_h, new_confidence,
                   iou, flag
            FROM stamp_prediction_drift
            """
        ).fetchall()
    out: dict[str, dict] = {}
    for r in rows:
        old = (
            None
            if r[1] is None
            else {"x": r[1], "y": r[2], "w": r[3], "h": r[4], "confidence": r[5]}
        )
        new = (
            None
            if r[6] is None
            else {"x": r[6], "y": r[7], "w": r[8], "h": r[9], "confidence": r[10]}
        )
        out[r[0]] = {"old": old, "new": new, "iou": r[11], "flag": r[12]}
    return out


def upsert_drift(items) -> int:
    """items: iterable of (stem, old_dict_or_none, new_dict_or_none, iou, flag)."""

    def unpack(b):
        if b is None:
            return (None, None, None, None, None)
        return (b["x"], b["y"], b["w"], b["h"], b["confidence"])

    rows = [
        (stem, *unpack(old), *unpack(new), iou, flag)
        for stem, old, new, iou, flag in items
    ]
    if not rows:
        return 0
    with get_db() as conn, conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO stamp_prediction_drift (
                stem,
                old_x, old_y, old_w, old_h, old_confidence,
                new_x, new_y, new_w, new_h, new_confidence,
                iou, flag
            )
            VALUES (%s, %s,%s,%s,%s,%s, %s,%s,%s,%s,%s, %s, %s)
            ON CONFLICT (stem) DO UPDATE SET
                old_x = EXCLUDED.old_x, old_y = EXCLUDED.old_y,
                old_w = EXCLUDED.old_w, old_h = EXCLUDED.old_h,
                old_confidence = EXCLUDED.old_confidence,
                new_x = EXCLUDED.new_x, new_y = EXCLUDED.new_y,
                new_w = EXCLUDED.new_w, new_h = EXCLUDED.new_h,
                new_confidence = EXCLUDED.new_confidence,
                iou = EXCLUDED.iou,
                flag = EXCLUDED.flag,
                updated_at = NOW()
            """,
            rows,
        )
        conn.commit()
    return len(rows)


# ---------- stamp_ocr (haiku model) ----------


def load_ocr_results(model: str = OCR_MODEL_HAIKU) -> dict[str, dict]:
    """Return {stem: {text, bbox_source, confidence, stage, review_status}} for the given model."""
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT stem, raw_text, bbox_source, confidence, stage, review_status
            FROM stamp_ocr WHERE model = %s
            """,
            (model,),
        ).fetchall()
    return {
        r[0]: {
            "text": r[1],
            "bbox_source": r[2],
            "confidence": r[3],
            "stage": r[4],
            "review_status": r[5],
        }
        for r in rows
    }


def upsert_ocr_result(
    stem: str,
    text: str,
    *,
    bbox_source: str | None = None,
    confidence: float | None = None,
    stage: int | None = None,
    review_status: str | None = None,
    model: str = OCR_MODEL_HAIKU,
) -> None:
    """Upsert a single OCR row, leaving unspecified fields untouched on conflict."""
    with get_db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO stamp_ocr (stem, raw_text, bbox_source, model, confidence, stage, review_status)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (stem, model) DO UPDATE SET
                raw_text     = EXCLUDED.raw_text,
                bbox_source  = COALESCE(EXCLUDED.bbox_source, stamp_ocr.bbox_source),
                confidence   = COALESCE(EXCLUDED.confidence, stamp_ocr.confidence),
                stage        = COALESCE(EXCLUDED.stage, stamp_ocr.stage),
                review_status= COALESCE(EXCLUDED.review_status, stamp_ocr.review_status),
                updated_at   = NOW()
            """,
            (stem, text, bbox_source, model, confidence, stage, review_status),
        )
        conn.commit()


def update_ocr_review_status(
    stem: str, review_status: str, model: str = OCR_MODEL_HAIKU
) -> None:
    """Set review_status without touching any other field."""
    with get_db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE stamp_ocr SET review_status = %s, updated_at = NOW()
            WHERE stem = %s AND model = %s
            """,
            (review_status, stem, model),
        )
        conn.commit()


def upsert_ocr_results_bulk(items, model: str = OCR_MODEL_HAIKU) -> int:
    """Bulk upsert (stem, text, bbox_source, confidence, stage) tuples.

    For bulk imports, simpler than the COALESCE single-row version; this
    overwrites with the supplied values, leaving review_status unchanged on
    conflict.
    """
    rows = [(s, t, b, model, c, st) for (s, t, b, c, st) in items]
    if not rows:
        return 0
    with get_db() as conn, conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO stamp_ocr (stem, raw_text, bbox_source, model, confidence, stage)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (stem, model) DO UPDATE SET
                raw_text    = EXCLUDED.raw_text,
                bbox_source = EXCLUDED.bbox_source,
                confidence  = EXCLUDED.confidence,
                stage       = EXCLUDED.stage,
                updated_at  = NOW()
            """,
            rows,
        )
        conn.commit()
    return len(rows)
