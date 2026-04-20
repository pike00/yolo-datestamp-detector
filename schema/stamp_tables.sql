-- stamp_* tables for the ScanMyPhotos date-stamp pipeline.
-- Column sets are inferred from scripts/_db.py usage; columns added beyond
-- what _db.py references are noted inline.
--
-- Re-created 2026-04-20 after the 2026-04-16 pgvector volume wipe.
-- Idempotent: safe to re-run.

-- ---------- stamp_predictions ----------
-- Bounding box predictions from YOLO. One row per stem per model label.
-- Only the latest prediction per stem is retained (upsert on conflict).
CREATE TABLE IF NOT EXISTS stamp_predictions (
    stem        text        PRIMARY KEY,
    x           integer     NOT NULL,
    y           integer     NOT NULL,
    w           integer     NOT NULL,
    h           integer     NOT NULL,
    confidence  real        NOT NULL,
    model       text        NOT NULL DEFAULT 'yolo26m-best',
    updated_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS stamp_predictions_model_idx ON stamp_predictions (model);
CREATE INDEX IF NOT EXISTS stamp_predictions_conf_idx ON stamp_predictions (confidence);


-- ---------- stamp_no_stamp ----------
-- Stems confirmed (by auto-signal or human) to have no visible date stamp.
-- source: 'user' (human via corrections dashboard), 'auto' (YOLO conf < thresh
-- AND OCR empty), 'skip' (explicit skip during annotation).
CREATE TABLE IF NOT EXISTS stamp_no_stamp (
    stem      text        PRIMARY KEY,
    source    text        NOT NULL DEFAULT 'user',
    added_at  timestamptz NOT NULL DEFAULT now()
);


-- ---------- stamp_prediction_drift ----------
-- Diff between old and new YOLO predictions when models are swapped.
-- Either old_* or new_* may be NULL (stem added/removed between runs).
CREATE TABLE IF NOT EXISTS stamp_prediction_drift (
    stem            text        PRIMARY KEY,
    old_x           integer,
    old_y           integer,
    old_w           integer,
    old_h           integer,
    old_confidence  real,
    new_x           integer,
    new_y           integer,
    new_w           integer,
    new_h           integer,
    new_confidence  real,
    iou             real,
    flag            text,
    updated_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS stamp_prediction_drift_flag_idx ON stamp_prediction_drift (flag);


-- ---------- stamp_ocr ----------
-- OCR results keyed by (stem, model) so Haiku and Gemma can coexist.
-- raw_text: what the OCR returned verbatim.
-- bbox_source: 'crop' (stage-1 tight crop), 'full' (stage-2 full-frame view),
--   or 'manual' (human entry).
-- stage: 1 = stage-1 pass, 2 = stage-2 reconciliation.
-- review_status: 'auto_accepted', 'needs_review', 'rejected', 'manual'.
--
-- parsed_date / parse_error added 2026-04-20 (not in pre-wipe schema):
-- store the parsed DATE once at merge time so join queries don't reparse
-- raw_text. Unparseable rows get parse_error populated instead.
CREATE TABLE IF NOT EXISTS stamp_ocr (
    stem           text        NOT NULL,
    model          text        NOT NULL,
    raw_text       text,
    bbox_source    text,
    confidence     real,
    stage          integer,
    review_status  text,
    parsed_date    date,
    parse_error    text,
    updated_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (stem, model)
);
CREATE INDEX IF NOT EXISTS stamp_ocr_review_idx ON stamp_ocr (review_status);
CREATE INDEX IF NOT EXISTS stamp_ocr_parsed_date_idx ON stamp_ocr (parsed_date) WHERE parsed_date IS NOT NULL;


-- ---------- scanmyphotos_manifest ----------
-- Join table: disc-prefixed stem <-> source file on HDD <-> sha256 of the
-- organized/deduplicated copy under originals/media/{sha256}.{ext}.
-- Populated by scripts/data/build_scanmyphotos_manifest.py.
CREATE TABLE IF NOT EXISTS scanmyphotos_manifest (
    stem         text        PRIMARY KEY,
    disc         integer     NOT NULL,
    source_path  text        NOT NULL,
    sha256       text        NOT NULL,
    size_bytes   bigint      NOT NULL,
    mtime        timestamptz,
    hashed_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS scanmyphotos_manifest_sha256_idx ON scanmyphotos_manifest (sha256);
CREATE INDEX IF NOT EXISTS scanmyphotos_manifest_disc_idx ON scanmyphotos_manifest (disc);
