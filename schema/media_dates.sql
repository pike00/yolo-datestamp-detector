-- Unified date sources for organized media.
-- Each media file keyed by sha256 can have zero or more date rows:
--   * exif_dates      — EXIF DateTimeOriginal/Digitized/DateTime from photo headers
--   * video_dates     — ffprobe creation_time from video containers
--   * stamp_ocr       — date parsed from ScanMyPhotos printed date-stamp OCR
--                       (joined via scanmyphotos_manifest to get sha256)
--
-- Consumers pick source priority as needed. To "does this file have ANY
-- known date?" query `media_dates` directly.

-- ---------- exif_dates ----------
-- Populated by scripts/data/extract_exif_dates.py
-- source: which EXIF tag was used ('DateTimeOriginal', 'DateTimeDigitized', 'DateTime')
CREATE TABLE IF NOT EXISTS exif_dates (
    sha256       text        PRIMARY KEY,
    date_taken   timestamptz NOT NULL,
    source       text        NOT NULL,
    raw          text,
    extracted_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS exif_dates_date_idx ON exif_dates (date_taken);


-- ---------- video_dates ----------
-- Populated by a loader that imports state/video_dates.json, produced by
-- scripts/extract_video_dates.py. Filename inside video_dates.json already
-- uses sha256.ext, so we parse out the sha256 on import.
CREATE TABLE IF NOT EXISTS video_dates (
    sha256       text        PRIMARY KEY,
    date_taken   timestamptz NOT NULL,
    source       text        NOT NULL,
    raw          text,
    extracted_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS video_dates_date_idx ON video_dates (date_taken);


-- ---------- media_dates view ----------
-- One row per (sha256, source_kind). Consumers can deduplicate with priority
-- or just query "is sha256 present at all" to know if the file is dated.
CREATE OR REPLACE VIEW media_dates AS
SELECT sha256,
       date_taken,
       source,
       'exif'::text AS kind,
       NULL::text AS review_status,
       NULL::real AS confidence
FROM exif_dates
UNION ALL
SELECT sha256,
       date_taken,
       source,
       'video'::text AS kind,
       NULL::text AS review_status,
       NULL::real AS confidence
FROM video_dates
UNION ALL
SELECT m.sha256,
       (so.parsed_date::timestamp AT TIME ZONE 'UTC') AS date_taken,
       ('stamp_ocr:' || so.model)::text AS source,
       'stamp_ocr'::text AS kind,
       so.review_status,
       so.confidence
FROM scanmyphotos_manifest m
JOIN stamp_ocr so ON so.stem = m.stem AND so.parsed_date IS NOT NULL;


-- ---------- convenience: media_has_date ----------
-- True/false set of sha256s with at least one known date. Makes the
-- undated-media query trivial:
--   SELECT sha256 FROM photo_embeddings
--   WHERE sha256 NOT IN (SELECT sha256 FROM media_has_date);
CREATE OR REPLACE VIEW media_has_date AS
SELECT DISTINCT sha256 FROM media_dates;
