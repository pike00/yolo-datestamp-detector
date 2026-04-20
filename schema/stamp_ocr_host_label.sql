-- Add host_label to stamp_ocr so multi-host bench runs of the same model
-- can be grouped in reports. Nullable; existing rows (currently none, post
-- data-loss) get NULL.

ALTER TABLE stamp_ocr ADD COLUMN IF NOT EXISTS host_label TEXT;

CREATE INDEX IF NOT EXISTS stamp_ocr_host_label_idx ON stamp_ocr(host_label);
