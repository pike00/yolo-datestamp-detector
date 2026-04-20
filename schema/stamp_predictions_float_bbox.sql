-- Migrate stamp_predictions bbox columns from integer to real.
-- Integer columns silently truncate the normalized-float YOLO outputs
-- to 0 or 1, corrupting every bbox. Existing rows are truncated garbage
-- and must be re-generated from YOLO inference after this migration.

BEGIN;

-- Drop the rows first so the type change has no data to cast.
TRUNCATE TABLE stamp_predictions;

-- Change column types. `real` matches the 6-decimal float output of
-- infer_all.py (see extract_best_prediction()).
ALTER TABLE stamp_predictions
    ALTER COLUMN x TYPE real,
    ALTER COLUMN y TYPE real,
    ALTER COLUMN w TYPE real,
    ALTER COLUMN h TYPE real;

COMMIT;
