import logging
import os
from utils.db import get_session, close_session
from utils.exif import read_exif_metadata, is_media_file
from utils.threading import ThreadedExecutor
from models import SourceFile, UniqueFile

logger = logging.getLogger(__name__)


def enrich_hash(args: tuple) -> dict:
    """Process a single unique hash: read EXIF and store metadata."""
    hash_val, staging_dir = args

    try:
        session = get_session()

        # Get all source files for this hash
        candidates = session.query(SourceFile).filter_by(sha256=hash_val).all()
        if not candidates:
            close_session(session)
            return {"sha256": hash_val, "status": "no_candidates"}

        # All candidates are byte-identical (same hash), so EXIF is identical.
        # Just find the first one that exists and is a media file.
        metadata = None
        for c in candidates:
            full_path = os.path.join(staging_dir, c.path)
            if os.path.exists(full_path) and is_media_file(c.path):
                metadata = read_exif_metadata(full_path)
                if metadata["exif_score"] > 0:
                    break  # Found good EXIF, stop

        if metadata is None:
            metadata = {
                "exif_score": 0.0,
                "exif_datetime": None,
                "exif_gps": None,
                "exif_fields_count": 0,
            }

        # Upsert unique_files entry with EXIF data
        unique_file = UniqueFile(
            sha256=hash_val,
            canonical_path=candidates[0].path,
            selection_reason="preliminary",
            exif_score=metadata["exif_score"],
            exif_datetime=metadata["exif_datetime"],
            exif_gps=metadata["exif_gps"],
            exif_fields_count=metadata["exif_fields_count"],
            duplicate_count=len(candidates) - 1,
            export_status="pending",
        )
        session.merge(unique_file)
        session.commit()
        close_session(session)

        return {
            "sha256": hash_val,
            "status": "success",
            "exif_score": metadata["exif_score"],
        }

    except Exception as e:
        logger.error(f"Error enriching hash {hash_val}: {e}")
        try:
            close_session(session)
        except Exception:
            pass
        return {
            "sha256": hash_val,
            "status": "error",
            "error": str(e),
        }


def enrich(
    staging_dir: str,
    thread_workers: int = 4,
) -> None:
    """
    Stage 1: Enrich phase - read EXIF metadata for all unique hashes.

    Creates/updates a unique_files entry per hash with EXIF data.
    Uses exiftool for broad format support (HEIC, MOV, JPEG, etc.).
    Re-runnable: uses merge() to update existing entries.
    """
    logger.info("Stage 1: Enriching with EXIF metadata")

    session = get_session()
    all_hashes = [row[0] for row in session.query(SourceFile.sha256).distinct().all()]
    close_session(session)

    logger.info(f"Found {len(all_hashes)} unique hashes to enrich")

    if not all_hashes:
        logger.info("Stage 1: No hashes to enrich")
        return

    items = [(hash_val, staging_dir) for hash_val in all_hashes]

    executor = ThreadedExecutor(max_workers=thread_workers)
    results = executor.execute_batch(
        items,
        fn=enrich_hash,
        task_name="Stage 1: Enrich with EXIF",
    )

    success_count = sum(1 for r in results if r.get("status") == "success")
    error_count = sum(1 for r in results if r.get("status") == "error")
    has_exif = sum(1 for r in results if r.get("exif_score", 0) > 0)
    logger.info(
        f"Stage 1 complete: {success_count} enriched, {error_count} errors, "
        f"{has_exif} with EXIF data"
    )
