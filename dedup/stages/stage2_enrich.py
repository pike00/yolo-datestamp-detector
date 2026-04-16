import logging
import os
from typing import List
from pathlib import Path
from utils.db import get_session, close_session
from utils.exif import read_exif_metadata
from utils.threading import ThreadedExecutor
from models.schema import FilePath, File

logger = logging.getLogger(__name__)


def get_exif_score_for_candidates(
    file_paths: List[str],
    staging_dir: str,
) -> dict:
    """
    Read EXIF from all candidate files for a hash group.
    Return highest score and metadata.
    """
    best_score = 0.0
    best_metadata = None
    best_path = None

    for rel_path in file_paths:
        full_path = os.path.join(staging_dir, rel_path.lstrip("img/"))
        if not os.path.exists(full_path):
            logger.debug(f"File not found in staging: {full_path}")
            continue

        metadata = read_exif_metadata(full_path)
        score = metadata.get("exif_score", 0.0)

        if score > best_score:
            best_score = score
            best_metadata = metadata
            best_path = rel_path

    return {
        "best_score": best_score,
        "best_path": best_path,
        "exif_datetime": best_metadata.get("exif_datetime") if best_metadata else None,
        "exif_gps": best_metadata.get("exif_gps") if best_metadata else None,
        "exif_fields_count": best_metadata.get("exif_fields_count", 0) if best_metadata else 0,
    }


def enrich_hash(args: tuple) -> dict:
    """Process a single unique hash: read EXIF, compute score, create File entry."""
    hash_val, staging_dir, db_path = args

    try:
        session = get_session(db_path)

        # Get all candidate paths for this hash
        candidates = session.query(FilePath).filter_by(hash=hash_val).all()
        if not candidates:
            logger.warning(f"Hash {hash_val} has no file_paths entries")
            close_session(session)
            return {"hash": hash_val, "status": "no_candidates"}

        # Read EXIF from candidates
        file_paths = [c.path for c in candidates]
        exif_info = get_exif_score_for_candidates(file_paths, staging_dir)

        # Create File entry
        file_entry = File(
            hash=hash_val,
            canonical_path=exif_info["best_path"] or candidates[0].path,
            exif_score=exif_info["best_score"],
            exif_datetime=exif_info["exif_datetime"],
            exif_gps=exif_info["exif_gps"],
            exif_fields_count=exif_info["exif_fields_count"],
            folder_source=candidates[0].source_folder,
            selected_reason="preliminary",
        )
        session.add(file_entry)
        session.commit()
        close_session(session)

        return {
            "hash": hash_val,
            "status": "success",
            "exif_score": exif_info["best_score"],
        }

    except Exception as e:
        logger.error(f"Error enriching hash {hash_val}: {e}")
        close_session(session)
        return {
            "hash": hash_val,
            "status": "error",
            "error": str(e),
        }


def enrich_with_exif(
    staging_dir: str,
    db_path: str,
    thread_workers: int = 4,
) -> None:
    """
    Stage 2: Read EXIF metadata for all unique hashes.
    Resumable: skips hashes already in File table.
    """
    logger.info("Stage 2: Enriching with EXIF metadata")

    session = get_session(db_path)

    # Get unique hashes not yet enriched
    enriched_hashes = set(row[0] for row in session.query(File.hash).all())
    all_hashes = set(row[0] for row in session.query(FilePath.hash).distinct().all())
    to_enrich = list(all_hashes - enriched_hashes)

    logger.info(
        f"Found {len(all_hashes)} unique hashes, "
        f"{len(enriched_hashes)} already enriched, "
        f"{len(to_enrich)} to process"
    )
    close_session(session)

    if not to_enrich:
        logger.info("Stage 2: All hashes already enriched, skipping")
        return

    # Prepare items for threading
    items = [(hash_val, staging_dir, db_path) for hash_val in to_enrich]

    # Process with threads
    executor = ThreadedExecutor(max_workers=thread_workers)
    results = executor.execute_batch(
        items,
        fn=enrich_hash,
        task_name="Stage 2: Enrich with EXIF",
    )

    # Summary
    success_count = sum(1 for r in results if r.get("status") == "success")
    error_count = sum(1 for r in results if r.get("status") == "error")
    logger.info(f"Stage 2 complete: {success_count} enriched, {error_count} errors")
