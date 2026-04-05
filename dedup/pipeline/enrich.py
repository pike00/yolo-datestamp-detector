import logging
import os
from typing import List, Dict, Any
from pathlib import Path
from utils.db import get_session, close_session
from utils.exif import read_exif_metadata
from utils.threading import ThreadedExecutor
from models import SourceFile, UniqueFile

logger = logging.getLogger(__name__)


def get_best_exif_for_hash(file_paths: List[str], staging_dir: str) -> Dict[str, Any]:
    """
    Read EXIF from all candidate files for a hash group.
    Return the candidate with the highest EXIF score and its metadata.
    """
    best_score = 0.0
    best_metadata = None
    best_path = None

    for rel_path in file_paths:
        full_path = os.path.join(staging_dir, rel_path)
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
        "best_path": best_path or (file_paths[0] if file_paths else None),
        "exif_datetime": best_metadata.get("exif_datetime") if best_metadata else None,
        "exif_gps": best_metadata.get("exif_gps") if best_metadata else None,
        "exif_fields_count": best_metadata.get("exif_fields_count", 0) if best_metadata else 0,
    }


def enrich_hash(args: tuple) -> dict:
    """Process a single unique hash: read EXIF and store metadata."""
    hash_val, staging_dir = args

    try:
        session = get_session()

        # Get all source files for this hash
        candidates = session.query(SourceFile).filter_by(sha256=hash_val).all()
        if not candidates:
            logger.warning(f"Hash {hash_val} has no source_files entries")
            close_session(session)
            return {"sha256": hash_val, "status": "no_candidates"}

        # Read EXIF from candidates, pick the best
        file_paths = [c.path for c in candidates]
        exif_info = get_best_exif_for_hash(file_paths, staging_dir)

        # Create unique_files entry with EXIF data (no canonical selection yet)
        unique_file = UniqueFile(
            sha256=hash_val,
            canonical_path=exif_info["best_path"] or candidates[0].path,
            selection_reason="preliminary",  # temporary, will be set in deduplicate stage
            exif_score=exif_info["best_score"],
            exif_datetime=exif_info["exif_datetime"],
            exif_gps=exif_info["exif_gps"],
            exif_fields_count=exif_info["exif_fields_count"],
            duplicate_count=len(candidates) - 1,
            export_status="pending",
        )
        session.add(unique_file)
        session.commit()
        close_session(session)

        return {
            "sha256": hash_val,
            "status": "success",
            "exif_score": exif_info["best_score"],
        }

    except Exception as e:
        logger.error(f"Error enriching hash {hash_val}: {e}")
        close_session(session)
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

    Creates a unique_files entry per hash with EXIF data.
    Canonical selection (priority-based) happens in the next stage.

    Resumable: skips hashes already in unique_files table.
    """
    logger.info("Stage 1: Enriching with EXIF metadata")

    session = get_session()

    # Get unique hashes not yet enriched
    enriched_hashes = set(row[0] for row in session.query(UniqueFile.sha256).all())
    all_hashes = set(row[0] for row in session.query(SourceFile.sha256).distinct().all())
    to_enrich = list(all_hashes - enriched_hashes)

    logger.info(
        f"Found {len(all_hashes)} unique hashes, "
        f"{len(enriched_hashes)} already enriched, "
        f"{len(to_enrich)} to process"
    )
    close_session(session)

    if not to_enrich:
        logger.info("Stage 1: All hashes already enriched, skipping")
        return

    # Prepare items for threading
    items = [(hash_val, staging_dir) for hash_val in to_enrich]

    # Process with threads
    executor = ThreadedExecutor(max_workers=thread_workers)
    results = executor.execute_batch(
        items,
        fn=enrich_hash,
        task_name="Stage 1: Enrich with EXIF",
    )

    # Summary
    success_count = sum(1 for r in results if r.get("status") == "success")
    error_count = sum(1 for r in results if r.get("status") == "error")
    logger.info(f"Stage 1 complete: {success_count} enriched, {error_count} errors")
