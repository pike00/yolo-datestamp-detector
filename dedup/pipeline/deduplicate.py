import logging
from typing import Tuple
from utils.db import get_session, close_session
from models import SourceFile, UniqueFile

logger = logging.getLogger(__name__)

# Folder priority ordering (from PLAN.md section 2.2)
FOLDER_PRIORITY = {
    "iCloudPhotos": 0,
    "Photos": 1,
    "20230513 ios Photos": 2,
    "Pictures": 3,
    "Desktop": 4,
}


def get_folder_priority(folder: str) -> int:
    """Get priority score for a folder. Lower = higher priority."""
    return FOLDER_PRIORITY.get(folder, 999)  # Unknown folders get lowest priority


def select_canonical_for_hash(
    sha256: str,
    session,
) -> Tuple[str, str]:
    """
    Select canonical file for a hash group using priority rules:
    1. Highest EXIF score (from unique_files.exif_score)
    2. Source folder priority (iCloudPhotos > Photos > ...)
    3. Shortest path

    Returns: (canonical_path, selection_reason)
    """
    # Get all source files for this hash
    source_files = session.query(SourceFile).filter_by(sha256=sha256).all()
    unique_file = session.query(UniqueFile).filter_by(sha256=sha256).first()

    if not source_files or not unique_file:
        logger.warning(f"Hash {sha256} missing source_files or unique_files entries")
        return None, None

    # Priority 1: EXIF score is already computed per hash (not per file)
    # So all candidates share the same EXIF score (from the best one)
    exif_score = unique_file.exif_score

    # Priority 2: Apply folder priority and path length as tiebreakers
    def score_candidate(source_file):
        folder_priority = get_folder_priority(source_file.source_folder)
        path_length = len(source_file.path)
        # Return tuple for sorting (lower = better)
        # Primary: folder priority, Secondary: path length
        return (folder_priority, path_length)

    # Sort candidates by score
    candidates = sorted(source_files, key=score_candidate)
    winner = candidates[0]

    # Determine selection reason
    if len(candidates) == 1:
        reason = "only_copy"
    elif get_folder_priority(winner.source_folder) < 999:
        # Folder was in our priority list
        reason = "folder_priority"
    else:
        reason = "shortest_path"

    return winner.path, reason


def deduplicate() -> None:
    """
    Stage 2: Deduplicate phase - select canonical file for each unique hash.

    Uses priority rules:
    1. EXIF score (already picked best-EXIF candidate in enrichment)
    2. Source folder priority (iCloudPhotos > Photos > ...)
    3. Shortest file path (tiebreaker)

    Updates unique_files table with canonical_path and selection_reason.
    Idempotent: skips hashes that already have a selection_reason != "preliminary".
    """
    logger.info("Stage 2: Deduplicating - selecting canonicals")

    session = get_session()

    # Get hashes that need canonical selection
    unresolved = session.query(UniqueFile).filter_by(selection_reason="preliminary").all()

    logger.info(f"Found {len(unresolved)} hashes to deduplicate")

    if not unresolved:
        logger.info("Stage 2: All hashes already deduplicated, skipping")
        close_session(session)
        return

    # Process each hash
    updated = 0
    for unique_file in unresolved:
        canonical_path, selection_reason = select_canonical_for_hash(
            unique_file.sha256, session
        )

        if not canonical_path:
            logger.warning(f"Could not select canonical for hash {unique_file.sha256}")
            continue

        # Update with final canonical selection
        unique_file.canonical_path = canonical_path
        unique_file.selection_reason = selection_reason
        updated += 1

        if updated % 5000 == 0:
            session.commit()
            logger.debug(f"Updated {updated} canonicals")

    session.commit()
    close_session(session)

    logger.info(f"Stage 2 complete: {updated} canonicals selected")
