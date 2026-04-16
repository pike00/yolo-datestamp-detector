import logging
from typing import Tuple
from utils.db import get_session, close_session
from models.schema import FilePath, File, Canonical, CopyProgress

logger = logging.getLogger(__name__)


def compute_canonical_for_hash(
    hash_val: str,
    session,
) -> Tuple[str, str, int]:
    """
    Compute canonical selection for a hash group.
    Returns: (canonical_path, selected_reason, duplicate_count)
    """
    # Get all candidates for this hash
    file_paths = session.query(FilePath).filter_by(hash=hash_val).all()
    file_entry = session.query(File).filter_by(hash=hash_val).first()

    if not file_paths or not file_entry:
        return None, None, 0

    # Primary: highest EXIF score (from File entry)
    canonical_path = file_entry.canonical_path
    folder_source = file_entry.folder_source

    # Tiebreaker: folder cohesion
    folder_counts = {}
    for fp in file_paths:
        folder = fp.source_folder
        folder_counts[folder] = folder_counts.get(folder, 0) + 1

    # If best EXIF is not from the dominant folder, check tiebreaker
    dominant_folder = max(folder_counts, key=folder_counts.get)
    dominant_count = folder_counts[dominant_folder]

    if folder_source != dominant_folder and dominant_count > 1:
        # Prefer file from dominant folder as tiebreaker
        for fp in file_paths:
            if fp.source_folder == dominant_folder:
                canonical_path = fp.path
                folder_source = dominant_folder
                selected_reason = "folder_cohesion_tiebreak"
                break
    else:
        selected_reason = "best_exif"

    duplicate_count = len(file_paths) - 1
    return canonical_path, selected_reason, duplicate_count


def select_canonicals(db_path: str) -> None:
    """
    Stage 3: Select canonical file for each unique hash.
    Create Canonical and CopyProgress entries.
    Idempotent: skips if Canonical already exists.
    """
    logger.info("Stage 3: Selecting canonicals")

    session = get_session(db_path)

    # Get unique hashes not yet processed
    processed_hashes = set(row[0] for row in session.query(Canonical.hash).all())
    all_hashes = set(row[0] for row in session.query(File.hash).all())
    to_process = list(all_hashes - processed_hashes)

    logger.info(
        f"Found {len(all_hashes)} unique hashes, "
        f"{len(processed_hashes)} already processed, "
        f"{len(to_process)} to process"
    )

    if not to_process:
        logger.info("Stage 3: All hashes already processed, skipping")
        close_session(session)
        return

    # Process each hash
    inserted = 0
    for hash_val in to_process:
        canonical_path, selected_reason, duplicate_count = compute_canonical_for_hash(
            hash_val, session
        )

        if not canonical_path:
            logger.warning(f"Could not select canonical for hash {hash_val}")
            continue

        # Extract extension from canonical path
        extension = canonical_path.split(".")[-1].lower()

        # Create Canonical entry
        canonical = Canonical(
            hash=hash_val,
            canonical_path=canonical_path,
            duplicate_count=duplicate_count,
            total_size_saved_bytes=0,
            verified=False,
        )
        session.add(canonical)

        # Create CopyProgress entry (initialization for Stage 4)
        copy_progress = CopyProgress(
            hash=hash_val,
            status="pending",
            copied_path=None,
            bytes_copied=0,
            retry_count=0,
        )
        session.add(copy_progress)

        inserted += 1
        if inserted % 1000 == 0:
            session.commit()
            logger.debug(f"Inserted {inserted} canonicals")

    session.commit()
    close_session(session)

    logger.info(f"Stage 3 complete: {inserted} canonicals created")
