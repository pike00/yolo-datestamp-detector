import os
import shutil
import logging
from utils.db import get_session, close_session
from utils.threading import ThreadedExecutor
from models.schema import Canonical, CopyProgress

logger = logging.getLogger(__name__)


def copy_single_file(args: tuple) -> dict:
    """Copy a single canonical file from staging to originals with hash-based name."""
    hash_val, canonical_path, extension, staging_dir, originals_dir, db_path, retry_limit = args

    try:
        # Determine source and destination paths
        source_file = os.path.join(staging_dir, canonical_path.lstrip("img/"))
        dest_file = os.path.join(originals_dir, f"{hash_val}.{extension}")

        # Verify source exists
        if not os.path.exists(source_file):
            logger.warning(f"Source file not found: {source_file}")
            raise FileNotFoundError(f"Source file not found: {source_file}")

        # Copy file
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
        shutil.copy2(source_file, dest_file)
        file_size = os.path.getsize(dest_file)

        # Update DB
        session = get_session(db_path)
        progress = session.query(CopyProgress).filter_by(hash=hash_val).first()
        if progress:
            progress.status = "done"
            progress.copied_path = dest_file
            progress.bytes_copied = file_size
        session.commit()
        close_session(session)

        return {
            "hash": hash_val,
            "status": "success",
            "dest_path": dest_file,
            "bytes": file_size,
        }

    except Exception as e:
        logger.error(f"Error copying {hash_val}: {e}")

        # Update DB with error
        session = get_session(db_path)
        progress = session.query(CopyProgress).filter_by(hash=hash_val).first()
        if progress:
            progress.retry_count += 1
            if progress.retry_count >= retry_limit:
                progress.status = "error"
                progress.error_msg = str(e)
            else:
                progress.status = "pending"
        session.commit()
        close_session(session)

        return {
            "hash": hash_val,
            "status": "error",
            "error": str(e),
        }


def copy_to_originals(
    staging_dir: str,
    originals_dir: str,
    db_path: str,
    thread_workers: int = 4,
    retry_limit: int = 2,
) -> None:
    """
    Stage 4: Copy canonical files from staging to originals/ with hash-based names.
    Resumable: skips files already copied or marked error.
    """
    logger.info(f"Stage 4: Copying canonicals from {staging_dir} to {originals_dir}")

    session = get_session(db_path)

    # Get all canonicals with pending status
    canonicals = session.query(Canonical).all()
    pending_hashes = session.query(CopyProgress).filter(
        CopyProgress.status.in_(["pending"])
    ).all()

    logger.info(f"Found {len(canonicals)} canonicals, {len(pending_hashes)} pending")

    # Prepare copy items
    copy_items = []
    for cp in pending_hashes:
        canonical = session.query(Canonical).filter_by(hash=cp.hash).first()
        if not canonical:
            logger.warning(f"Canonical not found for hash {cp.hash}")
            continue

        # Extract extension from canonical_path
        extension = canonical.canonical_path.split(".")[-1].lower()

        copy_items.append((
            cp.hash,
            canonical.canonical_path,
            extension,
            staging_dir,
            originals_dir,
            db_path,
            retry_limit,
        ))

    close_session(session)

    if not copy_items:
        logger.info("Stage 4: No pending items, skipping")
        return

    # Copy with threads
    executor = ThreadedExecutor(max_workers=thread_workers)
    results = executor.execute_batch(
        copy_items,
        fn=copy_single_file,
        task_name="Stage 4: Copy to Originals",
    )

    # Summary
    success_count = sum(1 for r in results if r.get("status") == "success")
    error_count = sum(1 for r in results if r.get("status") == "error")
    logger.info(f"Stage 4 complete: {success_count} copied, {error_count} errors")
