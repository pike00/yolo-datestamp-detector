import os
import shutil
import logging
from pathlib import Path
from utils.db import get_session, close_session
from utils.threading import ThreadedExecutor
from models.schema import StagingProgress

logger = logging.getLogger(__name__)


def calculate_expected_size(source_dir: str) -> int:
    """Calculate total size of all files in directory."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total += os.path.getsize(filepath)
    return total


def copy_file(args: tuple) -> dict:
    """Copy a single file with progress tracking. Returns result dict."""
    source_path, staging_path, db_path = args

    try:
        # Create staging subdirectory if needed
        os.makedirs(os.path.dirname(staging_path), exist_ok=True)

        # Copy file
        shutil.copy2(source_path, staging_path)
        file_size = os.path.getsize(staging_path)

        # Update DB
        session = get_session(db_path)
        progress = session.query(StagingProgress).filter_by(source_path=source_path).first()
        if progress is None:
            progress = StagingProgress(
                source_path=source_path,
                staging_path=staging_path,
                status="done",
                bytes_copied=file_size,
            )
            session.add(progress)
        else:
            progress.status = "done"
            progress.bytes_copied = file_size
        session.commit()
        close_session(session)

        return {
            "source_path": source_path,
            "status": "success",
            "bytes": file_size,
        }

    except Exception as e:
        logger.error(f"Error copying {source_path}: {e}")

        # Log error to DB
        session = get_session(db_path)
        progress = session.query(StagingProgress).filter_by(source_path=source_path).first()
        if progress is None:
            progress = StagingProgress(
                source_path=source_path,
                staging_path=staging_path,
                status="error",
                error_msg=str(e),
            )
            session.add(progress)
        else:
            progress.status = "error"
            progress.error_msg = str(e)
        session.commit()
        close_session(session)

        return {
            "source_path": source_path,
            "status": "error",
            "error": str(e),
        }


def copy_files_to_staging(
    source_dir: str,
    staging_dir: str,
    db_path: str,
    thread_workers: int = 4,
) -> None:
    """
    Copy all files from HDD source_dir to SSD staging_dir.
    Resumable via DB tracking.
    """
    logger.info(f"Stage 0: Copying files from {source_dir} to {staging_dir}")

    # Collect all file paths
    all_files = []
    for dirpath, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            source_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(source_path, source_dir)
            staging_path = os.path.join(staging_dir, rel_path)
            all_files.append((source_path, staging_path, db_path))

    logger.info(f"Found {len(all_files)} files to copy")

    # Skip function: check if already done in DB
    def skip_if_done(item):
        source_path = item[0]
        session = get_session(db_path)
        progress = session.query(StagingProgress).filter_by(source_path=source_path).first()
        already_done = progress and progress.status == "done"
        close_session(session)
        return already_done

    # Copy with threads
    executor = ThreadedExecutor(max_workers=thread_workers)
    results = executor.execute_batch(
        all_files,
        fn=copy_file,
        task_name="Stage 0: Copy HDD→SSD",
        skip_fn=skip_if_done,
    )

    # Summary
    success_count = sum(1 for r in results if r.get("status") == "success")
    error_count = sum(1 for r in results if r.get("status") == "error")
    logger.info(f"Stage 0 complete: {success_count} succeeded, {error_count} failed")
