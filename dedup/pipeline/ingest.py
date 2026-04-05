import hashlib
import logging
import os
from pathlib import Path
from utils.db import get_session, close_session
from utils.threading import ThreadedExecutor
from models import SourceFile

logger = logging.getLogger(__name__)
BUFFER_SIZE = 65536  # 64KB read buffer for hashing


def hash_and_register(args: tuple) -> dict:
    """Hash a file in staging and insert into DB."""
    staging_path, staging_dir = args

    try:
        rel_path = os.path.relpath(staging_path, staging_dir)
        file_size = os.path.getsize(staging_path)

        # Compute SHA-256 hash
        sha256 = hashlib.sha256()
        with open(staging_path, "rb") as f:
            while True:
                data = f.read(BUFFER_SIZE)
                if not data:
                    break
                sha256.update(data)

        hash_val = sha256.hexdigest()

        # Parse file metadata
        path_obj = Path(rel_path)
        filename = path_obj.name
        extension = path_obj.suffix.lstrip(".").lower()
        source_folder = path_obj.parts[0] if path_obj.parts else "unknown"

        # Store in database
        session = get_session()
        entry = SourceFile(
            path=rel_path,
            sha256=hash_val,
            size=file_size,
            source_folder=source_folder,
            filename=filename,
            extension=extension,
        )
        session.merge(entry)
        session.commit()
        close_session(session)

        return {
            "rel_path": rel_path,
            "sha256": hash_val,
            "status": "success",
            "size": file_size,
        }

    except Exception as e:
        logger.error(f"Error hashing {staging_path}: {e}")
        return {
            "staging_path": staging_path,
            "status": "error",
            "error": str(e),
        }


def ingest(
    source_dir: str,
    staging_dir: str,
    thread_workers: int = 4,
) -> None:
    """
    Stage 0: Ingest -- hash all files in staging and register in DB.

    Scans staging_dir (SSD) directly. Files should already be present in staging
    from a prior copy. Resumable via DB tracking.
    """
    logger.info(f"Stage 0: Scanning staging directory {staging_dir}")

    # Scan staging (SSD) -- much faster than scanning HDD
    all_files = []
    for dirpath, _, filenames in os.walk(staging_dir):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            all_files.append((full_path, staging_dir))

    logger.info(f"Found {len(all_files)} files in staging")

    # Pre-load completed paths for O(1) skip logic
    session = get_session()
    completed_paths = set(row[0] for row in session.query(SourceFile.path).all())
    close_session(session)
    logger.info(f"Found {len(completed_paths)} already ingested files in DB")

    # Skip function: check if already processed
    def skip_if_done(item):
        full_path, staging_dir = item
        rel_path = os.path.relpath(full_path, staging_dir)
        return rel_path in completed_paths

    # Process with threads
    executor = ThreadedExecutor(max_workers=thread_workers)
    results = executor.execute_batch(
        all_files,
        fn=hash_and_register,
        task_name="Stage 0: Ingest (Hash + Register)",
        skip_fn=skip_if_done,
    )

    # Summary
    success_count = sum(1 for r in results if r.get("status") == "success")
    error_count = sum(1 for r in results if r.get("status") == "error")

    # Database summary
    session = get_session()
    total_entries = session.query(SourceFile).count()
    unique_hashes = session.query(SourceFile.sha256).distinct().count()
    close_session(session)

    logger.info(
        f"Stage 0 complete: {success_count} ingested, {error_count} errors, "
        f"{total_entries} total files, {unique_hashes} unique hashes"
    )
