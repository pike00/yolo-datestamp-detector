import hashlib
import logging
import os
import shutil
from pathlib import Path
from utils.db import get_session, close_session
from utils.threading import ThreadedExecutor
from models import SourceFile

logger = logging.getLogger(__name__)
BUFFER_SIZE = 65536  # 64KB read buffer for hashing


def copy_and_hash_file(args: tuple) -> dict:
    """Copy file from HDD to staging, compute SHA-256 hash, insert into DB."""
    source_path, source_dir, staging_dir = args

    try:
        # Compute relative path from source root
        rel_path = os.path.relpath(source_path, source_dir)
        staging_path = os.path.join(staging_dir, rel_path)

        # Create staging subdirectories
        os.makedirs(os.path.dirname(staging_path), exist_ok=True)

        # Copy file if not already present with matching size
        source_size = os.path.getsize(source_path)
        if os.path.exists(staging_path) and os.path.getsize(staging_path) == source_size:
            # File already copied
            copied_size = source_size
        else:
            # Copy file
            shutil.copy2(source_path, staging_path)
            copied_size = os.path.getsize(staging_path)

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
            size=source_size,
            source_folder=source_folder,
            filename=filename,
            extension=extension,
        )
        session.merge(entry)  # merge handles duplicates by PK
        session.commit()
        close_session(session)

        return {
            "rel_path": rel_path,
            "sha256": hash_val,
            "status": "success",
            "size": source_size,
            "staging_path": staging_path,
        }

    except Exception as e:
        logger.error(f"Error copying/hashing {source_path}: {e}")
        return {
            "source_path": source_path,
            "status": "error",
            "error": str(e),
        }


def ingest(
    source_dir: str,
    staging_dir: str,
    thread_workers: int = 4,
) -> None:
    """
    Stage 0: Ingest phase - copy all files from HDD to SSD staging and compute hashes.

    Preserves directory structure: if source_dir=`/mnt/hdd` and file is `/mnt/hdd/Desktop/photo.jpg`,
    staging_path will be `staging/Desktop/photo.jpg`.

    Resumable: uses DB to track completed files.
    """
    logger.info(f"Stage 0: Ingesting files from {source_dir} to {staging_dir}")

    # Collect all file paths
    all_files = []
    dir_count = 0
    for dirpath, dirnames, filenames in os.walk(source_dir):
        dir_count += 1
        if dir_count % 100 == 0:
            logger.debug(f"Scanning: {dirpath} ({len(all_files)} files so far)")
        for filename in filenames:
            source_path = os.path.join(dirpath, filename)
            all_files.append((source_path, source_dir, staging_dir))

    logger.info(f"Found {len(all_files)} files to ingest ({dir_count} directories scanned)")

    # Pre-load completed paths for O(1) skip logic
    session = get_session()
    completed_paths = set(row[0] for row in session.query(SourceFile.path).all())
    close_session(session)
    logger.info(f"Found {len(completed_paths)} already ingested files in DB")

    # Skip function: check if already processed
    def skip_if_done(item):
        source_path, source_dir, _ = item
        rel_path = os.path.relpath(source_path, source_dir)
        return rel_path in completed_paths

    # Process with threads
    executor = ThreadedExecutor(max_workers=thread_workers)
    results = executor.execute_batch(
        all_files,
        fn=copy_and_hash_file,
        task_name="Stage 0: Ingest (Copy + Hash)",
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
