import hashlib
import logging
import os
import shutil
from utils.db import get_session, close_session
from utils.threading import ThreadedExecutor
from models import SourceFile, UniqueFile

logger = logging.getLogger(__name__)


def copy_canonical_file(args: tuple) -> dict:
    """Copy a canonical file from staging to originals with hash-based name."""
    sha256, canonical_path, extension, staging_dir, originals_dir, retry_limit = args

    try:
        # Build full source path
        source_file = os.path.join(staging_dir, canonical_path)

        # Verify source exists
        if not os.path.exists(source_file):
            raise FileNotFoundError(f"Source file not found: {source_file}")

        # Build destination: originals/<hash>.<ext>
        dest_file = os.path.join(originals_dir, f"{sha256}.{extension}")

        # Copy file
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
        shutil.copy2(source_file, dest_file)
        file_size = os.path.getsize(dest_file)

        # Update DB
        session = get_session()
        unique_file = session.query(UniqueFile).filter_by(sha256=sha256).first()
        if unique_file:
            unique_file.export_status = "copied"
            unique_file.export_path = dest_file
        session.commit()
        close_session(session)

        return {
            "sha256": sha256,
            "status": "copied",
            "dest_path": dest_file,
            "bytes": file_size,
        }

    except Exception as e:
        logger.error(f"Error copying {sha256}: {e}")

        # Update DB with error
        session = get_session()
        unique_file = session.query(UniqueFile).filter_by(sha256=sha256).first()
        if unique_file:
            unique_file.retry_count += 1
            if unique_file.retry_count >= retry_limit:
                unique_file.export_status = "error"
                unique_file.error_msg = str(e)
            else:
                # Keep as pending for retry
                pass
        session.commit()
        close_session(session)

        return {
            "sha256": sha256,
            "status": "error",
            "error": str(e),
        }


def verify_copied_file(args: tuple) -> dict:
    """Re-hash a copied file and compare to original hash."""
    sha256, dest_file = args

    try:
        # Verify file exists
        if not os.path.exists(dest_file):
            raise FileNotFoundError(f"Exported file not found: {dest_file}")

        # Compute SHA-256 hash of copied file
        sha256_hash = hashlib.sha256()
        with open(dest_file, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        computed_hash = sha256_hash.hexdigest()

        # Compare to original hash
        if computed_hash == sha256:
            status = "verified"
        else:
            logger.warning(f"Hash mismatch for {sha256}: computed {computed_hash}")
            status = "mismatch"

        # Update DB
        session = get_session()
        unique_file = session.query(UniqueFile).filter_by(sha256=sha256).first()
        if unique_file:
            if status == "verified":
                unique_file.export_status = "verified"
            else:
                unique_file.export_status = "mismatch"
            unique_file.verified_hash = computed_hash
        session.commit()
        close_session(session)

        return {
            "sha256": sha256,
            "status": status,
            "verification_hash": computed_hash,
        }

    except Exception as e:
        logger.error(f"Error verifying {sha256}: {e}")

        session = get_session()
        unique_file = session.query(UniqueFile).filter_by(sha256=sha256).first()
        if unique_file:
            unique_file.export_status = "error"
            unique_file.error_msg = str(e)
        session.commit()
        close_session(session)

        return {
            "sha256": sha256,
            "status": "error",
            "error": str(e),
        }


def export_canonicals(
    staging_dir: str,
    originals_dir: str,
    thread_workers: int = 4,
    retry_limit: int = 2,
) -> None:
    """
    Stage 3: Export phase - copy canonicals to originals/ and verify.

    Copies each unique_file's canonical to originals/<hash>.<extension> and re-hashes
    to verify integrity.

    Resumable: skips files already exported (export_status != pending).
    """
    logger.info(f"Stage 3: Exporting canonicals from {staging_dir} to {originals_dir}")

    session = get_session()

    # Get canonicals to export (status pending)
    pending = session.query(UniqueFile).filter_by(export_status="pending").all()

    logger.info(f"Found {len(pending)} canonicals pending export")

    # Prepare copy items
    copy_items = []
    for unique_file in pending:
        # Extract extension from canonical_path
        extension = unique_file.canonical_path.split(".")[-1].lower()

        copy_items.append((
            unique_file.sha256,
            unique_file.canonical_path,
            extension,
            staging_dir,
            originals_dir,
            retry_limit,
        ))

    close_session(session)

    if not copy_items:
        logger.info("Stage 3: No pending items, skipping")
        return

    # Copy with threads
    executor = ThreadedExecutor(max_workers=thread_workers)
    results = executor.execute_batch(
        copy_items,
        fn=copy_canonical_file,
        task_name="Stage 3a: Copy to Originals",
    )

    # Summary
    copied_count = sum(1 for r in results if r.get("status") == "copied")
    error_count = sum(1 for r in results if r.get("status") == "error")
    logger.info(f"Stage 3a complete: {copied_count} copied, {error_count} errors")

    # Now verify all copied files
    session = get_session()
    copied_files = session.query(UniqueFile).filter_by(export_status="copied").all()

    verify_items = [
        (uf.sha256, uf.export_path)
        for uf in copied_files
        if uf.export_path
    ]

    close_session(session)

    logger.info(f"Verifying {len(verify_items)} copied files")

    if verify_items:
        executor = ThreadedExecutor(max_workers=thread_workers)
        results = executor.execute_batch(
            verify_items,
            fn=verify_copied_file,
            task_name="Stage 3b: Verify Exports",
        )

        # Summary
        verified_count = sum(1 for r in results if r.get("status") == "verified")
        mismatch_count = sum(1 for r in results if r.get("status") == "mismatch")
        error_count = sum(1 for r in results if r.get("status") == "error")

        logger.info(
            f"Stage 3b complete: {verified_count} verified, "
            f"{mismatch_count} mismatches, {error_count} errors"
        )
    else:
        logger.info("Stage 3b: No files to verify")

    # Final summary
    session = get_session()
    total_unique = session.query(UniqueFile).count()
    verified = session.query(UniqueFile).filter_by(export_status="verified").count()
    mismatch = session.query(UniqueFile).filter_by(export_status="mismatch").count()
    export_error = session.query(UniqueFile).filter_by(export_status="error").count()
    close_session(session)

    logger.info(
        f"Stage 3 complete: {total_unique} unique files, "
        f"{verified} verified, {mismatch} mismatches, {export_error} errors"
    )
