import hashlib
import logging
import os
from utils.db import get_session, close_session
from utils.threading import ThreadedExecutor
from models.schema import Canonical, FinalReport

logger = logging.getLogger(__name__)


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def verify_single_file(args: tuple) -> dict:
    """Re-hash a copied file and compare to original hash."""
    hash_val, copied_path, db_path = args

    try:
        # Verify file exists
        if not os.path.exists(copied_path):
            raise FileNotFoundError(f"Copied file not found: {copied_path}")

        # Compute hash of copied file
        computed_hash = compute_file_hash(copied_path)

        # Compare to original hash
        if computed_hash == hash_val:
            status = "verified"
        else:
            logger.warning(f"Hash mismatch for {hash_val}: computed {computed_hash}")
            status = "mismatch"

        # Update DB
        session = get_session(db_path)
        canonical = session.query(Canonical).filter_by(hash=hash_val).first()
        if canonical:
            canonical.verified = (status == "verified")
            canonical.verification_hash = computed_hash
        session.commit()
        close_session(session)

        return {
            "hash": hash_val,
            "status": status,
            "verification_hash": computed_hash,
        }

    except Exception as e:
        logger.error(f"Error verifying {hash_val}: {e}")

        session = get_session(db_path)
        canonical = session.query(Canonical).filter_by(hash=hash_val).first()
        if canonical:
            canonical.verified = False
        session.commit()
        close_session(session)

        return {
            "hash": hash_val,
            "status": "error",
            "error": str(e),
        }


def verify_copies(
    originals_dir: str,
    db_path: str,
    thread_workers: int = 4,
) -> None:
    """
    Stage 5: Re-hash all files in originals/ and verify against original hashes.
    Resumable: skips files already verified.
    """
    logger.info("Stage 5: Verifying copied files")

    session = get_session(db_path)

    # Get canonicals not yet verified
    unverified = session.query(Canonical).filter_by(verified=False).all()
    logger.info(f"Found {len(unverified)} unverified canonicals")

    if not unverified:
        logger.info("Stage 5: All files already verified, skipping")
        close_session(session)
        return

    # Prepare verification items
    verify_items = []
    for canonical in unverified:
        # Construct path to copied file
        extension = canonical.canonical_path.split(".")[-1].lower()
        copied_path = os.path.join(originals_dir, f"{canonical.hash}.{extension}")

        verify_items.append((canonical.hash, copied_path, db_path))

    close_session(session)

    # Verify with threads
    executor = ThreadedExecutor(max_workers=thread_workers)
    results = executor.execute_batch(
        verify_items,
        fn=verify_single_file,
        task_name="Stage 5: Verify Copies",
    )

    # Summary
    verified_count = sum(1 for r in results if r.get("status") == "verified")
    mismatch_count = sum(1 for r in results if r.get("status") == "mismatch")
    error_count = sum(1 for r in results if r.get("status") == "error")

    logger.info(
        f"Stage 5 complete: {verified_count} verified, "
        f"{mismatch_count} mismatches, {error_count} errors"
    )

    # Generate final report
    session = get_session(db_path)
    total_unique = session.query(Canonical).count()
    verified = session.query(Canonical).filter_by(verified=True).count()

    final_report = FinalReport(
        stage_completed="5",
        total_files_analyzed=0,
        unique_files=total_unique,
        duplicate_files_removed=0,
        total_space_saved_gb=0.0,
        files_by_source=None,
        duration_seconds=0,
        errors_encountered=error_count,
        verified_copies=verified_count,
        failed_verifications=mismatch_count,
    )
    session.add(final_report)
    session.commit()
    close_session(session)
