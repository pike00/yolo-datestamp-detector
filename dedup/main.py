#!/usr/bin/env python
import logging
import sys
import time
from datetime import datetime
from config import Config
from utils.db import init_db
from utils.notifications import send_telegram_notification
from stages import (
    copy_files_to_staging,
    load_sha256sums,
    enrich_with_exif,
    select_canonicals,
    copy_to_originals,
    verify_copies,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_stage(stage_num: int, fn, *args, **kwargs) -> bool:
    """Run a single stage with error handling and logging."""
    logger.info(f"{'='*60}")
    logger.info(f"Starting Stage {stage_num}")
    logger.info(f"{'='*60}")

    try:
        fn(*args, **kwargs)
        logger.info(f"✓ Stage {stage_num} completed successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Stage {stage_num} failed: {e}", exc_info=True)
        return False


def main():
    """Orchestrate the full dedup pipeline."""
    Config.validate()

    logger.info("Starting Phase 2 Dedup Pipeline")
    logger.info(f"Config: HDD={Config.HDD_SOURCE_PATH}")
    logger.info(f"Config: Staging={Config.SSD_STAGING_PATH}")
    logger.info(f"Config: Originals={Config.SSD_ORIGINALS_PATH}")
    logger.info(f"Config: DB={Config.SSD_DB_PATH}")
    logger.info(f"Config: Threads={Config.THREAD_WORKERS}, Retries={Config.RETRY_LIMIT}")
    logger.info(f"Config: Stages to run: {Config.RUN_STAGES}")

    # Initialize database
    init_db(Config.SSD_DB_PATH)

    start_time = time.time()
    stage_results = {}

    # Stage 0: Preflight copy
    if 0 in Config.RUN_STAGES:
        stage_results[0] = run_stage(
            0,
            copy_files_to_staging,
            source_dir=Config.HDD_SOURCE_PATH,
            staging_dir=Config.SSD_STAGING_PATH,
            db_path=Config.SSD_DB_PATH,
            thread_workers=Config.THREAD_WORKERS,
        )
        if not stage_results[0]:
            logger.error("Stage 0 failed, aborting pipeline")
            notify_failure(stage_results, start_time)
            sys.exit(1)

    # Stage 1: Load
    if 1 in Config.RUN_STAGES:
        stage_results[1] = run_stage(
            1,
            load_sha256sums,
            manifest_path=Config.SHA256SUMS_PATH,
            db_path=Config.SSD_DB_PATH,
        )
        if not stage_results[1]:
            logger.error("Stage 1 failed, aborting pipeline")
            notify_failure(stage_results, start_time)
            sys.exit(1)

    # Stage 2: Enrich
    if 2 in Config.RUN_STAGES:
        stage_results[2] = run_stage(
            2,
            enrich_with_exif,
            staging_dir=Config.SSD_STAGING_PATH,
            db_path=Config.SSD_DB_PATH,
            thread_workers=Config.THREAD_WORKERS,
        )
        if not stage_results[2]:
            logger.error("Stage 2 failed, aborting pipeline")
            notify_failure(stage_results, start_time)
            sys.exit(1)

    # Stage 3: Select
    if 3 in Config.RUN_STAGES:
        stage_results[3] = run_stage(
            3,
            select_canonicals,
            db_path=Config.SSD_DB_PATH,
        )
        if not stage_results[3]:
            logger.error("Stage 3 failed, aborting pipeline")
            notify_failure(stage_results, start_time)
            sys.exit(1)

    # Stage 4: Copy
    if 4 in Config.RUN_STAGES:
        stage_results[4] = run_stage(
            4,
            copy_to_originals,
            staging_dir=Config.SSD_STAGING_PATH,
            originals_dir=Config.SSD_ORIGINALS_PATH,
            db_path=Config.SSD_DB_PATH,
            thread_workers=Config.THREAD_WORKERS,
            retry_limit=Config.RETRY_LIMIT,
        )
        if not stage_results[4]:
            logger.error("Stage 4 failed, aborting pipeline")
            notify_failure(stage_results, start_time)
            sys.exit(1)

    # Stage 5: Verify
    if 5 in Config.RUN_STAGES:
        stage_results[5] = run_stage(
            5,
            verify_copies,
            originals_dir=Config.SSD_ORIGINALS_PATH,
            db_path=Config.SSD_DB_PATH,
            thread_workers=Config.THREAD_WORKERS,
        )
        if not stage_results[5]:
            logger.error("Stage 5 failed, aborting pipeline")
            notify_failure(stage_results, start_time)
            sys.exit(1)

    # Success
    duration = time.time() - start_time
    logger.info(f"{'='*60}")
    logger.info(f"✓ All stages completed successfully!")
    logger.info(f"Duration: {duration/3600:.1f} hours ({duration/60:.0f} minutes)")
    logger.info(f"{'='*60}")

    notify_success(stage_results, duration)


def notify_success(stage_results: dict, duration: float):
    """Send success notification."""
    message = (
        f"✅ Phase 2 Dedup Pipeline Complete\n\n"
        f"Duration: {duration/3600:.1f}h ({duration/60:.0f}m)\n\n"
        f"All stages completed. Check DB for details."
    )
    send_telegram_notification(message, "✓ Dedup Success")


def notify_failure(stage_results: dict, duration: float):
    """Send failure notification."""
    failed_stages = [s for s, result in stage_results.items() if not result]
    message = (
        f"❌ Phase 2 Dedup Pipeline Failed\n\n"
        f"Failed stages: {failed_stages}\n"
        f"Duration: {duration/60:.0f} minutes\n\n"
        f"Check logs for details."
    )
    send_telegram_notification(message, "✗ Dedup Failure")


if __name__ == "__main__":
    main()
