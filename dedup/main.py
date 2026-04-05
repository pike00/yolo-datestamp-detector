#!/usr/bin/env python
import logging
import sys
import time
from datetime import datetime
from config import Config
from utils.db import init_db, close_all
from utils.notifications import send_telegram_notification
from pipeline import ingest, enrich, deduplicate, export_canonicals

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_stage(stage_num: int, stage_name: str, fn, *args, **kwargs) -> bool:
    """Run a single stage with error handling and logging."""
    logger.info(f"{'='*60}")
    logger.info(f"Starting Stage {stage_num}: {stage_name}")
    logger.info(f"{'='*60}")

    try:
        fn(*args, **kwargs)
        logger.info(f"✓ Stage {stage_num} ({stage_name}) completed successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Stage {stage_num} ({stage_name}) failed: {e}", exc_info=True)
        return False


def main():
    """Orchestrate the full dedup pipeline."""
    Config.validate()
    Config.setup_staging_paths()

    logger.info("Starting Dedup Pipeline (PostgreSQL)")
    logger.info(f"Config: HDD={Config.HDD_SOURCE_PATH}")
    logger.info(f"Config: Staging={Config.SSD_STAGING_PATH}")
    logger.info(f"Config: Originals={Config.SSD_ORIGINALS_PATH}")
    logger.info(f"Config: DB={Config.DB_HOST}:{Config.DB_PORT}/{Config.DB_NAME}")
    logger.info(f"Config: Threads={Config.THREAD_WORKERS}, Retries={Config.RETRY_LIMIT}")
    logger.info(f"Config: Stages to run: {Config.RUN_STAGES}")

    # Initialize database
    init_db()

    start_time = time.time()
    stage_results = {}

    try:
        # Stage 0: Ingest (copy + hash)
        if 0 in Config.RUN_STAGES:
            stage_results[0] = run_stage(
                0,
                "Ingest",
                ingest,
                source_dir=Config.HDD_SOURCE_PATH,
                staging_dir=Config.SSD_STAGING_PATH,
                thread_workers=Config.THREAD_WORKERS,
            )
            if not stage_results[0]:
                logger.error("Stage 0 failed, aborting pipeline")
                notify_failure(stage_results, start_time)
                sys.exit(1)

        # Stage 1: Enrich (EXIF)
        if 1 in Config.RUN_STAGES:
            stage_results[1] = run_stage(
                1,
                "Enrich",
                enrich,
                staging_dir=Config.SSD_STAGING_PATH,
                thread_workers=Config.THREAD_WORKERS,
            )
            if not stage_results[1]:
                logger.error("Stage 1 failed, aborting pipeline")
                notify_failure(stage_results, start_time)
                sys.exit(1)

        # Stage 2: Deduplicate (canonical selection)
        if 2 in Config.RUN_STAGES:
            stage_results[2] = run_stage(
                2,
                "Deduplicate",
                deduplicate,
            )
            if not stage_results[2]:
                logger.error("Stage 2 failed, aborting pipeline")
                notify_failure(stage_results, start_time)
                sys.exit(1)

        # Stage 3: Export (copy + verify)
        if 3 in Config.RUN_STAGES:
            stage_results[3] = run_stage(
                3,
                "Export",
                export_canonicals,
                staging_dir=Config.SSD_STAGING_PATH,
                originals_dir=Config.SSD_ORIGINALS_PATH,
                thread_workers=Config.THREAD_WORKERS,
                retry_limit=Config.RETRY_LIMIT,
            )
            if not stage_results[3]:
                logger.error("Stage 3 failed, aborting pipeline")
                notify_failure(stage_results, start_time)
                sys.exit(1)

        # Success
        duration = time.time() - start_time
        logger.info(f"{'='*60}")
        logger.info(f"✓ All stages completed successfully!")
        logger.info(f"Duration: {duration/3600:.1f} hours ({duration/60:.0f} minutes)")
        logger.info(f"{'='*60}")

        notify_success(stage_results, duration)

    finally:
        close_all()


def notify_success(stage_results: dict, duration: float):
    """Send success notification."""
    message = (
        f"✅ Dedup Pipeline Complete\n\n"
        f"Duration: {duration/3600:.1f}h ({duration/60:.0f}m)\n\n"
        f"All stages completed. Check logs for details."
    )
    send_telegram_notification(message, "✓ Dedup Success")


def notify_failure(stage_results: dict, start_time: float):
    """Send failure notification."""
    duration = time.time() - start_time
    failed_stages = [s for s, result in stage_results.items() if not result]
    message = (
        f"❌ Dedup Pipeline Failed\n\n"
        f"Failed stages: {failed_stages}\n"
        f"Duration: {duration/60:.0f} minutes\n\n"
        f"Check logs for details."
    )
    send_telegram_notification(message, "✗ Dedup Failure")


if __name__ == "__main__":
    main()
