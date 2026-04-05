import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration from environment variables or defaults."""

    # Storage paths
    HDD_SOURCE_PATH = os.getenv(
        "HDD_SOURCE_PATH",
        "/mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/Photos/img/",
    )
    SSD_STAGING_PATH = os.getenv(
        "SSD_STAGING_PATH",
        "/home/will/photo_project/staging/",
    )
    SSD_ORIGINALS_PATH = os.getenv(
        "SSD_ORIGINALS_PATH",
        "/home/will/photo_project/originals/",
    )

    # Derived paths
    SHA256SUMS_PATH = os.path.join(
        os.path.dirname(HDD_SOURCE_PATH),
        "SHA256SUMS.txt",
    )

    # Database (PostgreSQL)
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "dedup")
    DB_USER = os.getenv("DB_USER", "dedup")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "dedup_local_dev")

    # Threading
    THREAD_WORKERS = int(os.getenv("THREAD_WORKERS", "4"))
    RETRY_LIMIT = int(os.getenv("RETRY_LIMIT", "2"))

    # Stages to run (comma-separated: 0=ingest, 1=enrich, 2=deduplicate, 3=export)
    RUN_STAGES = [
        int(s.strip())
        for s in os.getenv("RUN_STAGES", "0,1,2,3").split(",")
    ]

    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    @classmethod
    def validate(cls):
        """Validate configuration on startup."""
        if not os.path.exists(cls.HDD_SOURCE_PATH):
            raise ValueError(f"HDD_SOURCE_PATH does not exist: {cls.HDD_SOURCE_PATH}")
        if not os.path.exists(os.path.dirname(cls.SHA256SUMS_PATH)):
            raise ValueError(f"SHA256SUMS parent dir does not exist: {cls.SHA256SUMS_PATH}")

    @classmethod
    def setup_staging_paths(cls):
        """Create required staging directories."""
        os.makedirs(cls.SSD_STAGING_PATH, exist_ok=True)
        os.makedirs(cls.SSD_ORIGINALS_PATH, exist_ok=True)
