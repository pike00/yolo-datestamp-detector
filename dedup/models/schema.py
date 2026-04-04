from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    JSON,
    Index,
)
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class FilePath(Base):
    """All file paths from SHA256SUMS.txt, indexed by hash."""
    __tablename__ = "file_paths"

    hash = Column(String(64), primary_key=True, nullable=False)
    path = Column(String(1024), nullable=False, unique=True)
    size = Column(Integer, nullable=False)
    source_folder = Column(String(256), nullable=False, index=True)
    filename = Column(String(256), nullable=False)
    extension = Column(String(32), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_filePath_hash_source", "hash", "source_folder"),
    )


class File(Base):
    """EXIF-enriched metadata per unique hash."""
    __tablename__ = "files"

    hash = Column(String(64), primary_key=True, nullable=False, index=True)
    canonical_path = Column(String(1024), nullable=False)
    exif_score = Column(Float, nullable=False, default=0.0)
    exif_datetime = Column(DateTime, nullable=True)
    exif_gps = Column(String(128), nullable=True)
    exif_fields_count = Column(Integer, nullable=False, default=0)
    folder_source = Column(String(256), nullable=False)
    selected_reason = Column(String(128), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_file_hash", "hash"),
        Index("idx_file_folder_source", "folder_source"),
    )


class Canonical(Base):
    """Canonical file selections, one per unique hash."""
    __tablename__ = "canonicals"

    hash = Column(String(64), primary_key=True, nullable=False, index=True)
    canonical_path = Column(String(1024), nullable=False)
    duplicate_count = Column(Integer, nullable=False, default=0)
    total_size_saved_bytes = Column(Integer, nullable=False, default=0)
    verified = Column(Boolean, nullable=False, default=False)
    verification_hash = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_canonical_hash", "hash"),
        Index("idx_canonical_verified", "verified"),
    )


class CopyProgress(Base):
    """Track copy status for each canonical file (Stage 4)."""
    __tablename__ = "copy_progress"

    hash = Column(String(64), primary_key=True, nullable=False, index=True)
    status = Column(String(32), nullable=False, default="pending")
    copied_path = Column(String(1024), nullable=True)
    bytes_copied = Column(Integer, nullable=False, default=0)
    error_msg = Column(String(512), nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_copyProgress_status", "status"),
        Index("idx_copyProgress_hash", "hash"),
    )


class StagingProgress(Base):
    """Track staging copy status (Stage 0)."""
    __tablename__ = "staging_progress"

    source_path = Column(String(1024), primary_key=True, nullable=False)
    staging_path = Column(String(1024), nullable=False)
    status = Column(String(32), nullable=False, default="pending")
    bytes_copied = Column(Integer, nullable=False, default=0)
    error_msg = Column(String(512), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_stagingProgress_status", "status"),
    )


class FinalReport(Base):
    """Summary statistics after pipeline completion."""
    __tablename__ = "final_report"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stage_completed = Column(String(32), nullable=False)
    total_files_analyzed = Column(Integer, nullable=False, default=0)
    unique_files = Column(Integer, nullable=False, default=0)
    duplicate_files_removed = Column(Integer, nullable=False, default=0)
    total_space_saved_gb = Column(Float, nullable=False, default=0.0)
    files_by_source = Column(JSON, nullable=True)
    duration_seconds = Column(Integer, nullable=False, default=0)
    errors_encountered = Column(Integer, nullable=False, default=0)
    verified_copies = Column(Integer, nullable=False, default=0)
    failed_verifications = Column(Integer, nullable=False, default=0)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
