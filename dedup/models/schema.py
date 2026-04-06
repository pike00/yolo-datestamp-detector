from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Index,
    BigInteger,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class SourceFile(Base):
    """All discovered source files with their content hash."""
    __tablename__ = "source_files"

    path = Column(String(1024), primary_key=True, nullable=False)
    sha256 = Column(String(64), nullable=False, index=True)
    size = Column(BigInteger, nullable=False)
    source_folder = Column(String(256), nullable=False, index=True)
    filename = Column(String(256), nullable=False)
    extension = Column(String(32), nullable=False, index=True)
    ingested_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_source_files_sha256_folder", "sha256", "source_folder"),
    )


class UniqueFile(Base):
    """Deduplicated files: one row per unique SHA-256 hash."""
    __tablename__ = "unique_files"

    sha256 = Column(String(64), primary_key=True, nullable=False, index=True)
    canonical_path = Column(String(1024), nullable=False)
    selection_reason = Column(String(64), nullable=False)  # "exif", "folder_priority", "shortest_path"
    duplicate_count = Column(Integer, nullable=False, default=0)

    # EXIF metadata
    exif_score = Column(Float, nullable=False, default=0.0)
    exif_datetime = Column(DateTime, nullable=True)
    exif_gps = Column(String(128), nullable=True)
    exif_fields_count = Column(Integer, nullable=False, default=0)
    exif_data = Column(JSONB, nullable=True)  # Full exiftool output
    camera_make = Column(String(128), nullable=True)
    camera_model = Column(String(128), nullable=True)
    image_width = Column(Integer, nullable=True)
    image_height = Column(Integer, nullable=True)
    mime_type = Column(String(64), nullable=True)

    # Export/verification tracking
    export_status = Column(String(32), nullable=False, default="pending")  # pending/copied/verified/error
    export_path = Column(String(1024), nullable=True)  # path in originals/
    verified_hash = Column(String(64), nullable=True)
    error_msg = Column(String(512), nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_unique_files_export_status", "export_status"),
    )


class RotationPrediction(Base):
    """Rotation predictions for images from orientation detection model."""
    __tablename__ = "rotation_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_path = Column(String(1024), nullable=False)
    filename = Column(String(256), nullable=False, index=True)
    sha256 = Column(String(64), nullable=True, index=True)  # links to unique_files if available
    model_name = Column(String(128), nullable=False)
    predicted_class = Column(Integer, nullable=False)  # 0, 1, 2, 3
    rotation_needed = Column(Integer, nullable=False)  # 0, 90, 180, 270 degrees CW
    confidence = Column(Float, nullable=False)
    # Store all class probabilities for downstream analysis
    prob_0 = Column(Float, nullable=False)  # P(upright)
    prob_90 = Column(Float, nullable=False)  # P(needs 90 CW)
    prob_180 = Column(Float, nullable=False)  # P(needs 180)
    prob_270 = Column(Float, nullable=False)  # P(needs 270 CW)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_rotation_file_sha", "sha256", "model_name"),
    )


class StampRotation(Base):
    """User-confirmed rotation for scanned photos during stamp review."""
    __tablename__ = "stamp_rotations"

    stem = Column(String(256), primary_key=True, nullable=False)
    rotation = Column(Integer, nullable=False, default=0)  # 0, 90, 180, 270 degrees CW
    source = Column(String(32), nullable=False, default="user")  # "predicted", "user"
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
