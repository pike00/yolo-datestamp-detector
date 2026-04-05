import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from models.schema import Base, SourceFile, UniqueFile


@pytest.fixture
def db_session():
    """In-memory SQLite DB for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_source_file_creation(db_session):
    """Test SourceFile table creation and insertion."""
    sf = SourceFile(
        path="Desktop/photo.jpg",
        sha256="abc123def456",
        size=1024000,
        source_folder="Desktop",
        filename="photo.jpg",
        extension="jpg",
    )
    db_session.add(sf)
    db_session.commit()

    result = db_session.query(SourceFile).filter_by(sha256="abc123def456").first()
    assert result.filename == "photo.jpg"
    assert result.source_folder == "Desktop"
    assert result.size == 1024000


def test_unique_file_creation(db_session):
    """Test UniqueFile table creation and EXIF metadata storage."""
    uf = UniqueFile(
        sha256="abc123def456",
        canonical_path="Desktop/photo.jpg",
        selection_reason="folder_priority",
        exif_score=0.85,
        exif_datetime=datetime(2020, 1, 15),
        exif_gps="37.7749,-122.4194",
        exif_fields_count=18,
        duplicate_count=2,
        export_status="pending",
    )
    db_session.add(uf)
    db_session.commit()

    result = db_session.query(UniqueFile).filter_by(sha256="abc123def456").first()
    assert result.exif_score == 0.85
    assert result.exif_fields_count == 18
    assert result.duplicate_count == 2
    assert result.export_status == "pending"


def test_unique_file_export_tracking(db_session):
    """Test UniqueFile export status tracking."""
    uf = UniqueFile(
        sha256="def456ghi789",
        canonical_path="Photos/image.jpg",
        selection_reason="shortest_path",
        export_status="pending",
        retry_count=0,
    )
    db_session.add(uf)
    db_session.commit()

    result = db_session.query(UniqueFile).filter_by(sha256="def456ghi789").first()
    assert result.export_status == "pending"
    assert result.retry_count == 0

    # Simulate export
    result.export_status = "copied"
    result.export_path = "/originals/def456ghi789.jpg"
    db_session.commit()

    updated = db_session.query(UniqueFile).filter_by(sha256="def456ghi789").first()
    assert updated.export_status == "copied"
    assert updated.export_path == "/originals/def456ghi789.jpg"


def test_unique_file_verification(db_session):
    """Test UniqueFile verification tracking."""
    uf = UniqueFile(
        sha256="xyz123abc456",
        canonical_path="iCloudPhotos/sunset.jpg",
        selection_reason="exif",
        export_status="copied",
        export_path="/originals/xyz123abc456.jpg",
        verified_hash=None,
    )
    db_session.add(uf)
    db_session.commit()

    # Simulate verification
    result = db_session.query(UniqueFile).filter_by(sha256="xyz123abc456").first()
    result.export_status = "verified"
    result.verified_hash = "xyz123abc456"
    db_session.commit()

    verified = db_session.query(UniqueFile).filter_by(sha256="xyz123abc456").first()
    assert verified.export_status == "verified"
    assert verified.verified_hash == "xyz123abc456"
