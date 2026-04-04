import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from models.schema import Base, FilePath, File, Canonical, CopyProgress, StagingProgress, FinalReport


@pytest.fixture
def db_session():
    """In-memory SQLite DB for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_filePath_creation(db_session):
    """Test FilePath table creation and insertion."""
    fp = FilePath(
        hash="abc123def456",
        path="/mnt/hdd/Photos/img/Desktop/photo.jpg",
        size=1024000,
        source_folder="Desktop",
        filename="photo.jpg",
        extension="jpg",
    )
    db_session.add(fp)
    db_session.commit()

    result = db_session.query(FilePath).filter_by(hash="abc123def456").first()
    assert result.filename == "photo.jpg"
    assert result.source_folder == "Desktop"
    assert result.size == 1024000


def test_file_creation(db_session):
    """Test File table creation and EXIF metadata storage."""
    f = File(
        hash="abc123def456",
        canonical_path="/mnt/staging/Desktop/photo.jpg",
        exif_score=0.85,
        exif_datetime=datetime(2020, 1, 15),
        exif_gps="37.7749,-122.4194",
        exif_fields_count=18,
        folder_source="Desktop",
        selected_reason="best_exif",
    )
    db_session.add(f)
    db_session.commit()

    result = db_session.query(File).filter_by(hash="abc123def456").first()
    assert result.exif_score == 0.85
    assert result.exif_fields_count == 18


def test_canonical_creation(db_session):
    """Test Canonical table for dedup results."""
    c = Canonical(
        hash="abc123def456",
        canonical_path="/mnt/staging/Desktop/photo.jpg",
        duplicate_count=3,
        total_size_saved_bytes=3072000,
        verified=False,
    )
    db_session.add(c)
    db_session.commit()

    result = db_session.query(Canonical).filter_by(hash="abc123def456").first()
    assert result.duplicate_count == 3
    assert result.verified is False


def test_copyProgress_creation(db_session):
    """Test CopyProgress table for tracking Stage 4."""
    cp = CopyProgress(
        hash="abc123def456",
        status="pending",
        retry_count=0,
    )
    db_session.add(cp)
    db_session.commit()

    result = db_session.query(CopyProgress).filter_by(hash="abc123def456").first()
    assert result.status == "pending"
    assert result.retry_count == 0


def test_stagingProgress_creation(db_session):
    """Test StagingProgress table for tracking Stage 0."""
    sp = StagingProgress(
        source_path="/mnt/hdd/Photos/img/Desktop/photo.jpg",
        staging_path="/home/will/photo_project/staging/Desktop/photo.jpg",
        status="done",
        bytes_copied=1024000,
    )
    db_session.add(sp)
    db_session.commit()

    result = db_session.query(StagingProgress).filter_by(
        source_path="/mnt/hdd/Photos/img/Desktop/photo.jpg"
    ).first()
    assert result.status == "done"


def test_finalReport_creation(db_session):
    """Test FinalReport table."""
    fr = FinalReport(
        stage_completed="5",
        total_files_analyzed=95519,
        unique_files=45838,
        duplicate_files_removed=49681,
        total_space_saved_gb=280.0,
        files_by_source={"Desktop": 41797, "iCloudPhotos": 9373},
        duration_seconds=28800,
        errors_encountered=0,
        verified_copies=45838,
        failed_verifications=0,
    )
    db_session.add(fr)
    db_session.commit()

    result = db_session.query(FinalReport).first()
    assert result.unique_files == 45838
    assert result.total_space_saved_gb == 280.0
