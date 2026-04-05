import pytest
import tempfile
import os
from pathlib import Path
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch
from models.schema import Base, SourceFile, UniqueFile
from pipeline import ingest, enrich, deduplicate, export_canonicals


@pytest.fixture
def integration_env():
    """Set up temp directories and in-memory SQLite DB for integration tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source_dir = os.path.join(tmpdir, "source")
        staging_dir = os.path.join(tmpdir, "staging")
        originals_dir = os.path.join(tmpdir, "originals")

        os.makedirs(source_dir)
        os.makedirs(staging_dir)
        os.makedirs(originals_dir)

        # Create in-memory SQLite database for testing
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)

        yield {
            "source_dir": source_dir,
            "staging_dir": staging_dir,
            "originals_dir": originals_dir,
            "engine": engine,
            "tmpdir": tmpdir,
        }


def test_full_pipeline_smoke(integration_env):
    """Smoke test: run all stages on minimal test data."""
    env = integration_env

    # Create test source files
    test_dir = os.path.join(env["source_dir"], "Desktop")
    os.makedirs(test_dir)

    # Create a single test JPEG
    import hashlib
    img_path = os.path.join(test_dir, "photo_0.jpg")
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_path, "JPEG")

    # Compute hash
    sha256_hash = hashlib.sha256()
    with open(img_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    actual_hash = sha256_hash.hexdigest()

    # Mock the database to use our test engine
    with patch("utils.db._engine", env["engine"]):
        with patch("utils.db._SessionLocal", sessionmaker(bind=env["engine"], expire_on_commit=False)):
            # Stage 0: Ingest (copy + hash)
            ingest(
                source_dir=env["source_dir"],
                staging_dir=env["staging_dir"],
                thread_workers=1,
            )

            # Verify ingest
            Session = sessionmaker(bind=env["engine"])
            session = Session()
            assert session.query(SourceFile).count() == 1
            source_file = session.query(SourceFile).first()
            assert source_file.sha256 == actual_hash
            assert source_file.filename == "photo_0.jpg"
            session.close()

            # Stage 1: Enrich (EXIF)
            enrich(
                staging_dir=env["staging_dir"],
                thread_workers=1,
            )

            # Verify enrich
            session = Session()
            assert session.query(UniqueFile).count() == 1
            unique_file = session.query(UniqueFile).first()
            assert unique_file.sha256 == actual_hash
            assert unique_file.selection_reason == "preliminary"
            session.close()

            # Stage 2: Deduplicate (canonical selection)
            deduplicate()

            # Verify dedup
            session = Session()
            unique_file = session.query(UniqueFile).filter_by(sha256=actual_hash).first()
            assert unique_file.selection_reason != "preliminary"
            assert unique_file.canonical_path is not None
            session.close()

            # Stage 3: Export (copy + verify)
            export_canonicals(
                staging_dir=env["staging_dir"],
                originals_dir=env["originals_dir"],
                thread_workers=1,
            )

            # Verify export
            session = Session()
            unique_file = session.query(UniqueFile).filter_by(sha256=actual_hash).first()
            assert unique_file.export_status == "verified"
            assert unique_file.verified_hash == actual_hash
            session.close()

            # Verify file exists in originals
            files_in_originals = os.listdir(env["originals_dir"])
            assert len(files_in_originals) == 1
            assert files_in_originals[0].startswith(actual_hash[:16])
