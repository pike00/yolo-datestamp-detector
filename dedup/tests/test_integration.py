import pytest
import tempfile
import os
from pathlib import Path
from PIL import Image
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from models.schema import Base, FilePath, File, Canonical, CopyProgress
from stages import (
    load_sha256sums,
    enrich_with_exif,
    select_canonicals,
    copy_to_originals,
    verify_copies,
)


@pytest.fixture
def integration_env():
    """Set up temp directories and database for integration tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        staging_dir = os.path.join(tmpdir, "staging")
        originals_dir = os.path.join(tmpdir, "originals")
        db_path = os.path.join(tmpdir, "test.duckdb")
        manifest_path = os.path.join(tmpdir, "SHA256SUMS.txt")

        os.makedirs(staging_dir)
        os.makedirs(originals_dir)

        # Create database with required tables
        engine = create_engine(f"duckdb:///{db_path}", echo=False)
        # Create tables manually to avoid DuckDB SERIAL issue for FinalReport
        with engine.begin() as conn:
            # Create other tables
            for table in Base.metadata.tables.values():
                if table.name != "final_report":
                    table.create(conn, checkfirst=True)
            # Create final_report without autoincrement (workaround for DuckDB issues)
            conn.execute(text("""
            CREATE SEQUENCE IF NOT EXISTS final_report_id_seq START 1
            """))
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS final_report (
                id INTEGER PRIMARY KEY DEFAULT nextval('final_report_id_seq'),
                stage_completed VARCHAR(32) NOT NULL,
                total_files_analyzed INTEGER NOT NULL DEFAULT 0,
                unique_files INTEGER NOT NULL DEFAULT 0,
                duplicate_files_removed INTEGER NOT NULL DEFAULT 0,
                total_space_saved_gb FLOAT NOT NULL DEFAULT 0.0,
                files_by_source JSON,
                duration_seconds INTEGER NOT NULL DEFAULT 0,
                errors_encountered INTEGER NOT NULL DEFAULT 0,
                verified_copies INTEGER NOT NULL DEFAULT 0,
                failed_verifications INTEGER NOT NULL DEFAULT 0,
                timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """))

        yield {
            "staging_dir": staging_dir,
            "originals_dir": originals_dir,
            "db_path": db_path,
            "manifest_path": manifest_path,
            "tmpdir": tmpdir,
        }


def test_full_pipeline_smoke(integration_env):
    """Smoke test: run all stages on minimal test data."""
    env = integration_env

    # Create test files and manifest
    test_dir = os.path.join(env["staging_dir"], "Desktop")
    os.makedirs(test_dir)

    # Create a single test JPEG
    import hashlib
    img_path = os.path.join(test_dir, f"photo_0.jpg")
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_path, "JPEG")

    # Compute hash of the file
    sha256_hash = hashlib.sha256()
    with open(img_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    actual_hash = sha256_hash.hexdigest()

    # Create manifest with one file entry
    with open(env["manifest_path"], "w") as f:
        f.write(f"{actual_hash}  img/Desktop/photo_0.jpg\n")

    # Run stages
    load_sha256sums(env["manifest_path"], env["db_path"])
    enrich_with_exif(env["staging_dir"], env["db_path"], thread_workers=1)
    select_canonicals(env["db_path"])
    copy_to_originals(
        env["staging_dir"],
        env["originals_dir"],
        env["db_path"],
        thread_workers=1,
    )
    verify_copies(env["originals_dir"], env["db_path"], thread_workers=1)

    # Verify results
    engine = create_engine(f"duckdb:///{env['db_path']}", echo=False)
    Session = sessionmaker(bind=engine)
    session = Session()
    # One FilePath
    assert session.query(FilePath).count() == 1
    # One unique file
    assert session.query(File).count() == 1
    # One canonical
    assert session.query(Canonical).count() == 1
    # Canonical should be verified
    assert session.query(Canonical).filter_by(verified=True).count() == 1
    session.close()

    # Verify file exists in originals
    files_in_originals = os.listdir(env["originals_dir"])
    assert len(files_in_originals) == 1
    assert files_in_originals[0].startswith(actual_hash[:16])
