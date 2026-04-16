import logging
from pathlib import Path
from typing import Tuple
from utils.db import get_session, close_session
from models.schema import FilePath

logger = logging.getLogger(__name__)


def parse_sha256sums_line(line: str) -> Tuple[str, str]:
    """
    Parse a line from SHA256SUMS.txt.
    Format: "hash  path"
    Returns: (hash, path)
    """
    parts = line.strip().split(None, 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid SHA256SUMS line: {line}")
    return parts[0], parts[1]


def load_sha256sums(manifest_path: str, db_path: str) -> None:
    """
    Load SHA256SUMS.txt into file_paths table.
    Resumable: skips hashes already in DB.
    """
    logger.info(f"Stage 1: Loading {manifest_path}")

    session = get_session(db_path)

    # Get existing hashes to skip
    existing_hashes = set(row[0] for row in session.query(FilePath.hash).all())
    logger.info(f"Found {len(existing_hashes)} existing entries in file_paths")

    # Read manifest and insert new entries
    new_entries = []
    skipped = 0

    with open(manifest_path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                hash_val, path = parse_sha256sums_line(line)

                if hash_val in existing_hashes:
                    skipped += 1
                    continue

                # Extract metadata from path
                path_obj = Path(path)
                filename = path_obj.name
                extension = path_obj.suffix.lstrip(".").lower()

                # Infer source folder (first component in path)
                source_folder = path_obj.parts[1] if len(path_obj.parts) > 1 else "unknown"

                # Get file size (stat from staging if available, else 0)
                size = 0

                new_entry = FilePath(
                    hash=hash_val,
                    path=path,
                    size=size,
                    source_folder=source_folder,
                    filename=filename,
                    extension=extension,
                )
                new_entries.append(new_entry)

                if len(new_entries) >= 1000:
                    session.bulk_save_objects(new_entries)
                    session.commit()
                    logger.debug(f"Batch inserted {len(new_entries)} entries")
                    new_entries = []

            except Exception as e:
                logger.warning(f"Line {line_num}: {e}")

    # Insert remaining entries
    if new_entries:
        session.bulk_save_objects(new_entries)
        session.commit()

    total_entries = session.query(FilePath).count()
    unique_hashes = session.query(FilePath.hash).distinct().count()
    close_session(session)

    logger.info(
        f"Stage 1 complete: {total_entries} total entries, "
        f"{unique_hashes} unique hashes, {skipped} skipped"
    )
