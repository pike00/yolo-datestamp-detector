#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["psycopg2-binary"]
# ///
"""One-time setup: copy source image files to working directory.

Queries a PostgreSQL database for file metadata and copies deduplicated
originals into the scanmyphotos/ working directory with disc-prefixed names.

Configure via environment variables:
    ORIGINALS_DIR  Path to deduplicated originals (default: ../originals)
    DB_HOST        PostgreSQL host (default: localhost)
    DB_PORT        PostgreSQL port (default: 5432)
    DB_NAME        Database name (default: dedup)
    DB_USER        Database user (default: dedup)
    DB_PASSWORD    Database password (default: dedup_local_dev)
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
ORIGINALS_DIR = Path(os.environ.get("ORIGINALS_DIR", str(BASE_DIR.parent / "originals")))
WORKING_DIR = BASE_DIR / "scanmyphotos"
MANIFEST_FILE = BASE_DIR / "state" / "scanmyphotos_manifest.json"

# PostgreSQL connection (configurable via env vars)
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", "5432")),
    "dbname": os.environ.get("DB_NAME", "dedup"),
    "user": os.environ.get("DB_USER", "dedup"),
    "password": os.environ.get("DB_PASSWORD", "dedup_local_dev"),
}


def get_pg_port():
    """Get the mapped port for dedup-postgres container."""
    result = subprocess.run(
        ["docker", "port", "dedup-postgres", "5432"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        # Output like "0.0.0.0:5432\n" or "0.0.0.0:32768\n"
        return int(result.stdout.strip().split(":")[-1])
    return 5432


def query_scanmyphotos_mapping():
    """Query PostgreSQL for ScanMyPhotos file mapping."""
    import psycopg2

    port = get_pg_port()
    print(f"Connecting to PostgreSQL on port {port}...")

    conn = psycopg2.connect(**{**DB_CONFIG, "port": port})
    cur = conn.cursor()

    cur.execute("""
        SELECT DISTINCT ON (sf.sha256)
            SUBSTRING(sf.path FROM 'Disc ([0-9]+)') as disc_num,
            sf.filename,
            sf.sha256
        FROM source_files sf
        WHERE sf.path LIKE '%%ScanMyPhotos/Disc%%'
        AND sf.filename LIKE '%%.jpg'
        ORDER BY sf.sha256, sf.path
    """)

    rows = cur.fetchall()
    cur.close()
    conn.close()

    mapping = []
    for disc_num, filename, sha256 in rows:
        stem = Path(filename).stem
        working_name = f"d{disc_num}_{stem}.jpg"
        mapping.append({
            "disc": int(disc_num),
            "filename": filename,
            "sha256": sha256,
            "working_name": working_name,
        })

    return mapping


def copy_files(mapping):
    """Copy files from originals/ to scanmyphotos/ working directory."""
    WORKING_DIR.mkdir(exist_ok=True)

    existing = {f.name for f in WORKING_DIR.glob("*.jpg")}
    to_copy = [m for m in mapping if m["working_name"] not in existing]

    if not to_copy:
        print(f"All {len(mapping)} files already copied.")
        return

    print(f"Copying {len(to_copy)} files ({len(existing)} already exist)...")

    copied = 0
    missing = 0
    for i, entry in enumerate(to_copy):
        src = ORIGINALS_DIR / f"{entry['sha256']}.jpg"
        dst = WORKING_DIR / entry["working_name"]

        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing += 1
            if missing <= 5:
                print(f"  WARNING: missing {src.name}")

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(to_copy)} copied...")

    print(f"Done: {copied} copied, {missing} missing, {len(existing)} already existed.")


def main():
    print("=== ScanMyPhotos Setup ===\n")

    # Step 1: Query mapping
    print("Step 1: Querying PostgreSQL for ScanMyPhotos files...")
    mapping = query_scanmyphotos_mapping()
    print(f"  Found {len(mapping)} unique files across discs:")
    disc_counts = {}
    for m in mapping:
        disc_counts[m["disc"]] = disc_counts.get(m["disc"], 0) + 1
    for disc in sorted(disc_counts):
        print(f"    Disc {disc}: {disc_counts[disc]} files")

    # Step 2: Copy files
    print(f"\nStep 2: Copying to {WORKING_DIR}/...")
    copy_files(mapping)

    # Step 3: Save manifest
    print(f"\nStep 3: Saving manifest to {MANIFEST_FILE.name}...")
    with open(MANIFEST_FILE, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"  Manifest saved ({len(mapping)} entries).")

    print("\n=== Setup complete ===")


if __name__ == "__main__":
    main()
