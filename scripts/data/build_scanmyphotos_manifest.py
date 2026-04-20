# /// script
# requires-python = ">=3.12"
# dependencies = ["psycopg[binary]>=3.1.0"]
# ///
"""Build the scanmyphotos_manifest table.

For each symlink in scanmyphotos/, resolve the HDD source file, compute
sha256 of file contents, and upsert (stem, disc, source_path, sha256,
size_bytes, mtime) into scanmyphotos_manifest.

Also verifies that originals/media/{sha256}.jpg exists for each row;
prints a warning for misses (means dedup pipeline didn't ingest that file).

Usage:
    python scripts/data/build_scanmyphotos_manifest.py [--force] [--limit N] [--workers 12]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import psycopg

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCANMYPHOTOS_DIR = BASE_DIR / "scanmyphotos"
ORIGINALS_DIR = BASE_DIR / "originals" / "media"
DB_CONN_STRING = os.environ.get(
    "DATABASE_URL", "postgresql://dedup:dedup_local_dev@127.0.0.1:5432/dedup"
)

STEM_RE = re.compile(r"^d(\d+)_(\d+)$")


def hash_one(symlink_path: Path) -> dict | None:
    """Return a manifest row for one symlink, or None if unreadable."""
    stem = symlink_path.stem
    m = STEM_RE.match(stem)
    if not m:
        return {"stem": stem, "error": f"stem doesn't match d<disc>_<num> pattern"}
    disc = int(m.group(1))

    try:
        real = symlink_path.resolve(strict=True)
    except (FileNotFoundError, OSError) as e:
        return {"stem": stem, "error": f"resolve: {e}"}

    try:
        st = real.stat()
    except OSError as e:
        return {"stem": stem, "error": f"stat: {e}"}

    try:
        with open(real, "rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
    except OSError as e:
        return {"stem": stem, "error": f"read: {e}"}

    return {
        "stem": stem,
        "disc": disc,
        "source_path": str(real),
        "sha256": digest,
        "size_bytes": st.st_size,
        "mtime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rehash even if stem already in manifest")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    symlinks = sorted(p for p in SCANMYPHOTOS_DIR.iterdir() if p.is_symlink() or p.is_file())
    if args.limit:
        symlinks = symlinks[: args.limit]
    print(f"found {len(symlinks)} entries in {SCANMYPHOTOS_DIR}", file=sys.stderr)

    # Skip already-hashed stems unless --force
    with psycopg.connect(DB_CONN_STRING) as conn:
        existing = {
            r[0] for r in conn.execute("SELECT stem FROM scanmyphotos_manifest").fetchall()
        }
    pending = symlinks if args.force else [p for p in symlinks if p.stem not in existing]
    print(
        f"{len(existing)} already hashed, {len(pending)} to hash this run",
        file=sys.stderr,
    )
    if not pending:
        print("nothing to do", file=sys.stderr)
        return 0

    rows: list[dict] = []
    errors: list[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(hash_one, p): p for p in pending}
        for fut in as_completed(futures):
            r = fut.result()
            if r is None:
                continue
            if "error" in r:
                errors.append(r)
            else:
                rows.append(r)
            done += 1
            if done % 500 == 0:
                print(f"  hashed {done}/{len(pending)}", file=sys.stderr)

    print(f"hashed: {len(rows)}  errors: {len(errors)}", file=sys.stderr)

    # Upsert into Postgres
    if rows:
        with psycopg.connect(DB_CONN_STRING) as conn, conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO scanmyphotos_manifest
                  (stem, disc, source_path, sha256, size_bytes, mtime)
                VALUES (%(stem)s, %(disc)s, %(source_path)s, %(sha256)s, %(size_bytes)s, %(mtime)s)
                ON CONFLICT (stem) DO UPDATE SET
                    disc        = EXCLUDED.disc,
                    source_path = EXCLUDED.source_path,
                    sha256      = EXCLUDED.sha256,
                    size_bytes  = EXCLUDED.size_bytes,
                    mtime       = EXCLUDED.mtime,
                    hashed_at   = now()
                """,
                rows,
            )
            conn.commit()
        print(f"upserted {len(rows)} rows to scanmyphotos_manifest", file=sys.stderr)

    # Cross-check: every sha256 should have a matching originals/media/{sha256}.{ext}
    missing_in_media = []
    for r in rows:
        # Check common image extensions
        for ext in ("jpg", "jpeg", "png", "heic", "bmp", "tif", "tiff"):
            if (ORIGINALS_DIR / f"{r['sha256']}.{ext}").exists():
                break
        else:
            missing_in_media.append(r["stem"])
    if missing_in_media:
        print(
            f"WARN: {len(missing_in_media)} stems not found in {ORIGINALS_DIR} "
            f"(first 5: {missing_in_media[:5]})",
            file=sys.stderr,
        )
    else:
        print(f"cross-check passed: all {len(rows)} sha256s present in {ORIGINALS_DIR}", file=sys.stderr)

    if errors:
        print(f"\nERRORS (first 10):", file=sys.stderr)
        for e in errors[:10]:
            print(f"  {e['stem']}: {e['error']}", file=sys.stderr)

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
