#!/usr/bin/env bash
# Snapshot photo_project bench state to the homelab backup HDD.
#
# Usage:
#   bench_backup.sh [<label>]
#
# <label> is a free-form tag appended to the timestamp, e.g.:
#   bench_backup.sh post-yolo-refresh
#   bench_backup.sh post-ground-truth-freeze
#   bench_backup.sh post-bench-ares-kimi-vl
#
# Outputs to:
#   /mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/backups/photo_project_bench/<timestamp>_<label>/
#     ├── stamp_tables.sql.zst        # SQL dump of the three bench tables
#     └── state_bench/                # rsync of state/bench/
#
# Safe to run mid-session. Never deletes existing backups.

set -euo pipefail

LABEL="${1:-snapshot}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_ROOT="/mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/backups/photo_project_bench"
DEST="$BACKUP_ROOT/${TS}_${LABEL}"
DSN="${DATABASE_URL:-postgresql://dedup:dedup_local_dev@localhost:5432/dedup}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p "$DEST"

echo "==> Dumping stamp_* tables to $DEST/stamp_tables.sql.zst"
# pg_dump from the dockerized Postgres via docker exec is the most reliable
# path on ares. If the tool chain differs on this host, fall back to a
# Python-based COPY. We try pg_dump first.
if command -v pg_dump >/dev/null 2>&1; then
    pg_dump "$DSN" \
        --table=stamp_predictions \
        --table=stamp_ocr \
        --table=stamp_no_stamp \
        --no-owner --no-privileges \
        | zstd -T0 -19 -o "$DEST/stamp_tables.sql.zst"
else
    echo "pg_dump not on PATH; falling back to Python COPY"
    BENCH_DSN="$DSN" BENCH_DEST="$DEST/stamp_tables.sql.zst" \
    uv run --with zstandard --with 'psycopg[binary]>=3.1.0' python - <<'PY'
import io
import os
import psycopg
import zstandard as zstd_mod

dsn = os.environ["BENCH_DSN"]
dest = os.environ["BENCH_DEST"]
tables = ["stamp_predictions", "stamp_ocr", "stamp_no_stamp"]
buf = io.BytesIO()
with psycopg.connect(dsn) as conn:
    for t in tables:
        buf.write(f"-- TABLE {t}\n".encode())
        # Dump schema
        cols = conn.execute(
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_name=%s ORDER BY ordinal_position",
            (t,),
        ).fetchall()
        buf.write(f"-- columns: {cols}\n".encode())
        # Dump data as COPY-compatible text
        buf.write(f"COPY {t} FROM stdin;\n".encode())
        with conn.cursor() as cur, cur.copy(f"COPY {t} TO STDOUT") as copy:
            for chunk in copy:
                buf.write(bytes(chunk))
        buf.write(b"\\." + b"\n\n")
cctx = zstd_mod.ZstdCompressor(level=19)
with open(dest, "wb") as f:
    f.write(cctx.compress(buf.getvalue()))
print(f"wrote {dest}")
PY
fi

echo "==> Rsyncing state/bench/ to $DEST/state_bench/"
if [ -d state/bench ]; then
    rsync -a --info=stats2 state/bench/ "$DEST/state_bench/"
else
    echo "  (state/bench/ does not exist yet -- skipping)"
fi

echo
echo "==> Backup complete."
echo "    $DEST"
du -sh "$DEST"/* 2>/dev/null || true
