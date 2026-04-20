# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Extract QuickTime/ISOBMFF creation_time from video files via ffprobe.

Writes a JSON mapping {filename: {"date": [Y,M,D,h,m,s], "source": "ffprobe:format_tags.creation_time", "raw": iso_string}}
Filenames are basenames (sha256.ext) matching the organization_log.json key shape.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

VIDEO_EXTS = {".mov", ".mp4", ".3gp", ".mpg", ".m4v", ".avi"}
FIELDS = [
    "format_tags=creation_time",
    "stream_tags=creation_time",
]


def probe(path: Path) -> dict | None:
    """Return parsed metadata for one file or None if no date found."""
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_entries", "format_tags=creation_time:stream_tags=creation_time",
        str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        return {"name": path.name, "error": str(e)}

    data = json.loads(out.stdout or "{}")
    fmt_time = (data.get("format", {}).get("tags") or {}).get("creation_time")
    stream_time = None
    for s in data.get("streams", []) or []:
        t = (s.get("tags") or {}).get("creation_time")
        if t:
            stream_time = t
            break
    raw = fmt_time or stream_time
    if not raw:
        return {"name": path.name, "date": None, "raw": None}

    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    dt_utc = dt.astimezone(timezone.utc)
    return {
        "name": path.name,
        "date": [dt_utc.year, dt_utc.month, dt_utc.day, dt_utc.hour, dt_utc.minute, dt_utc.second],
        "raw": raw,
        "source": "ffprobe:format_tags.creation_time" if fmt_time else "ffprobe:stream_tags.creation_time",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="directory of video files")
    ap.add_argument("-o", "--out", type=Path, required=True, help="output JSON path")
    ap.add_argument("-j", "--jobs", type=int, default=8)
    args = ap.parse_args()

    files = sorted(p for p in args.src.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    print(f"probing {len(files)} video files with {args.jobs} workers...", file=sys.stderr)

    results: dict[str, dict] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(probe, p): p for p in files}
        for fut in as_completed(futures):
            r = fut.result()
            if r is not None:
                results[r["name"]] = r
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(files)}", file=sys.stderr)

    with_date = sum(1 for r in results.values() if r.get("date"))
    errors = sum(1 for r in results.values() if "error" in r)
    print(f"total: {len(results)}  with_date: {with_date}  errors: {errors}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
