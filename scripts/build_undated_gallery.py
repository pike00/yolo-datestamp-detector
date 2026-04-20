# /// script
# requires-python = ">=3.12"
# dependencies = ["pillow>=10", "pillow-heif>=1"]
# ///
"""Build a paginated HTML gallery of undated media.

Reads state/undated_media.json, generates 240px WebP thumbnails, writes
pages of 300 items each. Photos first (sorted), then videos.
"""
from __future__ import annotations

import html as _html
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

MEDIA_DIR = Path("/home/will/photo_project/originals/media")
UNDATED = Path("/home/will/photo_project/state/undated_media.json")
OUT_DIR = Path("/home/will/photo_project/output/undated_gallery")
THUMB_DIR = OUT_DIR / "thumbs"
PAGE_SIZE = 300
THUMB_WIDTH = 240


def thumb_photo(name: str) -> tuple[str, bool, str | None]:
    src = MEDIA_DIR / name
    dst = THUMB_DIR / f"{src.stem}.webp"
    if dst.exists():
        return name, True, None
    try:
        with Image.open(src) as img:
            img = img.convert("RGB")
            img.thumbnail((THUMB_WIDTH, THUMB_WIDTH * 3))
            img.save(dst, "WEBP", quality=78, method=4)
        return name, True, None
    except Exception as e:
        return name, False, f"{type(e).__name__}: {e}"


def thumb_video(name: str) -> tuple[str, bool, str | None]:
    src = MEDIA_DIR / name
    dst = THUMB_DIR / f"{src.stem}.jpg"
    if dst.exists():
        return name, True, None
    try:
        for ts in ("00:00:01", "00:00:00.5", "00:00:00"):
            res = subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-ss", ts, "-i", str(src),
                 "-frames:v", "1", "-vf", f"scale={THUMB_WIDTH}:-2",
                 "-q:v", "4", str(dst)],
                capture_output=True, timeout=30,
            )
            if res.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
                return name, True, None
        return name, False, "ffmpeg:no_frames"
    except subprocess.TimeoutExpired:
        return name, False, "timeout"
    except Exception as e:
        return name, False, f"{type(e).__name__}: {e}"


def render_page(
    page_num: int,
    total_pages: int,
    items: list[dict],
    title: str,
    total_photos: int,
    total_videos: int,
) -> str:
    cards = []
    for item in items:
        name = item["name"]
        stem = Path(name).stem
        ext = "jpg" if item["kind"] == "video" else "webp"
        thumb_rel = f"thumbs/{stem}.{ext}"
        full_rel = f"../../originals/media/{name}"
        kind = item["kind"]
        cards.append(
            f'<a class="card {kind}" href="{full_rel}" target="_blank" title="{_html.escape(name)}">'
            f'<img loading="lazy" src="{thumb_rel}" alt="">'
            f'<div class="label">{kind}</div></a>'
        )

    nav_links = []
    for n in range(1, total_pages + 1):
        label = str(n)
        cls = "current" if n == page_num else ""
        target = "index.html" if n == 1 else f"page{n}.html"
        nav_links.append(f'<a class="{cls}" href="{target}">{label}</a>')
    nav = f'<nav>{" ".join(nav_links)}</nav>'

    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>{_html.escape(title)} — page {page_num}/{total_pages}</title>
<style>
body {{ font-family: system-ui; background:#111; color:#ddd; margin:0; padding:16px; }}
h1 {{ margin:0 0 8px; font-size:18px; }}
.meta {{ color:#888; font-size:13px; margin-bottom:12px; }}
nav {{ margin:12px 0; display:flex; flex-wrap:wrap; gap:4px; }}
nav a {{ padding:4px 10px; background:#222; color:#aaa; text-decoration:none; border-radius:4px; font-size:13px; }}
nav a.current {{ background:#4a9; color:#000; font-weight:bold; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(200px,1fr)); gap:8px; }}
.card {{ position:relative; display:block; aspect-ratio:1; overflow:hidden; border-radius:4px; background:#222; }}
.card img {{ width:100%; height:100%; object-fit:cover; display:block; }}
.card .label {{ position:absolute; top:4px; left:4px; background:rgba(0,0,0,.7); padding:2px 6px; font-size:10px; border-radius:3px; text-transform:uppercase; }}
.card.video .label {{ background:rgba(180,50,50,.85); }}
.card.photo .label {{ background:rgba(50,120,180,.85); }}
.card:hover {{ outline:2px solid #4a9; }}
</style></head>
<body>
<h1>{_html.escape(title)}</h1>
<div class="meta">{total_photos:,} undated photos + {total_videos:,} undated videos &middot; page {page_num}/{total_pages}</div>
{nav}
<div class="grid">{"".join(cards)}</div>
{nav}
</body></html>"""


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_DIR.mkdir(exist_ok=True)

    data = json.loads(UNDATED.read_text())
    photos = data["undated_photos"]
    videos = data["undated_videos"]
    errors = data.get("photo_errors", [])

    total_photos = len(photos) + len(errors)
    total_videos = len(videos)

    # Build thumbnails
    photo_names = [p["name"] for p in photos] + [e["name"] for e in errors]
    video_names = [v["name"] for v in videos]

    print(f"Generating {len(photo_names)} photo thumbs...", file=sys.stderr)
    ok_photos, fail_photos = 0, []
    with ProcessPoolExecutor(max_workers=12) as pool:
        for i, (name, ok, err) in enumerate(pool.map(thumb_photo, photo_names, chunksize=32)):
            if ok:
                ok_photos += 1
            else:
                fail_photos.append((name, err))
            if (i + 1) % 2000 == 0:
                print(f"  photos: {i+1}/{len(photo_names)}", file=sys.stderr)

    print(f"Generating {len(video_names)} video thumbs...", file=sys.stderr)
    ok_videos, fail_videos = 0, []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i, (name, ok, err) in enumerate(pool.map(thumb_video, video_names, chunksize=4)):
            if ok:
                ok_videos += 1
            else:
                fail_videos.append((name, err))
            if (i + 1) % 200 == 0:
                print(f"  videos: {i+1}/{len(video_names)}", file=sys.stderr)

    print(f"thumbs: {ok_photos} photos ok, {len(fail_photos)} failed; {ok_videos} videos ok, {len(fail_videos)} failed", file=sys.stderr)

    # Assemble items: only ones with a thumbnail
    items: list[dict] = []
    for name in photo_names:
        if (THUMB_DIR / f"{Path(name).stem}.webp").exists():
            items.append({"name": name, "kind": "photo"})
    for name in video_names:
        if (THUMB_DIR / f"{Path(name).stem}.jpg").exists():
            items.append({"name": name, "kind": "video"})

    items.sort(key=lambda x: (x["kind"] != "photo", x["name"]))

    total_pages = max(1, (len(items) + PAGE_SIZE - 1) // PAGE_SIZE)
    title = "Undated media"
    for page_idx in range(total_pages):
        chunk = items[page_idx * PAGE_SIZE : (page_idx + 1) * PAGE_SIZE]
        page_num = page_idx + 1
        html_text = render_page(page_num, total_pages, chunk, title, total_photos, total_videos)
        out = OUT_DIR / ("index.html" if page_num == 1 else f"page{page_num}.html")
        out.write_text(html_text)

    # Write failures report
    report = OUT_DIR / "failures.json"
    report.write_text(json.dumps({
        "photo_failures": [{"name": n, "error": e} for n, e in fail_photos],
        "video_failures": [{"name": n, "error": e} for n, e in fail_videos],
    }, indent=2))

    print(f"wrote {total_pages} pages to {OUT_DIR}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
