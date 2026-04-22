# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "psycopg[binary]>=3.1.0",
#     "pillow>=10.0",
# ]
# ///
"""Render a static HTML dashboard of current stamp_predictions bboxes.

For each prediction row:
  - full-photo thumbnail (320 px wide) with the bbox drawn on top
  - zoomed crop of the bbox region (256 px wide) with 40% padding

Output:
  output/bbox_dashboard/
    index.html
    thumbs/<stem>_full.jpg
    thumbs/<stem>_crop.jpg

Run:
  .venv/bin/python scripts/annotate/render_bbox_dashboard.py

Serve (binds 0.0.0.0 so it's reachable on the tailnet):
  cd output/bbox_dashboard && python3 -m http.server --bind 0.0.0.0 8890
  open http://ares.savannah-mimosa.ts.net:8890/
"""

from __future__ import annotations

import html
import json
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from _db import get_db  # noqa: E402

SCANMYPHOTOS_DIR = BASE_DIR / "scanmyphotos"
OUT_DIR = BASE_DIR / "output" / "bbox_dashboard"
THUMBS_DIR = OUT_DIR / "thumbs"

FULL_WIDTH = 320
CROP_WIDTH = 256
CROP_PAD = 0.40


def fetch_rows() -> list[dict]:
    sql = """
        SELECT stem, x, y, w, h, confidence, model, updated_at
        FROM stamp_predictions
        ORDER BY confidence DESC
    """
    with get_db() as conn:
        rows = conn.execute(sql).fetchall()
    return [
        {
            "stem": r[0],
            "x": float(r[1]),
            "y": float(r[2]),
            "w": float(r[3]),
            "h": float(r[4]),
            "conf": float(r[5]),
            "model": r[6],
        }
        for r in rows
    ]


def find_image(stem: str) -> Path | None:
    p = SCANMYPHOTOS_DIR / f"{stem}.jpg"
    return p if p.exists() else None


def render_tile(row: dict, src: Path) -> bool:
    """Write full+crop thumbnails for this row. Returns True on success."""
    try:
        img = Image.open(src).convert("RGB")
    except Exception as e:
        print(f"  skip {row['stem']}: open failed ({e})")
        return False
    iw, ih = img.size

    cx, cy, bw, bh = row["x"], row["y"], row["w"], row["h"]
    x1 = (cx - bw / 2) * iw
    y1 = (cy - bh / 2) * ih
    x2 = (cx + bw / 2) * iw
    y2 = (cy + bh / 2) * ih

    full = img.copy()
    draw = ImageDraw.Draw(full)
    stroke = max(2, int(min(iw, ih) * 0.008))
    draw.rectangle([x1, y1, x2, y2], outline=(255, 80, 30), width=stroke)
    full.thumbnail((FULL_WIDTH, FULL_WIDTH * 4), Image.Resampling.LANCZOS)

    pad_x = bw * iw * CROP_PAD
    pad_y = bh * ih * CROP_PAD
    crop_box = (
        max(0, int(x1 - pad_x)),
        max(0, int(y1 - pad_y)),
        min(iw, int(x2 + pad_x)),
        min(ih, int(y2 + pad_y)),
    )
    crop = img.crop(crop_box)
    cw, ch = crop.size
    if cw > 0 and ch > 0:
        scale = CROP_WIDTH / max(cw, ch)
        crop = crop.resize(
            (max(1, int(cw * scale)), max(1, int(ch * scale))),
            Image.Resampling.LANCZOS,
        )
    else:
        crop = Image.new("RGB", (CROP_WIDTH, CROP_WIDTH // 2), (40, 40, 40))

    (THUMBS_DIR / f"{row['stem']}_full.jpg").parent.mkdir(parents=True, exist_ok=True)
    full.save(THUMBS_DIR / f"{row['stem']}_full.jpg", "JPEG", quality=78, optimize=True)
    crop.save(THUMBS_DIR / f"{row['stem']}_crop.jpg", "JPEG", quality=82, optimize=True)
    return True


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Stamp predictions — bbox dashboard</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; font-family: system-ui, sans-serif; background: #111; color: #ddd; }
  header { padding: 12px 18px; border-bottom: 1px solid #2a2a2a; display: flex; gap: 18px; align-items: center; flex-wrap: wrap; position: sticky; top: 0; background: #111; z-index: 10; }
  header h1 { font-size: 15px; margin: 0; font-weight: 600; color: #ffb27a; }
  header .meta { font-size: 12px; color: #888; }
  header label { font-size: 12px; color: #bbb; display: flex; align-items: center; gap: 6px; }
  header input[type=range] { width: 160px; }
  header select, header input[type=text] { background: #1a1a1a; color: #ddd; border: 1px solid #333; padding: 3px 6px; font-size: 12px; border-radius: 3px; }
  main { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 10px; padding: 10px; }
  .tile { background: #1a1a1a; border: 1px solid #262626; border-radius: 4px; overflow: hidden; display: flex; flex-direction: column; }
  .tile-imgs { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; padding: 4px; }
  .tile-imgs img { width: 100%; height: 160px; object-fit: contain; background: #000; border-radius: 2px; }
  .tile-meta { font-size: 11px; padding: 4px 8px 6px 8px; display: flex; justify-content: space-between; color: #aaa; }
  .tile-meta .stem { font-family: ui-monospace, monospace; color: #ddd; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .tile-meta .conf { flex-shrink: 0; margin-left: 8px; }
  .conf.hi { color: #7fdc7a; }
  .conf.mid { color: #f5c767; }
  .conf.lo { color: #e85c5c; }
</style>
</head>
<body>
<header>
  <h1>Stamp predictions — bbox dashboard</h1>
  <span class="meta" id="summary">__SUMMARY__</span>
  <label>sort
    <select id="sort">
      <option value="conf_desc">confidence ↓</option>
      <option value="conf_asc">confidence ↑</option>
      <option value="stem">stem</option>
    </select>
  </label>
  <label>min conf <span id="min_conf_val">0.00</span>
    <input id="min_conf" type="range" min="0" max="1" step="0.01" value="0">
  </label>
  <label>search
    <input id="search" type="text" placeholder="stem contains…">
  </label>
  <span class="meta" id="visible_count">—</span>
</header>
<main id="grid"></main>
<script id="rows_data" type="application/json">__ROWS_JSON__</script>
<script>
(() => {
  const ROWS = JSON.parse(document.getElementById('rows_data').textContent);
  const grid = document.getElementById('grid');
  const sortSel = document.getElementById('sort');
  const minConf = document.getElementById('min_conf');
  const minConfVal = document.getElementById('min_conf_val');
  const search = document.getElementById('search');
  const visibleCount = document.getElementById('visible_count');

  const confClass = c => c >= 0.7 ? 'hi' : (c >= 0.4 ? 'mid' : 'lo');

  function makeTile(r) {
    const tile = document.createElement('div'); tile.className = 'tile';
    const imgs = document.createElement('div'); imgs.className = 'tile-imgs';
    const full = document.createElement('img'); full.loading = 'lazy'; full.src = r.full; full.alt = '';
    const crop = document.createElement('img'); crop.loading = 'lazy'; crop.src = r.crop; crop.alt = '';
    imgs.append(full, crop);
    const meta = document.createElement('div'); meta.className = 'tile-meta';
    const stem = document.createElement('span'); stem.className = 'stem'; stem.title = r.stem; stem.textContent = r.stem;
    const conf = document.createElement('span'); conf.className = 'conf ' + confClass(r.conf); conf.textContent = r.conf.toFixed(3);
    meta.append(stem, conf);
    tile.append(imgs, meta);
    return tile;
  }

  function render() {
    const q = search.value.trim().toLowerCase();
    const mc = parseFloat(minConf.value);
    minConfVal.textContent = mc.toFixed(2);
    let rows = ROWS.filter(r => r.conf >= mc && (!q || r.stem.toLowerCase().includes(q)));
    const mode = sortSel.value;
    if (mode === 'conf_desc') rows.sort((a, b) => b.conf - a.conf);
    else if (mode === 'conf_asc') rows.sort((a, b) => a.conf - b.conf);
    else rows.sort((a, b) => a.stem.localeCompare(b.stem));

    visibleCount.textContent = rows.length + ' / ' + ROWS.length + ' shown';
    grid.replaceChildren(...rows.map(makeTile));
  }

  sortSel.addEventListener('change', render);
  minConf.addEventListener('input', render);
  search.addEventListener('input', render);
  render();
})();
</script>
</body>
</html>
"""


def write_index(records: list[dict], model_label: str) -> None:
    summary = html.escape(
        f"{len(records)} rows · model {model_label or '—'} · generated "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    out = INDEX_HTML.replace("__SUMMARY__", summary)
    out = out.replace("__ROWS_JSON__", json.dumps(records))
    (OUT_DIR / "index.html").write_text(out)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    THUMBS_DIR.mkdir(parents=True, exist_ok=True)

    rows = fetch_rows()
    if not rows:
        print("No stamp_predictions rows yet.")
        return

    print(f"Rendering {len(rows)} tiles…")
    model_label = rows[0]["model"]
    records: list[dict] = []
    missing_src = 0
    for i, row in enumerate(rows, 1):
        src = find_image(row["stem"])
        if src is None:
            missing_src += 1
            continue
        full_path = THUMBS_DIR / f"{row['stem']}_full.jpg"
        crop_path = THUMBS_DIR / f"{row['stem']}_crop.jpg"
        if not (full_path.exists() and crop_path.exists()):
            if not render_tile(row, src):
                continue
        records.append(
            {
                "stem": row["stem"],
                "conf": row["conf"],
                "full": f"thumbs/{row['stem']}_full.jpg",
                "crop": f"thumbs/{row['stem']}_crop.jpg",
            }
        )
        if i % 200 == 0:
            print(f"  {i}/{len(rows)}")

    write_index(records, model_label)
    print(
        f"Wrote {OUT_DIR / 'index.html'}  "
        f"({len(records)} tiles, {missing_src} missing sources)"
    )


if __name__ == "__main__":
    main()
