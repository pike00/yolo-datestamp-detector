# /// script
# requires-python = ">=3.14"
# ///
"""Generate an HTML review page for the stage-1 pilot results.

Renders one card per OCR result: original photo, stage-1 crop, and the
transcribed text, plus metadata (bbox source, confidence, trigger status).
Uses relative paths so the HTML can be opened directly from output/.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RESULTS_FILE = BASE_DIR / "state" / "ocr_results.json"
ROTATION_FILE = BASE_DIR / "state" / "rotation_predictions.json"
# Put the HTML at the repo root so both scanmyphotos/ and output/ are child
# paths — VS Code's HTML preview blocks parent-directory (../) paths.
OUTPUT_HTML = BASE_DIR / "pilot_review.html"

# Same regex the orchestrator uses
DATE_FORMAT_RE = re.compile(
    r"^(?:\d{1,2} \d{1,2} ?'\d{2}|'\d{2} \d{1,2} \d{1,2})$"
)
LOW_CONFIDENCE_THRESHOLD = 0.3


def should_review(text: str, confidence: float | None) -> bool:
    if confidence is not None and confidence < LOW_CONFIDENCE_THRESHOLD:
        return True
    if text is None:
        return True
    if "?" in text:
        return True
    if text == "NONE":
        return False
    if not DATE_FORMAT_RE.match(text):
        return True
    return False


def main() -> None:
    results = json.loads(RESULTS_FILE.read_text())
    rotations = {}
    if ROTATION_FILE.exists():
        rotations = json.loads(ROTATION_FILE.read_text())
    stems = sorted(results.keys())

    total = len(stems)
    clean = sum(1 for s in stems if not should_review(results[s]["text"], results[s].get("confidence")) and results[s]["text"] != "NONE")
    none_ct = sum(1 for s in stems if results[s]["text"] == "NONE")
    flagged = sum(1 for s in stems if should_review(results[s]["text"], results[s].get("confidence")))

    cards = []
    rotated_count = 0
    for stem in stems:
        entry = results[stem]
        text = entry.get("text", "")
        conf = entry.get("confidence")
        source = entry.get("bbox_source", "?")
        flag = should_review(text, conf)

        # Rotation: degrees CW required to make the photo upright.
        # CSS rotate must apply that many degrees CW to the displayed image.
        rot_entry = rotations.get(stem) or {}
        cnn_rot = int(rot_entry.get("rotation", 0))
        cnn_conf = rot_entry.get("confidence")
        if cnn_rot != 0:
            rotated_count += 1

        # Relative to yolo_finetune/pilot_review.html
        photo_rel = f"scanmyphotos/{stem}.jpg"
        crop_rel = f"output/ocr_crops_stage1/{stem}.jpg"

        badge_class = "flag" if flag else ("none" if text == "NONE" else "ok")
        badge_label = "REVIEW" if flag else ("NONE" if text == "NONE" else "OK")

        conf_str = f"{conf:.2f}" if conf is not None else "—"
        cnn_conf_str = f"{cnn_conf:.2f}" if cnn_conf is not None else "—"

        cards.append(f"""
<div class="card {badge_class}" data-stem="{html.escape(stem)}" data-cnn-rot="{cnn_rot}" data-cnn-conf="{cnn_conf_str}">
  <div class="meta">
    <label class="opus-toggle" title="Flag for Opus stage-2 review">
      <input type="checkbox" class="opus-flag" data-stem="{html.escape(stem)}"> Opus
    </label>
    <span class="stem">{html.escape(stem)}</span>
    <span class="badge">{badge_label}</span>
    <span class="src">{html.escape(source)}</span>
    <span class="conf">conf: {conf_str}</span>
    <span class="rot-info" title="EfficientNetV2 rotation prediction">rot: <span class="rot-display">{cnn_rot}°</span></span>
    <span class="rot-buttons">
      <button class="rot-btn" data-rot="0">0</button>
      <button class="rot-btn" data-rot="90">90</button>
      <button class="rot-btn" data-rot="180">180</button>
      <button class="rot-btn" data-rot="270">270</button>
    </span>
  </div>
  <div class="text">{html.escape(text) or "&nbsp;"}</div>
  <div class="images">
    <a href="{photo_rel}" target="_blank" class="photo-link"><img loading="lazy" src="{photo_rel}" alt="photo" class="photo-img"></a>
    <a href="{crop_rel}" target="_blank" class="crop-link"><img loading="lazy" src="{crop_rel}" alt="crop" class="crop crop-img"></a>
  </div>
</div>
""")

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Pilot OCR Review ({total} photos)</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 14px/1.4 system-ui, sans-serif; margin: 16px; background: #111; color: #eee; }}
  h1 {{ font-size: 18px; margin: 0 0 8px; }}
  .summary {{ margin-bottom: 16px; padding: 8px 12px; background: #222; border-radius: 6px; }}
  .summary span {{ margin-right: 20px; }}
  .filters {{ margin-bottom: 16px; }}
  .filters button {{ padding: 6px 12px; margin-right: 6px; background: #333; color: #eee; border: 1px solid #555; border-radius: 4px; cursor: pointer; font: inherit; }}
  .filters button.active {{ background: #4a9eff; color: #000; border-color: #4a9eff; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; }}
  .card {{ background: #1a1a1a; border: 1px solid #333; border-radius: 6px; padding: 8px; }}
  .card.flag {{ border-color: #c44; }}
  .card.none {{ border-color: #777; opacity: 0.6; }}
  .card.ok {{ border-color: #4c4; }}
  .meta {{ display: flex; gap: 8px; align-items: center; font-size: 12px; margin-bottom: 4px; flex-wrap: wrap; }}
  .stem {{ font-family: ui-monospace, monospace; color: #aaa; }}
  .badge {{ padding: 2px 6px; border-radius: 3px; font-weight: bold; font-size: 11px; }}
  .card.flag .badge {{ background: #c44; }}
  .card.none .badge {{ background: #555; }}
  .card.ok .badge {{ background: #4c4; color: #000; }}
  .src {{ color: #888; font-size: 11px; }}
  .conf {{ color: #888; font-size: 11px; }}
  .text {{ font: 16px ui-monospace, monospace; background: #000; color: #fa3; padding: 6px 8px; border-radius: 3px; margin: 4px 0; min-height: 24px; white-space: pre; }}
  .images {{ display: flex; gap: 4px; align-items: flex-start; }}
  .images img {{ max-width: 100%; max-height: 240px; object-fit: contain; background: #000; }}
  .images img.crop {{ max-height: 80px; }}
  .images a {{ display: block; flex: 1; min-width: 0; overflow: hidden; }}
  /* Containers must reserve square-ish space because rotated children change effective bounds. */
  .images a.photo-link {{ aspect-ratio: 4/3; display: flex; align-items: center; justify-content: center; }}
  .images a.crop-link {{ height: 80px; display: flex; align-items: center; justify-content: center; }}
  .photo-img, .crop-img {{ transition: transform 0.15s ease; transform-origin: center; }}
  .rot-info {{ color: #fa3; font-size: 11px; }}
  .rot-display {{ font-weight: bold; }}
  .rot-buttons {{ display: inline-flex; gap: 2px; margin-left: 4px; }}
  .rot-btn {{ padding: 1px 5px; font-size: 10px; background: #2a2a2a; color: #aaa; border: 1px solid #444; border-radius: 3px; cursor: pointer; font-family: ui-monospace, monospace; }}
  .rot-btn.active {{ background: #fa3; color: #000; border-color: #fa3; font-weight: bold; }}
  .rot-btn:hover {{ background: #444; }}
  .rot-btn.active:hover {{ background: #fb4; }}
  .card.rot-overridden {{ border-style: dashed; }}
  .opus-toggle {{ display: inline-flex; align-items: center; gap: 3px; padding: 2px 6px; background: #2a2a2a; border: 1px solid #555; border-radius: 3px; cursor: pointer; font-size: 11px; color: #ccc; user-select: none; }}
  .opus-toggle input {{ margin: 0; cursor: pointer; }}
  .card.opus-flagged {{ outline: 2px solid #fa3; outline-offset: -2px; }}
  .opus-bar {{ position: sticky; top: 0; z-index: 10; background: #1a1a1a; border: 1px solid #fa3; border-radius: 6px; padding: 8px 12px; margin-bottom: 12px; display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
  .opus-bar b {{ color: #fa3; font-size: 16px; }}
  .opus-bar button {{ padding: 6px 12px; background: #fa3; color: #000; border: none; border-radius: 4px; cursor: pointer; font: inherit; font-weight: bold; }}
  .opus-bar button.secondary {{ background: #444; color: #eee; }}
  .opus-bar textarea {{ flex: 1; min-width: 200px; background: #000; color: #fa3; border: 1px solid #555; border-radius: 3px; padding: 4px 6px; font: 11px ui-monospace, monospace; resize: vertical; min-height: 28px; max-height: 200px; }}
</style>
</head>
<body>
<h1>Pilot OCR Review</h1>
<div class="summary">
  <span><b>{total}</b> total</span>
  <span>OK: <b style="color:#4c4">{clean}</b></span>
  <span>NONE: <b style="color:#888">{none_ct}</b></span>
  <span>REVIEW: <b style="color:#c44">{flagged}</b></span>
  <span>Rotated (CNN): <b style="color:#fa3">{rotated_count}</b></span>
</div>
<div class="opus-bar">
  <span>Manual Opus flags: <b id="opus-count">0</b></span>
  <button id="opus-download">Download manual_opus_flags.json</button>
  <button id="opus-clear" class="secondary">Clear all</button>
  <button id="opus-filter" class="secondary">Show only flagged</button>
  <textarea id="opus-list" readonly placeholder="(no manual flags)"></textarea>
</div>
<div class="filters">
  <button data-filter="all" class="active">All</button>
  <button data-filter="ok">OK only</button>
  <button data-filter="flag">Review only</button>
  <button data-filter="none">NONE only</button>
</div>
<div class="grid">
{"".join(cards)}
</div>
<script>
const buttons = document.querySelectorAll('.filters button');
const cards = document.querySelectorAll('.card');
buttons.forEach(btn => btn.addEventListener('click', () => {{
  buttons.forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const f = btn.dataset.filter;
  cards.forEach(c => {{
    c.style.display = (f === 'all' || c.classList.contains(f)) ? '' : 'none';
  }});
}}));

// --- Manual Opus flagging (persisted in localStorage) ---
const STORAGE_KEY = 'opus_manual_flags_v1';
const opusCount = document.getElementById('opus-count');
const opusList = document.getElementById('opus-list');
const opusFilterBtn = document.getElementById('opus-filter');

function loadFlags() {{
  try {{ return new Set(JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')); }}
  catch (e) {{ return new Set(); }}
}}
function saveFlags(set) {{
  localStorage.setItem(STORAGE_KEY, JSON.stringify([...set].sort()));
}}
function refreshUI() {{
  const flags = loadFlags();
  opusCount.textContent = flags.size;
  opusList.value = [...flags].sort().join('\\n');
  document.querySelectorAll('.opus-flag').forEach(cb => {{
    const stem = cb.dataset.stem;
    cb.checked = flags.has(stem);
    cb.closest('.card').classList.toggle('opus-flagged', flags.has(stem));
  }});
}}

document.querySelectorAll('.opus-flag').forEach(cb => {{
  cb.addEventListener('change', (e) => {{
    const flags = loadFlags();
    const stem = e.target.dataset.stem;
    if (e.target.checked) flags.add(stem); else flags.delete(stem);
    saveFlags(flags);
    refreshUI();
  }});
}});

document.getElementById('opus-download').addEventListener('click', () => {{
  const flags = [...loadFlags()].sort();
  const blob = new Blob([JSON.stringify(flags, null, 2)], {{type: 'application/json'}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'manual_opus_flags.json';
  a.click();
  URL.revokeObjectURL(url);
}});

document.getElementById('opus-clear').addEventListener('click', () => {{
  if (confirm('Clear all manual Opus flags?')) {{
    localStorage.removeItem(STORAGE_KEY);
    refreshUI();
  }}
}});

let opusFilterActive = false;
opusFilterBtn.addEventListener('click', () => {{
  opusFilterActive = !opusFilterActive;
  opusFilterBtn.textContent = opusFilterActive ? 'Show all' : 'Show only flagged';
  cards.forEach(c => {{
    if (opusFilterActive) {{
      c.style.display = c.classList.contains('opus-flagged') ? '' : 'none';
    }} else {{
      c.style.display = '';
    }}
  }});
}});

refreshUI();

// --- Manual rotation overrides (persisted in localStorage) ---
const ROT_KEY = 'rotation_overrides_v1';

function loadRotOverrides() {{
  try {{ return JSON.parse(localStorage.getItem(ROT_KEY) || '{{}}'); }}
  catch (e) {{ return {{}}; }}
}}
function saveRotOverrides(map) {{
  localStorage.setItem(ROT_KEY, JSON.stringify(map));
}}
function effectiveRotation(card) {{
  const stem = card.dataset.stem;
  const overrides = loadRotOverrides();
  if (overrides[stem] !== undefined) return parseInt(overrides[stem], 10);
  return parseInt(card.dataset.cnnRot, 10) || 0;
}}
function applyRotation(card) {{
  const rot = effectiveRotation(card);
  const photo = card.querySelector('.photo-img');
  const crop = card.querySelector('.crop-img');
  // CSS rotates clockwise positive. Our stored rotation is "degrees CW required to make upright",
  // so we apply the same value as the rotate transform.
  if (photo) photo.style.transform = `rotate(${{rot}}deg)`;
  if (crop) crop.style.transform = `rotate(${{rot}}deg)`;
  card.querySelector('.rot-display').textContent = rot + '\\u00b0';
  card.querySelectorAll('.rot-btn').forEach(b => {{
    b.classList.toggle('active', parseInt(b.dataset.rot, 10) === rot);
  }});
  // Mark cards whose effective rotation differs from the CNN prediction
  const overrides = loadRotOverrides();
  card.classList.toggle('rot-overridden', overrides[card.dataset.stem] !== undefined);
}}

document.querySelectorAll('.card').forEach(card => {{
  applyRotation(card);
  card.querySelectorAll('.rot-btn').forEach(btn => {{
    btn.addEventListener('click', () => {{
      const newRot = parseInt(btn.dataset.rot, 10);
      const stem = card.dataset.stem;
      const cnnRot = parseInt(card.dataset.cnnRot, 10) || 0;
      const overrides = loadRotOverrides();
      if (newRot === cnnRot) {{
        // Reverting to CNN value clears the override
        delete overrides[stem];
      }} else {{
        overrides[stem] = newRot;
      }}
      saveRotOverrides(overrides);
      applyRotation(card);
    }});
  }});
}});
</script>
</body>
</html>
"""

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(html_doc)
    print(f"Wrote {OUTPUT_HTML.relative_to(BASE_DIR)}")
    print(f"  total={total} ok={clean} none={none_ct} review={flagged}")


if __name__ == "__main__":
    main()
