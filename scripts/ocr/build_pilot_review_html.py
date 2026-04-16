# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "psycopg[binary]>=3.1.0",
# ]
# ///
"""Generate an HTML review page for the stage-1 pilot results.

3x3 grid review flow:
- Cards auto-approve once scrolled above the viewport (unless flagged)
- Per-card buttons: approve / flag for rotation re-pass / flag for Opus re-pass
- Inline-editable OCR text (click to edit, blur to save)
- Manual rotation override (0/90/180/270)
- All state persisted in localStorage as a single unified review record
- Download review_state.json for downstream consumption
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

# Script lives at scripts/ocr/build_pilot_review_html.py
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from _db import load_ocr_results  # noqa: E402

ROTATION_FILE = BASE_DIR / "state" / "rotation_predictions.json"
OUTPUT_HTML = BASE_DIR / "output" / "pilot_review.html"

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
    results = load_ocr_results()
    rotations = {}
    if ROTATION_FILE.exists():
        rotations = json.loads(ROTATION_FILE.read_text())
    stems = sorted(results.keys())

    total = len(stems)
    clean = sum(1 for s in stems if not should_review(results[s]["text"], results[s].get("confidence")) and results[s]["text"] != "NONE")
    none_ct = sum(1 for s in stems if results[s]["text"] == "NONE")
    flagged_auto = sum(1 for s in stems if should_review(results[s]["text"], results[s].get("confidence")))

    cards = []
    rotated_count = 0
    for stem in stems:
        entry = results[stem]
        text = entry.get("text", "")
        conf = entry.get("confidence")
        source = entry.get("bbox_source", "?")
        auto_flag = should_review(text, conf)

        rot_entry = rotations.get(stem) or {}
        cnn_rot = int(rot_entry.get("rotation", 0))
        cnn_conf = rot_entry.get("confidence")
        if cnn_rot != 0:
            rotated_count += 1

        # Paths relative to output/pilot_review.html
        photo_rel = f"../scanmyphotos/{stem}.jpg"
        crop_rel = f"ocr_crops_stage1/{stem}.jpg"

        initial_class = "auto-flag" if auto_flag else ("none" if text == "NONE" else "auto-ok")
        initial_label = "REVIEW" if auto_flag else ("NONE" if text == "NONE" else "OK")

        conf_str = f"{conf:.2f}" if conf is not None else "—"
        cnn_conf_str = f"{cnn_conf:.2f}" if cnn_conf is not None else "—"

        text_escaped = html.escape(text) if text else ""
        text_attr = html.escape(text)

        cards.append(f"""
<div class="card {initial_class}" data-stem="{html.escape(stem)}" data-cnn-rot="{cnn_rot}" data-cnn-conf="{cnn_conf_str}" data-original-text="{text_attr}">
  <div class="status-pill"></div>
  <div class="meta">
    <span class="stem">{html.escape(stem)}</span>
    <span class="badge auto-badge">{initial_label}</span>
    <span class="src">{html.escape(source)}</span>
    <span class="conf">c:{conf_str}</span>
    <span class="rot-info">rot:<span class="rot-display">{cnn_rot}</span>°<span class="rot-conf">({cnn_conf_str})</span></span>
  </div>
  <div class="text" contenteditable="plaintext-only" spellcheck="false">{text_escaped}</div>
  <div class="controls">
    <button class="ctl-btn approve-btn" title="Approve (A)">✓ Approve</button>
    <button class="ctl-btn rotation-btn" title="Flag for rotation re-pass (R)">↻ Rotation</button>
    <button class="ctl-btn opus-btn" title="Flag for Opus re-pass (O)">⟲ Opus</button>
    <button class="ctl-btn reset-btn" title="Clear all review state">↶ Reset</button>
  </div>
  <div class="rot-override">
    <span class="rot-label">Override:</span>
    <button class="rot-btn" data-rot="0">0</button>
    <button class="rot-btn" data-rot="90">90</button>
    <button class="rot-btn" data-rot="180">180</button>
    <button class="rot-btn" data-rot="270">270</button>
  </div>
  <div class="images">
    <a href="{photo_rel}" target="_blank" class="photo-link"><img loading="lazy" src="{photo_rel}" alt="photo" class="photo-img"></a>
    <a href="{crop_rel}" target="_blank" class="crop-link"><img loading="lazy" src="{crop_rel}" alt="crop" class="crop-img"></a>
  </div>
</div>
""")

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Pilot OCR Review ({total} photos)</title>
<style>
  :root {{ color-scheme: dark; --bg: #0e0e10; --panel: #1a1a1d; --panel2: #242428; --border: #333; --text: #e8e8ea; --muted: #888; --ok: #4c4; --flag: #c44; --warn: #fa3; --opus: #f66; --rot: #4af; --edit: #0ce; }}
  * {{ box-sizing: border-box; }}
  body {{ font: 13px/1.4 system-ui, sans-serif; margin: 0; background: var(--bg); color: var(--text); }}
  h1 {{ font-size: 16px; margin: 0; }}
  .header {{ position: sticky; top: 0; z-index: 20; background: var(--panel); border-bottom: 1px solid var(--border); padding: 10px 16px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }}
  .counts {{ display: flex; gap: 14px; font-size: 12px; flex-wrap: wrap; }}
  .counts span b {{ font-weight: 700; }}
  .counts .unreviewed b {{ color: var(--muted); }}
  .counts .approved b {{ color: var(--ok); }}
  .counts .autoapproved b {{ color: #6b6; }}
  .counts .rotation b {{ color: var(--rot); }}
  .counts .opus b {{ color: var(--opus); }}
  .counts .edited b {{ color: var(--edit); }}
  .filters {{ display: flex; gap: 4px; }}
  .filters button, .actions button {{ padding: 5px 10px; background: var(--panel2); color: var(--text); border: 1px solid var(--border); border-radius: 4px; cursor: pointer; font: inherit; font-size: 12px; }}
  .filters button.active {{ background: #4a9eff; color: #000; border-color: #4a9eff; }}
  .actions {{ display: flex; gap: 6px; margin-left: auto; }}
  .actions button.primary {{ background: var(--warn); color: #000; font-weight: 600; border-color: var(--warn); }}
  .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; padding: 14px; max-width: 1800px; margin: 0 auto; }}
  .card {{ position: relative; background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 8px; display: flex; flex-direction: column; gap: 6px; }}
  .card.auto-flag {{ border-left: 3px solid var(--flag); }}
  .card.auto-ok {{ border-left: 3px solid var(--ok); }}
  .card.none {{ border-left: 3px solid var(--muted); }}
  .card.state-approved {{ outline: 2px solid var(--ok); outline-offset: -2px; }}
  .card.state-approved.auto-approved {{ outline-style: dashed; opacity: 0.55; }}
  .card.state-rotation-flag {{ outline: 2px solid var(--rot); outline-offset: -2px; }}
  .card.state-opus-flag {{ outline: 2px solid var(--opus); outline-offset: -2px; }}
  .card.state-edited .text {{ border-color: var(--edit); }}
  .status-pill {{ position: absolute; top: 4px; right: 4px; padding: 1px 6px; font-size: 10px; font-weight: 700; border-radius: 3px; display: none; }}
  .card.state-approved .status-pill {{ display: block; background: var(--ok); color: #000; }}
  .card.state-approved .status-pill::before {{ content: '✓ APPROVED'; }}
  .card.state-approved.auto-approved .status-pill::before {{ content: '✓ AUTO'; }}
  .card.state-rotation-flag .status-pill {{ display: block; background: var(--rot); color: #000; }}
  .card.state-rotation-flag .status-pill::before {{ content: '↻ ROT'; }}
  .card.state-opus-flag .status-pill {{ display: block; background: var(--opus); color: #000; }}
  .card.state-opus-flag .status-pill::before {{ content: '⟲ OPUS'; }}
  .meta {{ display: flex; gap: 6px; align-items: center; font-size: 11px; flex-wrap: wrap; }}
  .stem {{ font-family: ui-monospace, monospace; color: #aaa; }}
  .badge {{ padding: 1px 5px; border-radius: 3px; font-weight: 700; font-size: 10px; }}
  .auto-flag .badge {{ background: var(--flag); color: #fff; }}
  .auto-ok .badge {{ background: var(--ok); color: #000; }}
  .none .badge {{ background: #555; color: #ccc; }}
  .src, .conf, .rot-info {{ color: var(--muted); font-size: 10px; }}
  .rot-info {{ color: var(--warn); }}
  .rot-conf {{ color: var(--muted); margin-left: 2px; }}
  .text {{ font: 18px ui-monospace, monospace; background: #000; color: var(--warn); padding: 8px 10px; border-radius: 4px; min-height: 36px; white-space: pre; border: 1px solid #333; cursor: text; outline: none; }}
  .text:focus {{ border-color: var(--edit); background: #001a1a; }}
  .controls {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 3px; }}
  .ctl-btn {{ padding: 4px 2px; background: var(--panel2); color: var(--text); border: 1px solid var(--border); border-radius: 3px; cursor: pointer; font: inherit; font-size: 10px; }}
  .ctl-btn:hover {{ background: #333; }}
  .approve-btn.active {{ background: var(--ok); color: #000; border-color: var(--ok); font-weight: 700; }}
  .rotation-btn.active {{ background: var(--rot); color: #000; border-color: var(--rot); font-weight: 700; }}
  .opus-btn.active {{ background: var(--opus); color: #000; border-color: var(--opus); font-weight: 700; }}
  .rot-override {{ display: flex; gap: 3px; align-items: center; font-size: 10px; color: var(--muted); }}
  .rot-label {{ color: var(--muted); }}
  .rot-btn {{ padding: 1px 6px; font-size: 10px; background: var(--panel2); color: #aaa; border: 1px solid var(--border); border-radius: 3px; cursor: pointer; font-family: ui-monospace, monospace; }}
  .rot-btn.active {{ background: var(--warn); color: #000; border-color: var(--warn); font-weight: 700; }}
  .images {{ display: flex; gap: 4px; align-items: flex-start; }}
  .images a {{ display: block; overflow: hidden; }}
  .images a.photo-link {{ flex: 3; aspect-ratio: 4/3; display: flex; align-items: center; justify-content: center; background: #000; border-radius: 3px; }}
  .images a.crop-link {{ flex: 1; aspect-ratio: 4/1; display: flex; align-items: center; justify-content: center; background: #000; border-radius: 3px; }}
  .images img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
  .photo-img, .crop-img {{ transition: transform 0.15s ease; transform-origin: center; }}
  /* Filter modes — applied via body class */
  body.filter-unreviewed .card.state-approved,
  body.filter-unreviewed .card.state-rotation-flag,
  body.filter-unreviewed .card.state-opus-flag {{ display: none; }}
  body.filter-flagged .card:not(.state-rotation-flag):not(.state-opus-flag) {{ display: none; }}
  body.filter-edited .card:not(.state-edited) {{ display: none; }}
</style>
</head>
<body>
<div class="header">
  <h1>OCR Review</h1>
  <div class="counts">
    <span class="total"><b id="c-total">{total}</b> total</span>
    <span class="unreviewed">unreviewed <b id="c-unreviewed">0</b></span>
    <span class="approved">approved <b id="c-approved">0</b></span>
    <span class="autoapproved">auto <b id="c-auto">0</b></span>
    <span class="rotation">rot-flagged <b id="c-rotation">0</b></span>
    <span class="opus">opus-flagged <b id="c-opus">0</b></span>
    <span class="edited">edited <b id="c-edited">0</b></span>
  </div>
  <div class="filters">
    <button data-filter="all" class="active">All</button>
    <button data-filter="unreviewed">Unreviewed</button>
    <button data-filter="flagged">Flagged</button>
    <button data-filter="edited">Edited</button>
  </div>
  <div class="actions">
    <button id="download-btn" class="primary">Download review_state.json</button>
    <button id="clear-btn">Clear all</button>
  </div>
</div>
<div class="grid">
{"".join(cards)}
</div>
<script>
const STORAGE_KEY = 'pilot_review_state_v2';

function loadState() {{
  try {{ return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}'); }}
  catch (e) {{ return {{}}; }}
}}
function saveState(s) {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(s)); }}
function getCard(state, stem) {{ return state[stem] || {{}}; }}
function updateCard(stem, patch) {{
  const s = loadState();
  const merged = {{...getCard(s, stem), ...patch}};
  // Drop empty keys so the stored object stays compact
  for (const k of Object.keys(merged)) {{
    if (merged[k] === null || merged[k] === undefined || merged[k] === false) delete merged[k];
  }}
  if (Object.keys(merged).length === 0) delete s[stem];
  else s[stem] = merged;
  saveState(s);
  renderCard(document.querySelector(`.card[data-stem="${{CSS.escape(stem)}}"]`));
  renderCounts();
}}

function renderCard(card) {{
  if (!card) return;
  const stem = card.dataset.stem;
  const s = getCard(loadState(), stem);

  // Review state classes
  card.classList.toggle('state-approved', !!s.approved);
  card.classList.toggle('auto-approved', !!s.auto_approved);
  card.classList.toggle('state-rotation-flag', !!s.rotation_flag);
  card.classList.toggle('state-opus-flag', !!s.opus_flag);
  card.classList.toggle('state-edited', s.edited_text !== undefined && s.edited_text !== null);

  // Button active states
  card.querySelector('.approve-btn').classList.toggle('active', !!s.approved && !s.auto_approved);
  card.querySelector('.rotation-btn').classList.toggle('active', !!s.rotation_flag);
  card.querySelector('.opus-btn').classList.toggle('active', !!s.opus_flag);

  // Text content
  const textEl = card.querySelector('.text');
  const original = card.dataset.originalText;
  const display = s.edited_text !== undefined && s.edited_text !== null ? s.edited_text : original;
  if (textEl.innerText !== display) textEl.innerText = display;

  // Rotation
  const cnnRot = parseInt(card.dataset.cnnRot, 10) || 0;
  const effRot = s.rotation_override !== undefined && s.rotation_override !== null ? s.rotation_override : cnnRot;
  const photo = card.querySelector('.photo-img');
  const crop = card.querySelector('.crop-img');
  if (photo) photo.style.transform = `rotate(${{effRot}}deg)`;
  if (crop) crop.style.transform = `rotate(${{effRot}}deg)`;
  card.querySelector('.rot-display').textContent = effRot;
  card.querySelectorAll('.rot-btn').forEach(b => {{
    b.classList.toggle('active', parseInt(b.dataset.rot, 10) === effRot);
  }});
}}

function renderAll() {{
  document.querySelectorAll('.card').forEach(renderCard);
  renderCounts();
}}

function renderCounts() {{
  const state = loadState();
  const cards = document.querySelectorAll('.card');
  let approved = 0, auto = 0, rot = 0, opus = 0, edited = 0;
  cards.forEach(c => {{
    const s = getCard(state, c.dataset.stem);
    if (s.approved) {{
      approved++;
      if (s.auto_approved) auto++;
    }}
    if (s.rotation_flag) rot++;
    if (s.opus_flag) opus++;
    if (s.edited_text !== undefined && s.edited_text !== null) edited++;
  }});
  const total = cards.length;
  const touched = approved + rot + opus;
  document.getElementById('c-unreviewed').textContent = total - touched;
  document.getElementById('c-approved').textContent = approved;
  document.getElementById('c-auto').textContent = auto;
  document.getElementById('c-rotation').textContent = rot;
  document.getElementById('c-opus').textContent = opus;
  document.getElementById('c-edited').textContent = edited;
}}

// --- Per-card event wiring ---
document.querySelectorAll('.card').forEach(card => {{
  const stem = card.dataset.stem;

  card.querySelector('.approve-btn').addEventListener('click', () => {{
    const s = getCard(loadState(), stem);
    const currentlyManual = s.approved && !s.auto_approved;
    if (currentlyManual) {{
      updateCard(stem, {{approved: null, auto_approved: null}});
    }} else {{
      updateCard(stem, {{approved: true, auto_approved: null, rotation_flag: null, opus_flag: null}});
    }}
  }});
  card.querySelector('.rotation-btn').addEventListener('click', () => {{
    const s = getCard(loadState(), stem);
    if (s.rotation_flag) updateCard(stem, {{rotation_flag: null}});
    else updateCard(stem, {{rotation_flag: true, approved: null, auto_approved: null}});
  }});
  card.querySelector('.opus-btn').addEventListener('click', () => {{
    const s = getCard(loadState(), stem);
    if (s.opus_flag) updateCard(stem, {{opus_flag: null}});
    else updateCard(stem, {{opus_flag: true, approved: null, auto_approved: null}});
  }});
  card.querySelector('.reset-btn').addEventListener('click', () => {{
    updateCard(stem, {{approved: null, auto_approved: null, rotation_flag: null, opus_flag: null, edited_text: null, rotation_override: null}});
  }});

  // Inline text editing
  const textEl = card.querySelector('.text');
  textEl.addEventListener('blur', () => {{
    const newText = textEl.innerText.replace(/\\r?\\n/g, '').trim();
    const original = card.dataset.originalText;
    if (newText === original) {{
      updateCard(stem, {{edited_text: null}});
    }} else {{
      updateCard(stem, {{edited_text: newText}});
    }}
  }});
  textEl.addEventListener('keydown', (e) => {{
    if (e.key === 'Enter') {{ e.preventDefault(); textEl.blur(); }}
    if (e.key === 'Escape') {{
      const s = getCard(loadState(), stem);
      const current = s.edited_text !== undefined && s.edited_text !== null ? s.edited_text : card.dataset.originalText;
      textEl.innerText = current;
      textEl.blur();
    }}
  }});

  // Rotation override buttons
  card.querySelectorAll('.rot-btn').forEach(btn => {{
    btn.addEventListener('click', () => {{
      const newRot = parseInt(btn.dataset.rot, 10);
      const cnnRot = parseInt(card.dataset.cnnRot, 10) || 0;
      if (newRot === cnnRot) {{
        updateCard(stem, {{rotation_override: null}});
      }} else {{
        updateCard(stem, {{rotation_override: newRot}});
      }}
    }});
  }});
}});

// --- Auto-approve on scroll past ---
const scrollObserver = new IntersectionObserver((entries) => {{
  entries.forEach(entry => {{
    if (!entry.isIntersecting && entry.boundingClientRect.bottom < 0) {{
      const card = entry.target;
      const stem = card.dataset.stem;
      const s = getCard(loadState(), stem);
      // Only auto-approve if the card is untouched
      if (!s.approved && !s.rotation_flag && !s.opus_flag) {{
        updateCard(stem, {{approved: true, auto_approved: true}});
      }}
    }}
  }});
}}, {{ threshold: 0 }});
document.querySelectorAll('.card').forEach(c => scrollObserver.observe(c));

// --- Filter buttons ---
document.querySelectorAll('.filters button').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.filters button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const f = btn.dataset.filter;
    document.body.className = f === 'all' ? '' : `filter-${{f}}`;
  }});
}});

// --- Download + Clear ---
document.getElementById('download-btn').addEventListener('click', () => {{
  const state = loadState();
  const blob = new Blob([JSON.stringify(state, null, 2)], {{type: 'application/json'}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'review_state.json';
  a.click();
  URL.revokeObjectURL(url);
}});
document.getElementById('clear-btn').addEventListener('click', () => {{
  if (confirm('Clear ALL review state (approvals, flags, edits, rotation overrides)?')) {{
    localStorage.removeItem(STORAGE_KEY);
    renderAll();
  }}
}});

// --- Initial render ---
renderAll();
</script>
</body>
</html>
"""

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(html_doc)
    print(f"Wrote {OUTPUT_HTML.relative_to(BASE_DIR)}")
    print(f"  total={total} auto-ok={clean} auto-none={none_ct} auto-flagged={flagged_auto} cnn-rotated={rotated_count}")


if __name__ == "__main__":
    main()
