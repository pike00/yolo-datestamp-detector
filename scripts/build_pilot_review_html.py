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
    stems = sorted(results.keys())

    total = len(stems)
    clean = sum(1 for s in stems if not should_review(results[s]["text"], results[s].get("confidence")) and results[s]["text"] != "NONE")
    none_ct = sum(1 for s in stems if results[s]["text"] == "NONE")
    flagged = sum(1 for s in stems if should_review(results[s]["text"], results[s].get("confidence")))

    cards = []
    for stem in stems:
        entry = results[stem]
        text = entry.get("text", "")
        conf = entry.get("confidence")
        source = entry.get("bbox_source", "?")
        flag = should_review(text, conf)

        # Relative to yolo_finetune/pilot_review.html
        photo_rel = f"scanmyphotos/{stem}.jpg"
        crop_rel = f"output/ocr_crops_stage1/{stem}.jpg"

        badge_class = "flag" if flag else ("none" if text == "NONE" else "ok")
        badge_label = "REVIEW" if flag else ("NONE" if text == "NONE" else "OK")

        conf_str = f"{conf:.2f}" if conf is not None else "—"

        cards.append(f"""
<div class="card {badge_class}">
  <div class="meta">
    <span class="stem">{html.escape(stem)}</span>
    <span class="badge">{badge_label}</span>
    <span class="src">{html.escape(source)}</span>
    <span class="conf">conf: {conf_str}</span>
  </div>
  <div class="text">{html.escape(text) or "&nbsp;"}</div>
  <div class="images">
    <a href="{photo_rel}" target="_blank"><img loading="lazy" src="{photo_rel}" alt="photo"></a>
    <a href="{crop_rel}" target="_blank"><img loading="lazy" src="{crop_rel}" alt="crop" class="crop"></a>
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
  .images a {{ display: block; flex: 1; min-width: 0; }}
</style>
</head>
<body>
<h1>Pilot OCR Review</h1>
<div class="summary">
  <span><b>{total}</b> total</span>
  <span>OK: <b style="color:#4c4">{clean}</b></span>
  <span>NONE: <b style="color:#888">{none_ct}</b></span>
  <span>REVIEW: <b style="color:#c44">{flagged}</b></span>
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
