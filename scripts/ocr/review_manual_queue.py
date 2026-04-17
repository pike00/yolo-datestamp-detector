# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "psycopg[binary]>=3.1.0",
# ]
# ///
"""Manual review UI for the stage-2 OCR disagreement queue.

Serves a single-page app at http://localhost:8890/ that walks through
state/ocr_manual_queue.json one entry at a time. For each entry the user
sees the stage-2 crop and full-image views side-by-side and picks a final
answer (or types a correction). Decisions are written to stamp_ocr
(stage=3, review_status='confirmed' or 'no_stamp') and the entry is
removed from the queue file.
"""

from __future__ import annotations

import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from _db import update_ocr_review_status, upsert_ocr_result  # noqa: E402

QUEUE_FILE = BASE_DIR / "state" / "ocr_manual_queue.json"
CROP_DIR = BASE_DIR / "output" / "ocr_crops_stage2_crop"
FULL_DIR = BASE_DIR / "output" / "ocr_crops_stage2_full"

PORT = 8890


def load_queue() -> list[dict]:
    if not QUEUE_FILE.exists():
        return []
    with open(QUEUE_FILE) as f:
        return json.load(f)


def save_queue(queue: list[dict]) -> None:
    tmp = QUEUE_FILE.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(queue, f, indent=2)
    tmp.replace(QUEUE_FILE)


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OCR Manual Review</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; font-family: -apple-system, system-ui, sans-serif; background: #1a1a1a; color: #e0e0e0; }
  header { padding: 12px 20px; background: #222; border-bottom: 1px solid #333; display: flex; justify-content: space-between; align-items: center; }
  header h1 { margin: 0; font-size: 16px; font-weight: 500; }
  .progress { font-size: 13px; color: #888; }
  main { padding: 20px; display: grid; grid-template-columns: 1fr 1fr; gap: 20px; max-width: 1800px; margin: 0 auto; }
  .view { background: #222; border-radius: 6px; overflow: hidden; }
  .view-label { padding: 8px 12px; background: #2a2a2a; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: #888; display: flex; justify-content: space-between; }
  .view img { display: block; width: 100%; height: auto; max-height: 70vh; object-fit: contain; background: #000; }
  .view-text { padding: 10px 12px; font-family: ui-monospace, SFMono-Regular, monospace; font-size: 18px; color: #6cf; }
  .controls { grid-column: 1 / -1; background: #222; padding: 16px; border-radius: 6px; }
  .controls .meta { font-family: ui-monospace, monospace; font-size: 13px; color: #aaa; margin-bottom: 12px; }
  .controls .meta b { color: #e0e0e0; }
  .controls .stage1 { color: #d88; }
  .choices { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
  .choices button { background: #2d4a2d; color: #e0e0e0; border: 1px solid #3d5a3d; padding: 8px 14px; border-radius: 4px; font-size: 14px; cursor: pointer; font-family: ui-monospace, monospace; }
  .choices button:hover { background: #3d6a3d; }
  .choices button.none { background: #4a2d2d; border-color: #5a3d3d; }
  .choices button.none:hover { background: #6a3d3d; }
  .choices button.skip { background: #444; border-color: #555; }
  .key { display: inline-block; min-width: 18px; padding: 0 4px; background: #000; color: #6cf; border-radius: 3px; margin-right: 6px; font-size: 11px; vertical-align: middle; }
  .custom { display: flex; gap: 8px; }
  .custom input { flex: 1; background: #111; border: 1px solid #444; color: #e0e0e0; padding: 8px 10px; font-family: ui-monospace, monospace; font-size: 14px; border-radius: 4px; }
  .custom button { background: #2d3d5a; border: 1px solid #3d4d6a; color: #e0e0e0; padding: 8px 14px; border-radius: 4px; cursor: pointer; }
  .done { text-align: center; padding: 80px 20px; font-size: 18px; color: #888; }
</style>
</head>
<body>
<header>
  <h1>OCR Manual Review</h1>
  <div class="progress" id="progress">loading…</div>
</header>
<main id="main"></main>
<script>
let queue = [];
let idx = 0;

async function load() {
  const r = await fetch('/api/queue');
  queue = await r.json();
  idx = 0;
  render();
}

function el(tag, props, ...children) {
  const n = document.createElement(tag);
  if (props) {
    for (const [k, v] of Object.entries(props)) {
      if (k === 'class') n.className = v;
      else if (k === 'text') n.textContent = v;
      else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v);
    }
  }
  for (const c of children) if (c != null) n.append(c);
  return n;
}

function keyBadge(k) {
  return el('span', { class: 'key', text: k });
}

function choiceButton(label, key, cssClass, handler) {
  const b = el('button', { class: cssClass || '', onclick: handler });
  b.append(keyBadge(key));
  b.append(document.createTextNode(label));
  return b;
}

function buildView(title, keyLabel, imageUrl, viewText) {
  const v = el('div', { class: 'view' });
  const hdr = el('div', { class: 'view-label' });
  hdr.append(el('span', { text: title }), el('span', { text: '[' + keyLabel + ']' }));
  v.append(hdr);
  v.append(el('img', { src: imageUrl }));
  v.append(el('div', { class: 'view-text', text: viewText }));
  return v;
}

function render() {
  const main = document.getElementById('main');
  const prog = document.getElementById('progress');
  main.replaceChildren();
  if (queue.length === 0) {
    main.append(el('div', { class: 'done', text: '✓ queue empty' }));
    prog.textContent = 'done';
    return;
  }
  if (idx >= queue.length) idx = queue.length - 1;
  if (idx < 0) idx = 0;
  const e = queue[idx];
  prog.textContent = `${idx + 1} / ${queue.length} remaining — ${e.stem}`;
  const conf = e.confidence == null ? '—' : e.confidence.toFixed(3);

  main.append(
    buildView('crop view', '1', `/api/image?stem=${encodeURIComponent(e.stem)}&view=crop`, e.view_crop || ''),
    buildView('full image', '2', `/api/image?stem=${encodeURIComponent(e.stem)}&view=full`, e.view_full || '')
  );

  const controls = el('div', { class: 'controls' });
  const meta = el('div', { class: 'meta' });
  meta.append(
    document.createTextNode('stem: '),
    el('b', { text: e.stem }),
    document.createTextNode('   stage-1: '),
    el('span', { class: 'stage1', text: e.stage1_text || '' }),
    document.createTextNode(`   conf: ${conf}`)
  );
  controls.append(meta);

  const choices = el('div', { class: 'choices' });
  choices.append(
    choiceButton(e.view_crop || '', '1', '', () => decide('confirmed', e.view_crop)),
    choiceButton(e.view_full || '', '2', '', () => decide('confirmed', e.view_full)),
    choiceButton('stage-1: ' + (e.stage1_text || ''), '3', '', () => decide('confirmed', e.stage1_text)),
    choiceButton('NONE', 'N', 'none', () => decide('no_stamp', 'NONE')),
    choiceButton('skip', 'S', 'skip', skip)
  );
  controls.append(choices);

  const custom = el('div', { class: 'custom' });
  const input = el('input', { id: 'custom', type: 'text', placeholder: 'custom correction (Enter to submit)', autofocus: 'autofocus' });
  const submit = el('button', { onclick: submitCustom, text: 'submit' });
  custom.append(input, submit);
  controls.append(custom);

  main.append(controls);
  input.focus();
}

async function decide(action, text) {
  const e = queue[idx];
  await fetch('/api/decide', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ stem: e.stem, action, text }),
  });
  queue.splice(idx, 1);
  render();
}

function skip() {
  idx = (idx + 1) % Math.max(queue.length, 1);
  render();
}

function submitCustom() {
  const input = document.getElementById('custom');
  const v = input.value.trim();
  if (!v) return;
  const action = v.toUpperCase() === 'NONE' ? 'no_stamp' : 'confirmed';
  decide(action, v);
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') {
    if (e.key === 'Enter') { e.preventDefault(); submitCustom(); }
    return;
  }
  const cur = queue[idx];
  if (!cur) return;
  if (e.key === '1') decide('confirmed', cur.view_crop);
  else if (e.key === '2') decide('confirmed', cur.view_full);
  else if (e.key === '3') decide('confirmed', cur.stage1_text);
  else if (e.key.toLowerCase() === 'n') decide('no_stamp', 'NONE');
  else if (e.key.toLowerCase() === 's') skip();
  else if (e.key === 'ArrowRight') { idx++; render(); }
  else if (e.key === 'ArrowLeft') { idx--; render(); }
});

load();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write(f"[{self.log_date_time_string()}] {fmt % args}\n")

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self._send(HTTPStatus.OK, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if url.path == "/api/queue":
            data = json.dumps(load_queue()).encode("utf-8")
            self._send(HTTPStatus.OK, data, "application/json")
            return
        if url.path == "/api/image":
            qs = parse_qs(url.query)
            stem = (qs.get("stem") or [""])[0]
            view = (qs.get("view") or [""])[0]
            if not stem or view not in {"crop", "full"} or "/" in stem or ".." in stem:
                self._send(HTTPStatus.BAD_REQUEST, b"bad params", "text/plain")
                return
            path = (CROP_DIR if view == "crop" else FULL_DIR) / f"{stem}.jpg"
            if not path.exists():
                self._send(HTTPStatus.NOT_FOUND, b"missing image", "text/plain")
                return
            self._send(HTTPStatus.OK, path.read_bytes(), "image/jpeg")
            return
        self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/api/decide":
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            self._send(HTTPStatus.BAD_REQUEST, b"bad json", "text/plain")
            return

        stem = payload.get("stem")
        action = payload.get("action")
        text = payload.get("text")
        if not stem or action not in {"confirmed", "no_stamp"} or text is None:
            self._send(HTTPStatus.BAD_REQUEST, b"bad payload", "text/plain")
            return

        if action == "no_stamp":
            upsert_ocr_result(stem, "NONE", stage=3, review_status="no_stamp")
        else:
            upsert_ocr_result(stem, text, stage=3, review_status="confirmed")
        update_ocr_review_status(stem, "no_stamp" if action == "no_stamp" else "confirmed")

        queue = load_queue()
        queue = [e for e in queue if e.get("stem") != stem]
        save_queue(queue)

        self._send(HTTPStatus.OK, json.dumps({"ok": True, "remaining": len(queue)}).encode(), "application/json")


def main() -> int:
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Manual review UI: http://localhost:{PORT}/")
    print(f"Queue: {len(load_queue())} entries")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    return 0


if __name__ == "__main__":
    sys.exit(main())
