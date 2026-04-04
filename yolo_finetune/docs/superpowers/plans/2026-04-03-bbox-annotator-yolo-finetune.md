# Bounding Box Annotator + YOLO Fine-Tune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a keyboard-only bounding box annotation UI and YOLOv8 fine-tuning pipeline for labeling date stamp regions on ~100 scanned photos.

**Architecture:** Single Python package (`yolo_finetune/`) with two entry points: `annotate.py` (HTTP server + browser UI for labeling) and `train.py` (YOLO fine-tuning). The annotator writes YOLO-format labels directly — no conversion step. Browser UI uses vanilla JS + Canvas, keyboard-only control, with carry-forward box positioning.

**Tech Stack:** Python 3.14, `http.server` (stdlib), vanilla JS/Canvas, `ultralytics` (YOLOv8), `PIL`/`torch`/`torchvision` (already in venv)

---

## File Structure

```
yolo_finetune/
  annotate.py      # HTTP server: serves UI, images, and REST API for save/skip/state
  index.html       # Single-page annotation UI: Canvas rendering, keyboard controls
  train.py         # YOLO fine-tuning: dataset split, training, output
  dataset/         # Created by annotate.py and train.py
    images/        # Symlinks to source photos
    labels/        # YOLO-format .txt label files
    data.yaml      # YOLO dataset config (created by train.py)
  progress.json    # Annotation session state (created by annotate.py)
  skipped.txt      # Skipped filenames (created by annotate.py)
```

**Source photos:** `../photo_mapping_samples/` (100 JPGs, ~1784x1187px each)

**Existing venv:** `/home/will/photo_project/.venv/` — activate with `source /home/will/photo_project/.venv/bin/activate`

---

### Task 1: Annotation Server Backend (`annotate.py`)

**Files:**
- Create: `annotate.py`

The server handles: serving `index.html`, serving source images, and a REST API for annotation state management.

- [ ] **Step 1: Create `annotate.py` with image listing and serving**

```python
"""Annotation server — serves UI and REST API for bounding box labeling."""

import http.server
import json
import os
import socketserver
import sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs

PORT = 8888
BASE_DIR = Path(__file__).parent
IMAGE_DIR = BASE_DIR.parent / "photo_mapping_samples"
DATASET_DIR = BASE_DIR / "dataset"
LABELS_DIR = DATASET_DIR / "labels"
IMAGES_DIR = DATASET_DIR / "images"
PROGRESS_FILE = BASE_DIR / "progress.json"
SKIPPED_FILE = BASE_DIR / "skipped.txt"


def get_image_files():
    """Return sorted list of source JPGs, excluding _boxes variants."""
    files = sorted(
        f.name for f in IMAGE_DIR.glob("*.jpg")
        if "_boxes" not in f.name
    )
    # Also check .JPG
    files += sorted(
        f.name for f in IMAGE_DIR.glob("*.JPG")
        if "_boxes" not in f.name and f.name not in files
    )
    return files


def load_progress():
    """Load or initialize annotation progress."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {
        "current_index": 0,
        "last_box": {"x": 0.4, "y": 0.45, "w": 0.2, "h": 0.1},
        "labeled": [],
        "skipped": [],
    }


def save_progress(progress):
    """Write progress to disk immediately."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def save_label(filename, box):
    """Write a YOLO-format label file. box is {x, y, w, h} normalized."""
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem
    label_path = LABELS_DIR / f"{stem}.txt"
    cx, cy, bw, bh = box["x"], box["y"], box["w"], box["h"]
    label_path.write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    # Symlink source image into dataset
    img_link = IMAGES_DIR / filename
    if not img_link.exists():
        img_link.symlink_to((IMAGE_DIR / filename).resolve())


def save_skip(filename):
    """Append filename to skipped.txt."""
    with open(SKIPPED_FILE, "a") as f:
        f.write(filename + "\n")


def remove_label(filename):
    """Remove a label file if it exists (for undo/redo)."""
    stem = Path(filename).stem
    label_path = LABELS_DIR / f"{stem}.txt"
    if label_path.exists():
        label_path.unlink()
    img_link = IMAGES_DIR / filename
    if img_link.is_symlink():
        img_link.unlink()


def remove_skip(filename):
    """Remove a filename from skipped.txt if present."""
    if not SKIPPED_FILE.exists():
        return
    lines = SKIPPED_FILE.read_text().splitlines()
    lines = [l for l in lines if l.strip() != filename]
    SKIPPED_FILE.write_text("\n".join(lines) + "\n" if lines else "")


class AnnotationHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/state":
            self._json_response(self._get_state())
        elif path.startswith("/photos/"):
            # Serve source images
            filename = path[len("/photos/"):]
            filepath = IMAGE_DIR / filename
            if filepath.exists():
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "max-age=3600")
                self.end_headers()
                self.wfile.write(filepath.read_bytes())
            else:
                self.send_error(404)
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        content_len = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_len)) if content_len else {}

        if path == "/api/label":
            self._handle_label(body)
        elif path == "/api/skip":
            self._handle_skip(body)
        elif path == "/api/goto":
            self._handle_goto(body)
        else:
            self.send_error(404)

    def _get_state(self):
        images = get_image_files()
        progress = load_progress()
        return {
            "images": images,
            "total": len(images),
            "current_index": progress["current_index"],
            "last_box": progress["last_box"],
            "labeled": progress["labeled"],
            "skipped": progress["skipped"],
        }

    def _handle_label(self, body):
        """Save label for current image and advance."""
        filename = body["filename"]
        box = body["box"]  # {x, y, w, h} normalized
        progress = load_progress()

        # Remove from skipped if it was previously skipped
        if filename in progress["skipped"]:
            progress["skipped"].remove(filename)
            remove_skip(filename)

        save_label(filename, box)

        if filename not in progress["labeled"]:
            progress["labeled"].append(filename)

        progress["last_box"] = box
        progress["current_index"] = body.get("next_index", progress["current_index"] + 1)
        save_progress(progress)
        self._json_response({"ok": True})

    def _handle_skip(self, body):
        """Mark current image as skipped and advance."""
        filename = body["filename"]
        progress = load_progress()

        # Remove label if it was previously labeled
        if filename in progress["labeled"]:
            progress["labeled"].remove(filename)
            remove_label(filename)

        if filename not in progress["skipped"]:
            progress["skipped"].append(filename)
        save_skip(filename)

        progress["current_index"] = body.get("next_index", progress["current_index"] + 1)
        save_progress(progress)
        self._json_response({"ok": True})

    def _handle_goto(self, body):
        """Jump to a specific image index."""
        progress = load_progress()
        progress["current_index"] = body["index"]
        save_progress(progress)
        self._json_response({"ok": True})

    def _json_response(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        # Suppress request logs except errors
        if args and isinstance(args[0], str) and args[0].startswith("4"):
            super().log_message(format, *args)


def main():
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    images = get_image_files()
    progress = load_progress()
    labeled = len(progress["labeled"])
    skipped = len(progress["skipped"])

    print(f"Found {len(images)} images in {IMAGE_DIR}")
    if labeled or skipped:
        print(f"Resuming: {labeled} labeled, {skipped} skipped, at index {progress['current_index']}")
    print(f"Starting server on http://localhost:{PORT}")
    print("Open that URL in your browser to begin annotating.")

    with socketserver.TCPServer(("", PORT), AnnotationHandler) as httpd:
        httpd.allow_reuse_address = True
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify server starts and API responds**

Run:
```bash
cd /home/will/photo_project/yolo_finetune
source /home/will/photo_project/.venv/bin/activate
python annotate.py &
sleep 1
curl -s http://localhost:8888/api/state | python -m json.tool
kill %1
```

Expected: JSON with `images` (list of 100 filenames), `total: 100`, `current_index: 0`, `last_box`, empty `labeled`/`skipped` arrays.

- [ ] **Step 3: Commit**

```bash
git add annotate.py
git commit -m "feat: annotation server with REST API for labeling workflow"
```

---

### Task 2: Annotation UI Frontend (`index.html`)

**Files:**
- Create: `index.html`

Single-page app: loads state from API, renders photo on Canvas with bounding box overlay, handles all keyboard input, posts label/skip back to server.

- [ ] **Step 1: Create `index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Annotate</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #0f0f23;
    color: #e0e0e0;
    font-family: 'Courier New', monospace;
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
    user-select: none;
  }

  #status-bar {
    background: #16213e;
    padding: 6px 16px;
    font-size: 13px;
    display: flex;
    justify-content: space-between;
    border-bottom: 1px solid #333;
    flex-shrink: 0;
  }
  #status-bar .labeled { color: #4ade80; }
  #status-bar .skipped { color: #fbbf24; }

  #canvas-container {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 0;
    padding: 8px;
  }
  canvas {
    max-width: 100%;
    max-height: 100%;
  }

  #shortcut-bar {
    background: #16213e;
    padding: 6px 16px;
    font-size: 11px;
    color: #a0a0b0;
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    justify-content: center;
    border-top: 1px solid #333;
    flex-shrink: 0;
  }
  kbd {
    background: #2a2a4a;
    padding: 1px 5px;
    border-radius: 3px;
    color: #e0e0e0;
    border: 1px solid #444;
    font-family: inherit;
  }
  .confirm { color: #4ade80; }
  .skip-label { color: #fbbf24; }

  #done-overlay {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(15, 15, 35, 0.95);
    align-items: center;
    justify-content: center;
    flex-direction: column;
    font-size: 20px;
    gap: 16px;
  }
  #done-overlay.visible { display: flex; }
</style>
</head>
<body>

<div id="status-bar">
  <span id="filename">Loading...</span>
  <span id="progress"></span>
  <span class="labeled" id="labeled-count"></span>
  <span class="skipped" id="skipped-count"></span>
</div>

<div id="canvas-container">
  <canvas id="canvas"></canvas>
</div>

<div id="shortcut-bar">
  <span><kbd>&#8592;&#8593;&#8595;&#8594;</kbd> Move</span>
  <span><kbd>Shift</kbd> Fine</span>
  <span><kbd>[ ]</kbd> Width</span>
  <span><kbd>{ }</kbd> Height</span>
  <span><kbd>Enter</kbd> <span class="confirm">Confirm</span></span>
  <span><kbd>S</kbd> <span class="skip-label">Skip</span></span>
  <span><kbd>Z</kbd> Back</span>
  <span><kbd>R</kbd> Reset</span>
  <span><kbd>Q</kbd> Quit</span>
</div>

<div id="done-overlay">
  <div id="done-title">All images processed!</div>
  <div id="done-stats"></div>
</div>

<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

let state = null;       // server state
let img = null;         // current Image object
let imgW = 0, imgH = 0; // natural image dimensions

// Box in normalized coords [0,1] relative to image
let box = { x: 0.4, y: 0.45, w: 0.2, h: 0.1 };

const COARSE = 20; // px
const FINE = 3;    // px
const SIZE_STEP_COARSE = 10; // px
const SIZE_STEP_FINE = 3;    // px

async function loadState() {
  const resp = await fetch('/api/state');
  state = await resp.json();
  box = { ...state.last_box };
  loadCurrentImage();
}

function loadCurrentImage() {
  if (state.current_index >= state.total) {
    showDone();
    return;
  }

  const filename = state.images[state.current_index];
  updateStatus(filename);

  img = new Image();
  img.onload = () => {
    imgW = img.naturalWidth;
    imgH = img.naturalHeight;
    resizeCanvas();
    draw();
  };
  img.src = '/photos/' + filename;
}

function resizeCanvas() {
  const container = document.getElementById('canvas-container');
  const maxW = container.clientWidth - 16;
  const maxH = container.clientHeight - 16;

  const scale = Math.min(maxW / imgW, maxH / imgH);
  canvas.width = Math.floor(imgW * scale);
  canvas.height = Math.floor(imgH * scale);
}

function draw() {
  if (!img) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  // Draw bounding box
  const bx = (box.x - box.w / 2) * canvas.width;
  const by = (box.y - box.h / 2) * canvas.height;
  const bw = box.w * canvas.width;
  const bh = box.h * canvas.height;

  ctx.strokeStyle = '#00ff88';
  ctx.lineWidth = 2;
  ctx.strokeRect(bx, by, bw, bh);

  // Semi-transparent fill
  ctx.fillStyle = 'rgba(0, 255, 136, 0.08)';
  ctx.fillRect(bx, by, bw, bh);

  // Corner handles
  const hs = 6;
  ctx.fillStyle = '#00ff88';
  ctx.fillRect(bx - hs/2, by - hs/2, hs, hs);
  ctx.fillRect(bx + bw - hs/2, by - hs/2, hs, hs);
  ctx.fillRect(bx - hs/2, by + bh - hs/2, hs, hs);
  ctx.fillRect(bx + bw - hs/2, by + bh - hs/2, hs, hs);
}

function updateStatus(filename) {
  document.getElementById('filename').textContent = filename || '';
  document.getElementById('progress').textContent =
    (state.current_index + 1) + ' / ' + state.total;
  document.getElementById('labeled-count').textContent =
    state.labeled.length + ' labeled';
  document.getElementById('skipped-count').textContent =
    state.skipped.length + ' skipped';
}

function showDone() {
  document.getElementById('done-overlay').classList.add('visible');
  document.getElementById('done-stats').textContent =
    state.labeled.length + ' labeled, ' + state.skipped.length + ' skipped';
}

async function confirmLabel() {
  const filename = state.images[state.current_index];
  const nextIndex = state.current_index + 1;

  await fetch('/api/label', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename, box, next_index: nextIndex }),
  });

  if (!state.labeled.includes(filename)) state.labeled.push(filename);
  state.skipped = state.skipped.filter(function(f) { return f !== filename; });
  state.current_index = nextIndex;
  state.last_box = { x: box.x, y: box.y, w: box.w, h: box.h };
  loadCurrentImage();
}

async function skipImage() {
  const filename = state.images[state.current_index];
  const nextIndex = state.current_index + 1;

  await fetch('/api/skip', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename, next_index: nextIndex }),
  });

  if (!state.skipped.includes(filename)) state.skipped.push(filename);
  state.labeled = state.labeled.filter(function(f) { return f !== filename; });
  state.current_index = nextIndex;
  loadCurrentImage();
}

function goBack() {
  if (state.current_index <= 0) return;
  state.current_index--;

  fetch('/api/goto', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ index: state.current_index }),
  }).then(function() { loadCurrentImage(); });
}

function resetBox() {
  box = { x: 0.5, y: 0.5, w: 0.2, h: 0.1 };
  draw();
}

document.addEventListener('keydown', function(e) {
  if (state && state.current_index >= state.total) return;

  var step = e.shiftKey ? FINE : COARSE;
  var sizeStep = e.shiftKey ? SIZE_STEP_FINE : SIZE_STEP_COARSE;

  switch (e.key) {
    case 'ArrowLeft':
      e.preventDefault();
      box.x = Math.max(box.w / 2, box.x - step / imgW);
      draw();
      break;
    case 'ArrowRight':
      e.preventDefault();
      box.x = Math.min(1 - box.w / 2, box.x + step / imgW);
      draw();
      break;
    case 'ArrowUp':
      e.preventDefault();
      box.y = Math.max(box.h / 2, box.y - step / imgH);
      draw();
      break;
    case 'ArrowDown':
      e.preventDefault();
      box.y = Math.min(1 - box.h / 2, box.y + step / imgH);
      draw();
      break;
    case '[':
      box.w = Math.max(0.02, box.w - sizeStep / imgW);
      draw();
      break;
    case ']':
      box.w = Math.min(1, box.w + sizeStep / imgW);
      draw();
      break;
    case '{':
      box.h = Math.max(0.02, box.h - sizeStep / imgH);
      draw();
      break;
    case '}':
      box.h = Math.min(1, box.h + sizeStep / imgH);
      draw();
      break;
    case 'Enter':
      e.preventDefault();
      confirmLabel();
      break;
    case 's':
    case 'S':
      skipImage();
      break;
    case 'z':
    case 'Z':
      goBack();
      break;
    case 'r':
    case 'R':
      resetBox();
      break;
    case 'q':
    case 'Q':
      fetch('/api/goto', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ index: state.current_index }),
      }).then(function() {
        document.body.textContent = '';
        var msg = document.createElement('div');
        msg.style.cssText = 'display:flex;align-items:center;justify-content:center;height:100vh;font-size:20px;color:#e0e0e0;';
        msg.textContent = 'Progress saved. You can close this tab.';
        document.body.appendChild(msg);
      });
      break;
  }
});

// Auto-resize canvas on window resize
window.addEventListener('resize', function() {
  if (img && imgW) {
    resizeCanvas();
    draw();
  }
});

// Start
loadState();
</script>
</body>
</html>
```

- [ ] **Step 2: Test the full annotation workflow manually**

Run:
```bash
cd /home/will/photo_project/yolo_finetune
source /home/will/photo_project/.venv/bin/activate
python annotate.py
```

Open `http://localhost:8888` in browser. Verify:
1. First photo loads and fills most of the viewport
2. Green bounding box is visible
3. Arrow keys move the box, `[` `]` resize width, `{` `}` resize height
4. Enter saves and advances (check `dataset/labels/` for `.txt` file)
5. S skips and advances (check `skipped.txt`)
6. Z goes back
7. Keyboard input works immediately on load without clicking

Label 2-3 images, skip 1, press Q. Restart server and verify it resumes.

- [ ] **Step 3: Commit**

```bash
git add index.html
git commit -m "feat: browser annotation UI with keyboard-only bounding box controls"
```

---

### Task 3: YOLO Training Script (`train.py`)

**Files:**
- Create: `train.py`

Reads annotations from `dataset/`, splits into train/val, creates `data.yaml`, runs YOLOv8 fine-tuning.

- [ ] **Step 1: Install ultralytics**

```bash
cd /home/will/photo_project
source .venv/bin/activate
pip install ultralytics>=8.0
```

- [ ] **Step 2: Create `train.py`**

```python
"""YOLO fine-tuning on annotated bounding box data."""

import random
import shutil
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
SKIPPED_FILE = BASE_DIR / "skipped.txt"
IMAGE_SOURCE = BASE_DIR.parent / "photo_mapping_samples"

SEED = 42
VAL_RATIO = 0.2


def setup_dataset():
    """Split labeled + skipped images into train/val with YOLO directory structure."""
    # Collect labeled images (those with a .txt in labels/)
    labeled = sorted(p.stem for p in LABELS_DIR.glob("*.txt"))
    if not labeled:
        print("No labels found in dataset/labels/. Run annotate.py first.")
        raise SystemExit(1)

    # Collect skipped images (negative examples)
    skipped = []
    if SKIPPED_FILE.exists():
        skipped = [
            line.strip() for line in SKIPPED_FILE.read_text().splitlines()
            if line.strip()
        ]
    skipped_stems = [Path(f).stem for f in skipped]

    print(f"Found {len(labeled)} labeled images, {len(skipped_stems)} skipped (negative examples)")

    # Combine all stems for splitting
    all_stems = labeled + skipped_stems
    random.seed(SEED)
    random.shuffle(all_stems)

    val_count = max(1, int(len(all_stems) * VAL_RATIO))
    val_stems = set(all_stems[:val_count])
    train_stems = set(all_stems[val_count:])

    # Create train/val directory structure
    for split in ("train", "val"):
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Clear old split symlinks
    for split in ("train", "val"):
        for f in (DATASET_DIR / "images" / split).iterdir():
            f.unlink()
        for f in (DATASET_DIR / "labels" / split).iterdir():
            f.unlink()

    labeled_set = set(labeled)

    for stem in train_stems | val_stems:
        split = "val" if stem in val_stems else "train"

        # Find source image
        src_img = None
        for ext in (".jpg", ".JPG"):
            candidate = IMAGE_SOURCE / f"{stem}{ext}"
            if candidate.exists():
                src_img = candidate
                break

        if src_img is None:
            print(f"  Warning: no source image for {stem}, skipping")
            continue

        # Symlink image
        dst_img = DATASET_DIR / "images" / split / src_img.name
        if not dst_img.exists():
            dst_img.symlink_to(src_img.resolve())

        # Copy label if this is a labeled image (not a negative example)
        if stem in labeled_set:
            src_label = LABELS_DIR / f"{stem}.txt"
            dst_label = DATASET_DIR / "labels" / split / f"{stem}.txt"
            if src_label.exists() and not dst_label.exists():
                shutil.copy2(src_label, dst_label)
        # Negative examples: image present, no label file — YOLO handles this correctly

    # Write data.yaml
    data_yaml = {
        "path": str(DATASET_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "target"},
    }
    yaml_path = DATASET_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    train_labeled = len(train_stems & labeled_set)
    val_labeled = len(val_stems & labeled_set)
    train_neg = len(train_stems) - train_labeled
    val_neg = len(val_stems) - val_labeled
    print(f"Train: {train_labeled} labeled + {train_neg} negative = {len(train_stems)}")
    print(f"Val:   {val_labeled} labeled + {val_neg} negative = {len(val_stems)}")
    print(f"Dataset config: {yaml_path}")

    return yaml_path


def train(data_yaml):
    """Run YOLOv8 fine-tuning."""
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(data_yaml),
        epochs=100,
        patience=10,
        imgsz=640,
        batch=8,
        device="cpu",
        project=str(BASE_DIR / "runs" / "detect"),
        name="train",
        exist_ok=True,
        verbose=True,
    )

    best = BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
    if best.exists():
        print(f"\nTraining complete! Best weights: {best}")
    else:
        print("\nTraining finished but no best.pt found — check logs above.")


def main():
    print("Setting up dataset...")
    data_yaml = setup_dataset()
    print("\nStarting YOLOv8 fine-tuning (CPU, this will take a while)...\n")
    train(data_yaml)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify dataset setup works (dry run)**

After labeling at least 3 images and skipping 1 via the annotation UI:

```bash
cd /home/will/photo_project/yolo_finetune
source /home/will/photo_project/.venv/bin/activate
python -c "
from train import setup_dataset
setup_dataset()
"
```

Expected: prints train/val split counts, creates `dataset/images/train/`, `dataset/images/val/`, `dataset/labels/train/`, `dataset/labels/val/`, and `dataset/data.yaml`.

Verify:
```bash
cat dataset/data.yaml
ls dataset/images/train/ | head -5
ls dataset/labels/train/ | head -5
```

- [ ] **Step 4: Commit**

```bash
git add train.py
git commit -m "feat: YOLO fine-tuning script with train/val split and negative examples"
```

---

### Task 4: Add ultralytics dependency and .gitignore

**Files:**
- Modify: `/home/will/photo_project/pyproject.toml`
- Create or modify: `.gitignore`

- [ ] **Step 1: Add ultralytics to pyproject.toml**

Add `ultralytics>=8.0` to the dependencies list in `/home/will/photo_project/pyproject.toml`:

```toml
[project]
name = "photo-project"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.14"
dependencies = [
    "accelerate>=1.13.0",
    "einops>=0.8.0",
    "pillow>=12.2.0",
    "timm>=0.9.0",
    "torch>=2.11.0",
    "torchvision>=0.26.0",
    "transformers<4.50",
    "ultralytics>=8.0",
]
```

- [ ] **Step 2: Create `.gitignore` for yolo_finetune**

```
dataset/
runs/
progress.json
skipped.txt
__pycache__/
*.pyc
.superpowers/
```

- [ ] **Step 3: Commit**

```bash
git add .gitignore
cd /home/will/photo_project && git add pyproject.toml
cd /home/will/photo_project/yolo_finetune
git commit -m "chore: add ultralytics dep, gitignore for dataset and training artifacts"
```

---

### Task 5: End-to-end verification

- [ ] **Step 1: Clean slate test**

```bash
cd /home/will/photo_project/yolo_finetune
rm -rf dataset/ progress.json skipped.txt runs/
```

- [ ] **Step 2: Run annotator and label 5 images, skip 2**

```bash
source /home/will/photo_project/.venv/bin/activate
python annotate.py
```

Open browser, label 5 images, skip 2, press Q.

Verify files:
```bash
ls dataset/labels/*.txt | wc -l   # should be 5
cat dataset/labels/*.txt           # each line: 0 cx cy w h
cat skipped.txt                    # should have 2 filenames
cat progress.json | python -m json.tool  # current_index should be 7
```

- [ ] **Step 3: Run training (short test)**

Verify dataset setup works:

```bash
python -c "
from train import setup_dataset
setup_dataset()
"
```

Then run a quick 2-epoch training test:

```bash
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='dataset/data.yaml', epochs=2, imgsz=640, batch=4, device='cpu', project='runs/detect', name='test', exist_ok=True)
print('Training pipeline works!')
"
```

Expected: Training starts, runs 2 epochs, produces output in `runs/detect/test/`.

- [ ] **Step 4: Clean up test artifacts**

```bash
rm -rf runs/detect/test/
```
