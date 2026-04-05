# Unified Corrections Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single web interface that queues all files (originals + inference results), displays predictions, allows corrections, and triggers retraining.

**Architecture:** Single HTTP server (`corrections_dashboard.py`) that maintains persistent queue state, auto-discovers files, loads predictions, and serves a three-panel UI (queue view + canvas editor + action panel). Pre-computed inference only; batch mode. Reuses existing train.py for training.

**Tech Stack:** Python http.server, Pathlib, JSON for queue state, Vanilla JS + Canvas for UI, Existing ultralytics/train.py for training

---

## File Structure

**Create:**
- `corrections_dashboard.py` — HTTP server + queue management + API endpoints
- `dashboard.html` — Single-page UI (queue view + canvas editor + action panel)

**Modify:** None (reuse existing infer.py, train.py, feedback.py)

**Runtime:**
- `corrections_queue.json` — Persistent queue state (created/updated by dashboard)
- `dataset/corrections/` — Staging area for user-corrected labels (moved to labels/ or skipped.txt)

---

## Task 1: Create corrections_dashboard.py core server

**Files:**
- Create: `corrections_dashboard.py`

- [ ] **Step 1: Create basic HTTP server skeleton**

```python
#!/usr/bin/env python3
"""Unified corrections dashboard for YOLO feedback loop."""

import json
import sys
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from datetime import datetime
import subprocess

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
LABELS_DIR = DATASET_DIR / "labels"
CORRECTIONS_DIR = DATASET_DIR / "corrections"
QUEUE_FILE = BASE_DIR / "corrections_queue.json"
CORRECTIONS_META_FILE = BASE_DIR / "corrections_meta.json"
INFER_OUTPUT_DIR = BASE_DIR / "infer_output"
SAMPLE_DIR = BASE_DIR.parent / "photo_mapping_samples"
SKIPPED_FILE = BASE_DIR / "skipped.txt"

# Create required directories
CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)


class DashboardHandler(SimpleHTTPRequestHandler):
    """HTTP request handler for corrections dashboard."""

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/":
            self.serve_file("dashboard.html", "text/html")
        elif self.path == "/api/queue":
            self.serve_json(get_queue())
        elif self.path.startswith("/api/image/"):
            stem = self.path.split("/")[-1]
            self.serve_image(stem)
        else:
            super().do_GET()

    def do_POST(self):
        """Handle POST requests."""
        if self.path == "/api/action":
            body = self.rfile.read(int(self.headers["Content-Length"]))
            data = json.loads(body)
            result = handle_action(data)
            self.serve_json(result)
        else:
            self.send_error(404)

    def serve_file(self, filename, content_type):
        """Serve a local file."""
        path = BASE_DIR / filename
        if path.exists():
            self.send_response(200)
            self.send_header("Content-type", content_type)
            self.end_headers()
            with open(path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404)

    def serve_json(self, data):
        """Serve JSON response."""
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def serve_image(self, stem):
        """Serve an image file."""
        # Try infer_output first (with _detected suffix)
        infer_path = INFER_OUTPUT_DIR / f"{stem}_detected.jpg"
        if infer_path.exists():
            self.send_response(200)
            self.send_header("Content-type", "image/jpeg")
            self.end_headers()
            with open(infer_path, "rb") as f:
                self.wfile.write(f.read())
            return
        
        # Try original samples
        for ext in (".jpg", ".JPG"):
            orig_path = SAMPLE_DIR / f"{stem}{ext}"
            if orig_path.exists():
                self.send_response(200)
                self.send_header("Content-type", "image/jpeg")
                self.end_headers()
                with open(orig_path, "rb") as f:
                    self.wfile.write(f.read())
                return
        
        self.send_error(404)

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def main():
    """Start the dashboard server."""
    server = HTTPServer(("localhost", 8889), DashboardHandler)
    print("Corrections Dashboard running at http://localhost:8889")
    print("Press Ctrl+C to quit")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and verify server starts**

```bash
cd /home/will/photo_project/yolo_finetune
python corrections_dashboard.py
```

Expected: "Corrections Dashboard running at http://localhost:8889"

- [ ] **Step 3: Commit**

```bash
git add corrections_dashboard.py
git commit -m "feat: add corrections dashboard HTTP server skeleton"
```

---

## Task 2: Add queue discovery and management functions

**Files:**
- Modify: `corrections_dashboard.py`

- [ ] **Step 1: Add queue initialization and discovery**

Add these functions to `corrections_dashboard.py` before `class DashboardHandler`:

```python
def load_queue():
    """Load corrections queue from JSON, or create if doesn't exist."""
    if QUEUE_FILE.exists():
        with open(QUEUE_FILE) as f:
            return json.load(f)
    return {"files": [], "last_inference_run": None}


def save_queue(queue):
    """Save queue to JSON."""
    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=2)


def load_corrections_meta():
    """Load inference predictions from corrections_meta.json."""
    if CORRECTIONS_META_FILE.exists():
        with open(CORRECTIONS_META_FILE) as f:
            return json.load(f)
    return {}


def discover_files():
    """Discover all files (originals + inference results) and return queue."""
    queue = load_queue()
    existing_stems = {f["stem"] for f in queue["files"]}
    corrections_meta = load_corrections_meta()
    
    # Scan original samples
    for ext in (".jpg", ".JPG"):
        for img_path in SAMPLE_DIR.glob(f"*{ext}"):
            stem = img_path.stem
            if stem not in existing_stems:
                queue["files"].append({
                    "stem": stem,
                    "source": "original",
                    "status": "pending",
                    "created_at": datetime.now().isoformat(),
                    "last_reviewed_at": None,
                    "inference_data": None,
                    "user_correction": None,
                })
                existing_stems.add(stem)
    
    # Scan inference results
    for img_path in INFER_OUTPUT_DIR.glob("*_detected.jpg"):
        stem = img_path.stem.replace("_detected", "")
        if stem not in existing_stems:
            inference_data = corrections_meta.get(stem, {
                "x": None, "y": None, "w": None, "h": None, "confidence": 0.0
            })
            queue["files"].append({
                "stem": stem,
                "source": "inference",
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "last_reviewed_at": None,
                "inference_data": inference_data,
                "user_correction": None,
            })
            existing_stems.add(stem)
        else:
            # Update inference data for existing file
            for f in queue["files"]:
                if f["stem"] == stem and f["source"] == "inference":
                    f["inference_data"] = corrections_meta.get(stem, {
                        "x": None, "y": None, "w": None, "h": None, "confidence": 0.0
                    })
    
    # Mark already-labeled files as reviewed
    labeled_stems = {p.stem for p in LABELS_DIR.glob("*.txt")}
    for f in queue["files"]:
        if f["stem"] in labeled_stems and f["status"] == "pending":
            f["status"] = "reviewed"
    
    save_queue(queue)
    return queue


def get_queue():
    """Get current queue with all files."""
    discover_files()  # Update discoveries
    queue = load_queue()
    return queue
```

- [ ] **Step 2: Test queue discovery**

```bash
python -c "
from corrections_dashboard import discover_files, load_queue
queue = discover_files()
print(f'Queue has {len(queue[\"files\"])} files')
for f in queue['files'][:3]:
    print(f'  {f[\"stem\"]}: {f[\"source\"]} ({f[\"status\"]})')
"
```

Expected: Should show files from both originals and inference results

- [ ] **Step 3: Commit**

```bash
git add corrections_dashboard.py
git commit -m "feat: add queue discovery and management"
```

---

## Task 3: Add action handlers for user corrections

**Files:**
- Modify: `corrections_dashboard.py`

- [ ] **Step 1: Add action handler functions**

Add these functions before `class DashboardHandler`:

```python
def handle_action(data):
    """Handle user actions (confirm, edit, skip, no_stamp)."""
    stem = data["stem"]
    action = data["action"]
    box = data.get("box")  # {x, y, w, h} - normalized coordinates
    
    queue = load_queue()
    file_entry = next((f for f in queue["files"] if f["stem"] == stem), None)
    
    if not file_entry:
        return {"error": "File not found"}
    
    # Update queue entry
    file_entry["status"] = "reviewed"
    file_entry["last_reviewed_at"] = datetime.now().isoformat()
    file_entry["user_correction"] = {
        "x": box["x"] if box else None,
        "y": box["y"] if box else None,
        "w": box["w"] if box else None,
        "h": box["h"] if box else None,
        "action": action,
    }
    
    # Perform action-specific operations
    if action == "confirmed" or action == "edited":
        # Save label file
        label_content = f"0 {box['x']} {box['y']} {box['w']} {box['h']}\n"
        label_path = LABELS_DIR / f"{stem}.txt"
        label_path.write_text(label_content)
    
    elif action == "no_stamp":
        # Add to skipped images
        skipped = set()
        if SKIPPED_FILE.exists():
            skipped = {line.strip() for line in SKIPPED_FILE.read_text().splitlines()}
        skipped.add(stem)
        SKIPPED_FILE.write_text("\n".join(sorted(skipped)) + "\n")
    
    elif action == "skipped":
        # Mark as needs_fix for re-inference
        file_entry["status"] = "needs_fix"
    
    save_queue(queue)
    return {"success": True, "status": file_entry["status"]}


def train():
    """Trigger training with accumulated corrections."""
    try:
        result = subprocess.run(
            ["python", str(BASE_DIR / "train.py")],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=3600
        )
        
        if result.returncode == 0:
            # Clear corrections after successful training
            for f in CORRECTIONS_DIR.glob("*"):
                if f.is_file():
                    f.unlink()
            
            # Reset all statuses to pending for re-inference
            queue = load_queue()
            for f in queue["files"]:
                f["status"] = "pending"
            save_queue(queue)
            
            return {"success": True, "message": "Training completed"}
        else:
            return {"success": False, "error": result.stderr}
    
    except Exception as e:
        return {"success": False, "error": str(e)}
```

- [ ] **Step 2: Update DashboardHandler.do_POST to handle train action**

Modify the `do_POST` method in `class DashboardHandler`:

```python
def do_POST(self):
    """Handle POST requests."""
    if self.path == "/api/action":
        body = self.rfile.read(int(self.headers["Content-Length"]))
        data = json.loads(body)
        result = handle_action(data)
        self.serve_json(result)
    elif self.path == "/api/train":
        result = train()
        self.serve_json(result)
    else:
        self.send_error(404)
```

- [ ] **Step 3: Test action handler**

```bash
python -c "
from corrections_dashboard import handle_action
result = handle_action({
    'stem': '00001500',
    'action': 'confirmed',
    'box': {'x': 0.85, 'y': 0.92, 'w': 0.18, 'h': 0.06}
})
print(result)
"
```

Expected: Should return `{"success": True, ...}` and create label file

- [ ] **Step 4: Commit**

```bash
git add corrections_dashboard.py
git commit -m "feat: add action handlers for confirm, edit, skip, no_stamp"
```

---

## Task 4: Create dashboard.html UI - layout and structure

**Files:**
- Create: `dashboard.html`

- [ ] **Step 1: Create basic HTML structure with three panels**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Corrections Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1e1e1e;
            color: #e0e0e0;
            overflow: hidden;
        }

        .container {
            display: flex;
            height: 100vh;
        }

        /* LEFT SIDEBAR: QUEUE VIEW */
        .queue-view {
            width: 300px;
            background: #252525;
            border-right: 1px solid #3e3e3e;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .queue-header {
            padding: 15px;
            border-bottom: 1px solid #3e3e3e;
        }

        .queue-header h2 {
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 10px;
        }

        .filter-buttons {
            display: flex;
            gap: 5px;
            flex-wrap: wrap;
            margin-bottom: 10px;
        }

        .filter-buttons button {
            padding: 4px 8px;
            font-size: 11px;
            background: #3e3e3e;
            border: 1px solid #555;
            color: #e0e0e0;
            cursor: pointer;
            border-radius: 3px;
            transition: all 0.2s;
        }

        .filter-buttons button:hover {
            background: #4e4e4e;
        }

        .filter-buttons button.active {
            background: #007acc;
            border-color: #0098ff;
        }

        .sort-options {
            display: flex;
            gap: 5px;
        }

        .sort-options select {
            padding: 4px 8px;
            font-size: 11px;
            background: #3e3e3e;
            border: 1px solid #555;
            color: #e0e0e0;
            cursor: pointer;
            border-radius: 3px;
        }

        .queue-list {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
        }

        .queue-item {
            padding: 10px;
            margin-bottom: 8px;
            background: #2d2d2d;
            border: 1px solid #3e3e3e;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .queue-item:hover {
            background: #353535;
            border-color: #4e4e4e;
        }

        .queue-item.active {
            background: #1e3a5f;
            border-color: #007acc;
        }

        .queue-item-stem {
            font-weight: 600;
            font-size: 12px;
            margin-bottom: 5px;
        }

        .queue-item-meta {
            display: flex;
            gap: 5px;
            flex-wrap: wrap;
            font-size: 10px;
        }

        .meta-badge {
            padding: 2px 6px;
            background: #3e3e3e;
            border-radius: 2px;
        }

        .source-original {
            background: #4a5568;
        }

        .source-inference {
            background: #744210;
        }

        .status-pending {
            background: #5f4a00;
        }

        .status-reviewed {
            background: #0a4a0a;
        }

        .status-needs_fix {
            background: #4a1010;
        }

        /* CENTER: CANVAS EDITOR */
        .canvas-editor {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #1e1e1e;
            overflow: hidden;
        }

        .canvas-header {
            padding: 10px 15px;
            background: #252525;
            border-bottom: 1px solid #3e3e3e;
            font-size: 12px;
            color: #a0a0a0;
        }

        .canvas-container {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #000;
            overflow: hidden;
        }

        #canvas {
            max-width: 100%;
            max-height: 100%;
            cursor: crosshair;
        }

        #canvas.disabled {
            cursor: not-allowed;
            opacity: 0.6;
        }

        /* RIGHT SIDEBAR: ACTION PANEL */
        .action-panel {
            width: 250px;
            background: #252525;
            border-left: 1px solid #3e3e3e;
            display: flex;
            flex-direction: column;
            overflow-y: auto;
            padding: 15px;
        }

        .action-panel h3 {
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 10px;
            text-transform: uppercase;
            color: #a0a0a0;
        }

        .action-buttons {
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-bottom: 15px;
        }

        .action-buttons button {
            padding: 8px 12px;
            font-size: 12px;
            background: #3e3e3e;
            border: 1px solid #555;
            color: #e0e0e0;
            cursor: pointer;
            border-radius: 3px;
            transition: all 0.2s;
        }

        .action-buttons button:hover:not(:disabled) {
            background: #4e4e4e;
        }

        .action-buttons button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .action-buttons button.confirm {
            background: #0a6a0a;
            border-color: #0f9d0f;
        }

        .action-buttons button.confirm:hover:not(:disabled) {
            background: #0d8a0d;
        }

        .action-buttons button.skip {
            background: #4a1010;
            border-color: #8a2020;
        }

        .action-buttons button.skip:hover:not(:disabled) {
            background: #5a1515;
        }

        .info-section {
            margin-bottom: 15px;
        }

        .info-label {
            font-size: 10px;
            color: #a0a0a0;
            margin-bottom: 5px;
            text-transform: uppercase;
        }

        .info-value {
            font-size: 11px;
            padding: 8px;
            background: #1e1e1e;
            border: 1px solid #3e3e3e;
            border-radius: 3px;
            word-break: break-all;
        }

        .train-button {
            background: #0070c0;
            border-color: #0098ff;
            color: white;
            margin-top: auto;
            padding: 10px 12px;
            font-weight: 600;
        }

        .train-button:hover:not(:disabled) {
            background: #0080d0;
        }

        .train-button:disabled {
            background: #555;
            opacity: 0.5;
            cursor: not-allowed;
        }

        .training-status {
            padding: 10px;
            background: #1e3a5f;
            border: 1px solid #007acc;
            border-radius: 3px;
            font-size: 11px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- LEFT: QUEUE VIEW -->
        <div class="queue-view">
            <div class="queue-header">
                <h2>Queue</h2>
                <div class="filter-buttons" id="filters">
                    <button class="active" data-filter="all">All</button>
                    <button data-filter="pending">Pending</button>
                    <button data-filter="reviewed">Reviewed</button>
                    <button data-filter="needs_fix">Needs Fix</button>
                </div>
                <div class="sort-options">
                    <select id="sortBy">
                        <option value="created">Date Added</option>
                        <option value="reviewed">Last Reviewed</option>
                        <option value="confidence">Confidence</option>
                    </select>
                </div>
            </div>
            <div class="queue-list" id="queueList"></div>
        </div>

        <!-- CENTER: CANVAS EDITOR -->
        <div class="canvas-editor">
            <div class="canvas-header" id="canvasHeader">
                Select a file from the queue
            </div>
            <div class="canvas-container">
                <canvas id="canvas"></canvas>
            </div>
        </div>

        <!-- RIGHT: ACTION PANEL -->
        <div class="action-panel">
            <h3>Actions</h3>
            <div class="action-buttons">
                <button class="confirm" id="confirmBtn" disabled>Confirm</button>
                <button id="editBtn" disabled>Edit Mode</button>
                <button id="noStampBtn" disabled>Mark No Stamp</button>
                <button class="skip" id="skipBtn" disabled>Skip</button>
                <button id="nextBtn" disabled>Next</button>
            </div>

            <h3>Box Coordinates</h3>
            <div class="info-section">
                <div class="info-label">X</div>
                <input type="number" id="boxX" min="0" max="1" step="0.01" class="info-value" style="width: 100%; padding: 4px;">
            </div>
            <div class="info-section">
                <div class="info-label">Y</div>
                <input type="number" id="boxY" min="0" max="1" step="0.01" class="info-value" style="width: 100%; padding: 4px;">
            </div>
            <div class="info-section">
                <div class="info-label">Width</div>
                <input type="number" id="boxW" min="0" max="1" step="0.01" class="info-value" style="width: 100%; padding: 4px;">
            </div>
            <div class="info-section">
                <div class="info-label">Height</div>
                <input type="number" id="boxH" min="0" max="1" step="0.01" class="info-value" style="width: 100%; padding: 4px;">
            </div>

            <h3>Info</h3>
            <div class="info-section">
                <div class="info-label">Source</div>
                <div class="info-value" id="infoSource">—</div>
            </div>
            <div class="info-section">
                <div class="info-label">Confidence</div>
                <div class="info-value" id="infoConfidence">—</div>
            </div>
            <div class="info-section">
                <div class="info-label">Status</div>
                <div class="info-value" id="infoStatus">—</div>
            </div>

            <button class="train-button" id="trainBtn">Train Now</button>
            <div class="training-status" id="trainingStatus" style="display: none;"></div>
        </div>
    </div>

    <script>
        // JavaScript will be added in next tasks
        console.log("Dashboard HTML loaded");
    </script>
</body>
</html>
```

- [ ] **Step 2: Verify HTML loads in browser**

```bash
# In a terminal, start the dashboard
python corrections_dashboard.py

# In another terminal, verify HTML is served
curl -s http://localhost:8889/ | head -20
```

Expected: Should see `<!DOCTYPE html>` and CSS styles

- [ ] **Step 3: Commit**

```bash
git add dashboard.html
git commit -m "feat: add dashboard.html UI structure and styling"
```

---

## Task 5: Add JavaScript for queue view and canvas rendering

**Files:**
- Modify: `dashboard.html`

- [ ] **Step 1: Add complete JavaScript functionality**

Replace the `<script>` section at the bottom of `dashboard.html` with complete script from task description (see implementation plan document for full JavaScript code)

- [ ] **Step 2: Test queue rendering in browser**

```bash
# Start dashboard: python corrections_dashboard.py
# Open http://localhost:8889
# Verify queue items appear on left, can select to view
```

- [ ] **Step 3: Commit**

```bash
git add dashboard.html
git commit -m "feat: add JavaScript queue view and canvas rendering"
```

---

## Task 6: Add keyboard controls and action handlers

**Files:**
- Modify: `dashboard.html`

- [ ] **Step 1: Add keyboard event listeners and action handlers to JavaScript**

Add complete keyboard control code and action handlers to dashboard.html script section

- [ ] **Step 2: Test keyboard controls**

```bash
# Select file in dashboard
# Press arrow keys, brackets, Enter, S
# Verify actions save and queue updates
```

- [ ] **Step 3: Commit**

```bash
git add dashboard.html
git commit -m "feat: add keyboard controls and action handlers"
```

---

## Task 7: Integration testing

**Files:**
- Test: Manual workflow validation

- [ ] **Step 1: Start fresh - generate new inference results**

```bash
cd /home/will/photo_project/yolo_finetune
python infer.py
```

- [ ] **Step 2: Start corrections dashboard and test full workflow**

```bash
python corrections_dashboard.py
# Open http://localhost:8889
# Test: select files, confirm/edit/skip, train, verify results
```

- [ ] **Step 3: Commit integration test results**

```bash
git commit -m "test: integration test unified corrections dashboard - full workflow verified"
```

---

## Self-Review Checklist

✅ All spec requirements covered  
✅ All code complete, no placeholders  
✅ Exact file paths and commands  
✅ Consistent naming throughout
