# Continuous Review Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable continuous review of 7,455 ScanMyPhotos images with background training + re-inference for iterative model improvement.

**Architecture:** Extend existing corrections dashboard to use `scanmyphotos/` as image source (disc-prefixed names), add `infer_all.py` for batch inference, background worker subprocess for train+infer, progress polling via status file.

**Tech Stack:** Python http.server, ultralytics YOLO, vanilla JS + Canvas, subprocess for background work

---

## File Structure

**Create:**
- `infer_all.py` -- batch inference on scanmyphotos/, writes predictions JSON + worker status

**Modify:**
- `corrections_dashboard.py` -- new image source, queue from scanmyphotos/, worker status API, background train+infer
- `dashboard.html` -- progress bar, auto-select first pending, worker status polling, default filter pending
- `train.py` -- use scanmyphotos/ as IMAGE_SOURCE instead of photo_mapping_samples/

---

## Task 1: Create infer_all.py batch inference script

**Files:**
- Create: `infer_all.py`

- [ ] **Step 1: Create infer_all.py**

```python
#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["ultralytics>=8.0", "opencv-python-headless"]
# ///
"""Batch inference on all ScanMyPhotos images. Writes predictions + worker status."""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
SCANMYPHOTOS_DIR = BASE_DIR / "scanmyphotos"
LABELS_DIR = BASE_DIR / "dataset" / "labels"
SKIPPED_FILE = BASE_DIR / "skipped.txt"
PREDICTIONS_FILE = BASE_DIR / "scanmyphotos_predictions.json"
STATUS_FILE = BASE_DIR / "worker_status.json"
MODEL_PATH = BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"

INFERENCE_SIZE = 384
CONF_THRESHOLD = 0.3


def write_status(phase, progress, total, message):
    """Write worker status for dashboard polling."""
    with open(STATUS_FILE, "w") as f:
        json.dump({"phase": phase, "progress": progress, "total": total, "message": message}, f)


def get_pending_files():
    """Get list of files that need inference (no label, not skipped)."""
    labeled_stems = {p.stem for p in LABELS_DIR.glob("*.txt")}

    skipped_stems = set()
    if SKIPPED_FILE.exists():
        skipped_stems = {line.strip() for line in SKIPPED_FILE.read_text().splitlines() if line.strip()}

    reviewed = labeled_stems | skipped_stems

    all_images = sorted(SCANMYPHOTOS_DIR.glob("*.jpg"))
    pending = [img for img in all_images if img.stem not in reviewed]
    return pending


def run_inference():
    """Run YOLO inference on all pending files."""
    from ultralytics import YOLO

    if not MODEL_PATH.exists():
        write_status("error", 0, 0, f"Model not found: {MODEL_PATH}")
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return

    pending = get_pending_files()
    total = len(pending)

    if total == 0:
        write_status("done", 0, 0, "No pending files to infer.")
        print("No pending files to infer.")
        return

    write_status("inference", 0, total, f"Loading model...")
    print(f"Running inference on {total} pending files...")

    model = YOLO(str(MODEL_PATH))

    # Load existing predictions to merge with
    predictions = {}
    if PREDICTIONS_FILE.exists():
        with open(PREDICTIONS_FILE) as f:
            predictions = json.load(f)

    for i, img_path in enumerate(pending):
        stem = img_path.stem

        results = model(
            str(img_path),
            imgsz=INFERENCE_SIZE,
            conf=CONF_THRESHOLD,
            device="cpu",
            verbose=False,
        )

        result = results[0]
        if len(result.boxes) > 0:
            # Take highest confidence detection
            best_idx = result.boxes.conf.argmax()
            box = result.boxes[best_idx]

            # Convert to YOLO normalized format (center x, center y, w, h)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            img_h, img_w = result.orig_shape
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h
            conf = float(box.conf[0])

            predictions[stem] = {
                "x": round(float(cx), 6),
                "y": round(float(cy), 6),
                "w": round(float(bw), 6),
                "h": round(float(bh), 6),
                "confidence": round(conf, 4),
            }

        if (i + 1) % 50 == 0 or (i + 1) == total:
            write_status("inference", i + 1, total, f"{i + 1}/{total} inferred")
            # Save predictions incrementally every 50
            with open(PREDICTIONS_FILE, "w") as f:
                json.dump(predictions, f, indent=2)
            print(f"  [{i + 1}/{total}] {stem}")

    # Final save
    with open(PREDICTIONS_FILE, "w") as f:
        json.dump(predictions, f, indent=2)

    write_status("done", total, total, f"Inference complete. {len(predictions)} predictions saved.")
    print(f"Done. {len(predictions)} total predictions in {PREDICTIONS_FILE.name}")


if __name__ == "__main__":
    run_inference()
```

- [ ] **Step 2: Verify it runs (dry check -- model may not exist yet)**

```bash
uv run infer_all.py
```

Expected: Either runs inference or prints "Model not found" if no trained model exists yet.

- [ ] **Step 3: Commit**

```bash
git add infer_all.py
git commit -m "feat: add infer_all.py for batch inference on ScanMyPhotos"
```

---

## Task 2: Update train.py to use scanmyphotos/ as image source

**Files:**
- Modify: `train.py`

- [ ] **Step 1: Change IMAGE_SOURCE and update image lookup**

In `train.py`, change line 21:

```python
# Old:
IMAGE_SOURCE = BASE_DIR.parent / "photo_mapping_samples"

# New:
IMAGE_SOURCE = BASE_DIR / "scanmyphotos"
```

- [ ] **Step 2: Update image file extension search**

In `setup_dataset()`, the image lookup at lines 100-105 searches for `.jpg` and `.JPG`. Since scanmyphotos/ files are all lowercase `.jpg`, this still works. No change needed.

- [ ] **Step 3: Commit**

```bash
git add train.py
git commit -m "feat: update train.py to use scanmyphotos/ as image source"
```

---

## Task 3: Rewrite corrections_dashboard.py for continuous review

**Files:**
- Modify: `corrections_dashboard.py`

- [ ] **Step 1: Update constants and imports**

Replace the constants block (lines 14-22) with:

```python
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
LABELS_DIR = DATASET_DIR / "labels"
SCANMYPHOTOS_DIR = BASE_DIR / "scanmyphotos"
QUEUE_FILE = BASE_DIR / "corrections_queue.json"
PREDICTIONS_FILE = BASE_DIR / "scanmyphotos_predictions.json"
SKIPPED_FILE = BASE_DIR / "skipped.txt"
STATUS_FILE = BASE_DIR / "worker_status.json"

# Create required directories
LABELS_DIR.mkdir(parents=True, exist_ok=True)
```

Remove the old `CORRECTIONS_DIR`, `CORRECTIONS_META_FILE`, `INFER_OUTPUT_DIR`, `SAMPLE_DIR` constants and the `CORRECTIONS_DIR.mkdir()` call.

- [ ] **Step 2: Replace discover_files() and get_queue()**

Replace the `load_queue`, `save_queue`, `load_corrections_meta`, `discover_files`, and `get_queue` functions with:

```python
def load_queue():
    """Load corrections queue from JSON, or create if doesn't exist."""
    if QUEUE_FILE.exists():
        with open(QUEUE_FILE) as f:
            return json.load(f)
    return {"files": [], "stats": {}}


def save_queue(queue):
    """Save queue to JSON."""
    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=2)


def load_predictions():
    """Load model predictions from scanmyphotos_predictions.json."""
    if PREDICTIONS_FILE.exists():
        with open(PREDICTIONS_FILE) as f:
            return json.load(f)
    return {}


def discover_files():
    """Build queue from scanmyphotos/ directory."""
    queue = load_queue()
    existing_stems = {f["stem"] for f in queue["files"]}
    predictions = load_predictions()

    # Load reviewed status
    labeled_stems = {p.stem for p in LABELS_DIR.glob("*.txt")}
    skipped_stems = set()
    if SKIPPED_FILE.exists():
        skipped_stems = {line.strip() for line in SKIPPED_FILE.read_text().splitlines() if line.strip()}

    # Scan scanmyphotos/ for all images
    for img_path in sorted(SCANMYPHOTOS_DIR.glob("*.jpg")):
        stem = img_path.stem
        if stem in existing_stems:
            continue

        # Parse disc number from stem (e.g., d1_00000001 -> 1)
        disc = int(stem.split("_")[0][1:]) if stem.startswith("d") else 0
        original_num = stem.split("_", 1)[1] if "_" in stem else stem
        display_name = f"D{disc} - {original_num}"

        status = "pending"
        if stem in labeled_stems:
            status = "labeled"
        elif stem in skipped_stems:
            status = "no_stamp"

        queue["files"].append({
            "stem": stem,
            "disc": disc,
            "display_name": display_name,
            "status": status,
            "created_at": datetime.now().isoformat(),
            "last_reviewed_at": None,
            "prediction": predictions.get(stem),
            "user_correction": None,
        })
        existing_stems.add(stem)

    # Update predictions and statuses for existing entries
    for f in queue["files"]:
        # Refresh prediction data
        if f["stem"] in predictions:
            f["prediction"] = predictions[f["stem"]]

        # Refresh status from disk
        if f["stem"] in labeled_stems and f["status"] == "pending":
            f["status"] = "labeled"
        elif f["stem"] in skipped_stems and f["status"] == "pending":
            f["status"] = "no_stamp"

    # Compute stats
    statuses = [f["status"] for f in queue["files"]]
    queue["stats"] = {
        "total": len(queue["files"]),
        "pending": statuses.count("pending"),
        "labeled": statuses.count("labeled"),
        "no_stamp": statuses.count("no_stamp"),
        "skipped": statuses.count("skipped"),
    }

    save_queue(queue)
    return queue


def get_queue():
    """Get current queue, discovering new files."""
    return discover_files()
```

- [ ] **Step 3: Update handle_action()**

Replace handle_action with:

```python
def handle_action(data):
    """Handle user actions (confirm, edit, skip, no_stamp)."""
    stem = data["stem"]
    action = data["action"]
    box = data.get("box")

    queue = load_queue()
    file_entry = next((f for f in queue["files"] if f["stem"] == stem), None)

    if not file_entry:
        return {"error": "File not found"}

    file_entry["last_reviewed_at"] = datetime.now().isoformat()
    file_entry["user_correction"] = {
        "x": box["x"] if box else None,
        "y": box["y"] if box else None,
        "w": box["w"] if box else None,
        "h": box["h"] if box else None,
        "action": action,
    }

    if action in ("confirmed", "edited"):
        label_content = f"0 {box['x']} {box['y']} {box['w']} {box['h']}\n"
        label_path = LABELS_DIR / f"{stem}.txt"
        label_path.write_text(label_content)
        file_entry["status"] = "labeled"

    elif action == "no_stamp":
        skipped = set()
        if SKIPPED_FILE.exists():
            skipped = {line.strip() for line in SKIPPED_FILE.read_text().splitlines() if line.strip()}
        skipped.add(stem)
        SKIPPED_FILE.write_text("\n".join(sorted(skipped)) + "\n")
        file_entry["status"] = "no_stamp"

    elif action == "skipped":
        file_entry["status"] = "skipped"

    # Recompute stats
    statuses = [f["status"] for f in queue["files"]]
    queue["stats"] = {
        "total": len(queue["files"]),
        "pending": statuses.count("pending"),
        "labeled": statuses.count("labeled"),
        "no_stamp": statuses.count("no_stamp"),
        "skipped": statuses.count("skipped"),
    }

    save_queue(queue)
    return {"success": True, "status": file_entry["status"], "stats": queue["stats"]}
```

- [ ] **Step 4: Replace train() with background worker**

Replace the `train()` function with:

```python
# Global worker process reference
_worker_process = None


def start_worker():
    """Start background train + infer worker."""
    global _worker_process

    if _worker_process and _worker_process.poll() is None:
        return {"error": "Worker already running"}

    # Write initial status
    with open(STATUS_FILE, "w") as f:
        json.dump({"phase": "starting", "progress": 0, "total": 0, "message": "Starting training..."}, f)

    # Launch background subprocess that runs train.py then infer_all.py
    _worker_process = subprocess.Popen(
        [
            "bash", "-c",
            f"uv run {BASE_DIR / 'train.py'} && uv run {BASE_DIR / 'infer_all.py'}"
        ],
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    return {"success": True, "message": "Worker started"}


def get_worker_status():
    """Read current worker status from file."""
    global _worker_process

    status = {"phase": "idle", "progress": 0, "total": 0, "message": "No worker running"}

    if STATUS_FILE.exists():
        try:
            with open(STATUS_FILE) as f:
                status = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    # Check if process has exited
    if _worker_process and _worker_process.poll() is not None:
        _worker_process = None
        if status.get("phase") not in ("done", "error"):
            status = {"phase": "error", "progress": 0, "total": 0, "message": "Worker exited unexpectedly"}

    return status
```

- [ ] **Step 5: Update DashboardHandler routes**

Replace the `do_GET` and `do_POST` methods with:

```python
    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/":
            self.serve_file("dashboard.html", "text/html")
        elif self.path == "/api/queue":
            self.serve_json(get_queue())
        elif self.path == "/api/worker-status":
            self.serve_json(get_worker_status())
        elif self.path.startswith("/api/image/"):
            stem = self.path.split("/")[-1]
            self.serve_image(stem)
        else:
            super().do_GET()

    def do_POST(self):
        """Handle POST requests."""
        if self.path == "/api/action":
            content_length = self.headers.get("Content-Length")
            if not content_length:
                self.send_error(400, "Missing Content-Length header")
                return

            try:
                body = self.rfile.read(int(content_length))
                data = json.loads(body)
            except (ValueError, json.JSONDecodeError):
                self.send_error(400, "Invalid Content-Length or JSON")
                return

            result = handle_action(data)
            self.serve_json(result)
        elif self.path == "/api/train":
            result = start_worker()
            self.serve_json(result)
        else:
            self.send_error(404)
```

- [ ] **Step 6: Update serve_image() to use scanmyphotos/**

Replace `serve_image` with:

```python
    def serve_image(self, stem):
        """Serve an image file from scanmyphotos/."""
        img_path = SCANMYPHOTOS_DIR / f"{stem}.jpg"
        if img_path.exists():
            self.send_response(200)
            self.send_header("Content-type", "image/jpeg")
            self.end_headers()
            with open(img_path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404)
```

- [ ] **Step 7: Delete old corrections_queue.json to reset queue**

```bash
rm -f corrections_queue.json
```

- [ ] **Step 8: Verify server starts and queue loads**

```bash
uv run corrections_dashboard.py &
sleep 2
curl -s http://localhost:8889/api/queue | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Files: {d[\"stats\"][\"total\"]}, Pending: {d[\"stats\"][\"pending\"]}')"
kill %1
```

Expected: `Files: 7455, Pending: <some number>`

- [ ] **Step 9: Commit**

```bash
git add corrections_dashboard.py
git commit -m "feat: rewrite dashboard backend for continuous ScanMyPhotos review"
```

---

## Task 4: Rewrite dashboard.html for continuous review

**Files:**
- Modify: `dashboard.html`

- [ ] **Step 1: Add progress bar CSS**

After the `.training-status` CSS block (around line 300), add:

```css
        .progress-bar {
            padding: 8px 15px;
            background: #252525;
            border-bottom: 1px solid #3e3e3e;
            font-size: 11px;
            color: #a0a0a0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .progress-bar .stats {
            display: flex;
            gap: 15px;
        }

        .progress-bar .stat-value {
            color: #e0e0e0;
            font-weight: 600;
        }

        .worker-status {
            padding: 8px 15px;
            background: #1e3a5f;
            border-bottom: 1px solid #007acc;
            font-size: 11px;
            display: none;
        }

        .worker-status .progress-fill {
            height: 4px;
            background: #007acc;
            border-radius: 2px;
            margin-top: 4px;
            transition: width 0.3s;
        }
```

- [ ] **Step 2: Add progress bar and worker status HTML**

After the opening `<div class="container">` and before the queue-view div, add a wrapper. Replace the canvas-editor section's canvas-header div to include the progress bar above it.

Actually, insert the progress bar right after `<body>` and before `<div class="container">`:

```html
    <div class="progress-bar" id="progressBar">
        <div class="stats">
            <span>Reviewed: <span class="stat-value" id="statReviewed">0</span> / <span class="stat-value" id="statTotal">0</span></span>
            <span>Labels: <span class="stat-value" id="statLabeled">0</span></span>
            <span>No stamp: <span class="stat-value" id="statNoStamp">0</span></span>
            <span>Skipped: <span class="stat-value" id="statSkipped">0</span></span>
        </div>
        <span id="statPercent">0%</span>
    </div>
    <div class="worker-status" id="workerStatus">
        <span id="workerMessage">Idle</span>
        <div style="background: #333; border-radius: 2px; height: 4px;">
            <div class="progress-fill" id="workerProgress" style="width: 0%;"></div>
        </div>
    </div>
```

And update `.container` height to account for progress bar:

```css
        .container {
            display: flex;
            height: calc(100vh - 30px);
        }
```

- [ ] **Step 3: Update filter buttons -- default to pending, change "Reviewed" to "Labeled"**

Replace the filter-buttons div:

```html
                <div class="filter-buttons" id="filters">
                    <button data-filter="all">All</button>
                    <button class="active" data-filter="pending">Pending</button>
                    <button data-filter="labeled">Labeled</button>
                    <button data-filter="no_stamp">No Stamp</button>
                    <button data-filter="skipped">Skipped</button>
                </div>
```

- [ ] **Step 4: Update "Train Now" button text**

Change:
```html
            <button class="train-button" id="trainBtn">Train Now</button>
```
To:
```html
            <button class="train-button" id="trainBtn">Train + Re-infer</button>
```

- [ ] **Step 5: Replace the entire `<script>` section**

Replace everything between `<script>` and `</script>` with:

```javascript
        // Global state
        let queue = { files: [], stats: {} };
        let currentFilter = "pending";
        let currentSort = "created";
        let selectedFile = null;
        let currentImage = null;
        let currentBox = { x: 0.5, y: 0.9, w: 0.2, h: 0.06 };
        let workerPolling = null;

        const canvas = document.getElementById("canvas");
        const ctx = canvas.getContext("2d");

        // Initialize dashboard
        async function initDashboard() {
            try {
                const response = await fetch("/api/queue");
                queue = await response.json();
                renderQueueList();
                updateProgressBar();
            } catch (error) {
                console.error("Error loading queue:", error);
            }
        }

        // Update progress bar
        function updateProgressBar() {
            const s = queue.stats || {};
            const reviewed = (s.labeled || 0) + (s.no_stamp || 0) + (s.skipped || 0);
            const total = s.total || 0;
            const pct = total > 0 ? ((reviewed / total) * 100).toFixed(1) : "0.0";

            document.getElementById("statReviewed").textContent = reviewed;
            document.getElementById("statTotal").textContent = total;
            document.getElementById("statLabeled").textContent = s.labeled || 0;
            document.getElementById("statNoStamp").textContent = s.no_stamp || 0;
            document.getElementById("statSkipped").textContent = s.skipped || 0;
            document.getElementById("statPercent").textContent = pct + "%";
        }

        // Render queue list
        function renderQueueList() {
            const queueList = document.getElementById("queueList");
            queueList.textContent = "";

            const filteredQueue = queue.files.filter(f => {
                if (currentFilter === "all") return true;
                return f.status === currentFilter;
            });

            const sortedQueue = sortQueue(filteredQueue);

            for (const file of sortedQueue) {
                const item = document.createElement("div");
                item.className = "queue-item";
                if (selectedFile && selectedFile.stem === file.stem) {
                    item.classList.add("active");
                }

                const confidence = file.prediction?.confidence || 0;
                const confidenceStr = confidence > 0 ? (confidence * 100).toFixed(0) + "%" : "";

                const stemDiv = document.createElement("div");
                stemDiv.className = "queue-item-stem";
                stemDiv.textContent = file.display_name || file.stem;
                item.appendChild(stemDiv);

                const metaDiv = document.createElement("div");
                metaDiv.className = "queue-item-meta";

                const statusBadge = document.createElement("span");
                statusBadge.className = "meta-badge status-" + file.status;
                statusBadge.textContent = file.status;
                metaDiv.appendChild(statusBadge);

                if (confidenceStr) {
                    const confBadge = document.createElement("span");
                    confBadge.className = "meta-badge";
                    confBadge.textContent = confidenceStr;
                    metaDiv.appendChild(confBadge);
                }

                item.appendChild(metaDiv);
                item.addEventListener("click", () => selectFile(file));
                queueList.appendChild(item);
            }
        }

        // Sort queue
        function sortQueue(files) {
            const sorted = [...files];
            switch (currentSort) {
                case "created":
                    sorted.sort((a, b) => (a.stem > b.stem ? 1 : -1));
                    break;
                case "reviewed":
                    sorted.sort((a, b) => {
                        const aTime = a.last_reviewed_at ? new Date(a.last_reviewed_at) : new Date(0);
                        const bTime = b.last_reviewed_at ? new Date(b.last_reviewed_at) : new Date(0);
                        return bTime - aTime;
                    });
                    break;
                case "confidence":
                    sorted.sort((a, b) => {
                        const aConf = a.prediction?.confidence || 0;
                        const bConf = b.prediction?.confidence || 0;
                        return bConf - aConf;
                    });
                    break;
            }
            return sorted;
        }

        // Setup filter buttons
        function setupFilterButtons() {
            const filters = document.getElementById("filters");
            filters.querySelectorAll("button").forEach(btn => {
                btn.addEventListener("click", () => {
                    filters.querySelectorAll("button").forEach(b => b.classList.remove("active"));
                    btn.classList.add("active");
                    currentFilter = btn.dataset.filter;
                    renderQueueList();
                });
            });
        }

        // Setup sort dropdown
        function setupSortDropdown() {
            const sortBy = document.getElementById("sortBy");
            sortBy.addEventListener("change", (e) => {
                currentSort = e.target.value;
                renderQueueList();
            });
        }

        // Select a file from queue
        async function selectFile(file) {
            selectedFile = file;
            currentBox = file.prediction || { x: 0.5, y: 0.9, w: 0.2, h: 0.06 };
            if (file.user_correction && file.user_correction.x !== null) {
                currentBox = file.user_correction;
            }
            updateBoxInputs();
            updateInfoPanel();
            updateButtonStates();
            await loadImage();
            renderQueueList();
            updateHeader();
        }

        // Load image from server
        async function loadImage() {
            return new Promise((resolve) => {
                const img = new Image();
                img.onload = () => {
                    currentImage = img;
                    drawCanvas();
                    resolve();
                };
                img.onerror = () => {
                    console.error("Failed to load image");
                    resolve();
                };
                img.src = "/api/image/" + selectedFile.stem;
            });
        }

        // Draw canvas with image and box overlay
        function drawCanvas() {
            if (!currentImage) {
                canvas.width = 400;
                canvas.height = 300;
                ctx.fillStyle = "#333";
                ctx.fillRect(0, 0, 400, 300);
                ctx.fillStyle = "#999";
                ctx.font = "14px sans-serif";
                ctx.textAlign = "center";
                ctx.fillText("No image loaded", 200, 150);
                return;
            }

            canvas.width = currentImage.width;
            canvas.height = currentImage.height;
            ctx.drawImage(currentImage, 0, 0);

            // Only draw box if we have valid coordinates
            if (currentBox.x === null || currentBox.x === undefined) return;

            const x = currentBox.x * canvas.width;
            const y = currentBox.y * canvas.height;
            const w = currentBox.w * canvas.width;
            const h = currentBox.h * canvas.height;

            // Box color: yellow for prediction, green for user correction
            let boxColor = "#ffff00";
            if (selectedFile.user_correction && selectedFile.user_correction.x !== null) {
                boxColor = "#00ff00";
            }

            ctx.strokeStyle = boxColor;
            ctx.lineWidth = 2;
            ctx.strokeRect(x - w/2, y - h/2, w, h);

            // Draw corners
            const cs = 6;
            ctx.fillStyle = boxColor;
            ctx.fillRect(x - w/2 - cs/2, y - h/2 - cs/2, cs, cs);
            ctx.fillRect(x + w/2 - cs/2, y - h/2 - cs/2, cs, cs);
            ctx.fillRect(x - w/2 - cs/2, y + h/2 - cs/2, cs, cs);
            ctx.fillRect(x + w/2 - cs/2, y + h/2 - cs/2, cs, cs);

            // Draw confidence label
            if (selectedFile.prediction?.confidence) {
                const conf = (selectedFile.prediction.confidence * 100).toFixed(0) + "%";
                ctx.font = "12px sans-serif";
                ctx.fillStyle = boxColor;
                ctx.textAlign = "left";
                ctx.fillText(conf, x - w/2, y - h/2 - 5);
            }
        }

        // Update box coordinate inputs
        function updateBoxInputs() {
            document.getElementById("boxX").value = (currentBox.x || 0).toFixed(4);
            document.getElementById("boxY").value = (currentBox.y || 0).toFixed(4);
            document.getElementById("boxW").value = (currentBox.w || 0).toFixed(4);
            document.getElementById("boxH").value = (currentBox.h || 0).toFixed(4);
        }

        // Update info panel
        function updateInfoPanel() {
            document.getElementById("infoSource").textContent = "D" + (selectedFile.disc || "?");
            document.getElementById("infoStatus").textContent = selectedFile.status;
            const confidence = selectedFile.prediction?.confidence || 0;
            document.getElementById("infoConfidence").textContent = confidence > 0 ? (confidence * 100).toFixed(0) + "%" : "--";
        }

        // Update header
        function updateHeader() {
            const header = document.getElementById("canvasHeader");
            if (!selectedFile) {
                header.textContent = "Select a file from the queue";
            } else {
                header.textContent = (selectedFile.display_name || selectedFile.stem) + " -- " + selectedFile.status;
            }
        }

        // Setup action button handlers
        function setupActionButtons() {
            document.getElementById("confirmBtn").addEventListener("click", () => performAction("confirmed"));
            document.getElementById("noStampBtn").addEventListener("click", () => performAction("no_stamp"));
            document.getElementById("skipBtn").addEventListener("click", () => performAction("skipped"));
            document.getElementById("nextBtn").addEventListener("click", () => selectNextFile());
            document.getElementById("trainBtn").addEventListener("click", () => triggerWorker());

            document.getElementById("editBtn").addEventListener("click", () => {
                // Toggle edit mode visual indicator
                const btn = document.getElementById("editBtn");
                btn.classList.toggle("active");
            });
        }

        // Perform action
        async function performAction(action) {
            if (!selectedFile) return;

            try {
                const response = await fetch("/api/action", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        stem: selectedFile.stem,
                        action: action,
                        box: action !== "no_stamp" ? currentBox : null
                    })
                });

                const result = await response.json();
                if (result.success) {
                    // Update local state
                    if (result.stats) queue.stats = result.stats;
                    updateProgressBar();

                    // Mark current file in local queue
                    const idx = queue.files.findIndex(f => f.stem === selectedFile.stem);
                    if (idx >= 0) {
                        queue.files[idx].status = result.status;
                        queue.files[idx].user_correction = {
                            x: currentBox.x, y: currentBox.y,
                            w: currentBox.w, h: currentBox.h,
                            action: action
                        };
                    }

                    selectNextPending();
                } else {
                    console.error("Action failed:", result.error);
                }
            } catch (error) {
                console.error("Error performing action:", error);
            }
        }

        // Select next pending file
        function selectNextPending() {
            const pending = queue.files.filter(f => f.status === "pending");
            if (pending.length > 0) {
                selectFile(pending[0]);
            } else {
                selectedFile = null;
                currentImage = null;
                renderQueueList();
                updateHeader();
                drawCanvas();
            }
        }

        // Select next file in current filter
        function selectNextFile() {
            const filtered = queue.files.filter(f => {
                if (currentFilter === "all") return true;
                return f.status === currentFilter;
            });

            if (!selectedFile) {
                if (filtered.length > 0) selectFile(filtered[0]);
                return;
            }

            const idx = filtered.findIndex(f => f.stem === selectedFile.stem);
            if (idx < filtered.length - 1) {
                selectFile(filtered[idx + 1]);
            }
        }

        // Trigger background worker (train + re-infer)
        async function triggerWorker() {
            const trainBtn = document.getElementById("trainBtn");
            trainBtn.disabled = true;

            try {
                const response = await fetch("/api/train", { method: "POST" });
                const result = await response.json();

                if (result.success) {
                    startWorkerPolling();
                } else {
                    trainBtn.disabled = false;
                    console.error("Worker start failed:", result.error);
                }
            } catch (error) {
                trainBtn.disabled = false;
                console.error("Error starting worker:", error);
            }
        }

        // Poll worker status
        function startWorkerPolling() {
            const statusDiv = document.getElementById("workerStatus");
            const messageSpan = document.getElementById("workerMessage");
            const progressBar = document.getElementById("workerProgress");

            statusDiv.style.display = "block";

            workerPolling = setInterval(async () => {
                try {
                    const response = await fetch("/api/worker-status");
                    const status = await response.json();

                    messageSpan.textContent = status.message || status.phase;
                    const pct = status.total > 0 ? ((status.progress / status.total) * 100) : 0;
                    progressBar.style.width = pct + "%";

                    if (status.phase === "done" || status.phase === "error") {
                        clearInterval(workerPolling);
                        workerPolling = null;
                        document.getElementById("trainBtn").disabled = false;

                        // Reload queue to get new predictions
                        await initDashboard();
                        selectNextPending();

                        setTimeout(() => {
                            statusDiv.style.display = "none";
                        }, 5000);
                    }
                } catch (error) {
                    console.error("Error polling worker:", error);
                }
            }, 2000);
        }

        // Keyboard controls
        function setupKeyboardControls() {
            document.addEventListener("keydown", (e) => {
                // Don't capture when typing in inputs
                if (e.target.tagName === "INPUT") return;
                if (!selectedFile) return;

                const cw = currentImage ? currentImage.width : 1;
                const ch = currentImage ? currentImage.height : 1;
                const moveStep = e.shiftKey ? 3 / cw : 20 / cw;
                const moveStepY = e.shiftKey ? 3 / ch : 20 / ch;
                const sizeStep = 0.01;

                switch (e.key) {
                    case "ArrowLeft":
                        currentBox.x = Math.max(currentBox.w / 2, currentBox.x - moveStep);
                        drawCanvas(); updateBoxInputs(); e.preventDefault(); break;
                    case "ArrowRight":
                        currentBox.x = Math.min(1 - currentBox.w / 2, currentBox.x + moveStep);
                        drawCanvas(); updateBoxInputs(); e.preventDefault(); break;
                    case "ArrowUp":
                        currentBox.y = Math.max(currentBox.h / 2, currentBox.y - moveStepY);
                        drawCanvas(); updateBoxInputs(); e.preventDefault(); break;
                    case "ArrowDown":
                        currentBox.y = Math.min(1 - currentBox.h / 2, currentBox.y + moveStepY);
                        drawCanvas(); updateBoxInputs(); e.preventDefault(); break;
                    case "[":
                        currentBox.w = Math.max(0.02, currentBox.w - sizeStep);
                        drawCanvas(); updateBoxInputs(); e.preventDefault(); break;
                    case "]":
                        currentBox.w = Math.min(0.5, currentBox.w + sizeStep);
                        drawCanvas(); updateBoxInputs(); e.preventDefault(); break;
                    case "{":
                        currentBox.h = Math.max(0.02, currentBox.h - sizeStep);
                        drawCanvas(); updateBoxInputs(); e.preventDefault(); break;
                    case "}":
                        currentBox.h = Math.min(0.3, currentBox.h + sizeStep);
                        drawCanvas(); updateBoxInputs(); e.preventDefault(); break;
                    case "Enter":
                        performAction("confirmed"); e.preventDefault(); break;
                    case "s": case "S":
                        performAction("no_stamp"); e.preventDefault(); break;
                }
            });
        }

        // Update button states
        function updateButtonStates() {
            const hasFile = selectedFile !== null;
            document.getElementById("confirmBtn").disabled = !hasFile;
            document.getElementById("editBtn").disabled = !hasFile;
            document.getElementById("noStampBtn").disabled = !hasFile;
            document.getElementById("skipBtn").disabled = !hasFile;
            document.getElementById("nextBtn").disabled = !hasFile;
        }

        // Initialize on load
        window.addEventListener("load", async () => {
            setupFilterButtons();
            setupSortDropdown();
            setupActionButtons();
            setupKeyboardControls();
            await initDashboard();
            // Auto-select first pending file
            selectNextPending();
        });
```

- [ ] **Step 6: Commit**

```bash
git add dashboard.html
git commit -m "feat: rewrite dashboard UI for continuous review with progress and worker polling"
```

---

## Task 5: Integration testing

- [ ] **Step 1: Reset queue and start dashboard**

```bash
rm -f corrections_queue.json worker_status.json
uv run corrections_dashboard.py
```

Open http://localhost:8889 in browser.

- [ ] **Step 2: Verify queue loads with 7,455 files**

Check that the progress bar shows `0 / 7455` and the first pending file auto-loads.

- [ ] **Step 3: Test review actions**

- Press Enter to confirm a file (should save label and advance)
- Press S to mark no stamp (should advance)
- Click Skip button (should advance)
- Verify progress bar updates

- [ ] **Step 4: Test API endpoints**

```bash
curl -s http://localhost:8889/api/queue | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d['stats'], indent=2))"
curl -s http://localhost:8889/api/worker-status
```

- [ ] **Step 5: Commit**

```bash
git commit --allow-empty -m "test: integration test continuous review loop verified"
```
