# /// script
# requires-python = ">=3.14"
# ///
#!/usr/bin/env python3
"""Unified corrections dashboard for YOLO feedback loop."""

import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler

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


class DashboardHandler(SimpleHTTPRequestHandler):
    """HTTP request handler for corrections dashboard."""

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

    def log_message(self, format, *args):
        """Log HTTP requests and errors."""
        # Log errors (status >= 400) but suppress routine requests
        if len(args) > 0:
            try:
                status = int(args[0]) if isinstance(args[0], str) else args[0]
                if status >= 400:
                    print(f"HTTP {status}: {args[1] if len(args) > 1 else ''}")
            except (ValueError, IndexError, TypeError):
                pass


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

    labeled_stems = {p.stem for p in LABELS_DIR.glob("*.txt")}
    skipped_stems = set()
    if SKIPPED_FILE.exists():
        skipped_stems = {line.strip() for line in SKIPPED_FILE.read_text().splitlines() if line.strip()}

    for img_path in sorted(SCANMYPHOTOS_DIR.glob("*.jpg")):
        stem = img_path.stem
        if stem in existing_stems:
            continue

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

    for f in queue["files"]:
        if f["stem"] in predictions:
            f["prediction"] = predictions[f["stem"]]
        if f["stem"] in labeled_stems and f["status"] == "pending":
            f["status"] = "labeled"
        elif f["stem"] in skipped_stems and f["status"] == "pending":
            f["status"] = "no_stamp"

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


_worker_process = None


def start_worker():
    """Start background train + infer worker."""
    global _worker_process

    if _worker_process and _worker_process.poll() is None:
        return {"error": "Worker already running"}

    with open(STATUS_FILE, "w") as f:
        json.dump({"phase": "starting", "progress": 0, "total": 0, "message": "Starting training..."}, f)

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

    if _worker_process and _worker_process.poll() is not None:
        _worker_process = None
        if status.get("phase") not in ("done", "error"):
            status = {"phase": "error", "progress": 0, "total": 0, "message": "Worker exited unexpectedly"}

    return status


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
