# /// script
# requires-python = ">=3.14"
# ///
#!/usr/bin/env python3
"""Unified corrections dashboard for YOLO feedback loop."""

import json
import sys
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

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
        """Log HTTP requests and errors."""
        # Log errors (status >= 400) but suppress routine requests
        if len(args) > 1 and args[0] >= 400:
            print(f"HTTP {args[0]}: {args[1]}")


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
    from datetime import datetime

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


def handle_action(data):
    """Placeholder - will be implemented in Task 3."""
    return {"error": "Action handler not yet implemented"}


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
