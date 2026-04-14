# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "psycopg[binary]>=3.1.0",
# ]
# ///
#!/usr/bin/env python3
"""Unified corrections dashboard for YOLO feedback loop."""

import json
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler

BASE_DIR = Path(__file__).parent.parent
UI_DIR = BASE_DIR / "ui"
DATASET_DIR = BASE_DIR / "dataset"
LABELS_DIR = DATASET_DIR / "labels"
SCANMYPHOTOS_DIR = BASE_DIR / "scanmyphotos"
QUEUE_FILE = BASE_DIR / "state" / "corrections_queue.json"
PREDICTIONS_FILE = BASE_DIR / "state" / "scanmyphotos_predictions.json"
SKIPPED_FILE = BASE_DIR / "state" / "skipped.txt"
STATUS_FILE = BASE_DIR / "state" / "worker_status.json"

DB_CONN_STRING = os.environ.get(
    "DATABASE_URL",
    "postgresql://dedup:dedup_local_dev@localhost:5432/dedup",
)

# Create required directories
LABELS_DIR.mkdir(parents=True, exist_ok=True)


def get_db():
    """Get a psycopg connection to the dedup database."""
    import psycopg
    return psycopg.connect(DB_CONN_STRING)


def ensure_stamp_rotations_table():
    """Create stamp_rotations table if it doesn't exist."""
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stamp_rotations (
            stem VARCHAR(256) PRIMARY KEY,
            rotation INTEGER NOT NULL DEFAULT 0,
            source VARCHAR(32) NOT NULL DEFAULT 'user',
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """)
    conn.commit()
    conn.close()


def get_rotation_for_stem(stem):
    """Get rotation for a stem: user-saved first, then predicted, then 0."""
    conn = get_db()
    # Check user-saved rotation first
    row = conn.execute(
        "SELECT rotation FROM stamp_rotations WHERE stem = %s", (stem,)
    ).fetchone()
    if row:
        conn.close()
        return {"rotation": row[0], "source": "user"}

    # Fall back to predicted rotation (strip disc prefix for lookup)
    original_num = stem.split("_", 1)[1] if "_" in stem else stem
    row = conn.execute(
        "SELECT rotation_needed, confidence FROM rotation_predictions WHERE filename = %s AND filename NOT LIKE '%%_boxes%%' LIMIT 1",
        (f"{original_num}.jpg",)
    ).fetchone()
    conn.close()
    if row:
        return {"rotation": row[0], "source": "predicted", "confidence": round(row[1], 4)}

    return {"rotation": 0, "source": "none"}


def save_rotation_for_stem(stem, rotation):
    """Save user-confirmed rotation to stamp_rotations table."""
    conn = get_db()
    conn.execute("""
        INSERT INTO stamp_rotations (stem, rotation, source)
        VALUES (%s, %s, 'user')
        ON CONFLICT (stem) DO UPDATE SET rotation = %s, updated_at = NOW()
    """, (stem, rotation, rotation))
    conn.commit()
    conn.close()


class DashboardHandler(SimpleHTTPRequestHandler):
    """HTTP request handler for corrections dashboard."""

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/":
            self.serve_file("dashboard.html", "text/html")
        elif self.path == "/batch":
            self.serve_file("batch_review.html", "text/html")
        elif self.path == "/api/queue":
            self.serve_json(get_queue())
        elif self.path == "/api/worker-status":
            self.serve_json(get_worker_status())
        elif self.path.startswith("/api/rotation/"):
            stem = self.path.split("/")[-1]
            self.serve_json(get_rotation_for_stem(stem))
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
        elif self.path == "/api/bulk-action":
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

            result = handle_bulk_action(data)
            self.serve_json(result)
        elif self.path == "/api/bulk-approve":
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

            result = handle_bulk_approve(data)
            self.serve_json(result)
        elif self.path == "/api/rotation":
            content_length = self.headers.get("Content-Length")
            if not content_length:
                self.send_error(400, "Missing Content-Length header")
                return
            try:
                body = self.rfile.read(int(content_length))
                data = json.loads(body)
            except (ValueError, json.JSONDecodeError):
                self.send_error(400, "Invalid JSON")
                return
            stem = data.get("stem")
            rotation = data.get("rotation", 0)
            if not stem or rotation not in (0, 90, 180, 270):
                self.send_error(400, "Invalid stem or rotation")
                return
            save_rotation_for_stem(stem, rotation)
            self.serve_json({"success": True, "stem": stem, "rotation": rotation})
        elif self.path == "/api/train":
            result = start_worker()
            self.serve_json(result)
        else:
            self.send_error(404)

    def serve_file(self, filename, content_type):
        """Serve a local file."""
        path = UI_DIR / filename
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


def transform_bbox_to_original(cx, cy, w, h, rotation):
    """Transform bbox from rotated display space back to original image space.

    YOLO format: center_x, center_y, width, height (all normalized 0-1).
    Rotation is degrees CW that the image was rotated for display.
    """
    if rotation == 0:
        return cx, cy, w, h
    elif rotation == 90:
        # 90 CW display -> original: (cx,cy,w,h) -> (cy, 1-cx, h, w)
        return cy, 1 - cx, h, w
    elif rotation == 180:
        return 1 - cx, 1 - cy, w, h
    elif rotation == 270:
        # 270 CW display -> original: (cx,cy,w,h) -> (1-cy, cx, h, w)
        return 1 - cy, cx, h, w
    return cx, cy, w, h


def handle_action(data):
    """Handle user actions (confirm, edit, skip, no_stamp)."""
    stem = data["stem"]
    action = data["action"]
    box = data.get("box")
    rotation = data.get("rotation", 0)

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
        # Transform bbox from rotated display space to original image space
        ox, oy, ow, oh = transform_bbox_to_original(
            box['x'], box['y'], box['w'], box['h'], rotation
        )
        label_content = f"0 {ox} {oy} {ow} {oh}\n"
        label_path = LABELS_DIR / f"{stem}.txt"
        label_path.write_text(label_content)
        file_entry["status"] = "labeled"

        # Save rotation if non-zero
        if rotation != 0:
            save_rotation_for_stem(stem, rotation)

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


def handle_bulk_action(data):
    """Handle bulk actions on multiple files."""
    stems = data.get("stems", [])
    action = data.get("action")

    if not stems or action != "no_stamp":
        return {"error": "Invalid bulk action"}

    queue = load_queue()
    stem_set = set(stems)

    # Add all to skipped.txt
    skipped = set()
    if SKIPPED_FILE.exists():
        skipped = {line.strip() for line in SKIPPED_FILE.read_text().splitlines() if line.strip()}
    skipped |= stem_set
    SKIPPED_FILE.write_text("\n".join(sorted(skipped)) + "\n")

    # Update queue entries
    for f in queue["files"]:
        if f["stem"] in stem_set:
            f["status"] = "no_stamp"
            f["last_reviewed_at"] = datetime.now().isoformat()
            f["user_correction"] = {"x": None, "y": None, "w": None, "h": None, "action": "no_stamp"}

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
    return {"success": True, "count": len(stems), "stats": queue["stats"]}


def handle_bulk_approve(data):
    """Bulk-approve multiple files with their predicted bounding boxes in one call."""
    items = data.get("items", [])
    if not items:
        return {"error": "No items provided"}

    queue = load_queue()
    now = datetime.now().isoformat()
    stem_map = {f["stem"]: f for f in queue["files"]}
    count = 0

    for item in items:
        stem = item.get("stem")
        box = item.get("box")
        if not stem or not box:
            continue

        # Write YOLO label file
        label_content = f"0 {box['x']} {box['y']} {box['w']} {box['h']}\n"
        label_path = LABELS_DIR / f"{stem}.txt"
        label_path.write_text(label_content)

        # Update queue entry
        if stem in stem_map:
            entry = stem_map[stem]
            entry["status"] = "labeled"
            entry["last_reviewed_at"] = now
            entry["user_correction"] = {
                "x": box["x"], "y": box["y"],
                "w": box["w"], "h": box["h"],
                "action": "confirmed",
            }
        count += 1

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
    return {"success": True, "count": count, "stats": queue["stats"]}


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
            f"uv run {BASE_DIR / 'scripts' / 'train.py'} && uv run {BASE_DIR / 'scripts' / 'infer_all.py'}"
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
    ensure_stamp_rotations_table()
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
