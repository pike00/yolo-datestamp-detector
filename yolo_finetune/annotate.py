# /// script
# requires-python = ">=3.14"
# ///
"""Annotation server — serves UI and REST API for bounding box labeling."""

import argparse
import http.server
import json
import os
import socketserver
import sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs

PORT = 8888
BASE_DIR = Path(__file__).parent


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Annotate bounding boxes on images")
    parser.add_argument("--mode", choices=["annotate", "correct"], default="annotate",
                        help="annotate: label new images; correct: review model predictions")
    return parser.parse_args()


ARGS = parse_args()

# Determine image source based on mode
if ARGS.mode == "correct":
    IMAGE_DIR = BASE_DIR / "dataset" / "to_annotate"
    CORRECTIONS_METADATA_PATH = BASE_DIR / "dataset" / "corrections" / "corrections_meta.json"
else:
    IMAGE_DIR = BASE_DIR.parent / "photo_mapping_samples"
    CORRECTIONS_METADATA_PATH = None

DATASET_DIR = BASE_DIR / "dataset"
LABELS_DIR = DATASET_DIR / "labels"
IMAGES_DIR = DATASET_DIR / "images"

# Use different progress files for different modes
if ARGS.mode == "correct":
    PROGRESS_FILE = BASE_DIR / "progress_correct.json"
else:
    PROGRESS_FILE = BASE_DIR / "progress.json"

SKIPPED_FILE = BASE_DIR / "skipped.txt"


def load_predictions_metadata():
    """Load model predictions from corrections_meta.json if in correct mode."""
    if ARGS.mode == "correct" and CORRECTIONS_METADATA_PATH and CORRECTIONS_METADATA_PATH.exists():
        with open(CORRECTIONS_METADATA_PATH) as f:
            return json.load(f)
    return {}


PREDICTIONS_META = load_predictions_metadata()


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
            data = json.load(f)
    else:
        data = {
            "current_index": 0,
            "last_box": {"x": 0.4, "y": 0.45, "w": 0.2, "h": 0.1},
            "labeled": [],
            "skipped": [],
        }

    # In correct mode, pre-load predictions for all images
    if ARGS.mode == "correct":
        files = get_image_files()
        for filename in files:
            stem = Path(filename).stem
            if stem not in data.get("labeled", []) and stem not in data.get("skipped", []):
                if stem in PREDICTIONS_META:
                    pred = PREDICTIONS_META[stem]
                    if pred.get("x") is not None:
                        # Pre-populate with model prediction
                        data[f"prediction_{stem}"] = pred

    return data


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
        current_idx = progress.get("current_index", 0)

        # Load box: use prediction if in correct mode, else use last_box or default
        box = progress.get("last_box") or {"x": 0.4, "y": 0.45, "w": 0.2, "h": 0.1}

        if current_idx < len(images):
            current_file = images[current_idx]
            stem = Path(current_file).stem

            if ARGS.mode == "correct" and stem in PREDICTIONS_META:
                pred = PREDICTIONS_META[stem]
                if pred.get("x") is not None:
                    box = {
                        "x": pred["x"],
                        "y": pred["y"],
                        "w": pred["w"],
                        "h": pred["h"],
                    }
                    box["model_confidence"] = pred.get("confidence", 0)  # Mark as from model

        return {
            "images": images,
            "total": len(images),
            "current_index": current_idx,
            "box": box,
            "labeled": progress["labeled"],
            "skipped": progress["skipped"],
            "mode": ARGS.mode,
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
        # Suppress request logs except errors (args[1] is the status code)
        if len(args) >= 2 and str(args[1]).startswith(("4", "5")):
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

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), AnnotationHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
