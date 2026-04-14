# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "requests>=2.32",
# ]
# ///
"""Run a local Ollama vision model against a single stage-1 shard for accuracy comparison.

Reads a shard manifest, sends each cropped image to Ollama, and writes a
result JSON in the same shape as the Haiku/Sonnet subagent outputs so the
pilot review HTML can display it.

Usage:
    uv run scripts/ocr/ocr_ollama_bench.py --model gemma3:4b --shard 0000
    uv run scripts/ocr/ocr_ollama_bench.py --model qwen2.5vl:3b --shard 0000
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import requests

BASE_DIR = Path(__file__).parent.parent
STATE_DIR = BASE_DIR / "state"
STAGE1_SHARDS_DIR = STATE_DIR / "shards" / "stage1"

OLLAMA_URL = "http://localhost:11434"
REQUEST_TIMEOUT = 300

PROMPT = """This is a cropped photo showing a camera date stamp — small orange, red, or amber LED digits.

Transcribe EXACTLY what you see. Rules:
- Only the characters visible in the stamp. No reformatting.
- Preserve spacing and apostrophes exactly. Examples: "10 3 '99", "9 6'95", "'94 6 22".
- 7-segment LED digits can be confusing: 9 vs 5, 3 vs 8, 6 vs 0, 7 vs 1. Look carefully.
- If a character is unclear, use ? for it. Do not guess.
- If there is no date stamp visible, respond with exactly: NONE

Output ONLY the stamp text (or NONE), nothing else. No explanations, no prefixes."""


def ocr_one(model: str, crop_path: Path) -> tuple[str, float]:
    with open(crop_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    t0 = time.time()
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": PROMPT, "images": [b64]}],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 1024},
            "think": False,
        },
        timeout=REQUEST_TIMEOUT,
    )
    elapsed = time.time() - t0
    data = resp.json()
    text = data.get("message", {}).get("content", "").strip()
    return text, elapsed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Ollama model tag, e.g. gemma3:4b")
    p.add_argument("--shard", default="0000", help="Shard id, e.g. 0000")
    p.add_argument("--suffix", default=None, help="Output filename suffix (default: model tag sanitized)")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    manifest_path = STAGE1_SHARDS_DIR / f"shard_{args.shard}.json"
    manifest = json.loads(manifest_path.read_text())

    suffix = args.suffix or args.model.replace(":", "_").replace("/", "_")
    out_path = STAGE1_SHARDS_DIR / f"shard_{args.shard}_{suffix}_result.json"

    stems = manifest["stems"]
    if args.limit:
        stems = stems[: args.limit]

    print(f"Model: {args.model}")
    print(f"Shard: {args.shard} ({len(stems)} stems)")
    print(f"Output: {out_path.relative_to(BASE_DIR)}")
    print()

    results: dict[str, dict] = {}
    total_elapsed = 0.0
    t0 = time.time()

    for i, entry in enumerate(stems, 1):
        stem = entry["stem"]
        crop_path = BASE_DIR / entry["crop_path"]
        try:
            text, elapsed = ocr_one(args.model, crop_path)
        except Exception as e:
            print(f"  [{i}/{len(stems)}] {stem}: ERROR {e}")
            text = f"ERROR: {e}"
            elapsed = 0.0

        results[stem] = {
            "text": text,
            "bbox_source": entry.get("bbox_source"),
            "confidence": entry.get("confidence"),
            "elapsed_s": round(elapsed, 2),
        }
        total_elapsed += elapsed
        print(f"  [{i}/{len(stems)}] {stem}: {text!r}  ({elapsed:.1f}s)")

    out_path.write_text(json.dumps({
        "shard_id": args.shard,
        "stage": 1,
        "model": args.model,
        "results": results,
    }, indent=2))

    wall = time.time() - t0
    print()
    print(f"Total elapsed: {wall:.1f}s")
    print(f"Inference time: {total_elapsed:.1f}s")
    print(f"Avg per image: {total_elapsed / max(len(results), 1):.1f}s")
    print(f"Wrote {out_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
