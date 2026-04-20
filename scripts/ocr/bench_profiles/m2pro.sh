#!/usr/bin/env bash
# Bench profile: run on the user's M2 Pro MacBook with native Ollama.

set -euo pipefail

MODEL="${1:?usage: m2pro.sh <ollama-tag>}"
HOST="${OLLAMA_HOST:-http://localhost:11434}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"

cd "$REPO_ROOT"
source .venv/bin/activate

SAFE_TAG=$(echo "$MODEL" | tr '/:' '__')
OUT="state/bench/results/${SAFE_TAG}_m2pro.jsonl"

uv run scripts/ocr/bench_vlm_ocr.py \
    --model "$MODEL" \
    --host "$HOST" \
    --manifest state/bench/manifest.json \
    --crops-dir state/bench/crops \
    --output "jsonl://$OUT" \
    --host-label m2pro \
    --resume

echo
echo "Done. JSONL at: $OUT"
echo "Sync back to ares with: rsync $OUT ares:photo_project/state/bench/results/"
