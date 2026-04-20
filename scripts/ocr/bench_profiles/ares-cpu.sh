#!/usr/bin/env bash
# Bench profile: run bench_vlm_ocr.py on ares's Ollama container.
#
# Usage:
#   bench_profiles/ares-cpu.sh <ollama-tag>

set -euo pipefail

MODEL="${1:?usage: ares-cpu.sh <ollama-tag>}"
HOST="${OLLAMA_HOST:-http://ares.savannah-mimosa.ts.net:11434}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"

export OLLAMA_NUM_PARALLEL=1

cd "$REPO_ROOT"
source .venv/bin/activate

uv run scripts/ocr/bench_vlm_ocr.py \
    --model "$MODEL" \
    --host "$HOST" \
    --manifest state/bench/manifest.json \
    --crops-dir state/bench/crops \
    --output "postgres://${DATABASE_URL:-dedup:dedup_local_dev@localhost:5432/dedup}" \
    --host-label ares-cpu \
    --resume
