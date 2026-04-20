#!/usr/bin/env bash
# Bench profile: run on an ephemeral AWS GPU spot joined to the tailnet.

set -euo pipefail

MODEL="${1:?usage: cloud-gpu.sh <ollama-tag>}"
HOST="${OLLAMA_HOST:-http://localhost:11434}"
PG_DSN="${DATABASE_URL:-postgresql://dedup:dedup_local_dev@ares.savannah-mimosa.ts.net:5432/dedup}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"

cd "$REPO_ROOT"
source .venv/bin/activate

HOST_LABEL="${HOST_LABEL:-cloud-gpu-$(uname -m)}"

uv run scripts/ocr/bench_vlm_ocr.py \
    --model "$MODEL" \
    --host "$HOST" \
    --manifest state/bench/manifest.json \
    --crops-dir state/bench/crops \
    --output "postgres://${PG_DSN#postgresql://}" \
    --host-label "$HOST_LABEL" \
    --resume
