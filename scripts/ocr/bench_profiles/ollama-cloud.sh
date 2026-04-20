#!/usr/bin/env bash
# Bench profile: run against Ollama Cloud's free tier.
#
# Requires OLLAMA_API_KEY set in the environment (create one at
# https://ollama.com/settings/keys). The runner's _auth_headers()
# picks it up automatically.
#
# Usage:
#   OLLAMA_API_KEY=... bench_profiles/ollama-cloud.sh <ollama-tag>
#
# Example:
#   bench_profiles/ollama-cloud.sh gemini-3-pro:cloud
#   bench_profiles/ollama-cloud.sh kimi-vl:cloud

set -euo pipefail

MODEL="${1:?usage: ollama-cloud.sh <ollama-tag>}"
HOST="${OLLAMA_HOST:-https://ollama.com}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"

if [ -z "${OLLAMA_API_KEY:-}" ]; then
    echo "ERROR: OLLAMA_API_KEY not set. Create one at https://ollama.com/settings/keys" >&2
    exit 2
fi

cd "$REPO_ROOT"
source .venv/bin/activate

SAFE_TAG=$(echo "$MODEL" | tr '/:' '__')
OUT="state/bench/results/${SAFE_TAG}_ollama-cloud.jsonl"

uv run scripts/ocr/bench_vlm_ocr.py \
    --model "$MODEL" \
    --host "$HOST" \
    --manifest state/bench/manifest.json \
    --crops-dir state/bench/crops \
    --output "jsonl://$OUT" \
    --host-label ollama-cloud \
    --resume
