#!/usr/bin/env bash
# Sleep 5h, then attempt to finish the VLM OCR bench against Ollama Cloud
# via LiteLLM. Repeat until every model has 200 clean rows, or MAX_ROUNDS.
#
# Survives shell exit via nohup. Log at state/bench/results_litellm/loop.log.
#
# Rationale: Ollama Cloud's free-tier hourly quota rejects us once we've
# pushed ~300+ requests in a short window. --resume skips anything already
# clean in each model's JSONL, and bench_vlm_litellm.py no longer persists
# rate_limit rows, so each 5h round picks up exactly what's missing.
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
set -a && source .env && set +a

MODELS=(
  "gemma4-31b-cloud"
  "gemini-3-flash-preview-cloud"
  "qwen3-vl-235b-cloud"
  "qwen3-vl-235b-instruct-cloud"
  "gemma3-27b-cloud"
  "gemma3-12b-cloud"
  "kimi-k2.5-cloud"
  "qwen3.5-cloud"
  "gpt-oss-120b-cloud"
)

WAIT_SECONDS="${WAIT_SECONDS:-18000}"   # 5h
MAX_ROUNDS="${MAX_ROUNDS:-30}"          # safety cap: 30 rounds × 5h ≈ 6 days
TARGET_CLEAN=200

results_dir="state/bench/results_litellm"
mkdir -p "$results_dir"

all_models_complete() {
  for m in "${MODELS[@]}"; do
    local f="$results_dir/${m}.jsonl"
    if [ ! -f "$f" ]; then return 1; fi
    local clean
    clean=$({ grep -c '"error": null' "$f" || true; })
    if [ "$clean" -lt "$TARGET_CLEAN" ]; then return 1; fi
  done
  return 0
}

print_state() {
  echo "  state:"
  for m in "${MODELS[@]}"; do
    local f="$results_dir/${m}.jsonl"
    local clean=0 total=0
    if [ -f "$f" ]; then
      clean=$({ grep -c '"error": null' "$f" || true; })
      total=$(wc -l <"$f")
    fi
    printf "    %-40s  clean=%3d / %3d (total rows %3d)\n" "$m" "$clean" "$TARGET_CLEAN" "$total"
  done
}

for round in $(seq 1 "$MAX_ROUNDS"); do
  echo ""
  echo "########################################"
  echo "# ROUND $round — waiting ${WAIT_SECONDS}s until $(date -d "+${WAIT_SECONDS} seconds" -Iseconds)"
  echo "########################################"
  print_state

  if all_models_complete; then
    echo "ALL MODELS COMPLETE before round $round — exiting"
    exit 0
  fi

  sleep "$WAIT_SECONDS"

  echo ""
  echo "=== ROUND $round starting at $(date -Iseconds) ==="

  for m in "${MODELS[@]}"; do
    out="$results_dir/${m}.jsonl"
    clean=0
    if [ -f "$out" ]; then
      clean=$(grep -c '"error": null' "$out" || echo 0)
    fi
    if [ "$clean" -ge "$TARGET_CLEAN" ]; then
      echo "SKIP $m (clean=$clean)"
      continue
    fi
    echo ""
    echo "--- $m  (clean=$clean / $TARGET_CLEAN, $(date -Iseconds)) ---"
    uv run scripts/ocr/bench_vlm_litellm.py \
      --model "$m" \
      --base-url "$LITELLM_BASE_URL" \
      --manifest state/bench/manifest.json \
      --crops-dir state/bench/crops \
      --output "jsonl://$out" \
      --host-label litellm \
      --resume \
      || echo "!! $m exited non-zero; continuing round"
  done

  if all_models_complete; then
    echo ""
    echo "### ALL MODELS COMPLETE at $(date -Iseconds) after round $round"
    print_state
    exit 0
  fi
done

echo ""
echo "### MAX_ROUNDS ($MAX_ROUNDS) reached without completing all models at $(date -Iseconds)"
print_state
exit 1
