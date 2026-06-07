#!/usr/bin/env bash
set -euo pipefail

# ── qwen3.6-edge entrypoint ──────────────────────────────────────────────────
# Starts llama-server with TurboQuant + MTP on the RTX 5060 Ti.
#
# Model resolution:
#   1. If $MODEL exists, use it directly.
#   2. If $HF_REPO is set, download the GGUF from HuggingFace.
#   3. Otherwise, fail with a helpful message.
# ─────────────────────────────────────────────────────────────────────────────

MODEL="${MODEL:-/models/Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf}"
PORT="${PORT:-8081}"
HOST="${HOST:-0.0.0.0}"
CTX_SIZE="${CTX_SIZE:-131072}"
N_CPU_MOE="${N_CPU_MOE:-22}"
PARALLEL="${PARALLEL:-1}"
THREADS="${THREADS:-8}"
SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-2}"
SPEC_DRAFT_N_CPU_MOE="${SPEC_DRAFT_N_CPU_MOE:-0}"
CACHE_TYPE_K="${CACHE_TYPE_K:-turbo3}"
CACHE_TYPE_V="${CACHE_TYPE_V:-turbo3}"
NO_WARMUP="${NO_WARMUP:-true}"

# ── Model check ──────────────────────────────────────────────────────────────

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found at $MODEL"
    echo ""
    echo "Either:"
    echo "  1. Mount the model directory:  docker run -v /path/to/models:/models:ro ..."
    echo "  2. Set HF_REPO to auto-download:"
    echo "     docker run -e HF_REPO=unsloth/Qwen3.6-35B-A3B-MTP-GGUF ..."
    echo "  3. Set MODEL to your GGUF path:"
    echo "     docker run -e MODEL=/models/my-model.gguf ..."
    exit 1
fi

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  qwen3.6-edge — Qwen3.6-35B-A3B Q4_K_M + MTP + TurboQuant ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Model:  $MODEL"
echo "║  Port:   $PORT"
echo "║  Context: $CTX_SIZE"
echo "║  GPU MoE layers: $N_CPU_MOE / 40 (CPU offload)"
echo "║  TurboQuant KV: $CACHE_TYPE_K / $CACHE_TYPE_V"
echo "║  MTP drafts: $SPEC_DRAFT_N_MAX"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── Build llama-server command ───────────────────────────────────────────────

CMD=(
    llama-server
    -m "$MODEL"
    -ngl 99
    --n-cpu-moe "$N_CPU_MOE"
    -c "$CTX_SIZE"
    --no-mmap
    --cache-type-k "$CACHE_TYPE_K"
    --cache-type-v "$CACHE_TYPE_V"
    --threads "$THREADS"
    --parallel "$PARALLEL"
    --host "$HOST"
    --port "$PORT"
    --spec-type mtp
    --spec-draft-n-max "$SPEC_DRAFT_N_MAX"
    --spec-draft-n-cpu-moe "$SPEC_DRAFT_N_CPU_MOE"
)

if [ "$NO_WARMUP" = "true" ]; then
    CMD+=(--no-warmup)
fi

echo ""
echo "Starting llama-server..."
echo "Command: ${CMD[*]}"
echo ""

exec "${CMD[@]}"
