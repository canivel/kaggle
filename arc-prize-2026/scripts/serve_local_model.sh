#!/bin/bash
# Serve a local MLX model on the SAME endpoint the ARC harness already expects.
#
# The frozen fork's harness (duck_eval/private/bundle_20260815/.../kaggle.py)
# talks OpenAI-shaped /v1/chat/completions to http://127.0.0.1:1234/v1 and reads
# OPENAI_BASE_URL from the environment. mlx_lm.server speaks the same wire
# format, so serving on port 1234 is a DROP-IN -- no harness code changes.
#
#   ./scripts/serve_local_model.sh            # 8-bit Qwen3.8-27B, port 1234
#   ./scripts/serve_local_model.sh --port 8080
#   LOCAL_MODEL=<repo-or-path> ./scripts/serve_local_model.sh
#
# SCREENING ONLY. Everything this server produces is [MAC-SCREEN]: it ranks and
# eliminates candidates so Kaggle slots are not spent on arms that could have
# been killed locally. It never produces a verdict -- see MIGRATION_MACBOOK.md
# and the local_gate footer.

set -euo pipefail

MODEL="${LOCAL_MODEL:-mlx-community/Qwen3.8-27B-8bit}"
PORT="${LOCAL_MODEL_PORT:-1234}"
HOST="127.0.0.1"

# Field-floor sampler. Keep these matched to the arm you are screening against
# so local/Kaggle differ only by backend, not by decoding policy.
TEMP="${LOCAL_MODEL_TEMP:-0.7}"
TOP_P="${LOCAL_MODEL_TOP_P:-0.8}"
TOP_K="${LOCAL_MODEL_TOP_K:-20}"

# PROMPT CACHING -- the single biggest speed lever for agent loops.
# The harness resends a long, near-identical prefix (game state + history) on
# every turn, so without a KV cache each call re-prefills tens of thousands of
# tokens and prefill dominates wall-clock (measured ~2.6 min/call without it).
# This is the MLX equivalent of the `--enable-prefix-caching` the Kaggle vLLM
# rail already runs with. 8 distinct caches covers concurrent games; the byte
# cap keeps KV well clear of the ~29GB weights inside 64GB unified memory.
CACHE_SIZE="${LOCAL_MODEL_CACHE_SIZE:-8}"
CACHE_BYTES="${LOCAL_MODEL_CACHE_BYTES:-12000000000}"

while [ $# -gt 0 ]; do
    case "$1" in
        --port)  PORT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

MLX_SERVER="$(command -v mlx_lm.server || echo "$HOME/.local/bin/mlx_lm.server")"
if [ ! -x "$MLX_SERVER" ]; then
    echo "mlx_lm.server not found. Install it globally:  uv tool install mlx-lm" >&2
    exit 1
fi

cat <<BANNER
────────────────────────────────────────────────────────────────────
  LOCAL MODEL SERVER  [MAC-SCREEN]
  model : $MODEL
  url   : http://$HOST:$PORT/v1
  sampler: temp=$TEMP top_p=$TOP_P top_k=$TOP_K

  Point any project at it:
    export OPENAI_BASE_URL="http://$HOST:$PORT/v1"
    export LOCAL_LLM_BASE_URL="http://$HOST:$PORT/v1"

  Screening only. No verdict, no band read, no queue-head promotion.
────────────────────────────────────────────────────────────────────
BANNER

exec "$MLX_SERVER" \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --temp "$TEMP" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --prompt-cache-size "$CACHE_SIZE" \
    --prompt-cache-bytes "$CACHE_BYTES" \
    --log-level INFO
