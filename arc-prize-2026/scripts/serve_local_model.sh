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

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# DEFAULT IS 4-BIT. Measured on this box, same prompt, same sampler:
#            gen tok/s   peak mem   answer
#   8-bit         13.1     28.97GB   correct
#   4-bit         32.9     15.59GB   correct, same key insight
# 2.5x faster and 13GB less resident. The memory headroom matters as much as
# the speed: two hard machine hangs on 2026-08-27 came from running the 29GB
# build at sustained full GPU load.
# Quality was matched on ONE reasoning task, not established in general -- for
# work where fidelity to the Kaggle FP8 rail matters, pin the 8-bit build:
#   LOCAL_MODEL=mlx-community/Qwen3.8-27B-8bit ./scripts/serve_local_model.sh
MODEL="${LOCAL_MODEL:-mlx-community/Qwen3.8-27B-4bit}"
PORT="${LOCAL_MODEL_PORT:-1234}"
HOST="127.0.0.1"

# Field-floor sampler. Keep these matched to the arm you are screening against
# so local/Kaggle differ only by backend, not by decoding policy.
TEMP="${LOCAL_MODEL_TEMP:-0.7}"
TOP_P="${LOCAL_MODEL_TOP_P:-0.8}"
TOP_K="${LOCAL_MODEL_TOP_K:-20}"

# MAX TOKENS -- do not leave this at the server default.
# The harness sends `max_tokens: null`, so the SERVER's default applies, and
# mlx_lm.server defaults to 512. A thinking model blows through 512 mid-
# reasoning: measured 3 of 4 harness calls returning finish_reason="length"
# with exactly 512 completion tokens and NO tool_call emitted. That reads as a
# reasoning failure in the artifact when it is really a serving cap.
MAX_TOKENS="${LOCAL_MODEL_MAX_TOKENS:-8192}"

# THINKING -- the decisive knob for screening throughput on this hardware.
#
# MEASURED with thinking ON: reasoning length GROWS with context (76 -> 421 ->
# 7836 -> 9603 chars as the prompt went 4k -> 8k tokens), and generation runs
# ~13 tok/s, so a 4096-token reasoning response costs ~320 SECONDS and STILL
# truncates. Raising the cap only buys proportionally more wall-clock.
#
# For a rail whose job is USE-testing -- does the arm act, does tool-call
# parsing hold, does the model reach for the affordance -- the reasoning trace
# is not what is being measured, and thinking OFF is 10-20x faster with no
# truncation. Thinking ON remains right when the TRACE is the object of study.
#
# This is a deliberate deviation from the Kaggle config; label any run made
# with it, and never compare its latency or token counts to a Kaggle draw.
THINKING="${LOCAL_MODEL_THINKING:-1}"

# PROMPT CACHING -- the single biggest speed lever for agent loops.
# The harness resends a long, near-identical prefix (game state + history) on
# every turn, so without a KV cache each call re-prefills tens of thousands of
# tokens and prefill dominates wall-clock (measured ~2.6 min/call without it).
# This is the MLX equivalent of the `--enable-prefix-caching` the Kaggle vLLM
# rail already runs with. The byte cap must keep KV well clear of the ~29GB
# weights inside 64GB of UNIFIED memory -- there is no separate VRAM to spill to.
#
# SIZED DOWN 2026-08-27 AFTER A HARD MACHINE HANG. The previous values (8
# caches / 12GB) were reckless on a 64GB box: 29GB weights + 12GB KV = 41GB
# before the harness, Python, the browser and the OS. And 8 caches was 8x what
# the workload needs -- mac_screen runs concurrency=1, so one live cache plus
# one spare is the whole requirement. The hang was not proven to be memory
# pressure (no panic log survived the force-reboot) but this configuration
# should never have been set, and headroom is worth more than a marginal
# cache-hit rate.
CACHE_SIZE="${LOCAL_MODEL_CACHE_SIZE:-2}"
CACHE_BYTES="${LOCAL_MODEL_CACHE_BYTES:-4000000000}"

while [ $# -gt 0 ]; do
    case "$1" in
        --port)  PORT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# ---- MEMORY PREFLIGHT ------------------------------------------------------
# Unified memory means the model, the KV cache, the harness and the OS all draw
# on the same 64GB. Refuse to start rather than take the machine down: a hard
# hang on 2026-08-27 cost an overnight run and a force-reboot.
TOTAL_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
FREE_PAGES=$(vm_stat | awk '/Pages free/ {gsub(/\./,"",$3); print $3}')
INACTIVE_PAGES=$(vm_stat | awk '/Pages inactive/ {gsub(/\./,"",$3); print $3}')
PAGE=$(vm_stat | head -1 | grep -oE '[0-9]+')
AVAIL_GB=$(( (FREE_PAGES + INACTIVE_PAGES) * PAGE / 1073741824 ))
case "$MODEL" in
    *4bit*) WEIGHTS_GB=16 ;;
    *6bit*) WEIGHTS_GB=22 ;;
    *)      WEIGHTS_GB=29 ;;
esac
NEED_GB=$(( WEIGHTS_GB + CACHE_BYTES / 1073741824 + 6 ))   # weights + KV + headroom
echo "  memory  : ${AVAIL_GB}GB available of ${TOTAL_GB}GB; this config wants ~${NEED_GB}GB"
if [ "$AVAIL_GB" -lt "$NEED_GB" ]; then
    echo "" >&2
    echo "  REFUSING TO START: only ${AVAIL_GB}GB available, ~${NEED_GB}GB needed." >&2
    echo "  Close other apps, or lower LOCAL_MODEL_CACHE_BYTES (currently $CACHE_BYTES)." >&2
    echo "  Override deliberately with ARC_SKIP_MEM_CHECK=1." >&2
    [ "${ARC_SKIP_MEM_CHECK:-0}" = "1" ] || exit 3
fi

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

# TRACING: unless ARC_NO_TRACE=1, mlx binds an internal port and a logging
# proxy takes $PORT, so every request and every response body -- including the
# <think> block, which arrives inline in message.content because mlx_lm.server
# has no reasoning parser -- lands in runs/llm_traces/YYYY-MM-DD.jsonl.
THINK_ARGS=""
if [ "$THINKING" != "1" ]; then
    THINK_ARGS='{"enable_thinking": false}'
    echo "  thinking: OFF (fast screening; deviates from the Kaggle config)"
else
    echo "  thinking: ON  (max_tokens $MAX_TOKENS; expect ~5 min/call)"
fi

if [ "${ARC_NO_TRACE:-0}" != "1" ]; then
    UPSTREAM=$(( PORT + 1 ))
    "$REPO/.venv/bin/python" "$REPO/scripts/llm_proxy.py" \
        --listen "$PORT" --upstream "$UPSTREAM" &
    PROXY_PID=$!
    trap 'kill $PROXY_PID 2>/dev/null' EXIT INT TERM
    echo "  tracing : runs/traces.db  (proxy $PORT -> $UPSTREAM)"
    BIND_PORT="$UPSTREAM"
else
    echo "  tracing : DISABLED (ARC_NO_TRACE=1)"
    BIND_PORT="$PORT"
fi


# ---------------------------------------------------------------------------
# STABILITY GUARDS -- added 2026-08-27 after TWO hard hangs (display dies,
# blinks on wake, force reboot). No kernel panic, no GPU fault: a wedged
# graphics stack, not hardware damage.
#
# Cause was sustained maximum Metal load across repeated display sleep/wake
# transitions, made worse by running on BATTERY (where displaysleep is 2 min).
# It was NOT `pmset -c sleep 0` -- that is AC-only and the second hang happened
# on battery.
#
# caffeinate holds display AND system sleep for exactly this process's lifetime
# and releases on exit, so it fixes the trigger without touching global pmset.
# ---------------------------------------------------------------------------
if pmset -g ps 2>/dev/null | head -1 | grep -q "Battery Power"; then
    echo "  power   : ON BATTERY -- long GPU runs have hung this machine twice."
    if [ "${ARC_ALLOW_BATTERY:-0}" != "1" ]; then
        echo "            REFUSING to start. Plug in, or set ARC_ALLOW_BATTERY=1." >&2
        exit 3
    fi
    echo "            ARC_ALLOW_BATTERY=1 -- proceeding anyway."
else
    echo "  power   : AC (good)"
fi

CAFFEINATE=""
if [ "${ARC_NO_CAFFEINATE:-0}" != "1" ] && command -v caffeinate >/dev/null; then
    CAFFEINATE="/usr/bin/caffeinate -dims"
    echo "  sleep   : held by caffeinate for this process only (-dims)"
fi

exec $CAFFEINATE "$MLX_SERVER" \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$BIND_PORT" \
    --temp "$TEMP" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --max-tokens "$MAX_TOKENS" \
    ${THINK_ARGS:+--chat-template-args "$THINK_ARGS"} \
    --prompt-cache-size "$CACHE_SIZE" \
    --prompt-cache-bytes "$CACHE_BYTES" \
    --log-level INFO
