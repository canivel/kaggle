#!/usr/bin/env bash
# Pull + certify the Q38 FIELD-FLOOR arm, for the 18:00 EDT queue-head gate.
#
# Written 2026-08-20 ~09:40 EDT while the kernel was still RUNNING, so the 14:45 step is one
# command with the right flags rather than three improvised ones under time pressure.
#
# The gate (q38_field_prereg_2026-08-20.md sec 3) is COMPLETE + runtime certification sec 2 by
# 18:00 EDT. It is NOT the lc/score bands -- the board-verified 2.23 carries the draw decision.
# So this script runs `--certify-only`, which returns BEFORE lc_total/mean_score are computed:
# the operational call cannot be contaminated by having seen the effect size.
#
#   bash duck_eval/q38/q38field_pull_certify.sh            # pull + certify (the 18:00 call)
#   bash duck_eval/q38/q38field_pull_certify.sh --science  # full verdict, AFTER the queue call
set -uo pipefail

KERNEL="canivel/arc3-q38-field-eval"
DEST="runs/kernel_pulls/q38_field_v1"
K200="uvx --from kaggle==2.0.0 kaggle"     # 2.2.3 drops kernel logs; 2.0.0 is the sanctioned pull
SCIENCE=0
[ "${1:-}" = "--science" ] && SCIENCE=1

echo "== status =="
STATUS=$($K200 kernels status "$KERNEL" 2>&1 | grep -oE "COMPLETE|ERROR|CANCEL|RUNNING|QUEUED" | head -1)
echo "  $KERNEL -> ${STATUS:-UNKNOWN}"
if [ "$STATUS" != "COMPLETE" ]; then
  echo
  echo "  NOT COMPLETE. The queue-head gate is COMPLETE + certification by 18:00 EDT."
  echo "  If 18:00 passes without COMPLETE: filler stays as queue head one more night."
  echo "  Do NOT pull a non-terminal kernel and do NOT read a partial artifact as a result."
  exit 1
fi

mkdir -p "$DEST"
echo
echo "== pull artifacts =="
$K200 kernels output "$KERNEL" -p "$DEST" 2>&1 | grep -v "^Warning" | tail -5

echo
echo "== pull log =="
# The scorer normalises both plain-text and CLI-JSON-array log formats (the 08-19 instrument
# fix), so either shape here is safe to score.
$K200 kernels logs "$KERNEL" > "$DEST/q38field.log" 2>"$DEST/q38field.err" \
  && echo "  wrote $DEST/q38field.log ($(wc -c <"$DEST/q38field.log") bytes)" \
  || echo "  WARN: log pull failed -- see $DEST/q38field.err (certification will fail closed)"

echo
echo "== certify =="
if [ "$SCIENCE" = "1" ]; then
  echo "  (--science: full verdict. Only run this AFTER the queue-head call is recorded.)"
  uv run python duck_eval/q38/q38field_score.py "$DEST"
else
  uv run python duck_eval/q38/q38field_score.py "$DEST" --certify-only
fi
RC=$?

echo
echo "== what to do with this =="
echo "  CERTIFIED   -> queue head tonight (A21 exploration draw):"
echo "                 uv run python scripts/queue.py clear"
echo "                 uv run python scripts/queue.py add $KERNEL <version> \"<msg citing rethink + verified 2.23 + A21 budget>\""
echo "                 then re-list to confirm the head, and let ARCDailySubmit (18:37) fire."
echo "  INFRA DEATH -> NOT decisive about capability. Filler stays head. Log the certification"
echo "                 failure verbatim; do not fold it into any ledger."
exit $RC
