#!/usr/bin/env bash
# Pull + read the ARM 3 (Q38 x GRAFT compound) arm, for tonight's queue-head gate.
#
# Written 2026-08-21 ~08:45 EDT by the iterate session while the kernel was still RUNNING, so the
# ~09:05 read is one fail-closed command rather than three improvised ones. Modelled byte-for-byte
# on the proven duck_eval/q38/q38field_pull_certify.sh.
#
# THE GATE (ITERATION_LOG 08-21, coordinator): Arm 3 heads tonight IFF certified AND lc_total >= 28.
# Unlike the field arm, lc IS part of the operational gate here, so there is no --certify-only
# split to protect: the head call legitimately needs the effect size.
#
# TWO BINDING VERDICT-STRING CAVEATS, both recorded BEFORE the read (do not write the verdict
# without them):
#   (1) VEHICLE  - Arm 3 rides OUR v4 lineage carrying v21's deltas, not the author's vehicle, so a
#                  MISS is ambiguous between "compound doesn't lift" and "vehicle difference".
#   (2) HARNESS  - Arm 3 attaches the SHARE-FORK bundle (June-30-gen), NOT the 08-07 anim bundle the
#                  field-floor comparator (lc 28 / 6.173) ran. A below-floor read therefore confounds
#                  harness-generation x grafts and MUST NOT be written as "grafts dead".
#   => Only an above-floor SIGNAL (lc >= 33) is cleanly attributable.
#
#   bash duck_eval/graft/q38graft_pull_read.sh
set -uo pipefail

KERNEL="canivel/arc3-q38-graft-eval"
DEST="runs/kernel_pulls/q38_graft_v1"
K200="uvx --from kaggle==2.0.0 kaggle"     # 2.2.3 drops kernel logs; 2.0.0 is the sanctioned pull

echo "== status =="
STATUS=$($K200 kernels status "$KERNEL" 2>&1 | grep -oE "COMPLETE|ERROR|CANCEL|RUNNING|QUEUED" | head -1)
echo "  $KERNEL -> ${STATUS:-UNKNOWN}"
if [ "$STATUS" != "COMPLETE" ]; then
  echo
  echo "  NOT COMPLETE. Do NOT pull a non-terminal kernel; do NOT read a partial artifact."
  echo "  If 18:00 EDT passes without COMPLETE: the field-floor arm (arc3-q38-field-eval v1) is"
  echo "  tonight's head per the Arm-0 standing order, re-queued WITH the trusted-fork tag."
  exit 1
fi

mkdir -p "$DEST"
echo
echo "== pull artifacts =="
$K200 kernels output "$KERNEL" -p "$DEST" 2>&1 | grep -v "^Warning" | tail -5

echo
echo "== pull log =="
# graft_score._read_log normalises plain-text and CLI-JSON-array shapes (08-19 instrument fix) and
# fail-closes on an unparseable log -- both re-verified against real bytes on 08-21.
$K200 kernels logs "$KERNEL" > "$DEST/q38graft.log" 2>"$DEST/q38graft.err" \
  && echo "  wrote $DEST/q38graft.log ($(wc -c <"$DEST/q38graft.log") bytes)" \
  || echo "  WARN: log pull failed -- see $DEST/q38graft.err (certification will fail closed)"

echo
echo "== certify + read (sealed scorer, audited 08-21 against real log bytes) =="
uv run python duck_eval/graft/q38graft_score.py "$DEST"
RC=$?

echo
echo "== what to do with this =="
echo "  SIGNAL (lc >= 33)        -> cleanly attributable lift. Queue head tonight; both caveats moot."
echo "  NULL/HARM, certified     -> decisive on THIS vehicle only. Head stays field-floor v1."
echo "                              Verdict string MUST carry BOTH caveats above."
echo "  INFRA DEATH              -> NOT decisive about capability. Head stays field-floor v1."
echo "                              Log the certification failure verbatim; fold into no ledger."
echo "  head gate: certified AND lc_total >= 28 (see head_gate_lc28 in the JSON)."
exit $RC
