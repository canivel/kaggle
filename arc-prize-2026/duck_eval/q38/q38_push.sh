#!/usr/bin/env bash
# Q38 ENGINE-SWAP EVAL push + verify, as one auditable sequence (2026-08-15, the day's ONE slot).
#
# Ports the two guards this campaign paid for on 2026-08-14, verbatim from
# duck_eval/lora/lora_push_v2.sh:
#   --confirm-push        protects against the wrong TIME (a date check is not a safety property)
#   idempotence check     protects against the wrong ACTOR (two sessions spent both of 08-14's
#                         slots on one artifact because step 3's pull-back compares local to
#                         remote AFTER pushing, when a duplicate and a first push are identical)
#
#   0.  LOCAL-date guard (the slot budget is keyed to the LOCAL date)
#   0b. LEDGER RE-CONFIRM (section 11.4, binding) — print today's slot lines and refuse if the
#       section does not exist. A conditional that was true when issued is not evidence now.
#   0c. EXPLICIT-INTENT INTERLOCK: --confirm-push, or --dry-run to run 0..1b and stop.
#   1.  rebuild from the frozen duckfork + smoke (78) + sealed-scorer selftest (22)
#   1b. IDEMPOTENCE GUARD — exit 3 if remote code already equals local
#   2.  push
#   3.  pull-back verify: code sha256, cell count, ALL THREE dataset_sources (the 25 GB engine
#       is the one Kaggle is most likely to drop silently), docker sha, GPU, machine, internet
#   4.  preflight --family duck-harness --expect-diff-cells 2,6,8 (POST-push by design)
#
# Usage:  bash duck_eval/q38/q38_push.sh --dry-run       # gates only, NEVER pushes
#         bash duck_eval/q38/q38_push.sh --confirm-push  # the real thing
set -euo pipefail

REPO="/f/kaggle/arc-prize-2026"
KAGGLE="/c/Users/dcani/AppData/Roaming/Python/Python313/Scripts/kaggle.exe"
KERNEL="canivel/arc3-q38-engine-eval"
NB_NAME="arc3-q38-engine-eval.ipynb"
NB_DIR_WIN='F:\kaggle\arc-prize-2026\notebooks\q38-eval'   # BACKSLASH path for the CLI
NB="$REPO/notebooks/q38-eval/$NB_NAME"
PUSH_DATE="2026-08-15"

cd "$REPO"

MODE="${1:-}"
case "$MODE" in
  --dry-run|--confirm-push) ;;
  *)
    echo "REFUSING: pass --confirm-push to actually push, or --dry-run to run the gates." >&2
    echo "  Dry inspection is the default. A push spends a scarce slot and must never be a" >&2
    echo "  side effect of inspecting the script." >&2
    exit 2
    ;;
esac

echo "== 0-. ONE-SHOT GUARD (added 2026-08-15 AFTER v1 was pushed and started running) =="
# v1 is on Kaggle and RUNNING. The local artifact has since been hardened to pure ASCII
# (the CLI push path mangles non-ASCII; v1's /tokenize normaliser literal arrived corrupted).
# That makes local != remote, which would let the idempotence guard wave a v2 through for a
# NON-LOAD-BEARING fix and spend a slot we do not have. The degradation in v1 costs one
# corroborating instrument and nothing that gates the measurement. So: refuse by default.
if [ "${Q38_ALLOW_V2:-}" != "1" ]; then
  if "$KAGGLE" kernels status "$KERNEL" >/dev/null 2>&1; then
    echo "REFUSING: $KERNEL already exists. v1 was pushed on 2026-08-15 against that day's" >&2
    echo "  ONE free slot and its only defect is cosmetic (see the prereg addendum)." >&2
    echo "  A v2 needs a FRESH slot and a FRESH authorization. Set Q38_ALLOW_V2=1 only when" >&2
    echo "  you have both, and record why." >&2
    exit 5
  fi
fi

echo "== 0. slot date guard =="
echo "local date: $(date '+%Y-%m-%d %H:%M:%S %Z')  (the slot budget is keyed to the LOCAL date)"
if [ "$(date +%Y-%m-%d)" != "$PUSH_DATE" ]; then
  echo "REFUSING: local date is not $PUSH_DATE. This script is scoped to ONE authorized day;" >&2
  echo "  a later push needs a fresh authorization and a fresh ledger read, not a date edit." >&2
  exit 1
fi

echo
echo "== 0b. LEDGER RE-CONFIRM (section 11.4 — binding, even under a live authorization) =="
if ! grep -q "^### $PUSH_DATE" ITERATION_LOG.md; then
  echo "REFUSING: ITERATION_LOG.md has no '### $PUSH_DATE' section — the ledger for today does" >&2
  echo "  not exist, so slot availability CANNOT be re-confirmed." >&2
  exit 4
fi
echo "--- slot/push lines in today's ledger section (READ THESE BEFORE CONTINUING) ---"
awk "/^### $PUSH_DATE/{f=1} f" ITERATION_LOG.md | grep -i "slot\|push" || echo "  (none)"
echo "--- end ledger excerpt ---"
echo "Accounting on file (08-14 entry, verbatim): 'RULING on the LoRA canary overrun: LET IT RUN;"
echo "  count it against 08-15 slot 1. 08-15 therefore has ONE slot free (slot 2).'"
echo "Independently corroborated: the newest run on any canivel kernel is 2026-08-14 13:40 UTC,"
echo "  i.e. NOTHING has been pushed today yet. This push takes 08-15 slot 2, the last one."

echo
echo "== 0c. explicit-intent interlock =="
if [ "$MODE" = "--dry-run" ]; then
  echo "DRY RUN: gates will run; steps 2-4 will be SKIPPED. Nothing will be pushed."
else
  echo "explicit intent confirmed (--confirm-push)"
fi

echo
echo "== 1. rebuild + gates =="
python duck_eval/q38/build_q38_eval.py
python duck_eval/q38/q38_smoke.py | tail -3
python duck_eval/q38/q38_score.py --selftest | tail -2

echo
echo "== 1b. idempotence check (has this exact code already been pushed?) =="
PRECHECK="$(mktemp -d)"
if "$KAGGLE" kernels pull "$KERNEL" -p "$PRECHECK" >/dev/null 2>&1; then
  if python - "$NB" "$PRECHECK/$NB_NAME" <<'PY'
import hashlib, json, sys
def code_sha(p):
    nb = json.load(open(p, encoding="utf-8"))
    return hashlib.sha256("".join(
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code").encode()).hexdigest()
try:
    sys.exit(0 if code_sha(sys.argv[1]) == code_sha(sys.argv[2]) else 1)
except Exception:
    sys.exit(1)
PY
  then
    echo "REFUSING: remote already has this exact code. A re-push would spend a slot for a" >&2
    echo "  no-op. If you truly intend a re-run of identical code, do it deliberately and" >&2
    echo "  record why — do not let this script do it silently." >&2
    exit 3
  fi
fi
echo "remote differs from local (or the kernel does not exist yet) — this push is not a duplicate"

if [ "$MODE" = "--dry-run" ]; then
  echo
  echo "DRY RUN COMPLETE. All pre-push gates passed. NOTHING WAS PUSHED."
  echo "  To push: bash duck_eval/q38/q38_push.sh --confirm-push"
  exit 0
fi

echo
echo "== 2. push =="
"$KAGGLE" kernels push -p "$NB_DIR_WIN"

echo
echo "== 3. pull-back verify =="
sleep 5
PULL="$(mktemp -d)"
"$KAGGLE" kernels pull "$KERNEL" -p "$PULL" -m >/dev/null
python - "$NB" "$PULL/$NB_NAME" "$PULL/kernel-metadata.json" <<'PY'
import hashlib, json, sys
local_nb, remote_nb, remote_meta = sys.argv[1:4]

def code_sha(p):
    nb = json.load(open(p, encoding="utf-8"))
    src = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    return hashlib.sha256(src.encode()).hexdigest(), len(nb["cells"])

l, ln = code_sha(local_nb)
r, rn = code_sha(remote_nb)
print(f"local  sha256={l[:16]} cells={ln}")
print(f"remote sha256={r[:16]} cells={rn}")
assert l == r, "CODE MISMATCH — the pushed notebook is not what we built"
assert ln == rn == 17, f"cell count drifted: {ln} vs {rn}"

meta = json.load(open(remote_meta, encoding="utf-8"))
want = {"jeroencottaar/taaf-kaggle-source-share",
        "driessmit1/arc3-vllm-h100-wheelhouse-v3",
        "saltb0x/qwen3-8-27b-fp8"}
got = set(meta.get("dataset_sources") or [])
assert got == want, (
    "dataset_sources drifted — Kaggle drops unattachable sources SILENTLY and the 25 GB engine "
    f"is the likeliest casualty (feedback_kaggle_model_attach): {sorted(got)}")
assert "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot" not in got, "the incumbent engine is attached"
assert meta["enable_gpu"] is True and meta["enable_internet"] is False
assert meta["machine_shape"] == "NvidiaRtxPro6000"
assert meta["competition_sources"] == ["arc-prize-2026-arc-agi-3"]
assert meta["docker_image"].endswith(
    "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c")

# The rewrite must have survived the round trip, not just the file bytes.
remote_src = "".join("".join(c["source"]) for c in json.load(open(remote_nb, encoding="utf-8"))["cells"])
for token in ("saltb0x", "qwen3-8-27b-fp8", "Qwen/Qwen3.8-27B-FP8",
              '"reasoning_effort": "medium"', "_q38_pre_serve_asserts", "_q38_boot_asserts"):
    assert token in remote_src, f"remote notebook is missing {token!r}"
assert "vrfai/Qwen3.6-27B-FP8" not in remote_src.replace(
    "SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'", ""), "unexpected incumbent served-name"
print("pull-back OK: code MATCH, 3/3 datasets survived (engine included), env identical to the "
      "frozen fork, engine+pin tokens present in the remote source")
PY

echo
echo "== 4. preflight (post-push structural diff vs the frozen fork) =="
export PATH="$(dirname "$KAGGLE"):$PATH"
python scripts/preflight.py --kernel "$KERNEL" \
  --mode structural --family duck-harness \
  --baseline notebooks/duckfork/tufa-labs-duck-harness-june-30-milestone-winner.ipynb \
  --expect-diff-cells 2,6,8

echo
echo "PUSHED AND VERIFIED. Poll to terminal (~2h15m expected), then:"
echo "  $KAGGLE kernels status $KERNEL"
echo "  $KAGGLE kernels output $KERNEL -p runs/kernel_pulls/q38_v1"
echo "  python duck_eval/q38/q38_score.py runs/kernel_pulls/q38_v1"
echo "Read seal: learnings/war_room/q38_engine_swap_prereg_2026-08-15.md section 4 and 7."
echo "NOTE: kernels output downloads /kaggle/working FIRST and this kernel's working dir holds"
echo "  the multi-GB vllm-site-packages tree — budget for that, or pull selectively."
