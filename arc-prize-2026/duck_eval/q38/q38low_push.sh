#!/usr/bin/env bash
# Q38 TOKEN-COST ARM (reasoning_effort=low) push + verify -- 2026-08-17 slot 1.
# Single variable vs the completed engine arm: the effort value. Prereg sections 13-18.
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
KERNEL="canivel/arc3-q38-low-eval"
NB_NAME="arc3-q38-low-eval.ipynb"
NB_DIR_WIN='F:\kaggle\arc-prize-2026\notebooks\q38-low-eval'   # BACKSLASH path for the CLI
NB="$REPO/notebooks/q38-low-eval/$NB_NAME"
PUSH_DATE="2026-08-17"   # AUTHORIZED: 08-17 slot 1. Not today.

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

echo "== 0-. ONE-SHOT GUARD =="
# The arm is one slot. If the kernel already exists, a second push needs a fresh slot and a
# fresh authorization -- not a re-run of this script. (2026-08-14: two sessions spent both of
# that day's slots on one artifact because nothing checked this before pushing.)
if [ "${Q38LOW_ALLOW_V2:-}" != "1" ]; then
  if "$KAGGLE" kernels status "$KERNEL" >/dev/null 2>&1; then
    echo "REFUSING: $KERNEL already exists. A v2 needs a fresh slot, a fresh ledger read and" >&2
    echo "  Q38LOW_ALLOW_V2=1. Record why before you set it." >&2
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
echo "Accounting on file (08-16 morning check, verbatim): 'Kernel builds: both terminal, both"
echo "  ERROR, zero in flight.' 08-15 spent both its slots (LoRA-canary overrun = slot 1,"
echo "  Q38 v1 = slot 2). 08-16 therefore opens with BOTH slots free; this push takes slot 1."
echo "Corroborate before continuing: the excerpt printed above is today's ledger section, and"
echo "  no push line for $PUSH_DATE may appear in it. If one does, STOP -- this is slot 2 and"
echo "  it is the last one."

echo
echo "== 0c. explicit-intent interlock =="
if [ "$MODE" = "--dry-run" ]; then
  echo "DRY RUN: gates will run; steps 2-4 will be SKIPPED. Nothing will be pushed."
else
  echo "explicit intent confirmed (--confirm-push)"
fi

echo
echo "== 1. rebuild + gates =="
Q38_ARM=low python duck_eval/q38/build_q38_eval.py
Q38_ARM=low python duck_eval/q38/q38_smoke.py | tail -3
python duck_eval/q38/q38low_score.py --selftest | tail -2
# the two arms must be one variable apart and nothing more
python duck_eval/q38/q38_arm_diff.py

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


echo "== 1c. push-target integrity (the dir's metadata decides where the push goes) =="
python - "$KERNEL" <<'PY'
import json, sys
meta = json.load(open("F:/kaggle/arc-prize-2026/notebooks/q38-low-eval/kernel-metadata.json"))
want = sys.argv[1]
assert meta["id"] == want, (
    f"PUSH-TARGET MISMATCH: dir metadata says {meta['id']!r} but this script is for {want!r}. "
    "kaggle pushes what the DIRECTORY says, not what the script intends.")
print(f"push target verified: {meta['id']}")
PY

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
assert ln == rn == 17, f"cell count drifted: {ln} vs {rn}"

# VERIFIER FIX 1 (2026-08-16, after the v2 push): Kaggle's push path mangles non-ASCII on the
# round trip (UTF-8 bytes re-read as cp1252: U+2014 -> 'a-hat, euro, right-double-quote'). The
# em-dash that triggers it lives at offset 471 of BASELINE cell 16 -- it is the FROZEN FORK's
# own byte, not ours, and --expect-diff-cells is 2,6,8, so ASCII-hardening it would manufacture
# a fourth differing cell and break exactly the byte-identity D2/D3/D4 protect. The artifact is
# right and this check was wrong. preflight.py's D4 already treated this class as equal; step 3
# did not, and aborted BEFORE the load-bearing dataset/env checks below -- the dangerous part.
# So: exact match preferred; otherwise equality is accepted only when every surviving difference
# is confined to non-ASCII codepoints. Any ASCII-visible drift is still fatal.
if l != r:
    la, ra = (s.encode("ascii", "ignore") for s in (
        "".join("".join(c["source"]) for c in json.load(open(p, encoding="utf-8"))["cells"]
                if c["cell_type"] == "code") for p in (local_nb, remote_nb)))
    assert la == ra, "CODE MISMATCH — the pushed notebook is not what we built (ASCII-visible drift)"
    print("code MATCH after non-ASCII round-trip normalisation (ASCII-identical; see VERIFIER FIX 1)")
else:
    print("code MATCH exactly")

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
assert '"reasoning_effort": "medium"' not in remote_src, "MEDIUM literal in the LOW arm"
for token in ("saltb0x", "qwen3-8-27b-fp8", "Qwen/Qwen3.8-27B-FP8",
              '"reasoning_effort": "low"', "_q38_pre_serve_asserts", "_q38_boot_asserts"):
    assert token in remote_src, f"remote notebook is missing {token!r}"
# VERIFIER FIX 2 (2026-08-16): this assert predates v2's poisoning gate. The incumbent name is
# now SUPPOSED to appear a second time -- in `Q38_VETO`, the forbidden-served-name tuple whose
# whole job is to make a silently-served vrfai/Qwen3.6-27B-FP8 fatal (prereg 11.2's negative
# control). Asserting the string is absent would delete the gate that protects the measurement.
# So enumerate the two LICENSED sites and forbid a third, instead of forbidding the string.
LICENSED = ("SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'",       # rewrite table: the search side
            "Q38_VETO = ('vrfai-qwen3-6-27b-fp8-hf-snapshot', 'vrfai/Qwen3.6-27B-FP8')")
stripped = remote_src
for site in LICENSED:
    assert site in stripped, f"MISSING licensed site — the poisoning gate or rewrite is gone: {site!r}"
    stripped = stripped.replace(site, "", 1)
assert "vrfai/Qwen3.6-27B-FP8" not in stripped, (
    "unexpected incumbent served-name at an UNLICENSED site — the engine swap may be poisoned")
print("pull-back OK: code MATCH, 3/3 datasets survived (engine included), env identical to the "
      "frozen fork, engine+pin tokens present, poisoning gate (Q38_VETO) intact in the remote source")
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
echo "  $KAGGLE kernels output $KERNEL -p runs/kernel_pulls/q38low_v1"
echo "  python duck_eval/q38/q38low_score.py runs/kernel_pulls/q38low_v1"
echo "Read seal: learnings/war_room/q38_engine_swap_prereg_2026-08-15.md sections 15-17.
echo "  Pull selectively: kaggle kernels output $KERNEL -p <dir> --file-pattern '^(benchmark\.json|summary\.txt)$'""
echo "POST-MORTEM (if it ERRORs): do NOT start with kernels output - it front-loads the"
echo "  multi-GB vllm-site-packages tree. Use CLI 2.2.3 instead:"
echo "    kaggle kernels logs $KERNEL > runs/kernel_pulls/q38low_v1/q38.log"
echo "  which streams full stdout with per-line timestamps and never touches /kaggle/working."
