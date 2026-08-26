#!/usr/bin/env bash
# LORA SERVE CANARY v2 push + verify, as one auditable sequence (2026-08-15 slot 2).
#
# WHY THIS FILE EXISTS. On 2026-08-15 a readiness audit found that this lane had NO push
# script at all: v1 was pushed with an ad-hoc `kaggle kernels push`, and the two guards the
# campaign believes protect every push -- `--confirm-push` (wrong TIME) and the idempotence
# check (wrong ACTOR) -- live ONLY in `duck_eval/a17/b122_push_v2.sh`, which is hardcoded to
# the b122 kernel and to the date 2026-08-14 (its lane is CLOSED). The guards were doctrine,
# not code, on this path. This script ports them verbatim.
#
# Every step is a gate; a non-zero exit anywhere stops the sequence.
#
#   0.  LOCAL-date guard (slot budget is keyed to the LOCAL date)
#   0b. LEDGER RE-CONFIRM (section 11.4, binding): print the 2026-08-15 slot lines from
#       ITERATION_LOG.md and refuse if that section does not exist. A conditional that was
#       true when issued is not evidence that it is true now.
#   0c. EXPLICIT-INTENT INTERLOCK: --confirm-push, or --dry-run to run 0..1b and stop.
#   1.  rebuild from the frozen duckfork + smoke (75) + scorer selftest (35)
#   1b. IDEMPOTENCE GUARD -- exit 3 if remote code already equals local
#   2.  push
#   3.  pull-back verify: code sha256, all FOUR dataset_sources, env byte-match
#   4.  preflight --family duck-harness --expect-diff-cells 2,6,8,14 (POST-push by design)
#
# Usage:  bash duck_eval/lora/lora_push_v2.sh --dry-run       # gates only, NEVER pushes
#         bash duck_eval/lora/lora_push_v2.sh --confirm-push  # the real thing
set -euo pipefail

REPO="/f/kaggle/arc-prize-2026"
KAGGLE="/c/Users/dcani/AppData/Roaming/Python/Python313/Scripts/kaggle.exe"
KERNEL="canivel/arc3-lora-serve-canary"
NB_NAME="arc3-lora-serve-canary.ipynb"
NB_DIR_WIN='F:\kaggle\arc-prize-2026\notebooks\lora-serve-canary'   # BACKSLASH path for the CLI
NB="$REPO/notebooks/lora-serve-canary/$NB_NAME"
PUSH_DATE="2026-08-15"

cd "$REPO"

MODE="${1:-}"
case "$MODE" in
  --dry-run|--confirm-push) ;;
  *)
    echo "REFUSING: pass --confirm-push to actually push, or --dry-run to run the gates." >&2
    echo "  Dry inspection is the default. A push spends a scarce slot and must never be" >&2
    echo "  a side effect of inspecting the script." >&2
    exit 2
    ;;
esac

echo "== 0. slot date guard =="
echo "local date: $(date '+%Y-%m-%d %H:%M:%S %Z')  (slot budget is keyed to the LOCAL date)"
if [ "$(date +%Y-%m-%d)" != "$PUSH_DATE" ]; then
  echo "REFUSING: local date is not $PUSH_DATE. This script is scoped to ONE authorized day;" >&2
  echo "  a later push needs a fresh authorization and a fresh ledger read, not a date edit." >&2
  exit 1
fi

echo
echo "== 0b. LEDGER RE-CONFIRM (section 11.4 -- binding, even under a live authorization) =="
if ! grep -q "^### $PUSH_DATE" ITERATION_LOG.md; then
  echo "REFUSING: ITERATION_LOG.md has no '### $PUSH_DATE' section -- the ledger for today" >&2
  echo "  does not exist, so slot availability CANNOT be re-confirmed." >&2
  exit 4
fi
echo "--- slot/push lines in today's ledger section (READ THESE BEFORE CONTINUING) ---"
awk "/^### $PUSH_DATE/{f=1} f" ITERATION_LOG.md | grep -i "slot\|push" || echo "  (none)"
echo "--- end ledger excerpt ---"
echo "Authorization on file (08-14 entry): '08-15 slot 1 already counted (the ERRORed canary,"
echo "  per the 08-14 overrun ruling) => 08-15 has ONE free slot. AUTHORIZED: LoRA serve"
echo "  canary v2 takes 08-15 slot 2, ledger re-confirmed immediately before push.'"

echo
echo "== 0c. explicit-intent interlock =="
if [ "$MODE" = "--dry-run" ]; then
  echo "DRY RUN: gates will run; steps 2-4 will be SKIPPED. Nothing will be pushed."
else
  echo "explicit intent confirmed (--confirm-push)"
fi

echo
echo "== 1. rebuild + gates =="
python duck_eval/lora/build_lora_serve_canary.py
python duck_eval/lora/lora_canary_smoke.py | tail -2
python duck_eval/lora/lora_serve_score.py --selftest | tail -2

# IDEMPOTENCE GUARD (ported from b122_push_v2.sh). The pull-back in step 3 compares local to
# remote AFTER pushing -- at which point a duplicate push and a first push look identical.
# The check has to happen BEFORE. On 2026-08-14 two sessions spent both of the day's slots
# on one screen because nothing did this.
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
    echo "REFUSING: remote already has this exact code. A re-push would spend a slot for" >&2
    echo "  a no-op. If you truly intend a re-run of identical code, do it deliberately" >&2
    echo "  and record why -- do not let this script do it silently." >&2
    exit 3
  fi
fi
echo "remote differs from local (or kernel absent) -- push is not a duplicate"

if [ "$MODE" = "--dry-run" ]; then
  echo
  echo "DRY RUN COMPLETE. All pre-push gates passed. NOTHING WAS PUSHED."
  echo "  To push: bash duck_eval/lora/lora_push_v2.sh --confirm-push"
  exit 0
fi

echo
echo "== 2. push =="
"$KAGGLE" kernels push -p "$NB_DIR_WIN"

echo
echo "== 3. pull-back verify =="
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
assert l == r, "CODE MISMATCH -- the pushed notebook is not what we built"
assert ln == rn == 17, f"cell count drifted: {ln} vs {rn}"

meta = json.load(open(remote_meta, encoding="utf-8"))
want = {"jeroencottaar/taaf-kaggle-source-share",
        "driessmit1/arc3-vllm-h100-wheelhouse-v3",
        "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot",
        "canivel/arc3-lora-probe-adapters"}
got = set(meta.get("dataset_sources") or [])
assert got == want, f"dataset_sources drifted (Kaggle drops unattachable silently): {sorted(got)}"
assert meta["enable_gpu"] is True and meta["enable_internet"] is False
assert meta["machine_shape"] == "NvidiaRtxPro6000"
assert meta["competition_sources"] == ["arc-prize-2026-arc-agi-3"]
assert meta["docker_image"].endswith(
    "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c")
print("pull-back OK: code MATCH, 4/4 datasets survived, env identical to the family")
PY

echo
echo "== 4. preflight (post-push) =="
# preflight.py shells out to a BARE `kaggle`, so it needs the Scripts dir on PATH.
export PATH="$(dirname "$KAGGLE"):$PATH"
python scripts/preflight.py --kernel "$KERNEL" \
  --mode structural --family duck-harness \
  --baseline notebooks/duckfork/tufa-labs-duck-harness-june-30-milestone-winner.ipynb \
  --expect-diff-cells 2,6,8,14

echo
echo "PUSHED AND VERIFIED. Now poll to terminal, then:"
echo "  $KAGGLE kernels output $KERNEL -p <dir>"
echo "  python duck_eval/lora/lora_serve_score.py <dir>"
echo "NOTE: kernels output downloads /kaggle/working FIRST and this kernel's working dir"
echo "  holds the multi-GB vllm-site-packages tree -- budget for that, or pull selectively."
