#!/usr/bin/env bash
# B122 v2 push + verify, as one auditable sequence (2026-08-14 slot 1).
#
# Written BEFORE the push so the midnight action is atomic, reproducible and reviewable
# rather than a sequence of ad-hoc commands typed at 00:00. Every step is a gate:
# a non-zero exit anywhere stops the sequence.
#
#   1. rebuild from the frozen duckfork and re-run the smoke (86 checks) + scorer selftest (16)
#   2. push the kernel
#   3. PULL BACK and verify the code-cell sha256 and that all three dataset_sources survived
#      (Kaggle silently drops unattachable sources — feedback_kaggle_model_attach)
#   4. run preflight --family duck-harness (post-push verifier; a fresh slug 403s pre-push)
#
# Usage:  bash duck_eval/a17/b122_push_v2.sh
set -euo pipefail

REPO="/f/kaggle/arc-prize-2026"
KAGGLE="/c/Users/dcani/AppData/Roaming/Python/Python313/Scripts/kaggle.exe"
KERNEL="canivel/arc3-b122-boot-canary"
NB_DIR_WIN='F:\kaggle\arc-prize-2026\notebooks\b122-canary'   # BACKSLASH path for the CLI
NB="$REPO/notebooks/b122-canary/arc3-b122-boot-canary.ipynb"
BASE="$REPO/notebooks/duckfork/tufa-labs-duck-harness-june-30-milestone-winner.ipynb"
PULL="$(mktemp -d)"

cd "$REPO"

echo "== 0. slot check =="
echo "local date: $(date '+%Y-%m-%d %H:%M:%S %Z')  (slot budget is keyed to the LOCAL date)"
if [ "$(date +%Y-%m-%d)" != "2026-08-14" ]; then
  echo "REFUSING: local date is not 2026-08-14; 08-13 slots are spent (2/2)." >&2
  exit 1
fi

echo "== 1. rebuild + gates =="
python duck_eval/a17/build_b122_boot_canary.py
python duck_eval/a17/b122_canary_smoke.py | tail -2
python duck_eval/a17/b122_score.py --selftest | tail -2

echo "== 2. push =="
"$KAGGLE" kernels push -p "$NB_DIR_WIN"

echo "== 3. pull-back verify =="
"$KAGGLE" kernels pull "$KERNEL" -p "$PULL" -m >/dev/null
python - "$NB" "$PULL/arc3-b122-boot-canary.ipynb" "$PULL/kernel-metadata.json" <<'PY'
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
        "jcole75/arc3-qwen36-runtime-wheels",
        "ippeiogawa/qwen35-122b-a10b-nvfp4"}
got = set(meta.get("dataset_sources") or [])
assert got == want, f"dataset_sources drifted (Kaggle drops unattachable silently): {sorted(got)}"
assert meta["enable_gpu"] is True and meta["enable_internet"] is False
assert meta["machine_shape"] == "NvidiaRtxPro6000"
assert meta["docker_image"].endswith(
    "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c")
print("pull-back OK: code MATCH, 3/3 datasets survived, env identical to the family")
PY

echo "== 4. preflight (post-push) =="
python scripts/preflight.py --kernel "$KERNEL" \
  --mode structural --family duck-harness \
  --baseline notebooks/duckfork/tufa-labs-duck-harness-june-30-milestone-winner.ipynb \
  --expect-diff-cells 2,6,8,14 | head -20

echo
echo "PUSHED AND VERIFIED. Now poll to terminal, then:"
echo "  $KAGGLE kernels output $KERNEL -p <dir>"
echo "  python duck_eval/a17/b122_score.py <dir>"
