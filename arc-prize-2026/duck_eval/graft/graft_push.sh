#!/usr/bin/env bash
# GRAFT FLOOR ARM push + verify -- 2026-08-18 slot 1.
# PREREG: learnings/war_room/graft_floor_prereg_2026-08-17.md (sealed 08-17, pre-push).
# AUDIT:  duck_eval/graft/bundle_audit_2026-08-17.md  (read this before you push).
#
# ONE VARIABLE: the source-bundle dataset gains the public taaf_grafts layer and cell 12 calls
# install() with thtennant's published v19 flag set. Engine UNCHANGED (Qwen3.6).
#
# NOTE, PROMINENTLY: this is NOT the arm the 08-17 ruling authorized. That arm was
# banking+transfer+shortcircuit; the audit proved banking's trigger (run.state=="won") has
# never once fired in 470 recorded campaign game-runs and transfer needs clone siblings our
# 25-unique-game_id eval rail does not have. Both are held OFF and ASSERTED ABSENT. If the
# coordinator overrules and wants the literal authorized arm, that is a different script and a
# different prereg -- do not edit the FLAGS here to get there.
#
# Ports every guard this campaign has paid for:
#   0-. ONE-SHOT GUARD    -- a v2 needs a fresh slot + fresh authorization (08-14: two sessions
#                            spent both of one day's slots on a single artifact)
#   0.  LOCAL-date guard  -- the slot budget is keyed to the LOCAL date
#   0b. LEDGER RE-CONFIRM -- a conditional that was true when issued is not evidence now
#   0c. EXPLICIT-INTENT   -- --confirm-push, or --dry-run to run 0..1c and stop
#   1.  rebuild from the frozen duckfork + smoke + SEALED-scorer selftest
#   1b. IDEMPOTENCE GUARD -- step 3 compares local to remote AFTER pushing, so a duplicate and
#                            a first push are indistinguishable there
#   1c. PUSH-TARGET INTEGRITY (shipped 2026-08-17, the day a confirm-push went to the WRONG
#       KERNEL) -- `kaggle kernels push -p <dir>` obeys the DIRECTORY's metadata; the KERNEL
#       variable in a script like this one is otherwise just a comment.
#   2.  push
#   3.  pull-back verify: code sha, cell count, all 3 dataset_sources, docker sha, GPU, machine,
#       internet, and the graft/engine tokens in the REMOTE source
#   4.  preflight --expect-diff-cells 2,6,12 (POST-push by design: it pulls from Kaggle, so it
#       cannot run against a slug that does not exist yet)
#
# Usage:  bash duck_eval/graft/graft_push.sh --dry-run       # gates only, NEVER pushes
#         bash duck_eval/graft/graft_push.sh --confirm-push  # the real thing
set -euo pipefail

REPO="/f/kaggle/arc-prize-2026"
KAGGLE="/c/Users/dcani/AppData/Roaming/Python/Python313/Scripts/kaggle.exe"
KERNEL="canivel/arc3-graft-floor-eval"
NB_NAME="arc3-graft-floor-eval.ipynb"
NB_DIR_WIN='F:\kaggle\arc-prize-2026\notebooks\graft-floor-eval'   # BACKSLASH path for the CLI
NB_DIR_POSIX="$REPO/notebooks/graft-floor-eval"
NB="$NB_DIR_POSIX/$NB_NAME"
PUSH_DATE="2026-08-18"   # AUTHORIZED: 08-18 slot 1. 08-17 was 2/2 spent.

FORK_DS="thtennant/taaf-kaggle-source-share-fork"
STOCK_DS="jeroencottaar/taaf-kaggle-source-share"
WHEELS_DS="driessmit1/arc3-vllm-h100-wheelhouse-v3"
ENGINE_DS="driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"

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
if [ "${GRAFT_ALLOW_V2:-}" != "1" ]; then
  if "$KAGGLE" kernels status "$KERNEL" >/dev/null 2>&1; then
    echo "REFUSING: $KERNEL already exists. A v2 needs a fresh slot, a fresh ledger read and" >&2
    echo "  GRAFT_ALLOW_V2=1. Record why before you set it." >&2
    exit 5
  fi
fi
echo "fresh slug confirmed (feedback_fresh_kernel_slug)"

echo
echo "== 0. slot date guard =="
echo "local date: $(date '+%Y-%m-%d %H:%M:%S %Z')  (the slot budget is keyed to the LOCAL date)"
if [ "$(date +%Y-%m-%d)" != "$PUSH_DATE" ]; then
  echo "REFUSING: local date is not $PUSH_DATE. This script is scoped to ONE authorized day;" >&2
  echo "  a later push needs a fresh authorization and a fresh ledger read, not a date edit." >&2
  exit 1
fi

echo
echo "== 0b. LEDGER RE-CONFIRM (binding, even under a live authorization) =="
if ! grep -q "^### $PUSH_DATE" ITERATION_LOG.md; then
  echo "REFUSING: ITERATION_LOG.md has no '### $PUSH_DATE' section — the ledger for today does" >&2
  echo "  not exist, so slot availability CANNOT be re-confirmed." >&2
  exit 4
fi
echo "--- slot/push lines in today's ledger section (READ THESE BEFORE CONTINUING) ---"
awk "/^### $PUSH_DATE/{f=1} f" ITERATION_LOG.md | grep -i "slot\|push" || echo "  (none)"
echo "--- end ledger excerpt ---"
echo "Accounting on file (08-17, verbatim): '08-17 = 2 of 2 slots spent' (an unintended push to"
echo "  arc3-q38-engine-eval v3 counted as one, 08-14 precedent). 08-18 therefore opens with"
echo "  BOTH slots free; this push takes slot 1."
echo "Corroborate before continuing: no push line for $PUSH_DATE may appear above. If one does,"
echo "  STOP -- this is slot 2 and it is the last one."
echo "ALSO RE-READ THE LEDGER BAR, DO NOT CACHE IT: runs/ledger.json (it drifts every draw)."

echo
echo "== 0c. explicit-intent interlock =="
if [ "$MODE" = "--dry-run" ]; then
  echo "DRY RUN: gates will run; steps 2-4 will be SKIPPED. Nothing will be pushed."
else
  echo "explicit intent confirmed (--confirm-push)"
fi

echo
echo "== 1. rebuild + gates =="
python duck_eval/graft/build_graft_eval.py
python duck_eval/graft/graft_smoke.py | tail -3
python duck_eval/graft/graft_score.py --selftest | tail -4

echo
echo "== 1a. bundle re-audit (Kaggle attaches the LATEST version; metadata cannot pin one) =="
# The fork was republished 2026-08-17 00:26 and is actively maintained. If its bytes changed
# since the audit, the arm is no longer the arm that was sealed -- re-audit BEFORE spending the
# slot, not after reading the result.
python duck_eval/graft/graft_bundle_check.py

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

echo
echo "== 1c. push-target integrity (the dir's metadata decides where the push goes) =="
python - "$KERNEL" "$NB_DIR_POSIX/kernel-metadata.json" <<'PY'
import json, sys
want, meta_path = sys.argv[1], sys.argv[2]
meta = json.load(open(meta_path, encoding="utf-8"))
assert meta["id"] == want, (
    f"PUSH-TARGET MISMATCH: dir metadata says {meta['id']!r} but this script is for {want!r}. "
    "kaggle pushes what the DIRECTORY says, not what the script intends.")
print(f"push target verified: {meta['id']}  (metadata read from {meta_path})")
PY

if [ "$MODE" = "--dry-run" ]; then
  echo
  echo "DRY RUN COMPLETE. All pre-push gates passed. NOTHING WAS PUSHED."
  echo "  To push: bash duck_eval/graft/graft_push.sh --confirm-push"
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
python - "$NB" "$PULL/$NB_NAME" "$PULL/kernel-metadata.json" \
        "$FORK_DS" "$STOCK_DS" "$WHEELS_DS" "$ENGINE_DS" <<'PY'
import hashlib, json, sys
local_nb, remote_nb, remote_meta, FORK_DS, STOCK_DS, WHEELS_DS, ENGINE_DS = sys.argv[1:8]

def code(p):
    nb = json.load(open(p, encoding="utf-8"))
    return "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"), len(nb["cells"])

lsrc, ln = code(local_nb)
rsrc, rn = code(remote_nb)
l, r = (hashlib.sha256(s.encode()).hexdigest() for s in (lsrc, rsrc))
print(f"local  sha256={l[:16]} cells={ln}")
print(f"remote sha256={r[:16]} cells={rn}")
assert ln == rn == 17, f"cell count drifted: {ln} vs {rn}"

# VERIFIER FIX 1 (inherited from q38, 2026-08-16): Kaggle's push path mangles non-ASCII on the
# round trip (UTF-8 re-read as cp1252). The em-dash that triggers it is the FROZEN FORK's own
# byte in cell 16, and --expect-diff-cells is 2,6,12, so ASCII-hardening it would manufacture a
# fourth differing cell. Exact match preferred; otherwise equality is accepted only when every
# surviving difference is confined to non-ASCII codepoints. ASCII-visible drift is still fatal.
if l != r:
    la, ra = (s.encode("ascii", "ignore") for s in (lsrc, rsrc))
    assert la == ra, "CODE MISMATCH — the pushed notebook is not what we built (ASCII-visible drift)"
    print("code MATCH after non-ASCII round-trip normalisation (ASCII-identical; VERIFIER FIX 1)")
else:
    print("code MATCH exactly")

meta = json.load(open(remote_meta, encoding="utf-8"))
want = {FORK_DS, WHEELS_DS, ENGINE_DS}
got = set(meta.get("dataset_sources") or [])
assert got == want, (
    "dataset_sources drifted — Kaggle drops unattachable sources SILENTLY "
    f"(feedback_kaggle_model_attach): got {sorted(got)}, want {sorted(want)}")
# THE CENTRAL ATTACHMENT ASSERT, and it is the opposite sign from the q38 arm: the graft fork
# REPLACES the stock bundle. If both were attached, _find_bundle_dir() takes the FIRST rglob
# match of taaf-kaggle-bundle.json (the fork carries that marker too), BUNDLE_DIR becomes
# ambiguous, and a stock resolution would run STOCK while looking like a clean arm.
assert STOCK_DS not in got, (
    "the STOCK source bundle is still attached alongside the fork — BUNDLE_DIR is ambiguous "
    "and a stock resolution would silently run stock (and we would score it as a NULL)")
# ... and unlike the q38 arm, the incumbent engine MUST be present: it is the unchanged control.
assert ENGINE_DS in got, "the incumbent Qwen3.6 engine is NOT attached — the arm has 2 variables"
assert meta["enable_gpu"] is True and meta["enable_internet"] is False
assert meta["machine_shape"] == "NvidiaRtxPro6000"
assert meta["competition_sources"] == ["arc-prize-2026-arc-agi-3"]
assert meta["docker_image"].endswith(
    "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c")

# The rewrite must have survived the round trip, not just the file bytes.
full = "".join("".join(c["source"]) for c in json.load(open(remote_nb, encoding="utf-8"))["cells"])
for token in ("taaf_grafts.composite import install", '"efficiency": True', '"retry_guard": True',
              '"shortcircuit": True', '"goalkeep": True', '"hudmask": True',
              "expected_version=1", FORK_DS, ENGINE_DS):
    assert token in full, f"remote notebook is missing {token!r}"
# The arm is DEFINED by the exclusion of the unreachable flags. Enumerate the LICENSED mentions
# (the banner and the explanatory comments name them on purpose) and forbid an ARMED one.
for flag in ("banking", "transfer"):
    assert f'"{flag}": True' not in full, (
        f"FORBIDDEN flag {flag!r} is ARMED in the remote notebook — banking's trigger has never "
        "fired in 470 recorded game-runs and transfer has no clone siblings on this rail; "
        "arming either makes the result uninterpretable")
# cell 6's attachment site must not carry the stock ref (the banner may NAME it: "X REPLACES Y")
cell6 = "".join(json.load(open(remote_nb, encoding="utf-8"))["cells"][6]["source"])
ds_lines = [x for x in cell6.splitlines() if x.strip().startswith("DATASET_SOURCES = ")]
assert len(ds_lines) == 1, f"cell 6 has {len(ds_lines)} DATASET_SOURCES assignments"
assert STOCK_DS not in ds_lines[0], "stock bundle ref survives in cell 6's DATASET_SOURCES"
assert ds_lines[0].strip().startswith(f'DATASET_SOURCES = ["{FORK_DS}"'), (
    "the graft fork must be index 0 — cell 6 maps index 0 to BUNDLE_DIR")
print("pull-back OK: code MATCH, 3/3 datasets (fork attached, stock ABSENT, engine PRESENT), "
      "env identical to the frozen fork, all 5 flags + expected_version present, "
      "banking/transfer NOT armed, fork at index 0")
PY

echo
echo "== 4. preflight (post-push structural diff vs the frozen fork) =="
export PATH="$(dirname "$KAGGLE"):$PATH"
python scripts/preflight.py --kernel "$KERNEL" \
  --mode structural --family duck-harness \
  --baseline notebooks/duckfork/tufa-labs-duck-harness-june-30-milestone-winner.ipynb \
  --expect-diff-cells 2,6,12

echo
echo "PUSHED AND VERIFIED. Poll to terminal (~2h15m expected), then:"
echo "  $KAGGLE kernels status $KERNEL"
echo "  $KAGGLE kernels output $KERNEL -p runs/kernel_pulls/graft_floor_v1 --file-pattern '^(benchmark\\.json|summary\\.txt)\$'"
echo "  python duck_eval/graft/graft_score.py runs/kernel_pulls/graft_floor_v1"
echo "Read seal: learnings/war_room/graft_floor_prereg_2026-08-17.md sections 4-5."
echo "  The scorer needs the LOG to certify the install, so also pull logs:"
echo "    kaggle kernels logs $KERNEL > runs/kernel_pulls/graft_floor_v1/graft.log   # CLI 2.2.3"
echo "POST-MORTEM (if it ERRORs): do NOT start with kernels output — it front-loads the"
echo "  multi-GB vllm-site-packages tree. Use the logs route above."
echo
echo "REMINDER OF THE READING RULE: an uncertifiable install is INFRA DEATH, never a NULL."
echo "  install() never raises and degrades SILENTLY to stock, so a stock run and a genuine"
echo "  null are byte-identical in benchmark.json. The banner is the only thing that separates"
echo "  them. Do not read the levels number before the scorer certifies the banner."
