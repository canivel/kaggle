"""Build the war EVAL kernels: byte-identical to the corresponding war
notebook except line(s) prepended to cell 2 that force the offline bench at
build time (WARPACK_FORCE_OFFLINE_BENCH=1).

Default (war-v1 eval, panel R10, llm-agents obj. 3 "banking canary" + prereg
2026-07-14 §7): source = notebooks/duckwar/arc3-duck-war.ipynb (the pushed
arc3-duck-war kernel, ledger flags OFF). Seeds 1-3 of this kernel are three
pushes of the identical notebook (the "seed" is the stochastic rerun index —
there is no explicit seed variable anywhere in the stack).

    python duck_eval/warpack/build_eval_notebook.py
    -> notebooks/duckwar-eval/{arc3-duck-war-eval.ipynb, kernel-metadata.json}

--v2 (war-v2 ledger-ON efficacy screen, panel R12 N6): source =
duck_eval/warpack/duck_warpack_v2.ipynb (war-v1 + ledger flags
{ledger,escalation} ON + LEDGER CANARY + gate detection-signal logging; built
by build_notebook.py --v2). Also stamps WAR_EVAL_SEED=1 + a greppable seed
banner: this build is ledger-ON seed 1 and pairs with ledger-OFF
arc3-duck-war-eval seed 1 (= version 1) for the paired-Δlc contrast; later
ledger-ON seeds are further pushes of this same kernel (same convention as
the ledger-OFF seeds).

    python duck_eval/warpack/build_eval_notebook.py --v2
    -> notebooks/duckwar-v2-eval/{arc3-duck-war-v2-eval.ipynb, kernel-metadata.json}

--w0 (W0 (f) game-over-continuation standalone hygiene, grinder design §5 Jul
18 row + §(f)): source = the SAME raw duckwar baseline as war-v1 eval
(notebooks/duckwar/arc3-duck-war.ipynb). Cell 2 gets the identical eval-force
line PLUS a W0 seed/banner stamp; cell 12 is REPLACED with a
continuation-patch graft that imports continuation_patch (from the attached
arc-war-kit dataset, marker-based mount-agnostic find), applies it, prints the
runtime banner, and NEVER touches warpack/ledger (W0 is standalone: duck
baseline + (f) only). Any patch failure -> VANILLA duck (never 0), exactly the
warpack graft's failure policy.

    python duck_eval/warpack/build_eval_notebook.py --w0
    -> notebooks/duckw0-eval/{arc3-duck-w0-continuation-eval.ipynb, kernel-metadata.json}

--sentinel (W2 (a) budget-sentinel window, grinder design §(a) + §5 Jul 22-24
row): source = the SAME raw duckwar baseline as war-v1 eval
(notebooks/duckwar/arc3-duck-war.ipynb). Cell 2 gets the identical eval-force
line PLUS a sentinel seed/banner stamp; cell 12 is REPLACED with a
budget-sentinel graft that imports budget_sentinel_patch (from the attached
arc-war-kit dataset, marker-based mount-agnostic find), applies it, prints the
runtime banner, and NEVER touches warpack/ledger (the (a) window is a single
flag on the duck baseline). Any patch failure -> VANILLA duck (never 0),
exactly the warpack/continuation graft's failure policy.

    python duck_eval/warpack/build_eval_notebook.py --sentinel
    -> notebooks/ducksentinel-eval/{arc3-duck-sentinel-eval.ipynb, kernel-metadata.json}

--a17-canary (A17 72B-VL throughput canary, a17_72b_screen_scope.md v1 SS2 +
v2 SS3/SS5/SS7, prereg amendment 2026-07-20 A17'): source = the SAME raw
duckwar baseline. Composition = W0 (duck + (f) continuation graft in cell 12,
NO warpack, NO ledger) because the frozen 27B numerator (480 actions/7920 s)
is w0_eval_s1, the (f) continuation run. Cell 2 gets the eval-force line + an
A17 seed/banner stamp. Cell 8 gains a marker-based, FAIL-LOUD rewrite of the
bundle's setup_commands.json serve config: Qwen2.5-VL-72B-Instruct-AWQ model
path (config.json marker find), served-model-name, max-model-len 32768,
--quantization awq_marlin, --tool-call-parser hermes, NO qwen3 reasoning
parser / thinking kwargs, LOCAL_ANALYZER_ENABLE_THINKING=false, a GPU-name
banner + RTX-PRO-6000 hard gate, boot serve asserts (tool-call round-trip +
MM image probe), and a persisted serve cmd for the liveness restart. Cell 14
gains the 4-game screen filter (ft09/sb26/lp85/vc33, versioned ids, drift
rule sec7.2), a log-only heartbeat + panel-R19 liveness gate (stall -> ONE
vLLM restart -> loud os._exit(70)), and a post-run rho_action report
(per-game N banners + denominator + MM-cache evidence).

--compaction (A22 compaction + retained-reasoning arm, prereg intent
learnings/war_room/a22_compaction_prereg_2026-08-01.md): source = the SAME raw
duckwar baseline as war-v1 eval. Cell 2 gets the eval-force line PLUS an A22
seed/banner stamp PLUS COMPACTION=1 (the arm flag). Cell 12 is REPLACED with a
compaction graft that imports compaction_patch (from the attached arc-war-kit
dataset, marker-based mount-agnostic find), applies it, prints the runtime
banner, and NEVER touches warpack/ledger-graft/sentinel (single flag on the
duck baseline; the (f) continuation default block still rides per the 07-23
amendment). compaction_patch internally reuses ledger_core as its mechanical
digester (pure logic; NOT the war-v2 ledger graft). Any patch failure ->
VANILLA duck (never 0), exactly the warpack graft's failure policy. Canary:
grep the build log for "COMPACTION " event lines (>=1/run) + the
"compaction v1: ACTIVE" banner.

    python duck_eval/warpack/build_eval_notebook.py --compaction
    -> notebooks/duckcompaction-eval/{arc3-duck-compaction-eval.ipynb, kernel-metadata.json}

--animation (animation-awareness arm, sweep 08-11 ADOPT #1; prereg intent
learnings/war_room/animation_prereg_2026-08-11.md): source = the SAME raw
duckwar baseline as war-v1 eval. Cell 2 gets the eval-force line PLUS an
animation seed/banner stamp PLUS ANIMATION_AWARE=1 (the arm flag). Cell 12 is
REPLACED with an animation graft that imports animation_patch (from the attached
arc-war-kit dataset, marker-based mount-agnostic find), applies it, prints the
runtime banner, and NEVER touches warpack/ledger-graft/sentinel/compaction (the
(f) continuation default block still rides per the 07-23 amendment). Cell 14
gains the post-run canary call (K-A1..K-A4). The arm carries NO no-op guard:
prereg sec2.2 keeps it strictly downstream and separately gated. Any patch
failure -> VANILLA duck (never 0). Canary: grep the build log for "ANIMATION "
event lines + the "animation v1: ACTIVE" banner + one "ANIMATION CANARY" line.

    python duck_eval/warpack/build_eval_notebook.py --animation
    -> notebooks/duckanimation-eval/{arc3-duck-animation-eval.ipynb, kernel-metadata.json}

--p1 (P1 zero-information action suppressor; diagnosis
learnings/war_room/efficiency_diagnosis_2026-08-12.md sec5 P1, prereg
learnings/war_room/p1_prereg_2026-08-12.md): source = the SAME raw duckwar
baseline as war-v1 eval. Cell 2 gets the eval-force line PLUS a P1 seed/banner
stamp PLUS P1_SUPPRESS=1 (the arm flag). Cell 12 is REPLACED with a P1 graft
that imports p1_suppressor_patch (from the attached arc-war-kit dataset,
marker-based mount-agnostic find), applies it, prints the runtime banner, and
NEVER touches warpack/ledger-graft/sentinel/compaction/animation (the (f)
continuation default block still rides per the 07-23 amendment). Cell 14 gains
the post-run canary call. Shipped defaults: memo_mode=noop, confirm=2,
abort_revisit=OFF -- the aggressive settings delete the level-completing batch
on tu93/sp80/ar25 in the recorded traces (see the prereg sec4). Any patch
failure -> VANILLA duck (never 0). Canary: grep the build log for "P1 " event
lines + the "p1 v1: ACTIVE" banner + one "P1 CANARY" line.

    python duck_eval/warpack/build_eval_notebook.py --p1
    -> notebooks/duckp1-eval/{arc3-duck-p1-eval.ipynb, kernel-metadata.json}

--effnote (EFFNOTE quantified per-turn efficiency note; spec
learnings/war_room/harness_diff_2026-08-13.md sec4 item #1, prereg
learnings/war_room/effnote_prereg_2026-08-13.md): source = the SAME raw duckwar
baseline as war-v1 eval. Cell 2 gets the eval-force line PLUS an EFFNOTE
seed/banner stamp PLUS EFFNOTE=1 (the arm flag). Cell 12 is REPLACED with an
EFFNOTE graft that imports effnote_patch (from the attached arc-war-kit
dataset, marker-based mount-agnostic find), applies it, prints the runtime
banner, and NEVER touches warpack/ledger-graft/sentinel/compaction/animation/p1
(the (f) continuation default block still rides per the 07-23 amendment). Cell
14 gains the post-run canary call. REPORT-ONLY: the arm never blocks, declines
or injects an action -- it only appends a bounded note (<=700 CHARACTERS, never
a token fraction) to the user turn. Target is the CLAMPED GAME-AGNOSTIC PROXY
only: no baseline table, no metadata read, no game id. Any patch failure ->
VANILLA duck (never 0). Canary: grep the build log for "EFFNOTE " event lines +
the "effnote v1: ACTIVE" banner + one "EFFNOTE CANARY" line.

    python duck_eval/warpack/build_eval_notebook.py --effnote
    -> notebooks/duckeffnote-eval/{arc3-duck-effnote-eval.ipynb, kernel-metadata.json}

POLICY INVERSION (a17 only): every other mode falls back to VANILLA duck on
graft failure (never 0). Here a vanilla run would SILENTLY SERVE THE 27B and
poison the rho_action denominator, so any rewrite/serve failure RAISES ->
kernel ERROR. A dead canary is a retry; a silently-27B canary is a poisoned
measurement.

    python duck_eval/warpack/build_eval_notebook.py --a17-canary
    -> notebooks/a17-canary/{arc3-a17-72b-canary.ipynb, kernel-metadata.json}

All write a dummy submission.parquet and are NEVER queued for submission
(the submission queue pins canivel/arc3-duck-war at version 1).

(f) DEFAULT (prereg amendment 2026-07-23 item 4; W0 screen PASS 49/49 —
pure hygiene, zero cost): every mode EXCEPT --w0 (which IS the continuation
graft) appends the game-over-continuation graft to cell 12 by default.
Runtime kill switch stays (CONTINUATION_DISABLE=1 -> apply() no-ops); build
opt-out `--no-continuation` reproduces the pre-Jul-23 compositions (e.g. the
live sentinel seed-1/2 ledger, which is sentinel-only).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC_META = REPO / "notebooks" / "duckwar" / "kernel-metadata.json"

# war-v1 eval (ledger OFF)
SRC_NB = REPO / "notebooks" / "duckwar" / "arc3-duck-war.ipynb"
OUT_DIR = REPO / "notebooks" / "duckwar-eval"
KERNEL_ID = "canivel/arc3-duck-war-eval"

# war-v2 eval (ledger ON + canary)
SRC_NB_V2 = HERE / "duck_warpack_v2.ipynb"
OUT_DIR_V2 = REPO / "notebooks" / "duckwar-v2-eval"
KERNEL_ID_V2 = "canivel/arc3-duck-war-v2-eval"

# W0 (f) game-over-continuation eval (standalone hygiene; NO warpack/ledger)
OUT_DIR_W0 = REPO / "notebooks" / "duckw0-eval"
KERNEL_ID_W0 = "canivel/arc3-duck-w0-continuation-eval"

# W2 (a) budget-sentinel eval (single flag on the duck baseline; NO warpack/ledger)
OUT_DIR_SENTINEL = REPO / "notebooks" / "ducksentinel-eval"
KERNEL_ID_SENTINEL = "canivel/arc3-duck-sentinel-eval"

# A22 compaction + retained-reasoning eval (single flag on the duck baseline;
# NO warpack/ledger-graft/sentinel; (f) default rides)
OUT_DIR_COMPACTION = REPO / "notebooks" / "duckcompaction-eval"
KERNEL_ID_COMPACTION = "canivel/arc3-duck-compaction-eval"

# Animation-awareness eval (single flag on the duck baseline; NO
# warpack/ledger-graft/sentinel/compaction; (f) default rides)
OUT_DIR_ANIMATION = REPO / "notebooks" / "duckanimation-eval"
KERNEL_ID_ANIMATION = "canivel/arc3-duck-animation-eval"

# P1 zero-information action suppressor eval (single flag on the duck baseline;
# NO warpack/ledger-graft/sentinel/compaction/animation; (f) default rides)
OUT_DIR_P1 = REPO / "notebooks" / "duckp1-eval"
KERNEL_ID_P1 = "canivel/arc3-duck-p1-eval"

# EFFNOTE quantified per-turn efficiency note eval (single flag on the duck
# baseline; NO warpack/ledger-graft/sentinel/compaction/animation/p1; (f)
# default rides)
OUT_DIR_EFFNOTE = REPO / "notebooks" / "duckeffnote-eval"
KERNEL_ID_EFFNOTE = "canivel/arc3-duck-effnote-eval"

# A17 72B-VL canary (W0 composition + 72B serve rewrite; free-build rail, never submitted)
OUT_DIR_A17 = REPO / "notebooks" / "a17-canary"
KERNEL_ID_A17 = "canivel/arc3-a17-72b-canary"
A17_MODEL_SOURCE = "qwen-lm/qwen2.5-vl/transformers/72b-instruct-awq/1"
A17_SERVED_MODEL_NAME = "Qwen2.5-VL-72B-Instruct-AWQ"
A17_SCREEN_GAMES = ["ft09-0d8bbf25", "sb26-7fbdac44", "lp85-305b61c3", "vc33-5430563c"]

EVAL_LINE = (
    'import os; os.environ["WARPACK_FORCE_OFFLINE_BENCH"] = "1"'
    "  # EVAL BUILD: run the offline bench at build time (see build_eval_notebook.py)\n"
)

# war-v2 extra prepended lines: seed tag (greppable in the build log) — same
# seed convention as the ledger-OFF eval seeds (seed N = push N of the
# identical notebook); the env stamp + banner just make the pairing explicit.
EVAL_SEED_LINES_V2 = (
    'os.environ["WAR_EVAL_SEED"] = "1"'
    "  # ledger-ON seed 1 (panel R12 N6): pairs with ledger-OFF war-eval seed 1\n"
    'print("war-v2-eval: SEED=1 ledger-ON (pairs with ledger-OFF '
    'arc3-duck-war-eval seed 1)")\n'
)

# W0 cell-2 extra prepended lines: seed tag (greppable) + W0 banner. Same seed
# convention as the war-eval seeds (seed N = push N of the identical notebook);
# W0 is the SAME offline bench as arc3-duck-war-eval seed 1 with (f) ON and NO
# warpack, so it pairs with that seed for the (f) quick screen.
EVAL_SEED_LINES_W0 = (
    'os.environ["W0_EVAL_SEED"] = "1"'
    "  # W0 (f) seed 1 (grinder design sec5 Jul 18): pairs with arc3-duck-war-eval seed 1\n"
    'print("w0-continuation-eval: SEED=1 (f) game-over-continuation ON, NO '
    'warpack (pairs with arc3-duck-war-eval seed 1)")\n'
)

# W0 cell-12 REPLACEMENT: the continuation-patch graft. Structurally mirrors the
# warpack graft (RUN_HEAVY gate, marker-based mount-agnostic find under
# /kaggle/input, sys.path insert, apply, bm.label stamp, VANILLA fallback on any
# failure) but imports ONLY continuation_patch -- warpack/ledger are never
# touched. The runtime banner "continuation v1: ... ACTIVE (2 modules patched)"
# is printed by continuation_patch.apply() itself; this cell re-echoes the
# VERSION + module count so the build log proves what ran.
CELL12_W0 = (
    "# ============================================================================\n"
    "# Cell 12 - Customization hook: W0 (f) game-over-continuation graft ONLY.\n"
    "# grinder_cracking_design.md sec(f) + sec5 (Jul 18 standalone hygiene window).\n"
    "# W0 is STANDALONE: duck baseline + (f) only -- NO warpack, NO ledger.\n"
    "# Runs AFTER `bm` is unpickled (cell 10) and the bundled sources are\n"
    "# importable (cell 8), BEFORE bm.run() builds any session system prompt\n"
    "# (cell 14), so the PYTHON_ADDENDUM rewrite lands before the prompt is built.\n"
    "#\n"
    "# Sourcing: continuation_patch.py ships in the attached arc-war-kit dataset\n"
    "# (marker-based find, mount-path agnostic), or a local dir for interactive\n"
    "# runs. Failure policy: unless CONTINUATION_STRICT=1, any failure prints a\n"
    "# traceback and the run continues as VANILLA duck (stock prompt, never 0).\n"
    "# Kill switch: CONTINUATION_DISABLE=1 -> continuation_patch.apply() no-ops.\n"
    "# ============================================================================\n"
    "import os\n"
    "import sys\n"
    "import traceback\n"
    "from pathlib import Path\n"
    "\n"
    'CONTINUATION_MARKER = "continuation_patch.py"\n'
    'CONTINUATION_STRICT = os.environ.get("CONTINUATION_STRICT", "").strip() in {"1", "true"}\n'
    "\n"
    "\n"
    "def _find_continuation_dir() -> Path | None:\n"
    "    candidates: list[Path] = []\n"
    '    kaggle_input = Path("/kaggle/input")\n'
    "    if kaggle_input.is_dir():\n"
    "        candidates.extend(marker.parent for marker in kaggle_input.rglob(CONTINUATION_MARKER))\n"
    "    here = Path.cwd()\n"
    '    for probe in (here / "continuation", here, here.parent / "continuation"):\n'
    "        if (probe / CONTINUATION_MARKER).is_file():\n"
    "            candidates.append(probe)\n"
    "    return candidates[0] if candidates else None\n"
    "\n"
    "\n"
    "if not RUN_HEAVY:  # noqa: F821 - defined in cell 2 (fast-submit gate)\n"
    '    print("continuation: fast-submit save (RUN_HEAVY=False) - customization skipped")\n'
    "else:\n"
    "    try:\n"
    "        continuation_dir = _find_continuation_dir()\n"
    "        if continuation_dir is None:\n"
    "            raise FileNotFoundError(\n"
    '                f"{CONTINUATION_MARKER} not found under /kaggle/input or ./continuation - "\n'
    '                "attach the arc-war-kit dataset (it carries continuation_patch.py)."\n'
    "            )\n"
    "        if str(continuation_dir) not in sys.path:\n"
    "            sys.path.insert(0, str(continuation_dir))\n"
    "        import continuation_patch\n"
    "\n"
    "        applied = continuation_patch.apply()  # rewrites PYTHON_ADDENDUM game-over line\n"
    '        version = getattr(continuation_patch, "VERSION", "v1")\n'
    "        if not applied and os.environ.get(\"CONTINUATION_DISABLE\") != \"1\":\n"
    '            raise RuntimeError("continuation_patch.apply() returned False "\n'
    '                               "(source drift/tamper) - vanilla fallback")\n'
    '        bm.label = f"{bm.label}-continuation-{version}"  # noqa: F821 - bm from cell 10\n'
    '        print(f"continuation {version}: (f) game-over-continuation graft applied "\n'
    '              f"from {continuation_dir} (applied={applied}); NO warpack/ledger")\n'
    "    except Exception:\n"
    "        if CONTINUATION_STRICT:\n"
    "            raise\n"
    '        print("continuation: PATCH FAILED - continuing with VANILLA duck harness")\n'
    "        traceback.print_exc()\n"
)


# sentinel cell-2 extra prepended lines: seed tag (greppable) + banner. Same
# seed convention as the war-eval seeds (seed N = push N of the identical
# notebook); the (a) window is the duck baseline + budget sentinel.
EVAL_SEED_LINES_SENTINEL = (
    'os.environ["SENTINEL_EVAL_SEED"] = "1"'
    "  # (a) budget-sentinel seed 1 (grinder design sec(a) + sec5 Jul 22-24)\n"
    'print("sentinel-eval: SEED=1 (a) budget sentinel ON, NO warpack '
    '(pairs with the prior-stack seed 1)")\n'
)

# sentinel cell-12 REPLACEMENT: the budget-sentinel graft. Byte-identical in
# STRUCTURE to the W0 continuation graft (RUN_HEAVY gate, marker-based
# mount-agnostic find under /kaggle/input, sys.path insert, apply, VANILLA
# fallback on any failure) -- only the module name, marker, env-var names and
# banner strings differ. Imports ONLY budget_sentinel_patch; warpack/ledger are
# never touched. The runtime banner "sentinel v1: budget sentinel ACTIVE ..." is
# printed by budget_sentinel_patch.apply() itself (which also stamps bm.label).
CELL12_SENTINEL = (
    "# ============================================================================\n"
    "# Cell 12 - Customization hook: (a) budget-sentinel graft ONLY.\n"
    "# grinder_cracking_design.md sec(a) + sec5 (Jul 22-24 (a) window).\n"
    "# Single flag on the duck baseline: duck + budget sentinel -- NO warpack,\n"
    "# NO ledger. Runs AFTER `bm` is unpickled (cell 10) and the bundled sources\n"
    "# are importable (cell 8), BEFORE bm.run() drives any session (cell 14), so\n"
    "# the ToolAgent/_HarnessGameSession monkeypatches land before play starts.\n"
    "#\n"
    "# Sourcing: budget_sentinel_patch.py ships in the attached arc-war-kit\n"
    "# dataset (marker-based find, mount-path agnostic), or a local dir for\n"
    "# interactive runs. Failure policy: unless SENTINEL_STRICT=1, any failure\n"
    "# prints a traceback and the run continues as VANILLA duck (never 0).\n"
    "# Kill switch: SENTINEL_DISABLE=1 -> budget_sentinel_patch.apply() no-ops.\n"
    "# ============================================================================\n"
    "import os\n"
    "import sys\n"
    "import traceback\n"
    "from pathlib import Path\n"
    "\n"
    'SENTINEL_MARKER = "budget_sentinel_patch.py"\n'
    'SENTINEL_STRICT = os.environ.get("SENTINEL_STRICT", "").strip() in {"1", "true"}\n'
    "\n"
    "\n"
    "def _find_sentinel_dir() -> Path | None:\n"
    "    candidates: list[Path] = []\n"
    '    kaggle_input = Path("/kaggle/input")\n'
    "    if kaggle_input.is_dir():\n"
    "        candidates.extend(marker.parent for marker in kaggle_input.rglob(SENTINEL_MARKER))\n"
    "    here = Path.cwd()\n"
    '    for probe in (here / "sentinel", here, here.parent / "sentinel"):\n'
    "        if (probe / SENTINEL_MARKER).is_file():\n"
    "            candidates.append(probe)\n"
    "    return candidates[0] if candidates else None\n"
    "\n"
    "\n"
    "if not RUN_HEAVY:  # noqa: F821 - defined in cell 2 (fast-submit gate)\n"
    '    print("sentinel: fast-submit save (RUN_HEAVY=False) - customization skipped")\n'
    "else:\n"
    "    try:\n"
    "        sentinel_dir = _find_sentinel_dir()\n"
    "        if sentinel_dir is None:\n"
    "            raise FileNotFoundError(\n"
    '                f"{SENTINEL_MARKER} not found under /kaggle/input or ./sentinel - "\n'
    '                "attach the arc-war-kit dataset (it carries budget_sentinel_patch.py)."\n'
    "            )\n"
    "        if str(sentinel_dir) not in sys.path:\n"
    "            sys.path.insert(0, str(sentinel_dir))\n"
    "        import budget_sentinel_patch\n"
    "\n"
    "        applied = budget_sentinel_patch.apply(bm)  # patches ToolAgent + session  # noqa: F821 - bm from cell 10\n"
    '        version = getattr(budget_sentinel_patch, "VERSION", "v1")\n'
    "        if not applied and os.environ.get(\"SENTINEL_DISABLE\") != \"1\":\n"
    '            raise RuntimeError("budget_sentinel_patch.apply() returned False "\n'
    '                               "(source drift/tamper) - vanilla fallback")\n'
    '        print(f"sentinel {version}: (a) budget-sentinel graft applied "\n'
    '              f"from {sentinel_dir} (applied={applied}); NO warpack/ledger")\n'
    "    except Exception:\n"
    "        if SENTINEL_STRICT:\n"
    "            raise\n"
    '        print("sentinel: PATCH FAILED - continuing with VANILLA duck harness")\n'
    "        traceback.print_exc()\n"
)


# (f)-default appended block (prereg amendment 2026-07-23 item 4): the same
# self-contained continuation graft the W0 build ships, appended AFTER the
# mode's own cell-12 graft (order-independent: (f) rewrites the PYTHON_ADDENDUM
# prompt constant; the other grafts monkeypatch harness/agent classes).
CONTINUATION_DEFAULT_BLOCK = (
    "\n\n"
    "# ============================================================================\n"
    "# (f) game-over-continuation graft -- DEFAULT ON in all builds since\n"
    "# 2026-07-23 (prereg amendment item 4; W0 standalone screen PASS 49/49).\n"
    "# Kill switch: CONTINUATION_DISABLE=1 (runtime) / --no-continuation (build).\n"
    "# ============================================================================\n"
    + CELL12_W0
)


# compaction cell-2 extra prepended lines: seed tag (greppable) + the A22 arm
# flag + banner. Same seed convention as the war-eval seeds (seed N = push N of
# the identical notebook); pairs with ledger-OFF arc3-duck-war-eval seeds for
# the paired-Δlc contrast (a22_compaction_prereg_2026-08-01.md).
EVAL_SEED_LINES_COMPACTION = (
    'os.environ["COMPACTION_EVAL_SEED"] = "1"'
    "  # A22 compaction+retained-reasoning seed 1 (a22 prereg intent 2026-08-01)\n"
    'os.environ["COMPACTION"] = "1"'
    "  # THE A22 arm flag (compaction_patch.apply() is a no-op without it)\n"
    'print("compaction-eval: SEED=1 A22 compaction+retained-reasoning ON, NO '
    "warpack/ledger-graft/sentinel (pairs with arc3-duck-war-eval seed 1); "
    'COMPACTION=1")\n'
)

# compaction cell-12 REPLACEMENT: the compaction graft. Byte-identical in
# STRUCTURE to the sentinel/W0 grafts (RUN_HEAVY gate, marker-based
# mount-agnostic find under /kaggle/input, sys.path insert, apply, VANILLA
# fallback on any failure) -- only the module name, marker, env-var names and
# banner strings differ. Imports ONLY compaction_patch (which itself reuses
# ledger_core as a library digester); the warpack/ledger GRAFTS are never
# touched. The runtime banner "compaction v1: ACTIVE ..." is printed by
# compaction_patch.apply() itself (which also stamps bm.label).
CELL12_COMPACTION = (
    "# ============================================================================\n"
    "# Cell 12 - Customization hook: A22 compaction + retained-reasoning graft ONLY.\n"
    "# a22_compaction_prereg_2026-08-01.md (OpenAI ARC-AGI-3 harness result:\n"
    "# retained reasoning + compaction-instead-of-eviction, 13.3%->38.3%).\n"
    "# Single flag on the duck baseline: duck + COMPACTION=1 -- NO warpack,\n"
    "# NO ledger graft, NO sentinel. Runs AFTER `bm` is unpickled (cell 10) and\n"
    "# the bundled sources are importable (cell 8), BEFORE bm.run() drives any\n"
    "# session (cell 14), so the ToolAgent monkeypatches land before play starts.\n"
    "#\n"
    "# Sourcing: compaction_patch.py + ledger_core.py ship in the attached\n"
    "# arc-war-kit dataset (marker-based find, mount-path agnostic), or a local\n"
    "# dir for interactive runs. Failure policy: unless COMPACTION_STRICT=1, any\n"
    "# failure prints a traceback and the run continues as VANILLA duck (never 0).\n"
    "# Kill switch: COMPACTION_DISABLE=1 -> compaction_patch.apply() no-ops.\n"
    "# ============================================================================\n"
    "import os\n"
    "import sys\n"
    "import traceback\n"
    "from pathlib import Path\n"
    "\n"
    'COMPACTION_MARKER = "compaction_patch.py"\n'
    'COMPACTION_STRICT = os.environ.get("COMPACTION_STRICT", "").strip() in {"1", "true"}\n'
    "\n"
    "\n"
    "def _find_compaction_dir() -> Path | None:\n"
    "    candidates: list[Path] = []\n"
    '    kaggle_input = Path("/kaggle/input")\n'
    "    if kaggle_input.is_dir():\n"
    "        candidates.extend(marker.parent for marker in kaggle_input.rglob(COMPACTION_MARKER))\n"
    "    here = Path.cwd()\n"
    '    for probe in (here / "compaction", here, here.parent / "compaction"):\n'
    "        if (probe / COMPACTION_MARKER).is_file():\n"
    "            candidates.append(probe)\n"
    "    return candidates[0] if candidates else None\n"
    "\n"
    "\n"
    "if not RUN_HEAVY:  # noqa: F821 - defined in cell 2 (fast-submit gate)\n"
    '    print("compaction: fast-submit save (RUN_HEAVY=False) - customization skipped")\n'
    "else:\n"
    "    try:\n"
    "        compaction_dir = _find_compaction_dir()\n"
    "        if compaction_dir is None:\n"
    "            raise FileNotFoundError(\n"
    '                f"{COMPACTION_MARKER} not found under /kaggle/input or ./compaction - "\n'
    '                "attach the arc-war-kit dataset (it carries compaction_patch.py + ledger_core.py)."\n'
    "            )\n"
    "        if str(compaction_dir) not in sys.path:\n"
    "            sys.path.insert(0, str(compaction_dir))\n"
    "        import compaction_patch\n"
    "\n"
    "        applied = compaction_patch.apply(bm)  # patches ToolAgent  # noqa: F821 - bm from cell 10\n"
    '        version = getattr(compaction_patch, "VERSION", "v1")\n'
    "        if not applied and os.environ.get(\"COMPACTION_DISABLE\") != \"1\":\n"
    '            raise RuntimeError("compaction_patch.apply() returned False "\n'
    '                               "(flag missing / source drift) - vanilla fallback")\n'
    '        print(f"compaction {version}: A22 graft applied "\n'
    '              f"from {compaction_dir} (applied={applied}); NO warpack/ledger-graft/sentinel")\n'
    "    except Exception:\n"
    "        if COMPACTION_STRICT:\n"
    "            raise\n"
    '        print("compaction: PATCH FAILED - continuing with VANILLA duck harness")\n'
    "        traceback.print_exc()\n"
)


# ---------------------------------------------------------------------------
# animation-awareness blocks (sweep 08-11 ADOPT #1; prereg
# learnings/war_room/animation_prereg_2026-08-11.md)
# ---------------------------------------------------------------------------

EVAL_SEED_LINES_ANIMATION = (
    'os.environ["ANIMATION_EVAL_SEED"] = "1"'
    "  # animation-awareness seed 1 (prereg animation_prereg_2026-08-11.md)\n"
    'os.environ["ANIMATION_AWARE"] = "1"'
    "  # THE arm flag (animation_patch.apply() is a no-op without it)\n"
    'print("animation-eval: SEED=1 animation-awareness ON, NO '
    "warpack/ledger-graft/sentinel/compaction (pairs with the "
    'duck-harness-kaggle-continuation-v1 family); ANIMATION_AWARE=1; NO no-op guard")\n'
)

# animation cell-12 REPLACEMENT: the animation graft. Byte-identical in
# STRUCTURE to the sentinel/W0/compaction grafts (RUN_HEAVY gate, marker-based
# mount-agnostic find under /kaggle/input, sys.path insert, apply, VANILLA
# fallback on any failure) -- only the module name, marker, env-var names and
# banner strings differ. Imports ONLY animation_patch; the warpack/ledger
# GRAFTS, the sentinel and compaction are never touched. The runtime banner
# "animation v1: ACTIVE (N seams patched) ..." is printed by
# animation_patch.apply() itself (which also stamps bm.label).
CELL12_ANIMATION = (
    "# ============================================================================\n"
    "# Cell 12 - Customization hook: animation-awareness graft ONLY.\n"
    "# animation_prereg_2026-08-11.md (sweep 08-11 ADOPT #1): taaf/game.py:170\n"
    "# returns only raw.frame[-1]; all_frames/animation_frames have ZERO consumers,\n"
    "# so 401/11104 audited actions (19.0% of apparent no-ops) were state-aliased.\n"
    "# Single flag on the duck baseline: duck + ANIMATION_AWARE=1 -- NO warpack,\n"
    "# NO ledger graft, NO sentinel, NO compaction, and explicitly NO no-op guard\n"
    "# (prereg sec2.2: strictly downstream, separately gated). Runs AFTER `bm` is\n"
    "# unpickled (cell 10) and the bundled sources are importable (cell 8), BEFORE\n"
    "# bm.run() drives any session (cell 14), so the solver/ToolAgent monkeypatches\n"
    "# land before play starts.\n"
    "#\n"
    "# Sourcing: animation_patch.py ships in the attached arc-war-kit dataset\n"
    "# (marker-based find, mount-path agnostic), or a local dir for interactive\n"
    "# runs. Failure policy: unless ANIMATION_STRICT=1, any failure prints a\n"
    "# traceback and the run continues as VANILLA duck (never 0).\n"
    "# Kill switch: ANIMATION_DISABLE=1 -> animation_patch.apply() no-ops.\n"
    "# ============================================================================\n"
    "import os\n"
    "import sys\n"
    "import traceback\n"
    "from pathlib import Path\n"
    "\n"
    'ANIMATION_MARKER = "animation_patch.py"\n'
    'ANIMATION_STRICT = os.environ.get("ANIMATION_STRICT", "").strip() in {"1", "true"}\n'
    "\n"
    "\n"
    "def _find_animation_dir() -> Path | None:\n"
    "    candidates: list[Path] = []\n"
    '    kaggle_input = Path("/kaggle/input")\n'
    "    if kaggle_input.is_dir():\n"
    "        candidates.extend(marker.parent for marker in kaggle_input.rglob(ANIMATION_MARKER))\n"
    "    here = Path.cwd()\n"
    '    for probe in (here / "animation", here, here.parent / "animation"):\n'
    "        if (probe / ANIMATION_MARKER).is_file():\n"
    "            candidates.append(probe)\n"
    "    return candidates[0] if candidates else None\n"
    "\n"
    "\n"
    "if not RUN_HEAVY:  # noqa: F821 - defined in cell 2 (fast-submit gate)\n"
    '    print("animation: fast-submit save (RUN_HEAVY=False) - customization skipped")\n'
    "else:\n"
    "    try:\n"
    "        animation_dir = _find_animation_dir()\n"
    "        if animation_dir is None:\n"
    "            raise FileNotFoundError(\n"
    '                f"{ANIMATION_MARKER} not found under /kaggle/input or ./animation - "\n'
    '                "attach the arc-war-kit dataset (it carries animation_patch.py)."\n'
    "            )\n"
    "        if str(animation_dir) not in sys.path:\n"
    "            sys.path.insert(0, str(animation_dir))\n"
    "        import animation_patch\n"
    "\n"
    "        applied = animation_patch.apply(bm)  # patches solver + ToolAgent  # noqa: F821 - bm from cell 10\n"
    '        version = getattr(animation_patch, "VERSION", "v1")\n'
    "        if not applied and os.environ.get(\"ANIMATION_DISABLE\") != \"1\":\n"
    '            raise RuntimeError("animation_patch.apply() returned False "\n'
    '                               "(flag missing / source drift) - vanilla fallback")\n'
    '        print(f"animation {version}: graft applied "\n'
    '              f"from {animation_dir} (applied={applied}); '
    'NO warpack/ledger-graft/sentinel/compaction/noop-guard")\n'
    "    except Exception:\n"
    "        if ANIMATION_STRICT:\n"
    "            raise\n"
    '        print("animation: PATCH FAILED - continuing with VANILLA duck harness")\n'
    "        traceback.print_exc()\n"
)

# Appended to cell 14 (after the run) so the prereg canary (K-A1..K-A4) is in
# the build log even when zero animations fire. Blanket-guarded: a canary must
# never be able to fail a completed run.
CELL14_ANIMATION_CANARY = (
    "\n"
    "        try:\n"
    "            import animation_patch as _anim\n"
    "            _anim.canary_report()\n"
    "        except Exception as _anim_exc:  # noqa: BLE001 - canary must never break the run\n"
    '            print(f"ANIMATION CANARY unavailable: {_anim_exc!r}")\n'
)


# ---------------------------------------------------------------------------
# P1 zero-information action suppressor blocks (diagnosis
# learnings/war_room/efficiency_diagnosis_2026-08-12.md sec5 P1; prereg
# learnings/war_room/p1_prereg_2026-08-12.md)
# ---------------------------------------------------------------------------

EVAL_SEED_LINES_P1 = (
    'os.environ["P1_EVAL_SEED"] = "1"'
    "  # P1 suppressor seed 1 (prereg p1_prereg_2026-08-12.md)\n"
    'os.environ["P1_SUPPRESS"] = "1"'
    "  # THE arm flag (p1_suppressor_patch.apply() is a no-op without it)\n"
    'print("p1-eval: SEED=1 zero-information action suppressor ON, NO '
    "warpack/ledger-graft/sentinel/compaction/animation (pairs with the "
    'duck-harness-kaggle-continuation-v1 family); P1_SUPPRESS=1; '
    'shipped defaults memo_mode=noop confirm=2 abort_revisit=OFF")\n'
)

# P1 cell-12 REPLACEMENT: the P1 graft. Byte-identical in STRUCTURE to the
# animation/compaction/sentinel grafts (RUN_HEAVY gate, marker-based
# mount-agnostic find under /kaggle/input, sys.path insert, apply, VANILLA
# fallback on any failure) -- only the module name, marker, env-var names and
# banner strings differ. Imports ONLY p1_suppressor_patch; the warpack/ledger
# GRAFTS, the sentinel, compaction and animation are never touched.
CELL12_P1 = (
    "# ============================================================================\n"
    "# Cell 12 - Customization hook: P1 zero-information action suppressor ONLY.\n"
    "# efficiency_diagnosis_2026-08-12.md sec5 P1 / p1_prereg_2026-08-12.md.\n"
    "# 10.5% of the actions on our cleared levels re-execute a (board, action)\n"
    "# pair already executed on that level and 17.6% are fired inside a batch\n"
    "# that had already gone dead; the root cause is context truncation\n"
    "# (31744 tok / 33 history messages on a 225-action level), not a missing\n"
    "# loop detector. Single flag on the duck baseline: duck + P1_SUPPRESS=1 --\n"
    "# NO warpack, NO ledger graft, NO sentinel, NO compaction, NO animation.\n"
    "# Runs AFTER `bm` is unpickled (cell 10) and the bundled sources are\n"
    "# importable (cell 8), BEFORE bm.run() drives any session (cell 14), so the\n"
    "# solver/ToolAgent monkeypatches land before play starts.\n"
    "#\n"
    "# Sourcing: p1_suppressor_patch.py ships in the attached arc-war-kit dataset\n"
    "# (marker-based find, mount-path agnostic), or a local dir for interactive\n"
    "# runs. Failure policy: unless P1_STRICT=1, any failure prints a traceback\n"
    "# and the run continues as VANILLA duck (never 0).\n"
    "# Kill switch: P1_DISABLE=1 -> p1_suppressor_patch.apply() no-ops.\n"
    "# ============================================================================\n"
    "import os\n"
    "import sys\n"
    "import traceback\n"
    "from pathlib import Path\n"
    "\n"
    'P1_MARKER = "p1_suppressor_patch.py"\n'
    'P1_STRICT = os.environ.get("P1_STRICT", "").strip() in {"1", "true"}\n'
    "\n"
    "\n"
    "def _find_p1_dir() -> Path | None:\n"
    "    candidates: list[Path] = []\n"
    '    kaggle_input = Path("/kaggle/input")\n'
    "    if kaggle_input.is_dir():\n"
    "        candidates.extend(marker.parent for marker in kaggle_input.rglob(P1_MARKER))\n"
    "    here = Path.cwd()\n"
    '    for probe in (here / "p1", here, here.parent / "p1"):\n'
    "        if (probe / P1_MARKER).is_file():\n"
    "            candidates.append(probe)\n"
    "    return candidates[0] if candidates else None\n"
    "\n"
    "\n"
    "if not RUN_HEAVY:  # noqa: F821 - defined in cell 2 (fast-submit gate)\n"
    '    print("p1: fast-submit save (RUN_HEAVY=False) - customization skipped")\n'
    "else:\n"
    "    try:\n"
    "        p1_dir = _find_p1_dir()\n"
    "        if p1_dir is None:\n"
    "            raise FileNotFoundError(\n"
    '                f"{P1_MARKER} not found under /kaggle/input or ./p1 - "\n'
    '                "attach the arc-war-kit dataset (it carries p1_suppressor_patch.py)."\n'
    "            )\n"
    "        if str(p1_dir) not in sys.path:\n"
    "            sys.path.insert(0, str(p1_dir))\n"
    "        import p1_suppressor_patch\n"
    "\n"
    "        applied = p1_suppressor_patch.apply(bm)  # patches solver + ToolAgent  # noqa: F821 - bm from cell 10\n"
    '        version = getattr(p1_suppressor_patch, "VERSION", "v1")\n'
    "        if not applied and os.environ.get(\"P1_DISABLE\") != \"1\":\n"
    '            raise RuntimeError("p1_suppressor_patch.apply() returned False "\n'
    '                               "(flag missing / source drift) - vanilla fallback")\n'
    '        print(f"p1 {version}: graft applied "\n'
    '              f"from {p1_dir} (applied={applied}); '
    'NO warpack/ledger-graft/sentinel/compaction/animation")\n'
    "    except Exception:\n"
    "        if P1_STRICT:\n"
    "            raise\n"
    '        print("p1: PATCH FAILED - continuing with VANILLA duck harness")\n'
    "        traceback.print_exc()\n"
)

# Appended to cell 14 (after the run) so the prereg canaries are in the build
# log even when zero suppressions fire. Blanket-guarded.
CELL14_P1_CANARY = (
    "\n"
    "        try:\n"
    "            import p1_suppressor_patch as _p1\n"
    "            _p1.canary_report()\n"
    "        except Exception as _p1_exc:  # noqa: BLE001 - canary must never break the run\n"
    '            print(f"P1 CANARY unavailable: {_p1_exc!r}")\n'
)


# ---------------------------------------------------------------------------
# EFFNOTE quantified per-turn efficiency note blocks (spec
# learnings/war_room/harness_diff_2026-08-13.md sec4 item #1; prereg
# learnings/war_room/effnote_prereg_2026-08-13.md)
# ---------------------------------------------------------------------------

EVAL_SEED_LINES_EFFNOTE = (
    'os.environ["EFFNOTE_EVAL_SEED"] = "1"'
    "  # EFFNOTE seed 1 (prereg effnote_prereg_2026-08-13.md)\n"
    'os.environ["EFFNOTE"] = "1"'
    "  # THE arm flag (effnote_patch.apply() is a no-op without it)\n"
    'print("effnote-eval: SEED=1 quantified per-turn efficiency note ON, NO '
    "warpack/ledger-graft/sentinel/compaction/animation/p1 (pairs with the "
    'duck-harness-kaggle-continuation-v1 family); EFFNOTE=1; '
    'REPORT-ONLY; target=clamped game-agnostic proxy (NO baseline table); '
    'cost bound=700 CHARACTERS")\n'
)

# EFFNOTE cell-12 REPLACEMENT: the EFFNOTE graft. Byte-identical in STRUCTURE
# to the p1/animation/compaction/sentinel grafts (RUN_HEAVY gate, marker-based
# mount-agnostic find under /kaggle/input, sys.path insert, apply, VANILLA
# fallback on any failure) -- only the module name, marker, env-var names and
# banner strings differ. Imports ONLY effnote_patch.
CELL12_EFFNOTE = (
    "# ============================================================================\n"
    "# Cell 12 - Customization hook: EFFNOTE per-turn efficiency note ONLY.\n"
    "# harness_diff_2026-08-13.md sec4 #1 / effnote_prereg_2026-08-13.md.\n"
    "# The stock prompt's ENTIRE efficiency treatment is one unquantified\n"
    "# sentence (prompts.py:17): the model is never shown the scoring rule and\n"
    "# never sees its own action count, while the per-level score is\n"
    "# (baseline/actions)^2 -- quadratic in waste. This arm appends a bounded,\n"
    "# game-agnostic note to the USER turn: the scoring rule stated\n"
    "# quantitatively, the live action count vs a CLAMPED PROXY target, the\n"
    "# over-target ratio, three pure stall detectors, and a commit-don't-scan\n"
    "# reminder. REPORT-ONLY -- no action is ever blocked, declined or injected\n"
    "# and the hot step_env path is never touched. Single flag on the duck\n"
    "# baseline: duck + EFFNOTE=1 -- NO warpack, NO ledger graft, NO sentinel,\n"
    "# NO compaction, NO animation, NO p1.\n"
    "# Runs AFTER `bm` is unpickled (cell 10) and the bundled sources are\n"
    "# importable (cell 8), BEFORE bm.run() drives any session (cell 14), so the\n"
    "# solver/ToolAgent monkeypatches land before play starts.\n"
    "#\n"
    "# Sourcing: effnote_patch.py ships in the attached arc-war-kit dataset\n"
    "# (marker-based find, mount-path agnostic), or a local dir for interactive\n"
    "# runs. Failure policy: unless EFFNOTE_STRICT=1, any failure prints a\n"
    "# traceback and the run continues as VANILLA duck (never 0).\n"
    "# Kill switch: EFFNOTE_DISABLE=1 -> effnote_patch.apply() no-ops.\n"
    "# ============================================================================\n"
    "import os\n"
    "import sys\n"
    "import traceback\n"
    "from pathlib import Path\n"
    "\n"
    'EFFNOTE_MARKER = "effnote_patch.py"\n'
    'EFFNOTE_STRICT = os.environ.get("EFFNOTE_STRICT", "").strip() in {"1", "true"}\n'
    "\n"
    "\n"
    "def _find_effnote_dir() -> Path | None:\n"
    "    candidates: list[Path] = []\n"
    '    kaggle_input = Path("/kaggle/input")\n'
    "    if kaggle_input.is_dir():\n"
    "        candidates.extend(marker.parent for marker in kaggle_input.rglob(EFFNOTE_MARKER))\n"
    "    here = Path.cwd()\n"
    '    for probe in (here / "effnote", here, here.parent / "effnote"):\n'
    "        if (probe / EFFNOTE_MARKER).is_file():\n"
    "            candidates.append(probe)\n"
    "    return candidates[0] if candidates else None\n"
    "\n"
    "\n"
    "if not RUN_HEAVY:  # noqa: F821 - defined in cell 2 (fast-submit gate)\n"
    '    print("effnote: fast-submit save (RUN_HEAVY=False) - customization skipped")\n'
    "else:\n"
    "    try:\n"
    "        effnote_dir = _find_effnote_dir()\n"
    "        if effnote_dir is None:\n"
    "            raise FileNotFoundError(\n"
    '                f"{EFFNOTE_MARKER} not found under /kaggle/input or ./effnote - "\n'
    '                "attach the arc-war-kit dataset (it carries effnote_patch.py)."\n'
    "            )\n"
    "        if str(effnote_dir) not in sys.path:\n"
    "            sys.path.insert(0, str(effnote_dir))\n"
    "        import effnote_patch\n"
    "\n"
    "        applied = effnote_patch.apply(bm)  # patches solver + ToolAgent  # noqa: F821 - bm from cell 10\n"
    '        version = getattr(effnote_patch, "VERSION", "v1")\n'
    "        if not applied and os.environ.get(\"EFFNOTE_DISABLE\") != \"1\":\n"
    '            raise RuntimeError("effnote_patch.apply() returned False "\n'
    '                               "(flag missing / source drift) - vanilla fallback")\n'
    '        print(f"effnote {version}: graft applied "\n'
    '              f"from {effnote_dir} (applied={applied}); '
    'NO warpack/ledger-graft/sentinel/compaction/animation/p1")\n'
    "    except Exception:\n"
    "        if EFFNOTE_STRICT:\n"
    "            raise\n"
    '        print("effnote: PATCH FAILED - continuing with VANILLA duck harness")\n'
    "        traceback.print_exc()\n"
)

# Appended to cell 14 (after the run) so the prereg canaries are in the build
# log even when zero notes fire. Blanket-guarded.
CELL14_EFFNOTE_CANARY = (
    "\n"
    "        try:\n"
    "            import effnote_patch as _effnote\n"
    "            _effnote.canary_report()\n"
    "        except Exception as _effnote_exc:  # noqa: BLE001 - canary must never break the run\n"
    '            print(f"EFFNOTE CANARY unavailable: {_effnote_exc!r}")\n'
)


# ---------------------------------------------------------------------------
# A17 canary blocks
# ---------------------------------------------------------------------------

# A17 cell-2 extra prepended lines: seed tag (greppable) + banner. Same seed
# convention as the other eval kernels (seed N = push N of the identical
# notebook); the canary is the rho_action denominator run (scope v2 sec3).
EVAL_SEED_LINES_A17 = (
    'os.environ["A17_CANARY_SEED"] = "1"'
    "  # A17 72B-VL canary (scope v2 sec3 rho_action denominator; sec5 full 7920s window)\n"
    'print("A17-CANARY seed=1 mode=throughput-canary games=ft09-0d8bbf25,sb26-7fbdac44,'
    'lp85-305b61c3,vc33-5430563c composition=W0 (duck + (f) continuation, NO warpack); '
    '27B numerator frozen 480 actions/7920s (w0_eval_s1)")\n'
)

# Inserted into the PYSETUP serve script before run_vllm_api_smoke_test():
# boot-time serve asserts (scope v1 risks D/E, amendment sec7.1). FAIL-LOUD.
A17_SERVE_DEFS = r'''def _a17_png_b64(size: int = 64) -> str:
    # Dependency-free solid-colour PNG for the boot MM probe (risk E).
    import base64
    import struct
    import zlib

    raw = (b'\x00' + b'\xc8\x32\x32' * size) * size
    def _chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xFFFFFFFF)
    ihdr = struct.pack('>IIBBBBB', size, size, 8, 2, 0, 0, 0)
    png = b'\x89PNG\r\n\x1a\n' + _chunk(b'IHDR', ihdr) + _chunk(b'IDAT', zlib.compress(raw)) + _chunk(b'IEND', b'')
    return base64.b64encode(png).decode('ascii')


def _a17_serve_asserts() -> None:
    # A17-CANARY serve-config smoke (scope v1 risks D/E; amendment sec7.1).
    # FAIL-LOUD: any miss raises -> kernel ERROR. NO 27B fallback, ever.
    models = request_json(f'{VLLM_BASE_URL}/models', timeout=60)
    ids = sorted(m.get('id', '') for m in models.get('data', []))
    if ids != ['Qwen2.5-VL-72B-Instruct-AWQ']:
        raise RuntimeError('A17-CANARY FATAL: served model ids ' + repr(ids) + ' != [Qwen2.5-VL-72B-Instruct-AWQ] - a silent 27B run would poison rho_action; refusing to continue')
    print('A17-CANARY: model=Qwen2.5-VL-72B-Instruct-AWQ', flush=True)
    tools = [{'type': 'function', 'function': {'name': 'submit_action', 'description': 'Submit the next ARC action.', 'parameters': {'type': 'object', 'properties': {'action': {'type': 'string'}, 'x': {'type': 'integer'}, 'y': {'type': 'integer'}}, 'required': ['action']}}}]
    payload = {'model': SERVED_MODEL_NAME, 'messages': [{'role': 'user', 'content': 'Call the submit_action tool with action ACTION6, x 3, y 7.'}], 'tools': tools, 'tool_choice': {'type': 'function', 'function': {'name': 'submit_action'}}, 'temperature': 0.0, 'max_tokens': 256}
    response = request_json(f'{VLLM_BASE_URL}/chat/completions', payload=payload, timeout=300)
    calls = response['choices'][0]['message'].get('tool_calls') or []
    if not calls or calls[0].get('function', {}).get('name') != 'submit_action':
        raise RuntimeError('A17-CANARY FATAL: tool-call round-trip FAILED under hermes parser (risk D silent-zero class): ' + json.dumps(response)[:2000])
    args = json.loads(calls[0]['function'].get('arguments') or '{}')
    if 'action' not in args:
        raise RuntimeError('A17-CANARY FATAL: tool-call arguments missing required key action: ' + repr(args))
    print('A17-CANARY tool-call-roundtrip=OK parser=hermes name=submit_action args=' + json.dumps(args, sort_keys=True), flush=True)
    image_url = 'data:image/png;base64,' + _a17_png_b64()
    payload = {'model': SERVED_MODEL_NAME, 'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': image_url}}, {'type': 'text', 'text': 'Answer with one word: what colour is this image?'}]}], 'temperature': 0.0, 'max_tokens': 32}
    response = request_json(f'{VLLM_BASE_URL}/chat/completions', payload=payload, timeout=300)
    content = (response['choices'][0]['message'].get('content') or '').strip()
    if not content:
        raise RuntimeError('A17-CANARY FATAL: MM boot probe returned empty content - vision path broken (risk E)')
    print('A17-CANARY mm-image-roundtrip=OK reply=' + repr(content[:60]), flush=True)'''

# Replaces the 27B MODEL_PATH resolution inside PYSETUP: marker-based finder
# for the attached Kaggle Model (mount-path agnostic), FAIL-LOUD if absent.
A17_MODEL_FIND_BLOCK = r'''def _a17_find_72b_model() -> Path:
    # Marker-based, mount-path-agnostic (same discipline as the warpack /
    # continuation grafts). The ONLY acceptable model is the Qwen2.5-VL AWQ
    # artifact; if it is not attached, FAIL LOUDLY - a silent 27B fallback
    # would poison the rho_action denominator (scope v2 sec3).
    input_root = Path(os.environ.get('A17_INPUT_ROOT', '/kaggle/input'))
    hits = []
    for cfg in sorted(input_root.rglob('config.json')):
        try:
            text = cfg.read_text(encoding='utf-8', errors='ignore')
        except OSError:
            continue
        if 'Qwen2_5_VLForConditionalGeneration' in text and 'quantization_config' in text:
            if any(cfg.parent.glob('*.safetensors')):
                hits.append(cfg.parent)
    if not hits:
        raise RuntimeError('A17-CANARY FATAL: Qwen2.5-VL-72B-Instruct-AWQ not found under ' + str(input_root) + ' - attach Kaggle Model qwen-lm/qwen2.5-vl/transformers/72b-instruct-awq/1. REFUSING 27B fallback.')
    if len(hits) > 1:
        raise RuntimeError('A17-CANARY FATAL: multiple VL-AWQ candidate dirs: ' + ', '.join(str(h) for h in hits))
    return hits[0]


MODEL_PATH = _a17_find_72b_model()
print('A17-CANARY model_path=' + str(MODEL_PATH), flush=True)'''

# The exact-string rewrites applied AT KERNEL RUNTIME to the decoded
# setup_commands.json command (the vLLM serve script ships inside the attached
# jeroencottaar/taaf-kaggle-source-share dataset, so it can only be patched at
# runtime). Each `old` must occur EXACTLY once or the kernel dies loudly.
A17_SETUP_REWRITES: list[tuple[str, str]] = [
    # (1) served-model identity (also the smoke-test model id)
    ("SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'",
     "SERVED_MODEL_NAME = 'Qwen2.5-VL-72B-Instruct-AWQ'"),
    # (2) memory envelope: 43GB AWQ weights force max-model-len 65536 -> 32768
    #     (= ANALYZER_CONTEXT_WINDOW; zero behavioural cost, scope v1 sec0)
    ("VLLM_MAX_MODEL_LEN = 65536", "VLLM_MAX_MODEL_LEN = 32768"),
    # (3) model path: marker-based finder, fail-loud (no 27B fallback)
    ("MODEL_PATH = resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG)",
     A17_MODEL_FIND_BLOCK),
    # (4) risk D: hermes tool parser; DROP the qwen3 reasoning parser +
    #     preserve_thinking kwargs (Qwen2.5 has neither); explicit awq_marlin
    ("        '--tool-call-parser',\n"
     "        'qwen3_coder',\n"
     "        '--generation-config',\n"
     "        'vllm',\n"
     "        '--enable-prefix-caching',\n"
     "        '--default-chat-template-kwargs',\n"
     "        '{\"preserve_thinking\": true}',\n"
     "        '--reasoning-parser',\n"
     "        'qwen3',\n",
     "        '--tool-call-parser',\n"
     "        'hermes',\n"
     "        '--generation-config',\n"
     "        'vllm',\n"
     "        '--enable-prefix-caching',\n"
     "        '--quantization',\n"
     "        'awq_marlin',\n"),
    # (5) Qwen2.5 has no thinking mode: drop the boot-smoke thinking kwarg
    ("        'chat_template_kwargs': {'enable_thinking': False},\n", ""),
    # (6) analyzer thinking off (amendment sec7.1)
    ("'LOCAL_ANALYZER_ENABLE_THINKING': 'true',",
     "'LOCAL_ANALYZER_ENABLE_THINKING': 'false',"),
    # (7) GPU banner + hard SKU gate (scope sec0: different GPU = run VOID)
    ("assert_expected_cuda_gpu()\nmissing",
     "assert_expected_cuda_gpu()\n"
     "_a17_gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()\n"
     "print('A17-CANARY gpu=' + _a17_gpu, flush=True)\n"
     "if 'rtx pro 6000' not in _a17_gpu.lower():\n"
     "    raise RuntimeError('A17-CANARY FATAL: GPU ' + repr(_a17_gpu) + ' is not RTX PRO 6000 - run VOID per scope sec0 / amendment sec5.2')\n"
     "missing"),
    # (8) persist the exact serve cmd+env so the cell-14 liveness gate can
    #     restart the server identically (panel R19)
    ("    print('Starting vLLM OpenAI server:', ' '.join(cmd), flush=True)",
     "    (WORKING_DIR / 'a17_vllm_cmd.json').write_text(json.dumps({'cmd': [str(part) for part in cmd], 'env': vllm_env(), 'log': str(VLLM_SERVER_LOG), 'pid': str(VLLM_SERVER_PID), 'base_url': VLLM_BASE_URL}), encoding='utf-8')\n"
     "    print('A17-CANARY serve-cmd persisted to a17_vllm_cmd.json (liveness-gate restart source)', flush=True)\n"
     "    print('Starting vLLM OpenAI server:', ' '.join(cmd), flush=True)"),
    # (9) insert the boot serve asserts defs
    ("\n\ndef run_vllm_api_smoke_test() -> None:",
     "\n\n" + A17_SERVE_DEFS + "\n\n\ndef run_vllm_api_smoke_test() -> None:"),
    # (10) run the serve asserts right after the stock chat smoke
    ("start_vllm_server()\nrun_vllm_api_smoke_test()\nsetup_env = {",
     "start_vllm_server()\nrun_vllm_api_smoke_test()\n_a17_serve_asserts()\nsetup_env = {"),
]

# Cell-8 anchor: the stock setup-command loop (byte-exact, structural guard).
CELL8_ANCHOR = (
    "    # Solver setup commands (wheels, vLLM server startup, ...) run before the benchmark loads.\n"
    "    env = _command_env()\n"
    '    for command in json.loads((BUNDLE_DIR / "setup_commands.json").read_text()):\n'
)


def _cell8_a17_graft() -> str:
    return (
        "    # Solver setup commands (wheels, vLLM server startup, ...) run before the benchmark loads.\n"
        "    # --- A17-CANARY BEGIN serve-config rewrite (scope v1 sec2; FAIL-LOUD) ---\n"
        "    # POLICY INVERSION (documented): eval grafts normally fall back to vanilla\n"
        "    # duck on failure. Here a vanilla run would SILENTLY SERVE THE 27B and\n"
        "    # poison the rho_action denominator, so ANY rewrite failure raises ->\n"
        "    # kernel ERROR (a dead canary is a retry; a silent 27B is a poisoned\n"
        "    # measurement).\n"
        f"    A17_SETUP_REWRITES = {A17_SETUP_REWRITES!r}\n"
        "\n"
        "    def _a17_patch_setup_commands(commands):\n"
        "        if not isinstance(commands, list) or len(commands) != 1:\n"
        "            raise RuntimeError('A17-CANARY FATAL: expected exactly 1 setup command, got ' + repr(commands)[:200])\n"
        "        text = commands[0]\n"
        "        for old, new in A17_SETUP_REWRITES:\n"
        "            found = text.count(old)\n"
        "            if found != 1:\n"
        "                raise RuntimeError('A17-CANARY FATAL: serve-config anchor matched %d times (want 1): %r' % (found, old[:100]))\n"
        "            text = text.replace(old, new)\n"
        "        for veto in ('qwen3_coder', '--reasoning-parser', 'preserve_thinking', 'enable_thinking', \"'vrfai/Qwen3.6-27B-FP8'\"):\n"
        "            if veto in text:\n"
        "                raise RuntimeError('A17-CANARY FATAL: 27B serve artifact %r survived the rewrite' % veto)\n"
        "        for need in ('hermes', 'awq_marlin', 'Qwen2.5-VL-72B-Instruct-AWQ', '_a17_serve_asserts()', 'A17-CANARY gpu=', 'a17_vllm_cmd.json', 'VLLM_MAX_MODEL_LEN = 32768'):\n"
        "            if need not in text:\n"
        "                raise RuntimeError('A17-CANARY FATAL: required 72B serve token %r missing after rewrite' % need)\n"
        "        print('A17-CANARY setup-commands rewrite OK (%d anchors replaced; loud-fail mode, no 27B fallback)' % len(A17_SETUP_REWRITES), flush=True)\n"
        "        return [text]\n"
        "    # --- A17-CANARY END serve-config rewrite ---\n"
        "    env = _command_env()\n"
        '    for command in _a17_patch_setup_commands(json.loads((BUNDLE_DIR / "setup_commands.json").read_text())):\n'
    )


# Cell-14 anchors (byte-exact, all asserted unique before grafting).
CELL14_DEFS_ANCHOR = (
    "    # Print the run preamble and persist the launcher's git status for diagnostics.\n"
)
CELL14_GAMES_ANCHOR = "        bm.games = _offline_games(competition_env_files)\n"
CELL14_TRY_ANCHOR = (
    "    # Play the benchmark; teardown commands run even if the run raises.\n"
    "    try:\n"
)
CELL14_TRY_NEW = (
    "    # Play the benchmark; teardown commands run even if the run raises.\n"
    "    _a17_start_heartbeat()\n"
    "    try:\n"
)
CELL14_POSTRUN_ANCHOR = (
    "        if not TRUE_SUBMISSION:\n"
    "            # An offline run isn't scored, but Kaggle still expects a submission.parquet output.\n"
    "            _write_dummy_submission(WORKING_DIR)\n"
)

# Cell-14 defs: heartbeat (ops #684625) + liveness gate (panel R19) + post-run
# rho_action report. All log-only w.r.t. the harness (zero behaviour change to
# solver/reset/deadline logic; risk A / amendment sec7.2).
CELL14_A17_DEFS = r'''    # --- A17-CANARY BEGIN heartbeat/liveness/report defs ---
    # Ops #684625 (vLLM can silently hang on RTX Pro 6000 at >=8 concurrent
    # sessions) + panel R19 liveness directive. Log-only heartbeat every 120 s;
    # in-run liveness GATE: >=600 s with zero progress evidence -> ONE server
    # restart, second stall -> LOUD kernel death (os._exit(70)). The kill is
    # DISARMED in the final 10 min of the window and after it: the drain +
    # diagnostics phases have legitimately zero generation activity, and a
    # false kill there would destroy a completed run's artifacts.
    A17_SCREEN_GAMES = ["ft09-0d8bbf25", "sb26-7fbdac44", "lp85-305b61c3", "vc33-5430563c"]
    A17_WINDOW_S = 7920.0
    A17_HEARTBEAT_INTERVAL_S = 120.0
    A17_STALL_S = 600.0
    A17_KILL_DISARM_S = A17_WINDOW_S - 600.0

    def _a17_actions_total(bench_path):
        # Best-effort: the harness periodically saves benchmark.json in-run.
        try:
            data = json.loads(bench_path.read_text(encoding="utf-8"))
            return sum(len(gr.get("history", [])) for gr in data.get("game_runs", []))
        except Exception:
            return None

    def _a17_restart_vllm():
        info = json.loads((WORKING_DIR / "a17_vllm_cmd.json").read_text(encoding="utf-8"))
        try:
            os.kill(int(Path(info["pid"]).read_text().strip()), 9)
        except Exception:
            pass
        time.sleep(10)
        log_handle = open(info["log"], "a", encoding="utf-8")
        process = subprocess.Popen(info["cmd"], env=info["env"], stdout=log_handle, stderr=subprocess.STDOUT, text=True)
        Path(info["pid"]).write_text(str(process.pid), encoding="utf-8")
        deadline = time.monotonic() + 900.0
        while time.monotonic() < deadline:
            try:
                with urlopen(info["base_url"] + "/models", timeout=5) as response:
                    if response.status < 500:
                        return
            except Exception:
                time.sleep(5)
        raise RuntimeError("restarted vLLM server never answered /v1/models")

    def _a17_start_heartbeat(working_dir=None):
        import re
        import threading

        working_dir = working_dir or WORKING_DIR
        log_path = working_dir / "vllm-openai-server.log"
        bench_path = working_dir / "benchmark.json"
        start = time.monotonic()
        state = {"restarts": 0, "last_progress": start, "last_actions": None, "last_log_bytes": -1}

        def _beat():
            while True:
                time.sleep(A17_HEARTBEAT_INTERVAL_S)
                now = time.monotonic()
                elapsed = int(now - start)
                log_bytes = log_path.stat().st_size if log_path.exists() else 0
                gen_tps = None
                running = None
                try:
                    tail = log_path.read_bytes()[-20000:].decode("utf-8", errors="replace")
                    stats = re.findall(r"generation throughput: ([0-9.]+) tokens/s, Running: (\d+) reqs", tail)
                    if stats:
                        gen_tps = float(stats[-1][0])
                        running = int(stats[-1][1])
                except Exception:
                    pass
                actions = _a17_actions_total(bench_path) if bench_path.exists() else None
                progressed = False
                if actions is not None and state["last_actions"] is not None and actions > state["last_actions"]:
                    progressed = True
                if actions is not None:
                    state["last_actions"] = actions
                if gen_tps is not None and gen_tps > 0.0 and log_bytes != state["last_log_bytes"]:
                    progressed = True
                state["last_log_bytes"] = log_bytes
                if progressed:
                    state["last_progress"] = now
                stall_s = int(now - state["last_progress"])
                print("A17-CANARY HEARTBEAT t=%d actions_total=%s vllm_log_bytes=%d gen_tps=%s running_reqs=%s stall_s=%d restarts=%d"
                      % (elapsed, "NA" if actions is None else actions, log_bytes,
                         "NA" if gen_tps is None else gen_tps, "NA" if running is None else running,
                         stall_s, state["restarts"]), flush=True)
                if stall_s < A17_STALL_S:
                    continue
                if elapsed > A17_KILL_DISARM_S:
                    print("A17-CANARY LIVENESS-STALL-POSTWINDOW t=%d stall_s=%d (kill disarmed in final/post window)" % (elapsed, stall_s), flush=True)
                    continue
                if state["restarts"] == 0:
                    print("A17-CANARY LIVENESS-STALL t=%d stall_s=%d - attempting ONE vLLM restart (panel R19 gate)" % (elapsed, stall_s), flush=True)
                    try:
                        _a17_restart_vllm()
                        state["restarts"] = 1
                        state["last_progress"] = time.monotonic()
                        print("A17-CANARY LIVENESS-RESTART t=%d restarts=1 OK" % int(time.monotonic() - start), flush=True)
                    except Exception as exc:
                        print("A17-CANARY LIVENESS-FAIL t=%d restarts=1 (restart failed: %r) - hard-exiting so the window dies LOUDLY" % (int(time.monotonic() - start), exc), flush=True)
                        sys.stdout.flush()
                        os._exit(70)
                else:
                    print("A17-CANARY LIVENESS-FAIL t=%d restarts=%d - second stall; hard-exiting so the window dies LOUDLY" % (elapsed, state["restarts"]), flush=True)
                    sys.stdout.flush()
                    os._exit(70)

        threading.Thread(target=_beat, name="a17-heartbeat", daemon=True).start()

    def _a17_post_run_report(working_dir=None):
        import re

        working_dir = working_dir or WORKING_DIR
        bench_path = working_dir / "benchmark.json"
        if not bench_path.exists():
            raise RuntimeError("A17-CANARY FATAL: benchmark.json not written - rho_action denominator unmeasurable")
        data = json.loads(bench_path.read_text(encoding="utf-8"))
        by_id = {gr.get("game_id"): gr for gr in data.get("game_runs", [])}
        total = 0
        present = 0
        for gid in A17_SCREEN_GAMES:
            game_run = by_id.get(gid)
            if game_run is None:
                print("A17-CANARY N(%s)=MISSING" % gid, flush=True)
                continue
            n_actions = len(game_run.get("history", []))
            apl_sum = sum(game_run.get("actions_per_level") or [])
            window_s = float(game_run.get("final_wallclock_seconds") or 0.0)
            total += n_actions
            present += 1
            print("A17-CANARY N(%s)=%d actions_per_level_sum=%d lc=%s window_s=%.0f"
                  % (gid, n_actions, apl_sum, game_run.get("levels_completed"), window_s), flush=True)
            if abs(window_s - A17_WINDOW_S) > 0.05 * A17_WINDOW_S:
                print("A17-CANARY WARN window_drift game=%s window_s=%.0f expected~%.0f - null comparison VOID per amendment sec7.2 if the deadline regressed" % (gid, window_s, A17_WINDOW_S), flush=True)
        print("A17-CANARY rho_action_denominator=%d (games_present=%d/4; 27B numerator frozen at 480 actions/7920s, w0_eval_s1)" % (total, present), flush=True)
        print("A17-CANARY concurrency: harness solver params untouched; effective concurrent games this run = %d" % len(bm.games), flush=True)
        log_path = working_dir / "vllm-openai-server.log"
        rates = []
        if log_path.exists():
            for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
                match = re.search(r"MM cache hit rate: ([0-9.]+)%", line)
                if match:
                    rates.append(float(match.group(1)))
        if rates and max(rates) > 0.0:
            print("A17-CANARY mm_cache=NONZERO max_hit_rate=%.1f%% samples=%d" % (max(rates), len(rates)), flush=True)
        elif rates:
            print("A17-CANARY WARN mm_cache=ZERO across %d samples - vision path suspect; boot MM probe passed, but a 0%% MM run is discard-grade per scope v1 risk E" % len(rates), flush=True)
        else:
            print("A17-CANARY WARN mm_cache_evidence=NOT-FOUND in vllm log (format drift?) - rely on the boot MM probe", flush=True)
    # --- A17-CANARY END heartbeat/liveness/report defs ---
'''

# Cell-14 screen filter, appended right after the offline game-list line.
CELL14_A17_FILTER_BLOCK = r'''        # --- A17-CANARY BEGIN 4-game screen filter (scope v2 sec2 + sec7.2 drift rule) ---
        _a17_available = sorted(g.env_name for g in bm.games)
        bm.games = [g for g in bm.games if g.env_name in A17_SCREEN_GAMES]
        _a17_found = sorted(g.env_name for g in bm.games)
        _a17_missing = sorted(set(A17_SCREEN_GAMES) - set(_a17_found))
        for _a17_gid in _a17_missing:
            print("A17-CANARY DRIFT game=%s MISSING from bundled environments (dropped from both sides per scope v2 sec7.2)" % _a17_gid, flush=True)
        if len(_a17_missing) >= 2:
            raise RuntimeError("A17-CANARY FATAL: %d screen games missing (%s) - >=2 drops VOID the screen (scope v2 sec7.2); available=%s" % (len(_a17_missing), ",".join(_a17_missing), ",".join(_a17_available)))
        if not bm.games:
            raise RuntimeError("A17-CANARY FATAL: zero screen games resolved; available=%s" % ",".join(_a17_available))
        print("A17-CANARY games=%s (n=%d of 4; full %ds per-game window, games concurrent)" % (",".join(_a17_found), len(_a17_found), int(A17_WINDOW_S)), flush=True)
        # --- A17-CANARY END 4-game screen filter ---
'''


def main(v2: bool = False, w0: bool = False, sentinel: bool = False,
         sentinel_budget: int | None = None, continuation: bool = True,
         a17: bool = False, compaction: bool = False,
         animation: bool = False, p1: bool = False,
         effnote: bool = False) -> None:
    if a17:
        src_nb, out_dir, kernel_id = SRC_NB, OUT_DIR_A17, KERNEL_ID_A17
    elif effnote:
        src_nb, out_dir, kernel_id = SRC_NB, OUT_DIR_EFFNOTE, KERNEL_ID_EFFNOTE
    elif p1:
        src_nb, out_dir, kernel_id = SRC_NB, OUT_DIR_P1, KERNEL_ID_P1
    elif animation:
        src_nb, out_dir, kernel_id = SRC_NB, OUT_DIR_ANIMATION, KERNEL_ID_ANIMATION
    elif compaction:
        src_nb, out_dir, kernel_id = SRC_NB, OUT_DIR_COMPACTION, KERNEL_ID_COMPACTION
    elif sentinel:
        src_nb, out_dir, kernel_id = SRC_NB, OUT_DIR_SENTINEL, KERNEL_ID_SENTINEL
    elif w0:
        src_nb, out_dir, kernel_id = SRC_NB, OUT_DIR_W0, KERNEL_ID_W0
    elif v2:
        src_nb, out_dir, kernel_id = SRC_NB_V2, OUT_DIR_V2, KERNEL_ID_V2
    else:
        src_nb, out_dir, kernel_id = SRC_NB, OUT_DIR, KERNEL_ID
    stem = kernel_id.split("/", 1)[1]

    nb = json.loads(src_nb.read_text(encoding="utf-8"))
    cell2 = nb["cells"][2]
    src = "".join(cell2["source"])
    assert "WARPACK_FORCE_OFFLINE_BENCH" in src, "cell 2 is not the gate cell"
    assert not src.startswith(EVAL_LINE), "already an eval notebook?"
    seed_lines = (
        EVAL_SEED_LINES_A17 if a17
        else EVAL_SEED_LINES_EFFNOTE if effnote
        else EVAL_SEED_LINES_P1 if p1
        else EVAL_SEED_LINES_ANIMATION if animation
        else EVAL_SEED_LINES_COMPACTION if compaction
        else EVAL_SEED_LINES_SENTINEL if sentinel
        else EVAL_SEED_LINES_W0 if w0
        else EVAL_SEED_LINES_V2 if v2
        else "")
    if sentinel and sentinel_budget is not None:
        # The scored/eval regime is UNCAPPED (max_actions_per_game=None), so
        # without a forced budget the sentinel is a silent no-op (build report
        # "design decision 1" = the #1 pre-seal risk per prereg amendment
        # 2026-07-23 C7). C7 AS AMENDED 09:57 EDT 2026-07-23 rules
        # SENTINEL_BUDGET=150 (adopting the live two-seed 150-ledger; v2
        # game-envelope unit per R16 Q2 re-key) and requires the banner to
        # echo it.
        seed_lines += (
            f'os.environ["SENTINEL_BUDGET"] = "{sentinel_budget}"'
            "  # C7-ruled game-envelope budget (prereg amendment 2026-07-23, amended to 150; v2 unit=game-envelope)\n"
            f'print("sentinel-eval: SENTINEL_BUDGET={sentinel_budget} game-envelope (C7-as-amended, prereg amendment 2026-07-23)")\n'
        )
    prefix = EVAL_LINE + seed_lines
    cell2["source"] = (prefix + src).splitlines(keepends=True)
    cell2["outputs"] = []
    cell2["execution_count"] = None

    if sentinel:
        # (a)-defining diff: REPLACE the warpack graft (cell 12) with the
        # budget-sentinel-only graft. Sanity: the raw source's cell 12 IS the
        # warpack graft we are removing (so we never silently ship warpack).
        cell12_old = "".join(nb["cells"][12]["source"])
        assert "import warpack_patch" in cell12_old, \
            "sentinel source cell 12 is not the warpack graft (structural drift)"
        c12 = nb["cells"][12]
        c12["source"] = CELL12_SENTINEL.splitlines(keepends=True)
        c12["outputs"] = []
        c12["execution_count"] = None

    if animation:
        # animation-defining diff: REPLACE the warpack graft (cell 12) with the
        # animation-only graft. Sanity: the raw source's cell 12 IS the warpack
        # graft we are removing (so we never silently ship warpack).
        cell12_old = "".join(nb["cells"][12]["source"])
        assert "import warpack_patch" in cell12_old, \
            "animation source cell 12 is not the warpack graft (structural drift)"
        c12 = nb["cells"][12]
        c12["source"] = CELL12_ANIMATION.splitlines(keepends=True)
        c12["outputs"] = []
        c12["execution_count"] = None

        # Post-run canary (prereg sec3 K-A1..K-A4) into cell 14.
        c14 = nb["cells"][14]
        src14 = "".join(c14["source"])
        assert src14.count(CELL14_POSTRUN_ANCHOR) == 1, \
            "animation: cell 14 post-run anchor missing/ambiguous (structural drift)"
        c14["source"] = src14.replace(
            CELL14_POSTRUN_ANCHOR,
            CELL14_POSTRUN_ANCHOR + CELL14_ANIMATION_CANARY,
        ).splitlines(keepends=True)
        c14["outputs"] = []
        c14["execution_count"] = None

    if p1:
        # P1-defining diff: REPLACE the warpack graft (cell 12) with the
        # P1-only graft. Sanity: the raw source's cell 12 IS the warpack graft
        # we are removing (so we never silently ship warpack).
        cell12_old = "".join(nb["cells"][12]["source"])
        assert "import warpack_patch" in cell12_old, \
            "p1 source cell 12 is not the warpack graft (structural drift)"
        c12 = nb["cells"][12]
        c12["source"] = CELL12_P1.splitlines(keepends=True)
        c12["outputs"] = []
        c12["execution_count"] = None

        # Post-run canary into cell 14 (prereg sec3).
        c14 = nb["cells"][14]
        src14 = "".join(c14["source"])
        assert src14.count(CELL14_POSTRUN_ANCHOR) == 1, \
            "p1: cell 14 post-run anchor missing/ambiguous (structural drift)"
        c14["source"] = src14.replace(
            CELL14_POSTRUN_ANCHOR,
            CELL14_POSTRUN_ANCHOR + CELL14_P1_CANARY,
        ).splitlines(keepends=True)
        c14["outputs"] = []
        c14["execution_count"] = None

    if effnote:
        # EFFNOTE-defining diff: REPLACE the warpack graft (cell 12) with the
        # EFFNOTE-only graft. Sanity: the raw source's cell 12 IS the warpack
        # graft we are removing (so we never silently ship warpack).
        cell12_old = "".join(nb["cells"][12]["source"])
        assert "import warpack_patch" in cell12_old,             "effnote source cell 12 is not the warpack graft (structural drift)"
        c12 = nb["cells"][12]
        c12["source"] = CELL12_EFFNOTE.splitlines(keepends=True)
        c12["outputs"] = []
        c12["execution_count"] = None

        # Post-run canary into cell 14 (prereg sec3 K-E0/K-E1/K-E3).
        c14 = nb["cells"][14]
        src14 = "".join(c14["source"])
        assert src14.count(CELL14_POSTRUN_ANCHOR) == 1,             "effnote: cell 14 post-run anchor missing/ambiguous (structural drift)"
        c14["source"] = src14.replace(
            CELL14_POSTRUN_ANCHOR,
            CELL14_POSTRUN_ANCHOR + CELL14_EFFNOTE_CANARY,
        ).splitlines(keepends=True)
        c14["outputs"] = []
        c14["execution_count"] = None

    if compaction:
        # A22-defining diff: REPLACE the warpack graft (cell 12) with the
        # compaction-only graft. Sanity: the raw source's cell 12 IS the
        # warpack graft we are removing (so we never silently ship warpack).
        cell12_old = "".join(nb["cells"][12]["source"])
        assert "import warpack_patch" in cell12_old, \
            "compaction source cell 12 is not the warpack graft (structural drift)"
        c12 = nb["cells"][12]
        c12["source"] = CELL12_COMPACTION.splitlines(keepends=True)
        c12["outputs"] = []
        c12["execution_count"] = None

    if w0 or a17:
        # W0/A17-defining diff: REPLACE the warpack graft (cell 12) with the
        # continuation-only graft (A17 mirrors the W0 composition because the
        # frozen 27B numerator run w0_eval_s1 is duck + (f), NO warpack).
        # Sanity: the raw source's cell 12 IS the warpack graft we are
        # removing (so we never silently ship warpack).
        cell12_old = "".join(nb["cells"][12]["source"])
        assert "import warpack_patch" in cell12_old, \
            "W0/A17 source cell 12 is not the warpack graft (structural drift)"
        c12 = nb["cells"][12]
        c12["source"] = CELL12_W0.splitlines(keepends=True)
        c12["outputs"] = []
        c12["execution_count"] = None

    if a17:
        # Risk A / amendment sec7.2: the reset constant must ride through
        # untouched (we only PREPEND to cell 2, but assert it explicitly).
        assert 'os.environ["ONLY_RESET_LEVELS"] = "true"' in src, \
            "a17: cell 2 lost the ONLY_RESET_LEVELS pin (reset-path drift)"

        # Cell 8: FAIL-LOUD serve-config rewrite (72B model + serve flags).
        c8 = nb["cells"][8]
        src8 = "".join(c8["source"])
        assert src8.count(CELL8_ANCHOR) == 1, \
            "a17: cell 8 setup-commands anchor missing/ambiguous (structural drift)"
        c8["source"] = src8.replace(CELL8_ANCHOR, _cell8_a17_graft()).splitlines(keepends=True)
        c8["outputs"] = []
        c8["execution_count"] = None

        # Cell 14: screen filter + heartbeat/liveness gate + post-run report.
        c14 = nb["cells"][14]
        src14 = "".join(c14["source"])
        for anchor in (CELL14_DEFS_ANCHOR, CELL14_GAMES_ANCHOR, CELL14_TRY_ANCHOR, CELL14_POSTRUN_ANCHOR):
            assert src14.count(anchor) == 1, \
                f"a17: cell 14 anchor missing/ambiguous (structural drift): {anchor[:60]!r}"
        src14 = src14.replace(CELL14_DEFS_ANCHOR, CELL14_A17_DEFS + CELL14_DEFS_ANCHOR)
        src14 = src14.replace(CELL14_GAMES_ANCHOR, CELL14_GAMES_ANCHOR + CELL14_A17_FILTER_BLOCK)
        src14 = src14.replace(CELL14_TRY_ANCHOR, CELL14_TRY_NEW)
        src14 = src14.replace(CELL14_POSTRUN_ANCHOR,
                              CELL14_POSTRUN_ANCHOR + "            _a17_post_run_report()\n")
        c14["source"] = src14.splitlines(keepends=True)
        c14["outputs"] = []
        c14["execution_count"] = None

    if continuation and not (w0 or a17):
        # (f) default: append the continuation graft unless the mode already
        # carries it (idempotent — the v2 source gains it at build_notebook
        # time once that builder also defaults (f) ON).
        c12 = nb["cells"][12]
        src12 = "".join(c12["source"])
        if "import continuation_patch" not in src12:
            c12["source"] = (src12 + CONTINUATION_DEFAULT_BLOCK).splitlines(keepends=True)
            c12["outputs"] = []
            c12["execution_count"] = None
            print("cell-12: (f) continuation graft APPENDED (default ON since 2026-07-23; "
                  "--no-continuation to omit)")
        else:
            print("cell-12: (f) continuation graft already present (source carries it)")
    elif not (w0 or a17):
        print("cell-12: (f) continuation graft OMITTED (--no-continuation)")
    else:
        print("cell-12: continuation graft IS the mode's cell 12 (W0 composition)")

    if v2:  # sanity: the v2 source must actually carry the ledger-ON graft
        cell12 = "".join(nb["cells"][12]["source"])
        assert "ledger_patch" in cell12, "v2 source lacks the ledger graft"
        assert "LEDGER CANARY" in cell12, "v2 source lacks the ledger canary"
        assert "ledger_canary_report" in "".join(nb["cells"][14]["source"]), \
            "v2 source cell 14 lacks the canary summary"

    meta = json.loads(SRC_META.read_text(encoding="utf-8"))
    meta["id"] = kernel_id
    meta["title"] = stem
    meta["code_file"] = f"{stem}.ipynb"
    if a17:
        # kaggle_env_match discipline: ONLY delta vs the eval family metadata
        # (beyond id/title/code_file) is the attached 72B Kaggle Model source.
        meta["model_sources"] = [A17_MODEL_SOURCE]

    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{stem}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_dir}/{stem}.ipynb (+metadata); cell-2 delta = "
          f"{len(prefix.splitlines())} prepended line(s):")
    for line in prefix.splitlines():
        print(f"  {line}")


if __name__ == "__main__":
    args = sys.argv[1:]
    budget = None
    if "--sentinel-budget" in args:
        budget = int(args[args.index("--sentinel-budget") + 1])
    main(v2="--v2" in args, w0="--w0" in args, sentinel="--sentinel" in args,
         sentinel_budget=budget, continuation="--no-continuation" not in args,
         a17="--a17-canary" in args, compaction="--compaction" in args,
         animation="--animation" in args, p1="--p1" in args,
         effnote="--effnote" in args)
