"""GRAFT FLOOR EVAL builder (2026-08-17) — ONE variable: the public graft stack.

PREREG: `learnings/war_room/graft_floor_prereg_2026-08-17.md` (sealed before this build).
AUDIT:  `duck_eval/graft/bundle_audit_2026-08-17.md` (the verification that reshaped the arm).

WHY THIS ARM IS NOT THE AUTHORIZED ONE. The 08-17 ruling authorized
`banking + transfer + shortcircuit`. The audit falsified that arm's precondition:

  * `banking` gates on `run.state == "won"` (banking_solver.py:180). Across 23 pulled eval
    artifacts / 470 game-runs — the entire recorded campaign — runs reaching "won" = 0.
  * `transfer` needs clone siblings. Our eval rail is n_passes=1, 25 games, 25 UNIQUE
    game_ids. Its own docstring: a non-clone set makes the stack "a measured no-op".

So the authorized arm reduces to `shortcircuit` alone and its REFUTE would carry no
information — the A9/warpack error verbatim ("our gate measured LEVELS on an offline bench
where banking's conditions never fired"). This build substitutes the REACHABLE public floor:
thtennant's published v19 flag set, with banking/transfer OFF *and asserted absent*.

BUILT FROM: `notebooks/duckfork/tufa-labs-duck-harness-june-30-milestone-winner.ipynb` — the
frozen upstream fork, the exact bytes the `duck-harness-kaggle` baseline family (lc 18/19/21)
ran. NEVER hand-built (`feedback_arc_kernel_structural_drift`: 5 ERRORs, all hand-built).
Fresh slug (`feedback_fresh_kernel_slug`).

A PLAIN BUILD *IS* THE EVAL. The frozen fork's run cell branches on
TRUE_SUBMISSION = KAGGLE_IS_COMPETITION_RERUN, unset in any ordinary kernel BUILD, so it plays
the 25 bundled competition environments OFFLINE via `_offline_games()`, writes a dummy
submission.parquet, and is never scored. Same rail as the baseline trio.

WHAT CHANGES — exactly three code cells, and nothing else:
  cell 2   identity banner only (no behavioural change)
  cell 6   DATASET_SOURCES: the SOURCE-BUNDLE entry only, stock -> thtennant fork.
           index 0 must stay the bundle (cell 6 does `resolved = BUNDLE_DIR if i == 0`), and
           setup_commands.json resolves wheelhouse/model by exact ref string, so those two
           refs stay verbatim. ENGINE UNCHANGED (Qwen3.6) -> no engine confound.
  cell 12  the graft install (the customization hook the harness ships for exactly this).

WHY *REPLACE* THE BUNDLE RATHER THAN ATTACH BOTH — this is the build's subtlest hazard.
Cell 6 finds the bundle by marker: `Path("/kaggle/input").rglob("taaf-kaggle-bundle.json")`,
FIRST match wins, and rglob order is not contractual. The fork contains that marker too
(identical bytes). Attaching both makes BUNDLE_DIR ambiguous, and the failure is silent: if it
resolves to the stock dir, sys.path never gets taaf-grafts, cell 12's import raises
ModuleNotFoundError, the guarded except prints one line, and the run completes looking like a
NORMAL STOCK RUN which we would then score as a REFUTE.

Replacing is safe because the fork is verified stock + 16 additive files: recursive sha256 gave
73 stock files / 0 modified / 0 missing / 16 added under src/taaf-grafts/. Independently
corroborated by thtennant's own v18/v19 metadata, which attach exactly three datasets with the
fork REPLACING the stock source-share.

WHAT IS NOT CHANGED, and is asserted to survive (the one-variable proof): the engine
(vrfai/Qwen3.6-27B-FP8), the wheelhouse (vLLM 0.19.0), every setup command byte, every other
notebook cell, and every env field of the frozen fork's metadata.

FLAG-SET NOTES (from the audit, all verified in source):
  * ALL flags default OFF; `install(bm, {})` is a proven no-op.
  * `transfer` IMPLIES `banking` (composite.py:172) -> neither may appear.
  * `hudmask` is NESTED under `goalkeep` (composite.py:297) -> arming it alone does nothing.
  * UNKNOWN FLAG NAMES ARE SILENTLY IGNORED (no validation). A typo = a silent stock run that
    looks like a clean arm. This is why FLAGS is built from a checked table below and why the
    scorer asserts the runtime banner rather than the source.
  * `expected_version=1` makes a GRAFTS_API_VERSION bump fail CLOSED (to stock, loudly) — our
    only lever against an unpinnable dataset version. Kaggle attaches the LATEST version and
    kernel metadata cannot pin one; the fork was republished 2026-08-17 00:26.

Run:  uv run python duck_eval/graft/build_graft_eval.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC_NB = REPO / "notebooks" / "duckfork" / "tufa-labs-duck-harness-june-30-milestone-winner.ipynb"
SRC_META = REPO / "notebooks" / "duckfork" / "kernel-metadata.json"
OUT_DIR = REPO / "notebooks" / "graft-floor-eval"
OUT_NB = OUT_DIR / "arc3-graft-floor-eval.ipynb"
OUT_META = OUT_DIR / "kernel-metadata.json"

KERNEL_ID = "canivel/arc3-graft-floor-eval"
KERNEL_TITLE = "arc3-graft-floor-eval"

# --- the one variable ------------------------------------------------------
OLD_SOURCE_DS = "jeroencottaar/taaf-kaggle-source-share"        # stock bundle (frozen fork)
NEW_SOURCE_DS = "thtennant/taaf-kaggle-source-share-fork"       # stock + src/taaf-grafts/
WHEELS_DS = "driessmit1/arc3-vllm-h100-wheelhouse-v3"           # UNCHANGED
ENGINE_DS = "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"      # UNCHANGED (incumbent Qwen3.6)
SERVED_MODEL = "vrfai/Qwen3.6-27B-FP8"                          # UNCHANGED

# Audited manifest shas (path+sha256 manifest of the bundle, then sha256 of that manifest).
# Recorded so the pull-back gate can prove the attached version is the audited one.
AUDITED_BUNDLE_MANIFEST_SHA = "df447f61caa181cca68049e28b139e02"   # 89 files
AUDITED_GRAFTS_MANIFEST_SHA = "7705481551494b141d6a33ffec1d7a20"   # 16 files
AUDITED_DS_VERSION_DATE = "2026-08-17 00:26:06"

# --- the flag set: thtennant v19, verbatim ---------------------------------
# Every name here was read out of composite.py this session. A name not in VALID_FLAGS is a
# typo, and a typo is silently ignored by install() — so the builder fails closed instead.
VALID_FLAGS = (
    "efficiency", "retry_guard", "shortcircuit", "banking", "transfer", "recovery",
    "goalkeep", "hudmask", "schema_void", "schema_notes", "schema_helpers", "context_window",
)
# Reachable on our rail, and each with a named endpoint (prereg section 2).
FLAGS_ON = ("efficiency", "retry_guard", "shortcircuit", "goalkeep", "hudmask")
# Unreachable on this rail (banking needs a win: 0/470; transfer needs clones: 25 unique ids).
# The arm is DEFINED by their exclusion, so they are named here and asserted absent at read.
FLAGS_FORBIDDEN = ("banking", "transfer")
GRAFTS_API_VERSION = 1

for _f in FLAGS_ON:
    if _f not in VALID_FLAGS:
        raise SystemExit(f"BUILD FAIL: {_f!r} is not a flag install() reads (silent-ignore trap)")
if set(FLAGS_ON) & set(FLAGS_FORBIDDEN):
    raise SystemExit("BUILD FAIL: a forbidden flag is in FLAGS_ON")
if "hudmask" in FLAGS_ON and "goalkeep" not in FLAGS_ON:
    raise SystemExit("BUILD FAIL: hudmask is nested under goalkeep; alone it silently no-ops")

_FLAG_LITERAL = ", ".join(f'"{f}": True' for f in FLAGS_ON)

# ---------------------------------------------------------------------------
CELL2_ANCHOR = 'print(f"taaf.kaggle: TRUE_SUBMISSION={TRUE_SUBMISSION}")'
CELL2_NEW = f'''print(f"taaf.kaggle: TRUE_SUBMISSION={{TRUE_SUBMISSION}}")
print(
    "GRAFT-EVAL seed=1 mode=graft-floor-local25 "
    "grafts={NEW_SOURCE_DS} REPLACES {OLD_SOURCE_DS} "
    "(verified stock + 16 additive files, 0 stock bytes changed) "
    "flags={'+'.join(FLAGS_ON)} FORBIDDEN={'+'.join(FLAGS_FORBIDDEN)} "
    "engine={ENGINE_DS} (UNCHANGED, {SERVED_MODEL}) "
    "wheels={WHEELS_DS} (UNCHANGED, vLLM 0.19.0) "
    "audited_bundle_sha={AUDITED_BUNDLE_MANIFEST_SHA} "
    "baseline=duck-harness-kaggle m=3 lc 18/19/21 "
    "primary=mean_dlc HARM<=-0.286320 SIGNAL>=+0.286320 "
    "secondary=mean_score NON-INFERENTIAL",
    flush=True,
)'''

CELL6_ANCHOR = (
    'DATASET_SOURCES = ["jeroencottaar/taaf-kaggle-source-share", '
    '"driessmit1/arc3-vllm-h100-wheelhouse-v3", '
    '"driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"]'
)
CELL6_NEW = (
    "# GRAFT-EVAL: the SOURCE BUNDLE entry is the ONE substitution — stock -> thtennant fork\n"
    "# (verified stock + 16 additive files under src/taaf-grafts/, 0 stock files modified).\n"
    "# REPLACED, never attached alongside: _find_bundle_dir() takes the FIRST rglob match of\n"
    "# taaf-kaggle-bundle.json and the fork carries that marker too, so attaching both makes\n"
    "# BUNDLE_DIR ambiguous — and a stock resolution would silently run stock (cell 12's\n"
    "# import would fail into its guarded except). Order preserved: index 0 stays the bundle,\n"
    "# and setup_commands.json resolves the wheelhouse/engine by exact ref string.\n"
    "# ENGINE AND WHEELHOUSE UNCHANGED -> the graft stack is the only variable.\n"
    f'DATASET_SOURCES = ["{NEW_SOURCE_DS}", "{WHEELS_DS}", "{ENGINE_DS}"]'
)

CELL12_ANCHOR = (
    "# Make one-off changes to `bm`, `bm.games`, or `bm.solver` here before the run starts.\n"
    "# Example:\n"
    "# bm.label = f\"{bm.label}-debug\""
)
CELL12_NEW = f'''{CELL12_ANCHOR}

# GRAFT-EVAL cell 12 — the single entry point the graft stack ships for, called exactly as
# thtennant/arc3-duck-v19 and kevin250304/arc3-duck-v9b-recovery-banking call it (verbatim
# idiom, including the blanket except: install() itself never raises, but a bad IMPORT does).
#
# install() lives in taaf_grafts.composite, NOT taaf_grafts.__init__ (which exports only
# BankingHarnessSolver) — the wrong import path is a silent stock fallback.
#
# FLAGS ON:  {' '.join(FLAGS_ON)}
#   goalkeep(+hudmask, nested under it) -> LEVELS. Stops the carried world model being wiped
#     on game-over/level-change; the authors measured stock carrying a non-empty model on only
#     33 of 481 turns. That is OUR OWN documented root cause ("the agent FORGOT"), already
#     fixed and measured by someone else.
#   shortcircuit -> SCORE. Trims no-op overshoot from repeated-action batches; every no-op
#     otherwise increments the scored action counter and score is
#     min(115,(baseline/actions)**2*100) — quadratic (taaf/game.py:403, verified).
#   efficiency, retry_guard -> the proven public floor riders (report-only analyzer note plus
#     a pass-through chain layer).
#
# FLAGS DELIBERATELY OFF: {' '.join(FLAGS_FORBIDDEN)} — NOT refuted, UNREACHABLE on this rail.
#   banking gates on run.state == "won" (banking_solver.py:180) and this campaign has 0 wins
#   in 470 recorded game-runs; transfer needs clone siblings and this rail has 25 unique
#   game_ids. Arming them would test nothing and would license a meaningless REFUTE.
#
# expected_version pins the graft API: a GRAFTS_API_VERSION bump fails CLOSED (to stock,
# loudly) instead of silently changing what these flag names mean. Kaggle attaches the LATEST
# dataset version and kernel metadata cannot pin one, so this is the only in-band lever.
try:
    from taaf_grafts.composite import install
    install(bm, flags={{{_FLAG_LITERAL}}}, expected_version={GRAFTS_API_VERSION})
except Exception as exc:  # noqa: BLE001 - any graft failure must fall back to stock
    print(f"[taaf_grafts] cell-12 graft failed, running stock: {{type(exc).__name__}}: {{exc}}")'''


def _cell_source(cell: dict) -> str:
    return "".join(cell["source"])


def _set_source(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def _replace_once(text: str, old: str, new: str, where: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"BUILD FAIL: anchor in {where} matched {count} times (want 1): {old[:90]!r}")
    return text.replace(old, new)


def build() -> tuple[Path, Path]:
    nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    if len(cells) != 17:
        raise SystemExit(f"BUILD FAIL: frozen fork has {len(cells)} cells, expected 17 (drift)")
    if "kaggle" in (nb.get("metadata") or {}):
        raise SystemExit("BUILD FAIL: frozen fork gained a metadata.kaggle block (preflight D2)")

    _set_source(cells[2], _replace_once(_cell_source(cells[2]), CELL2_ANCHOR, CELL2_NEW, "cell 2"))
    _set_source(cells[6], _replace_once(_cell_source(cells[6]), CELL6_ANCHOR, CELL6_NEW, "cell 6"))
    _set_source(cells[12], _replace_once(_cell_source(cells[12]), CELL12_ANCHOR, CELL12_NEW, "cell 12"))

    # Every other cell must be byte-identical to the frozen fork.
    pristine = json.loads(SRC_NB.read_text(encoding="utf-8"))["cells"]
    changed = [i for i, (a, b) in enumerate(zip(pristine, cells))
               if _cell_source(a) != _cell_source(b)]
    if changed != [2, 6, 12]:
        raise SystemExit(f"BUILD FAIL: differing cells {changed}, expected [2, 6, 12]")

    # The built artifact must contain the arm's flags and must NOT contain the forbidden ones.
    code = "".join(_cell_source(c) for c in cells if c["cell_type"] == "code")
    for flag in FLAGS_ON:
        if f'"{flag}": True' not in code:
            raise SystemExit(f"BUILD FAIL: flag {flag!r} missing from the built install() call")
    for flag in FLAGS_FORBIDDEN:
        if f'"{flag}": True' in code:
            raise SystemExit(f"BUILD FAIL: FORBIDDEN flag {flag!r} is armed in the built artifact")
    # The stock ref may legitimately be NAMED in cell 2's identity banner ("X REPLACES Y").
    # What must not survive is the ATTACHMENT SITE: cell 6's DATASET_SOURCES assignment. Check
    # that exact executable line, not any mention, or the banner trips its own gate.
    ds_lines = [ln for ln in _cell_source(cells[6]).splitlines()
                if ln.strip().startswith("DATASET_SOURCES = ")]
    if len(ds_lines) != 1:
        raise SystemExit(f"BUILD FAIL: cell 6 has {len(ds_lines)} DATASET_SOURCES assignments (want 1)")
    if OLD_SOURCE_DS in ds_lines[0]:
        raise SystemExit("BUILD FAIL: the stock source-bundle ref survives in cell 6's "
                         "DATASET_SOURCES (attaching both bundles makes BUNDLE_DIR ambiguous)")
    if NEW_SOURCE_DS not in ds_lines[0]:
        raise SystemExit("BUILD FAIL: cell 6's DATASET_SOURCES does not attach the graft fork")
    if not ds_lines[0].strip().startswith(f'DATASET_SOURCES = ["{NEW_SOURCE_DS}"'):
        raise SystemExit("BUILD FAIL: the graft fork must be index 0 — cell 6 maps index 0 to "
                         "BUNDLE_DIR, so any other position silently resolves the wrong bundle")
    if "taaf_grafts.composite import install" not in code:
        raise SystemExit("BUILD FAIL: install() must be imported from taaf_grafts.composite")
    if SERVED_MODEL not in code and ENGINE_DS not in code:
        raise SystemExit("BUILD FAIL: the incumbent engine reference vanished (engine confound)")

    meta = json.loads(SRC_META.read_text(encoding="utf-8"))
    meta["id"] = KERNEL_ID
    meta["title"] = KERNEL_TITLE
    meta["code_file"] = OUT_NB.name
    sources = list(meta["dataset_sources"])
    if OLD_SOURCE_DS not in sources:
        raise SystemExit("BUILD FAIL: frozen-fork metadata no longer attaches the stock bundle")
    meta["dataset_sources"] = [NEW_SOURCE_DS if s == OLD_SOURCE_DS else s for s in sources]
    if OLD_SOURCE_DS in meta["dataset_sources"]:
        raise SystemExit("BUILD FAIL: stock bundle still attached alongside the fork")
    for required in (NEW_SOURCE_DS, WHEELS_DS, ENGINE_DS):
        if required not in meta["dataset_sources"]:
            raise SystemExit(f"BUILD FAIL: {required} not attached")
    if len(meta["dataset_sources"]) != 3:
        raise SystemExit(f"BUILD FAIL: expected 3 dataset_sources, got {meta['dataset_sources']}")

    # Env fields must be byte-identical to the frozen fork (feedback_kaggle_env_match).
    ref = json.loads(SRC_META.read_text(encoding="utf-8"))
    for key in ("enable_gpu", "enable_tpu", "enable_internet", "machine_shape", "docker_image",
                "competition_sources", "kernel_sources", "model_sources", "language",
                "kernel_type", "is_private", "keywords"):
        if meta.get(key) != ref.get(key):
            raise SystemExit(f"BUILD FAIL: env field {key} drifted from the frozen fork")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nb_text = json.dumps(nb, ensure_ascii=False)
    meta_text = json.dumps(meta, indent=2, ensure_ascii=False) + "\n"

    # Idempotence: deterministic-from-pristine-base. Re-running reproduces byte-for-byte.
    for path, text in ((OUT_NB, nb_text), (OUT_META, meta_text)):
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")

    return OUT_NB, OUT_META


if __name__ == "__main__":
    import hashlib

    nb_path, meta_path = build()
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    print(f"built {nb_path}")
    print(f"built {meta_path}")
    print(f"cells={len(nb['cells'])} code_sha256={hashlib.sha256(code.encode()).hexdigest()[:16]}")
    print(f"flags ON        : {' '.join(FLAGS_ON)}")
    print(f"flags FORBIDDEN : {' '.join(FLAGS_FORBIDDEN)} (unreachable on this rail)")
    print(f"bundle          : {OLD_SOURCE_DS} -> {NEW_SOURCE_DS}")
    print(f"engine          : {ENGINE_DS} (UNCHANGED)")
    print("differing cells vs frozen fork: [2, 6, 12]")
