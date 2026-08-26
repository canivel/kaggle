"""A17 canary v6 — FULL-WINDOW 72B throughput bench on the DATASET-weights route (2026-07-28 slot-1, staged 2026-07-27).

Why v6 exists: v5 (boot canary, window 1500 s) is the runtime test of the
dataset-weights route (canivel/qwen25-vl-72b-awq). Contingent on its PASS,
the queued next push (ITERATION_LOG 2026-07-27; build memo
learnings/war_room/a17_v5_dataset_route_2026-07-27.md "Contingent next push")
is v6 = v5 with the window restored to the sealed bench config. v6 re-runs
the 07-26 v4 pre-registration (learnings/war_room/a17_v4_prereg_2026-07-26.md):
G1 recovery >= 0.95, G2 >= 100 executed actions, G3 cadence measurement,
G4 = NO capability interpretation — and delivers the rho_action denominator
(scope v2 sec3: 480 / Sigma N_72B). MEASUREMENT ONLY: no GO/NO-GO reading in
code or at k=1 (learnings/a17_error_model.md); interpretation belongs to the
sealed walk + Sunday panel.

v6 = the staged v5 notebook with exactly the window/soft_end/banner strings
changed back per the pre-registered spec:

  1. cell 2: seed-line comment and banner. Banner mode becomes
     mode=throughput-canary-v6-dataset-weights; the window clause becomes
     7920s FULL BENCH (prereg G1-G4 + rho_action denominator); the
     dataset-route provenance and MEASUREMENT ONLY notice are kept verbatim.
     The seed-line comment is restored to the original (pre-v5) string
     preserved verbatim in build_v5_boot_canary.py.
  2. cell 14: `A17_WINDOW_S = 1500.0` (+ v5 comment) -> the original
     `A17_WINDOW_S = 7920.0`, restored VERBATIM from build_v5_boot_canary.py.
  3. cell 14: the v5 now()-anchored soft_end block -> the original
     budget-derived NOTEBOOK_START_EPOCH rule, restored VERBATIM from
     build_v5_boot_canary.py.

NO metadata change: the v5 metadata (model_sources REMOVED, weights dataset
canivel/qwen25-vl-72b-awq attached) is exactly what v6 needs. This script
asserts that state and does not write META_PATH. Model pin and serve config
(cell 8) are UNTOUCHED — the harness is MULTIMODAL (Qwen2.5-VL-72B-AWQ).

Every rewrite anchor must match EXACTLY once or this script raises
(same discipline as build_v5_boot_canary.py). Idempotence: running twice
raises (the v5 anchors are gone after the first run).
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NB_PATH = REPO / "notebooks" / "a17-canary" / "arc3-a17-72b-canary.ipynb"
META_PATH = REPO / "notebooks" / "a17-canary" / "kernel-metadata.json"
WEIGHTS_DATASET = "canivel/qwen25-vl-72b-awq"

CELL2_REWRITES: list[tuple[str, str]] = [
    (
        'os.environ["A17_CANARY_SEED"] = "1"  # A17 72B-VL v5 BOOT CANARY: '
        "dataset-route serve validation, SHORT 1500s window (NOT the sealed bench)",
        # Original pre-v5 comment, restored verbatim (build_v5_boot_canary.py
        # CELL2_REWRITES[0][0]) — it describes exactly what v6 is.
        'os.environ["A17_CANARY_SEED"] = "1"  # A17 72B-VL canary (scope v2 sec3 '
        "rho_action denominator; sec5 full 7920s window)",
    ),
    (
        'print("A17-CANARY seed=1 mode=boot-canary-v5-dataset-weights '
        "games=ft09-0d8bbf25,sb26-7fbdac44,lp85-305b61c3,vc33-5430563c "
        "composition=W0+fencedrec; weights=DATASET canivel/qwen25-vl-72b-awq "
        "(model_sources route DEAD, silently dropped at push 07-25/07-26); "
        "window=1500s BOOT TEST, output MEASUREMENT ONLY (a17_error_model.md: "
        'no NO-GO reading at k=1)")',
        'print("A17-CANARY seed=1 mode=throughput-canary-v6-dataset-weights '
        "games=ft09-0d8bbf25,sb26-7fbdac44,lp85-305b61c3,vc33-5430563c "
        "composition=W0+fencedrec; weights=DATASET canivel/qwen25-vl-72b-awq "
        "(model_sources route DEAD, silently dropped at push 07-25/07-26); "
        "window=7920s FULL BENCH (07-26 v4 prereg G1-G4 + rho_action denominator "
        "480/Sigma N_72B), output MEASUREMENT ONLY (a17_error_model.md: "
        'no NO-GO reading at k=1)")',
    ),
]

CELL14_REWRITES: list[tuple[str, str]] = [
    (
        "    A17_WINDOW_S = 1500.0  # v5 BOOT CANARY short window "
        "(dataset-route serve test; sealed bench stays 7920s)",
        # Original sealed-bench window, restored verbatim
        # (build_v5_boot_canary.py CELL14_REWRITES[0][0]).
        "    A17_WINDOW_S = 7920.0",
    ),
    (
        "    if not TRUE_SUBMISSION:\n"
        "        # v5 BOOT CANARY: fixed short window anchored at bm.run entry, so the\n"
        "        # 72B load time (before this cell) is excluded from the game window.\n"
        "        soft_end = min(soft_end, datetime.now() + timedelta(seconds=A17_WINDOW_S))\n",
        # Original budget-derived soft_end rule, restored verbatim
        # (build_v5_boot_canary.py CELL14_REWRITES[1][0]).
        "    if not TRUE_SUBMISSION:\n"
        '        budget = float(getattr(target, "max_runtime_s", 0.0) or 0.0)\n'
        "        if budget > 0:\n"
        "            soft_end = min(soft_end, datetime.fromtimestamp(NOTEBOOK_START_EPOCH)"
        " + timedelta(seconds=budget - min(600.0, budget / 2)))\n",
    ),
]


def _apply(text: str, rewrites: list[tuple[str, str]], label: str) -> str:
    for old, new in rewrites:
        found = text.count(old)
        if found != 1:
            raise SystemExit(
                f"FATAL {label}: anchor matched {found} times (want 1): {old[:90]!r}"
            )
        text = text.replace(old, new)
    return text


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    # Preconditions: this must be the v5 (dataset-weights boot canary) notebook.
    cell2 = "".join(nb["cells"][2]["source"])
    if "mode=throughput-canary-v6-dataset-weights" in cell2:
        raise SystemExit("FATAL: notebook is already v6 (idempotence guard)")
    if "mode=boot-canary-v5-dataset-weights" not in cell2:
        raise SystemExit("FATAL: cell 2 lacks the v5 banner — not the v5 notebook")
    cell12 = "".join(nb["cells"][12]["source"])
    if "fenced_recovery_patch" not in cell12:
        raise SystemExit("FATAL: cell 12 lacks the fenced-recovery graft — wrong composition")

    for idx, rewrites in ((2, CELL2_REWRITES), (14, CELL14_REWRITES)):
        cell = nb["cells"][idx]
        src = _apply("".join(cell["source"]), rewrites, f"cell {idx}")
        if idx == 14:
            if "A17_WINDOW_S = 1500.0" in src:
                raise SystemExit("FATAL: 1500.0 window survived the cell-14 rewrite")
            if "datetime.now() + timedelta(seconds=A17_WINDOW_S)" in src:
                raise SystemExit("FATAL: v5 now()-anchored soft_end survived the rewrite")
        # Compile check (top-level await is legal in the notebook runtime).
        compile(src, f"cell{idx}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        cell["source"] = src.splitlines(keepends=True)
        cell["outputs"] = []
        cell["execution_count"] = None

    # Metadata: ASSERT the v5 dataset-weights state; v6 must NOT touch it.
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    if "model_sources" in meta:
        raise SystemExit("FATAL: metadata still has model_sources — not the v5 metadata")
    if WEIGHTS_DATASET not in meta.get("dataset_sources", []):
        raise SystemExit(f"FATAL: weights dataset {WEIGHTS_DATASET} not attached")

    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"v6 full-window bench written: {NB_PATH}")
    print(f"  metadata UNTOUCHED (asserted: no model_sources; {WEIGHTS_DATASET} attached)")
    print(f"  cell 2: {len(CELL2_REWRITES)} rewrites; cell 14: {len(CELL14_REWRITES)} rewrites")


if __name__ == "__main__":
    sys.exit(main())
