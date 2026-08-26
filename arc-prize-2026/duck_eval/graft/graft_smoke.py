"""Structural smoke for the GRAFT FLOOR arm's built notebook. Runs BEFORE every push.

`feedback_test_before_submit` (v38 scored 0.00 on a missing import) and
`feedback_arc_kernel_structural_drift` (5 ERRORs, all hand-built). This asserts the built
artifact IS the sealed arm and nothing else, on the frozen fork's bytes.

It is a STRUCTURAL smoke, deliberately: the graft modules import `arcengine` and
`inference.framework.solver`, neither of which exists locally, so the install path itself can
only be exercised on Kaggle. That is precisely why the runtime banner assertions in
`graft_score.py` exist — read its docstring before trusting a levels number.

    python duck_eval/graft/graft_smoke.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NB = REPO / "notebooks" / "graft-floor-eval" / "arc3-graft-floor-eval.ipynb"
META = REPO / "notebooks" / "graft-floor-eval" / "kernel-metadata.json"
FROZEN = REPO / "notebooks" / "duckfork" / "tufa-labs-duck-harness-june-30-milestone-winner.ipynb"

FORK_DS = "thtennant/taaf-kaggle-source-share-fork"
STOCK_DS = "jeroencottaar/taaf-kaggle-source-share"
WHEELS_DS = "driessmit1/arc3-vllm-h100-wheelhouse-v3"
ENGINE_DS = "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"
SERVED_MODEL = "vrfai/Qwen3.6-27B-FP8"
FLAGS_ON = ("efficiency", "retry_guard", "shortcircuit", "goalkeep", "hudmask")
FLAGS_FORBIDDEN = ("banking", "transfer", "clickmap")  # clickmap: v21 flag, not this arm (2026-08-20)
EXPECT_DIFF_CELLS = [2, 4, 6, 12, 14]

_checks: list[tuple[bool, str]] = []


def check(ok: bool, label: str) -> None:
    _checks.append((bool(ok), label))


def main() -> int:
    if not NB.exists():
        raise SystemExit("SMOKE FAIL: build first — python duck_eval/graft/build_graft_eval.py")

    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    src = ["".join(c["source"]) for c in cells]
    code = "".join(s for s, c in zip(src, cells) if c["cell_type"] == "code")
    meta = json.loads(META.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))["cells"]
    fsrc = ["".join(c["source"]) for c in frozen]

    # -- 1. structure vs the frozen fork -----------------------------------
    check(len(cells) == 17, "notebook has 17 cells (frozen-fork structure)")
    check("kaggle" not in (nb.get("metadata") or {}), "no metadata.kaggle block (preflight D2)")
    diff = [i for i, (a, b) in enumerate(zip(fsrc, src)) if a != b]
    check(diff == EXPECT_DIFF_CELLS, f"differing cells == {EXPECT_DIFF_CELLS} (got {diff})")
    check(all(cells[i]["cell_type"] == frozen[i]["cell_type"] for i in range(17)),
          "no cell_type drift")

    # -- 2. the one variable: the bundle swap ------------------------------
    ds_lines = [x for x in src[6].splitlines() if x.strip().startswith("DATASET_SOURCES = ")]
    check(len(ds_lines) == 1, "cell 6 has exactly one DATASET_SOURCES assignment")
    line = ds_lines[0] if ds_lines else ""
    check(STOCK_DS not in line, "stock bundle ref ABSENT from cell 6 (BUNDLE_DIR unambiguous)")
    check(line.strip().startswith(f'DATASET_SOURCES = ["{FORK_DS}"'),
          "graft fork is index 0 (cell 6 maps index 0 -> BUNDLE_DIR)")
    check(WHEELS_DS in line and ENGINE_DS in line,
          "wheelhouse + incumbent engine still attached in cell 6 (no engine confound)")

    # -- 3. metadata -------------------------------------------------------
    check(meta["id"] == "canivel/arc3-graft-floor-eval", "kernel id is the fresh graft slug")
    check(set(meta["dataset_sources"]) == {FORK_DS, WHEELS_DS, ENGINE_DS},
          "metadata attaches exactly fork + wheels + engine")
    check(STOCK_DS not in meta["dataset_sources"], "metadata does NOT attach the stock bundle")
    ref = json.loads(FROZEN.parent.joinpath("kernel-metadata.json").read_text(encoding="utf-8"))
    env_keys = ("enable_gpu", "enable_tpu", "enable_internet", "machine_shape", "docker_image",
                "competition_sources", "kernel_sources", "model_sources", "language",
                "kernel_type", "is_private", "keywords")
    drift = [k for k in env_keys if meta.get(k) != ref.get(k)]
    check(not drift, f"env byte-identical to the frozen fork (drift: {drift})")
    check(meta["enable_gpu"] is True and meta["enable_internet"] is False,
          "GPU on, internet off")

    # -- 4. the install call ----------------------------------------------
    check("from taaf_grafts.composite import install" in code,
          "install imported from taaf_grafts.composite (NOT __init__, which lacks it)")
    for f in FLAGS_ON:
        check(f'"{f}": True' in code, f"flag armed: {f}")
    for f in FLAGS_FORBIDDEN:
        check(f'"{f}": True' not in code, f"FORBIDDEN flag NOT armed: {f}")
    check("expected_version=1" in code,
          "expected_version=1 (an API bump fails CLOSED, not silently)")
    check("except Exception as exc" in src[12] and "running stock" in src[12],
          "cell 12 keeps the public guarded idiom (a bad import must not kill the run)")
    # hudmask is nested under goalkeep in composite.py:297 — alone it silently no-ops.
    check(not ('"hudmask": True' in code and '"goalkeep": True' not in code),
          "hudmask is not armed without goalkeep")

    # -- 5. the identity banner (what the log will have to show) -----------
    check("GRAFT-EVAL" in src[2], "cell 2 prints the GRAFT-EVAL identity banner")
    check("HARM<=-0.286" in src[2] and "SIGNAL>=+0.286" in src[2],
          "banner records the sealed decision lines")
    check("m=3 lc 18/19/21" in src[2], "banner records the m=3 baseline")

    # -- 6. the eval rail is untouched ------------------------------------
    # A plain BUILD is the eval: TRUE_SUBMISSION is unset outside a competition rerun, so the
    # run cell plays the 25 bundled environments offline. That branch must be frozen-fork bytes.
    # MOUNTCHECK v2 (2026-08-19): the batch env's competition mount layout changed, so the run
    # cell's ONE hardcoded path is substituted with the cell-2-resolved GRAFT_COMP_ROOT. The
    # check's logic is unchanged -- byte-identity -- but its expected value follows the arm:
    # fork cell 14 with EXACTLY that one line substituted. Any OTHER drift still fails.
    _c14_anchor = '    competition_env_files = str(Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels").parent / "environment_files")'
    _c14_new = '    competition_env_files = str(Path(GRAFT_COMP_ROOT) / "environment_files")  # resolved in cell 2 (MOUNTCHECK v2)'
    check(fsrc[14].count(_c14_anchor) == 1, "fork run cell carries the expected single hardcoded env-files path")
    check(src[14] == fsrc[14].replace(_c14_anchor, _c14_new),
          "run cell (14) is fork bytes + EXACTLY the one resolved-path substitution")
    check(src[10] == fsrc[10], "benchmark-load cell (10) is byte-identical")
    check(src[8] == fsrc[8], "source-import/setup cell (8) is byte-identical (no engine rewrite)")
    check("TRUE_SUBMISSION" in src[2], "TRUE_SUBMISSION branch still present")

    # -- 7. the sealed scorer agrees with this artifact --------------------
    sys.path.insert(0, str(Path(__file__).parent))
    import graft_score as gs
    check(tuple(gs.FLAGS_ON) == FLAGS_ON, "scorer FLAGS_ON matches the built arm")
    check(tuple(gs.FLAGS_FORBIDDEN) == FLAGS_FORBIDDEN, "scorer FLAGS_FORBIDDEN matches")
    check(gs.SERVED_MODEL == SERVED_MODEL, "scorer expects the incumbent served model")
    check(abs(gs.HARM_LINE + gs.C_M3 * gs.SIGMA) < 1e-12, "scorer HARM line is derived, not typed")
    check(gs.N_GAMES == 25, "scorer expects 25 games")

    n_pass = sum(1 for ok, _ in _checks if ok)
    n_fail = len(_checks) - n_pass
    for ok, label in _checks:
        if not ok:
            print(f"  [FAIL] {label}")
    print(f"graft smoke: {n_pass}/{len(_checks)} checks passed, {n_fail} failed")
    if n_fail:
        print("SMOKE FAILED — do not push.")
        return 1
    print(f"arm = fork REPLACES stock | flags {'+'.join(FLAGS_ON)} | "
          f"FORBIDDEN {'+'.join(FLAGS_FORBIDDEN)} | engine UNCHANGED")
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
