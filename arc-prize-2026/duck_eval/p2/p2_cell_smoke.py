#!/usr/bin/env python3
"""P2 PATCH-CELL smoke -- execute the notebook's cell 6 for real, off-Kaggle.

WHY. `feedback_test_before_submit.md`: v38 scored 0.00 on Kaggle because of a
missing import that no local test ever executed. The two P2 smokes exercise
`p2_patch.apply_patch`, and local_gate statically parses the notebook -- but
NOTHING had ever RUN the patch cell end to end: the base64 decode, the bundle
discovery walk, the boot check, the module write + import, the anchor
application, and the sys.path shadow. Those are exactly the steps that fail
silently at 03:00 on a rented GPU.

So this pulls the REAL cell source out of the REAL built notebook, rewrites only
the two absolute Kaggle roots to local paths, and executes it.

Run:  uv run python duck_eval/p2/p2_cell_smoke.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NB = REPO / "notebooks" / "p2-retry-eval" / "arc3-p2-retry-eval.ipynb"
BUNDLE_PARENT = REPO / "runs" / "harness_diff_0813" / "ds"
PATCH_CELL_INDEX = 6

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(("  PASS  " if ok else "  FAIL  ") + name + (("  -- " + detail) if detail else ""))


def cell_source() -> str:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cell = nb["cells"][PATCH_CELL_INDEX]
    src = "".join(cell["source"])
    assert "P2 RESET-ANCHORED EPISODIC RETRY" in src, "cell 6 is not the P2 patch cell"
    return src


def run_cell(src: str, working: Path, inputs: Path, globals_extra=None):
    """Execute the cell with only the two Kaggle roots rewritten."""
    localised = (src
                 .replace('Path("/kaggle/input")', "Path(%r)" % str(inputs))
                 .replace('"/kaggle/working/', '"%s/' % str(working).replace("\\", "/")))
    g = {"Path": Path, "os": __import__("os"), "__name__": "__main__"}
    if globals_extra:
        g.update(globals_extra)
    exec(compile(localised, "<p2-cell-6>", "exec"), g)  # noqa: S102
    return g


def main() -> int:
    print("P2 PATCH-CELL SMOKE (executes notebook cell %d for real)" % PATCH_CELL_INDEX)
    print("=" * 70)

    if not NB.is_file():
        print("FATAL: build the notebook first (duck_eval/p2/build_p2_eval.py)")
        return 2

    src = cell_source()
    check("C0 cell 6 extracted from the built notebook", bool(src), "%d chars" % len(src))

    # The cell must not be able to run after `inference` is already imported --
    # that assert is the whole reason the cell sits at position 6.
    saved = {k: v for k, v in sys.modules.items() if k == "inference" or k.startswith("inference.")}
    for k in list(saved):
        del sys.modules[k]

    tmp = Path(tempfile.mkdtemp(prefix="p2cell-"))
    try:
        working = tmp / "working"
        working.mkdir()

        print("\n-- C1: the cell RUNS against a real bundle layout --")
        g = run_cell(src, working, BUNDLE_PARENT)
        check("C1a cell executed without raising", True)
        check("C1b bundle discovered", g.get("_p2_src_root") is not None,
              str(g.get("_p2_src_root")))
        check("C1c boot check passed", g.get("_p2_reset_ok") is True)
        info = g.get("_p2_info") or {}
        check("C1d 6 anchors applied", info.get("anchors_applied") == 6, str(info))
        check("C1e sandbox is the vehicle generation",
              bool(info.get("sandbox_is_vehicle_generation")), str(info.get("sandbox_md5_before")))
        dst = Path(g["_p2_dst"])
        check("C1f patched tree materialised", (dst / "inference" / "agent" / "tool_agent.py").is_file())
        check("C1g embedded module written to disk",
              (working / "p2_patch_embedded.py").is_file())

        print("\n-- C2: the patched module is the one that gets IMPORTED --")
        chk = g.get("_p2_chk")
        check("C2a inference.agent.tool_agent resolved from the patched tree",
              chk is not None and str(dst) in str(Path(chk.__file__)),
              str(getattr(chk, "__file__", None)))
        ta_src = (dst / "inference" / "agent" / "tool_agent.py").read_text(encoding="utf-8")
        for m in ("_p2_note_acting_turn", "_p2_retry_armed", "_p2_count_attempt_calls",
                  "_p2_flush", "_p2_game_key"):
            check("C2b %s present in the patched file" % m, m in ta_src)
        sb_src = (dst / "inference" / "agent" / "python_tool_sandbox.py").read_text(encoding="utf-8")
        check("C2c attempt() exported in the sandbox",
              'runtime_globals["attempt"] = attempt' in sb_src)

        print("\n-- C3: the class really carries the methods (not just the text) --")
        classes = [getattr(chk, n) for n in dir(chk) if isinstance(getattr(chk, n), type)]
        bound = [c for c in classes if "_p2_retry_armed" in dir(c)]
        check("C3a a class on the imported module has the trigger bound",
              len(bound) >= 1, "classes carrying it: %s" % [c.__name__ for c in bound])

        print("\n-- C4: NEGATIVE CONTROL -- the cell REFUSES a late run --")
        # If `inference` is already imported, the shadow cannot take effect and the
        # run would be silently STOCK. That must be a loud death, not a warning.
        tmp2 = tmp / "w2"
        tmp2.mkdir()
        import types
        sys.modules["inference"] = types.ModuleType("inference")
        try:
            run_cell(src, tmp2, BUNDLE_PARENT)
            check("C4a refuses to run after `inference` is imported", False,
                  "the cell RAN -- a stock run could ship silently")
        except AssertionError as exc:
            check("C4a refuses to run after `inference` is imported",
                  "P2 FATAL" in str(exc), str(exc)[:70])
        finally:
            del sys.modules["inference"]

        print("\n-- C5: NEGATIVE CONTROL -- a missing bundle dies LOUDLY --")
        empty = tmp / "no-bundle"
        empty.mkdir()
        tmp3 = tmp / "w3"
        tmp3.mkdir()
        try:
            run_cell(src, tmp3, empty)
            check("C5a missing bundle dies loudly", False, "the cell RAN with no bundle")
        except AssertionError as exc:
            check("C5a missing bundle dies loudly", "P2 FATAL" in str(exc), str(exc)[:70])

        print("\n-- C6: NEGATIVE CONTROL -- a broken RESET invariant dies LOUDLY --")
        # attempt() is only sound while RESET is always-legal. Break that in a
        # copy of the bundle and the boot check must refuse to arm the arm.
        fake_in = tmp / "fake-input"
        fake_bundle = fake_in / "jakobbrggen_taaf-kaggle-source-anim-20260807-anim"
        shutil.copytree(BUNDLE_PARENT / "jakobbrggen_taaf-kaggle-source-anim-20260807-anim",
                        fake_bundle)
        gp = fake_bundle / "src" / "tufa-arc-agi-framework" / "src" / "taaf" / "game.py"
        gp.write_text(gp.read_text(encoding="utf-8").replace("RESET (0) always present",
                                                             "RESET is sometimes present"),
                      encoding="utf-8")
        tmp4 = tmp / "w4"
        tmp4.mkdir()
        try:
            run_cell(src, tmp4, fake_in)
            check("C6a broken RESET invariant dies loudly", False,
                  "the cell ARMED on a bundle where attempt() is unsound")
        except AssertionError as exc:
            check("C6a broken RESET invariant dies loudly",
                  "RESET-always-legal" in str(exc), str(exc)[:80])

        print("\n-- C7: NEGATIVE CONTROL -- tampered embedded module dies LOUDLY --")
        tampered = src.replace("_P2_SHA = '", "_P2_SHA = 'ff", 1)
        tmp5 = tmp / "w5"
        tmp5.mkdir()
        try:
            run_cell(tampered, tmp5, BUNDLE_PARENT)
            check("C7a sha mismatch dies loudly", False, "the cell ran with a wrong sha")
        except AssertionError as exc:
            check("C7a sha mismatch dies loudly", "P2 FATAL" in str(exc), str(exc)[:70])

    finally:
        for k in [k for k in sys.modules if k == "inference" or k.startswith("inference.")]:
            del sys.modules[k]
        sys.modules.update(saved)
        sys.modules.pop("p2_patch_embedded", None)
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 70)
    print("PASS %d  FAIL %d" % (len(PASS), len(FAIL)))
    for f in FAIL:
        print("  FAILED: " + f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
