"""LIVE sentinel submission-kernel smoke -- CPU only, no GPU/LLM.

Runtime-tests the ASSEMBLED live notebook notebooks/ducksentinel/
arc3-duck-sentinel.ipynb (feedback_test_before_submit: always runtime-test the
exact artifact before a kernel push). The live arm (exploration draw #1,
amendment 2026-07-23 A21/C4) is: vanilla duck + (f) continuation
(hygiene-default) + budget sentinel @ SENTINEL_BUDGET=150 ONLY -- NO warpack,
NO ledger.

  S*  structural: 17 cells; cell 2 = vanilla + fast-submit gate + LIVE budget
      stamp (SENTINEL_BUDGET=150; NOT an eval build -- no forced offline
      bench); heavy gates on 4/6/8/10; cell 12 = sentinel graft + continuation
      graft, NO warpack/ledger; cell 14 keeps the fast-submit path;
      kernel-metadata matches the duckwar submission metadata except
      id/title/code_file; ASCII-only modified cells.
  I1  exec cell 2 with WARPACK_FORCE_OFFLINE_BENCH=1: RUN_HEAVY True,
      SENTINEL_BUDGET=150 exported, both live banners printed.
  I1f exec cell 2 with the gate cold (no force, no rerun signal): RUN_HEAVY
      False, then cell 14 fast path writes the dummy submission.parquet.
  I2  exec cell 12 (real source) against the runtime module copies
      (--warkit <dir>, default duck_eval/warpack/_kaggle_dataset): sentinel v2
      ACTIVE banner + continuation applied banner, bm.label carries BOTH
      markers, no warpack/ledger imported.
  K*  kill switches (subprocess): SENTINEL_DISABLE=1 + CONTINUATION_DISABLE=1
      -> harness + prompt left unpatched, no graft banners.

Run:  uv run python duck_eval/sentinel/sentinel_live_smoke.py [--warkit <dir>]
"""
from __future__ import annotations

import ast
import asyncio
import contextlib
import inspect
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUNDLE = REPO / "duck_eval" / "taaf_bundle" / "src"
NB_PATH = REPO / "notebooks" / "ducksentinel" / "arc3-duck-sentinel.ipynb"
META_PATH = REPO / "notebooks" / "ducksentinel" / "kernel-metadata.json"
WAR_META_PATH = REPO / "notebooks" / "duckwar" / "kernel-metadata.json"
WARKIT_DEFAULT = REPO / "duck_eval" / "warpack" / "_kaggle_dataset"

sys.path.insert(0, str(BUNDLE / "ARC3-Inference"))
sys.path.insert(0, str(BUNDLE / "tufa-arc-agi-framework" / "src"))

os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "sentinel-live-smoke")
os.environ.setdefault("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:9/v1")

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}" + (f"  [{detail[:90]}]" if detail else ""))
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail[:300]}")


def cell_src(nb: dict, i: int) -> str:
    return "".join(nb["cells"][i]["source"])


def exec_cell(src: str, ns: dict) -> str:
    buf = io.StringIO()
    code = compile(src, "<cell>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    with contextlib.redirect_stdout(buf):
        if code.co_flags & inspect.CO_COROUTINE:
            asyncio.run(eval(code, ns))  # noqa: S307 - our own notebook source
        else:
            exec(code, ns)  # noqa: S102 - our own notebook source
    out = buf.getvalue()
    if out.strip():
        for line in out.splitlines():
            print(f"    | {line}")
    return out


def structural(nb: dict) -> None:
    print("S: structural checks on the assembled LIVE notebook")
    check("S1 17 cells", len(nb["cells"]) == 17, str(len(nb["cells"])))
    c2 = cell_src(nb, 2)
    check("S2 cell 2 is NOT an eval build (no forced offline bench line)",
          not c2.startswith('import os; os.environ["WARPACK_FORCE_OFFLINE_BENCH"]'))
    check("S2b cell 2 carries the fast-submit gate",
          "RUN_HEAVY = TRUE_SUBMISSION or FORCE_OFFLINE_BENCH" in c2)
    check("S2c cell 2 exports SENTINEL_BUDGET=150 + live banners (C7-as-amended; "
          "scored regime uncapped -> without the export the sentinel is inert)",
          'os.environ["SENTINEL_BUDGET"] = "150"' in c2
          and "sentinel-live: SENTINEL_BUDGET=150" in c2
          and "NO warpack, NO ledger" in c2)
    gated = all(cell_src(nb, i).lstrip("#").strip().startswith("Warpack fast-submit gate")
                and "if RUN_HEAVY:" in cell_src(nb, i) for i in (4, 6, 8, 10))
    check("S3 heavy cells 4/6/8/10 gated on RUN_HEAVY", gated)
    c12 = cell_src(nb, 12)
    check("S4 cell 12 carries the sentinel graft",
          "import budget_sentinel_patch" in c12
          and "budget_sentinel_patch.apply(bm)" in c12)
    check("S4b cell 12 carries the (f) continuation graft (hygiene default)",
          "import continuation_patch" in c12 and "continuation_patch.apply()" in c12)
    check("S4c cell 12 does NOT ship warpack or ledger (one-flag discipline)",
          "import warpack_patch" not in c12 and "import ledger" not in c12
          and "warpack_patch.apply" not in c12)
    c14 = cell_src(nb, 14)
    check("S5 cell 14 keeps the fast-submit path", "_write_dummy_submission" in c14)
    check("S5b no non-ASCII in cells 2/12 (round-trip safe)",
          not any(ord(ch) > 127 for ch in c2 + c12))
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    base = json.loads(WAR_META_PATH.read_text(encoding="utf-8"))
    delta = {k for k in set(meta) | set(base) if meta.get(k) != base.get(k)}
    check("S6 kernel-metadata matches duckwar submission metadata except "
          "id/title/code_file",
          delta == {"id", "title", "code_file"}
          and meta["id"] == "canivel/arc3-duck-sentinel",
          f"delta={sorted(delta)}")


def run_integration(warkit: Path) -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    structural(nb)

    for mod in ("budget_sentinel_patch.py", "continuation_patch.py"):
        assert (warkit / mod).is_file(), f"missing {mod} under {warkit}"
    print(f"module source under test: warkit={warkit}")

    tmp_root = Path(tempfile.mkdtemp(prefix="senlive-"))
    run_dir = tmp_root / "run"
    (run_dir / "sentinel").mkdir(parents=True)
    (run_dir / "continuation").mkdir()
    shutil.copy(warkit / "budget_sentinel_patch.py",
                run_dir / "sentinel" / "budget_sentinel_patch.py")
    shutil.copy(warkit / "continuation_patch.py",
                run_dir / "continuation" / "continuation_patch.py")

    cwd0 = Path.cwd()
    os.chdir(run_dir)
    try:
        print("I1: exec cell 2 with the offline-bench force (heavy path)")
        os.environ["WARPACK_FORCE_OFFLINE_BENCH"] = "1"
        ns: dict = {}
        out2 = exec_cell(cell_src(nb, 2), ns)
        check("I1 RUN_HEAVY True under force", ns.get("RUN_HEAVY") is True)
        check("I1b SENTINEL_BUDGET=150 exported + live banners printed",
              os.environ.get("SENTINEL_BUDGET") == "150"
              and "sentinel-live: SENTINEL_BUDGET=150" in out2
              and "NO warpack, NO ledger" in out2)

        print("I1f: gate cold (no force/rerun) -> fast-submit dummy parquet")
        os.environ.pop("WARPACK_FORCE_OFFLINE_BENCH", None)
        os.environ.pop("KAGGLE_IS_COMPETITION_RERUN", None)
        ns_cold: dict = {}
        exec_cell(cell_src(nb, 2), ns_cold)
        check("I1f RUN_HEAVY False when gate cold", ns_cold.get("RUN_HEAVY") is False)
        work = run_dir / "working"
        work.mkdir(exist_ok=True)
        ns_cold["WORKING_DIR"] = work
        out14 = exec_cell(cell_src(nb, 14), ns_cold)
        check("I1g dummy submission.parquet written on the fast path",
              (work / "submission.parquet").is_file()
              and "FAST-SUBMIT" in out14)

        print("I2: exec cell 12 (sentinel + continuation grafts) heavy path")
        os.environ["WARPACK_FORCE_OFFLINE_BENCH"] = "1"
        ns2: dict = {}
        exec_cell(cell_src(nb, 2), ns2)
        ns2["bm"] = SimpleNamespace(label="sentinel-live-smoke")
        out12 = exec_cell(cell_src(nb, 12), ns2)
        check("I2 sentinel v2 ACTIVE banner",
              "sentinel v2: budget sentinel ACTIVE" in out12
              and "unit=game-envelope" in out12)
        check("I2b continuation graft applied",
              "game-over-continuation graft applied" in out12)
        check("I2c bm.label carries BOTH markers",
              "-sentinel-v2" in ns2["bm"].label
              and "-continuation-" in ns2["bm"].label, ns2["bm"].label)
        check("I2d no fallback-to-vanilla traceback", "PATCH FAILED" not in out12)
        check("I2e no warpack/ledger imported",
              "warpack_patch" not in sys.modules and "ledger_patch" not in sys.modules
              and "ledger_core" not in sys.modules)
    finally:
        os.chdir(cwd0)
        shutil.rmtree(tmp_root, ignore_errors=True)


def run_killswitch(warkit: Path) -> None:
    """Subprocess arm: SENTINEL_DISABLE=1 + CONTINUATION_DISABLE=1 -> no-ops."""
    import inference.framework.solver as solver_mod
    from inference.agent.tool_agent import ToolAgent
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    tmp_root = Path(tempfile.mkdtemp(prefix="senlivekill-"))
    run_dir = tmp_root / "run"
    (run_dir / "sentinel").mkdir(parents=True)
    (run_dir / "continuation").mkdir()
    shutil.copy(warkit / "budget_sentinel_patch.py",
                run_dir / "sentinel" / "budget_sentinel_patch.py")
    shutil.copy(warkit / "continuation_patch.py",
                run_dir / "continuation" / "continuation_patch.py")
    cwd0 = Path.cwd()
    os.chdir(run_dir)
    try:
        os.environ["WARPACK_FORCE_OFFLINE_BENCH"] = "1"
        ns: dict = {}
        exec_cell(cell_src(nb, 2), ns)
        pre_exec = solver_mod._HarnessGameSession._execute_action
        pre_prompt = ToolAgent._build_user_prompt
        ns["bm"] = SimpleNamespace(label="sentinel-live-kill")
        out12 = exec_cell(cell_src(nb, 12), ns)
        check("K1 sentinel kill switch: _execute_action unpatched",
              solver_mod._HarnessGameSession._execute_action is pre_exec)
        check("K1b sentinel kill switch: _build_user_prompt unpatched",
              ToolAgent._build_user_prompt is pre_prompt)
        check("K2 no ACTIVE banner under kill switches",
              "budget sentinel ACTIVE" not in out12)
    finally:
        os.chdir(cwd0)
        shutil.rmtree(tmp_root, ignore_errors=True)


def main() -> int:
    global PASS, FAIL
    args = sys.argv[1:]
    warkit = WARKIT_DEFAULT
    if "--warkit" in args:
        warkit = Path(args[args.index("--warkit") + 1]).resolve()
    warkit = warkit.resolve()

    if "--killswitch-child" in args:
        run_killswitch(warkit)
        print(f"\nKILLSWITCH RESULT: {PASS} passed, {FAIL} failed")
        return 1 if FAIL else 0

    print(f"LIVE sentinel submission-kernel smoke | nb={NB_PATH}")
    run_integration(warkit)

    print("K: kill-switch arm (subprocess, SENTINEL_DISABLE=1 + CONTINUATION_DISABLE=1)")
    env = dict(os.environ)
    env["SENTINEL_DISABLE"] = "1"
    env["CONTINUATION_DISABLE"] = "1"
    child = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--killswitch-child",
         "--warkit", str(warkit)],
        capture_output=True, text=True, env=env, timeout=600)
    for line in child.stdout.splitlines():
        if line.strip().startswith(("PASS", "FAIL", "KILLSWITCH")):
            print(f"  {line.strip()}")
    kill_pass = child.stdout.count("  PASS")
    kill_fail = child.stdout.count("  FAIL") + (1 if child.returncode not in (0, 1) else 0)
    if child.returncode not in (0, 1):
        print(child.stdout[-2000:])
        print(child.stderr[-2000:])
    PASS += kill_pass
    FAIL += kill_fail

    print(f"\nRESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
