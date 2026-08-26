"""W0 (f) game-over-continuation eval integration smoke -- CPU only, no GPU/LLM.

Runtime-tests the ASSEMBLED notebook notebooks/duckw0-eval/
arc3-duck-w0-continuation-eval.ipynb end-to-end (feedback_test_before_submit:
v38 scored 0.00 from a missing import -- always runtime-test the exact
artifact). W0 is STANDALONE: duck baseline + (f) only, NO warpack, NO ledger.

  S1-S6  structural: 17 cells, eval+W0-seed prefix on cell 2, heavy gates on
         cells 4/6/8/10, continuation graft (NOT warpack/ledger) in cell 12,
         cell 14 keeps the fast-submit path, kernel-metadata byte-parity with
         the war-eval kernel (id/title/code_file excepted).
  I1     exec cell 2 (real source): RUN_HEAVY forced by the eval line,
         W0_EVAL_SEED=1 stamped, W0 seed banner printed.
  I2     exec cell 12 (real source) against the module copy the kernel loads
         at runtime (--warkit <dir> = downloaded canivel/arc-war-kit copy;
         defaults to the repo dataset copy): continuation applied, banner
         "continuation v1: ... ACTIVE (2 modules patched)" printed, bm.label
         carries the continuation marker, NO warpack/ledger imported.
  I3     the REAL bundled _build_system_prompt reflects the patch: the OLD
         "stop acting immediately" phrase is gone and both NEW continuation
         lines are present (this is the (f) observable in the prompt).
  K*     kill switch (subprocess): CONTINUATION_DISABLE=1 -> graft no-op and
         the stock prompt is preserved (vanilla).

Run:  uv run python duck_eval/warpack/w0_eval_smoke.py [--warkit <dir>]
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
NB_PATH = REPO / "notebooks" / "duckw0-eval" / "arc3-duck-w0-continuation-eval.ipynb"
META_PATH = REPO / "notebooks" / "duckw0-eval" / "kernel-metadata.json"
BASE_META_PATH = REPO / "notebooks" / "duckwar-eval" / "kernel-metadata.json"
WARKIT_DEFAULT = REPO / "duck_eval" / "warpack" / "_kaggle_dataset"

sys.path.insert(0, str(BUNDLE / "ARC3-Inference"))

os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "w0-eval-smoke")
os.environ.setdefault("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:9/v1")

PASS = 0
FAIL = 0

OLD_PHRASE = "stop acting immediately"


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


def _prompt() -> str:
    import inference.agent.tool_agent as tool_agent
    return tool_agent._build_system_prompt(tool_output_tokens=1024)


def structural(nb: dict) -> None:
    print("S: structural checks on the assembled notebook")
    check("S1 17 cells", len(nb["cells"]) == 17, str(len(nb["cells"])))
    c2 = cell_src(nb, 2)
    check("S2 cell 2 forces the offline bench (eval line first)",
          c2.startswith('import os; os.environ["WARPACK_FORCE_OFFLINE_BENCH"] = "1"'))
    check("S2b cell 2 stamps W0_EVAL_SEED=1 + W0 seed banner",
          'os.environ["W0_EVAL_SEED"] = "1"' in c2
          and "pairs with arc3-duck-war-eval seed 1" in c2)
    gated = all(cell_src(nb, i).lstrip("#").strip().startswith("Warpack fast-submit gate")
                and "if RUN_HEAVY:" in cell_src(nb, i) for i in (4, 6, 8, 10))
    check("S3 heavy cells 4/6/8/10 gated on RUN_HEAVY", gated)
    c12 = cell_src(nb, 12)
    check("S4 cell 12 is the continuation graft (imports continuation_patch)",
          "import continuation_patch" in c12 and "continuation_patch.apply()" in c12
          and "if not RUN_HEAVY:" in c12)
    check("S4b cell 12 does NOT ship warpack or ledger (W0 standalone)",
          "import warpack_patch" not in c12 and "import ledger" not in c12
          and "warpack_patch.apply" not in c12)
    c14 = cell_src(nb, 14)
    check("S5 cell 14 keeps the fast-submit path", "_write_dummy_submission" in c14)
    check("S5b no non-ASCII in cells 2/12 (round-trip safe)",
          not any(ord(ch) > 127 for ch in c2 + c12))
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    base = json.loads(BASE_META_PATH.read_text(encoding="utf-8"))
    delta = {k for k in set(meta) | set(base) if meta.get(k) != base.get(k)}
    check("S6 kernel-metadata matches war-eval except id/title/code_file",
          delta == {"id", "title", "code_file"}
          and meta["id"] == "canivel/arc3-duck-w0-continuation-eval",
          f"delta={sorted(delta)}")


def run_integration(warkit: Path) -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    structural(nb)

    src_cont = warkit / "continuation_patch.py"
    assert src_cont.is_file(), f"no continuation_patch.py under {warkit}"
    print(f"module source under test: continuation={src_cont}")

    tmp_root = Path(tempfile.mkdtemp(prefix="w0smoke-"))
    run_dir = tmp_root / "run"
    # Mirror the kernel layout: continuation_patch.py discoverable next to cwd
    # (the graft also searches /kaggle/input; here the ./ probe path is used).
    run_dir.mkdir(parents=True)
    shutil.copy(src_cont, run_dir / "continuation_patch.py")

    cwd0 = Path.cwd()
    os.chdir(run_dir)
    try:
        ns: dict = {}
        print("I1: exec cell 2 (eval gate + W0 seed stamp)")
        out2 = exec_cell(cell_src(nb, 2), ns)
        check("I1 RUN_HEAVY forced True by the eval line",
              ns.get("RUN_HEAVY") is True and ns.get("FORCE_OFFLINE_BENCH") is True)
        check("I1b W0_EVAL_SEED=1 stamped + banner printed",
              os.environ.get("W0_EVAL_SEED") == "1"
              and "w0-continuation-eval: SEED=1" in out2)

        print("I3-pre: capture the stock prompt BEFORE the graft")
        pre = _prompt()
        check("I3-pre stock prompt carries the OLD game-over phrase",
              OLD_PHRASE in pre, "OLD phrase absent before graft")

        print("I2: exec cell 12 (continuation graft) against the runtime module copy")
        ns["bm"] = SimpleNamespace(label="w0-eval-smoke")
        out12 = exec_cell(cell_src(nb, 12), ns)
        check("I2 continuation graft applied (banner printed)",
              "continuation v1: game-over-continuation ACTIVE (2 modules patched)" in out12
              and "graft applied" in out12)
        check("I2b bm.label carries the continuation marker",
              "-continuation-v1" in ns["bm"].label, ns["bm"].label)
        check("I2c NO fallback-to-vanilla traceback",
              "PATCH FAILED" not in out12)
        check("I2d graft did NOT import warpack/ledger",
              "warpack_patch" not in sys.modules and "ledger_patch" not in sys.modules
              and "ledger_core" not in sys.modules)

        print("I3: the REAL bundled _build_system_prompt now reflects (f)")
        post = _prompt()
        check("I3 patched prompt drops the OLD 'stop acting immediately' phrase",
              OLD_PHRASE not in post)
        check("I3b patched prompt carries BOTH new continuation lines",
              "game_over` is NOT terminal for the run" in post
              and "keep playing immediately" in post)
    finally:
        os.chdir(cwd0)
        shutil.rmtree(tmp_root, ignore_errors=True)


def run_killswitch(warkit: Path) -> None:
    """Subprocess arm: CONTINUATION_DISABLE=1 -> graft no-ops, stock prompt kept."""
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    src_cont = warkit / "continuation_patch.py"
    tmp_root = Path(tempfile.mkdtemp(prefix="w0kill-"))
    run_dir = tmp_root / "run"
    run_dir.mkdir(parents=True)
    shutil.copy(src_cont, run_dir / "continuation_patch.py")
    cwd0 = Path.cwd()
    os.chdir(run_dir)
    try:
        ns: dict = {}
        exec_cell(cell_src(nb, 2), ns)
        pre = _prompt()
        ns["bm"] = SimpleNamespace(label="w0-kill-smoke")
        out12 = exec_cell(cell_src(nb, 12), ns)
        post = _prompt()
        # With CONTINUATION_STRICT unset, apply()==False under the kill switch is
        # tolerated (the graft prints nothing terminal and leaves the prompt vanilla).
        check("K1 kill switch: OLD phrase still present (vanilla prompt kept)",
              OLD_PHRASE in post and post == pre)
        check("K2 kill switch: continuation banner NOT printed",
              "ACTIVE (2 modules patched)" not in out12)
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
    assert (warkit / "continuation_patch.py").is_file(), f"bad --warkit: {warkit}"

    if "--killswitch-child" in args:
        run_killswitch(warkit)
        print(f"\nKILLSWITCH RESULT: {PASS} passed, {FAIL} failed")
        return 1 if FAIL else 0

    print(f"W0 (f) continuation-eval integration smoke | nb={NB_PATH}")
    run_integration(warkit)

    print("K: kill-switch arm (subprocess, CONTINUATION_DISABLE=1)")
    env = dict(os.environ)
    env["CONTINUATION_DISABLE"] = "1"
    env.pop("WARPACK_FORCE_OFFLINE_BENCH", None)
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
