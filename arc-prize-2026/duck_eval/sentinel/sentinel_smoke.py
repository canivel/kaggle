"""(a) budget-sentinel eval integration smoke -- CPU only, no GPU/LLM.

Runtime-tests the ASSEMBLED notebook notebooks/ducksentinel-eval/
arc3-duck-sentinel-eval.ipynb end-to-end (feedback_test_before_submit: v38
scored 0.00 from a missing import -- always runtime-test the exact artifact).
The (a) window is STANDALONE on the duck baseline: duck + budget sentinel, NO
warpack, NO ledger.

  S1-S6  structural: 17 cells, eval+sentinel-seed prefix on cell 2, heavy gates
         on cells 4/6/8/10, budget-sentinel graft (NOT warpack/ledger) in cell
         12, cell 14 keeps the fast-submit path, kernel-metadata byte-parity
         with the war-eval kernel (id/title/code_file excepted).
  I1     exec cell 2 (real source): RUN_HEAVY forced by the eval line,
         SENTINEL_EVAL_SEED=1 stamped, seed banner printed.
  I2     exec cell 12 (real source) against the module copy the kernel loads at
         runtime (--warkit <dir>): sentinel applied, banner
         "sentinel v2: budget sentinel ACTIVE ..." printed, bm.label carries the
         sentinel marker, NO warpack/ledger imported.
  M*     mechanism against the REAL bundled harness classes: drive the patched
         _HarnessGameSession._execute_action over a synthetic compressed budget
         and confirm (a) SENTINEL trigger events fire on threshold crossings,
         (b) the patched ToolAgent._build_user_prompt injects the budget FACT on
         a crossing turn and NOT on a non-crossing turn (the token-cost prong),
         (c) v2 unit semantics (R16 repair): a fresh level attempt does NOT
         re-arm the thresholds; crossings spanning an attempt boundary fire at
         the correct CUMULATIVE game actions.
  K*     kill switch (subprocess): SENTINEL_DISABLE=1 -> graft no-op, banner
         absent, harness unpatched (vanilla).

Run:  uv run python duck_eval/sentinel/sentinel_smoke.py [--warkit <dir>]
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
NB_PATH = REPO / "notebooks" / "ducksentinel-eval" / "arc3-duck-sentinel-eval.ipynb"
META_PATH = REPO / "notebooks" / "ducksentinel-eval" / "kernel-metadata.json"
BASE_META_PATH = REPO / "notebooks" / "duckwar-eval" / "kernel-metadata.json"
WARKIT_DEFAULT = HERE  # budget_sentinel_patch.py lives next to this smoke

sys.path.insert(0, str(BUNDLE / "ARC3-Inference"))

os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "sentinel-eval-smoke")
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
    print("S: structural checks on the assembled notebook")
    check("S1 17 cells", len(nb["cells"]) == 17, str(len(nb["cells"])))
    c2 = cell_src(nb, 2)
    check("S2 cell 2 forces the offline bench (eval line first)",
          c2.startswith('import os; os.environ["WARPACK_FORCE_OFFLINE_BENCH"] = "1"'))
    check("S2b cell 2 stamps SENTINEL_EVAL_SEED=1 + seed banner",
          'os.environ["SENTINEL_EVAL_SEED"] = "1"' in c2
          and "pairs with the prior-stack seed 1" in c2)
    check("S2c cell 2 exports SENTINEL_BUDGET=150 + budget banner (C7 as "
          "amended 2026-07-23: eval regime is UNCAPPED; without the export "
          "the sentinel is inert)",
          'os.environ["SENTINEL_BUDGET"] = "150"' in c2
          and "SENTINEL_BUDGET=150" in c2)
    gated = all(cell_src(nb, i).lstrip("#").strip().startswith("Warpack fast-submit gate")
                and "if RUN_HEAVY:" in cell_src(nb, i) for i in (4, 6, 8, 10))
    check("S3 heavy cells 4/6/8/10 gated on RUN_HEAVY", gated)
    c12 = cell_src(nb, 12)
    check("S4 cell 12 is the sentinel graft (imports budget_sentinel_patch)",
          "import budget_sentinel_patch" in c12
          and "budget_sentinel_patch.apply(bm)" in c12
          and "if not RUN_HEAVY:" in c12)
    check("S4b cell 12 does NOT ship warpack or ledger (single flag)",
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
          and meta["id"] == "canivel/arc3-duck-sentinel-eval",
          f"delta={sorted(delta)}")


class _FakeSolver:
    """Stands in for HarnessSolver: only max_actions_per_game is read."""
    def __init__(self, budget):
        self.max_actions_per_game = budget


class _FakeSession:
    """Minimal stand-in for _HarnessGameSession exposing exactly what the
    sentinel hook reads: .solver, .state_path, .action_count. We call the
    PATCHED classmethod with self=this so the real graft runs against a real
    payload shape."""
    def __init__(self, solver, state_path):
        self.solver = solver
        self.state_path = state_path
        self.action_count = 0


def _orig_execute_stub(payloads):
    """A fake original _execute_action: pops the next canned payload and bumps
    action_count. The graft wraps THIS and reads the returned payload."""
    it = iter(payloads)

    def _inner(self, action, *, batch_index, batch_size,
               generated_tokens=None, flush_viewer_payload=True):
        p = next(it)
        self.action_count = p["action_num"]
        return p
    return _inner


def _install_over_stub(sen, solver_mod, ToolAgent, orig_prompt, orig_analyze,
                       payloads):
    """(Re)install the graft with a canned-payload stub standing in for the
    real _execute_action, so we can drive the REAL wrapper over a compressed
    budget without a live solver run. The graft closes over whatever
    _execute_action is set at apply() time -> set the stub, reset the prompt/
    analyze patches to their originals, then apply()."""
    sen._APPLIED = False
    solver_mod._HarnessGameSession._execute_action = _orig_execute_stub(payloads)
    ToolAgent._build_user_prompt = orig_prompt
    ToolAgent.analyze = orig_analyze
    return sen.apply(SimpleNamespace(label="sentinel-mech"))


def run_mechanism(warkit: Path, run_dir: Path) -> None:
    """Drive the REAL patched harness classes over a compressed budget."""
    import inference.framework.solver as solver_mod
    from inference.agent.tool_agent import ToolAgent
    import budget_sentinel_patch as sen

    # Force a small budget via env override so we don't need a real solver run.
    os.environ["SENTINEL_BUDGET"] = "20"          # 20-action compressed budget
    os.environ["SENTINEL_THRESHOLDS"] = "0.5,0.75,0.9"

    orig_exec = solver_mod._HarnessGameSession._execute_action
    orig_prompt = ToolAgent._build_user_prompt
    orig_analyze = ToolAgent.analyze

    state_path = run_dir / "lp85_p0_runtime_state.json"

    # Canned payloads: actions 1..20 on level 1, all non-completing.
    # Crossings at 10 (50%), 15 (75%), 18 (90%).
    payloads = [{"action_num": a, "level": 1, "level_completed": False,
                 "game_over": False, "board_changed": True} for a in range(1, 21)]
    applied = _install_over_stub(sen, solver_mod, ToolAgent, orig_prompt,
                                 orig_analyze, payloads)
    check("M0 sentinel.apply() returned True", applied is True)

    sess = _FakeSession(_FakeSolver(20), state_path)
    fire_events = []
    for a in range(1, 21):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            solver_mod._HarnessGameSession._execute_action(
                sess, object(), batch_index=1, batch_size=1)
        for ln in buf.getvalue().splitlines():
            if ln.startswith("SENTINEL v=2 kind=budget_threshold"):
                fire_events.append(ln)
    print(f"    | fired {len(fire_events)} SENTINEL events over 20 actions")
    for ln in fire_events:
        print(f"    | {ln}")

    check("M1 three threshold events fired (50/75/90%)", len(fire_events) == 3,
          f"{len(fire_events)} events")
    check("M1b events carry game/action_num/threshold",
          all("game=lp85" in e and "action_num=" in e and "threshold=" in e
              for e in fire_events))
    thr = sorted(float(e.split("threshold=")[1].split()[0]) for e in fire_events)
    check("M1c thresholds are 0.50/0.75/0.90", thr == [0.5, 0.75, 0.9], str(thr))
    acts = sorted(int(e.split("action_num=")[1].split()[0]) for e in fire_events)
    check("M1d crossings at actions 10/15/18", acts == [10, 15, 18], str(acts))

    # events.jsonl sidecar written per game
    sidecar = state_path.parent / "lp85_p0_sentinel_events.jsonl"
    check("M2 per-game sentinel_events.jsonl written", sidecar.is_file())
    recs = [json.loads(l) for l in sidecar.read_text().splitlines() if l.strip()]
    check("M2b sidecar has 3 budget_threshold records",
          len(recs) == 3 and all(r["kind"] == "budget_threshold" for r in recs))

    # ---- prompt injection prong: FACT only on crossing turns ----------------
    # Bind the per-game store into a fake ToolAgent, then drive the prompt hook.
    key = sen._store_key(state_path)
    st = sen._STORES[key]

    class _FakeAgent:
        _sentinel_state_path = state_path
    agent = _FakeAgent()

    # A crossing just happened (action 18 set a pending FACT). Draining it once
    # must append the FACT; a second drain must NOT (one-shot, no per-turn tax).
    st.pending_fact = sen._build_fact("lp85", 18, 0.9, 20, 2)
    base_prompt = "BASE-PROMPT-BODY"

    # Reinstall the prompt patch over a stub original that returns base_prompt,
    # so we test ONLY the sentinel's append behaviour deterministically.
    def stub_prompt(self, action_num, **kw):
        return base_prompt
    sen._APPLIED = False
    ToolAgent._build_user_prompt = stub_prompt
    ToolAgent.analyze = orig_analyze
    sen.apply(SimpleNamespace(label="sentinel-prompt"))
    p1 = ToolAgent._build_user_prompt(agent, 18)
    p2 = ToolAgent._build_user_prompt(agent, 19)
    check("M3 FACT injected on the crossing turn",
          "budget sentinel" in p1 and p1 != base_prompt
          and "actions remain for ALL remaining levels" in p1)
    check("M3b FACT NOT re-injected next turn (one-shot, zero token tax)",
          p2 == base_prompt)

    # ---- v2 unit semantics (R16 repair): NO re-arm on a fresh attempt -------
    # Same game continues past a level-up: every threshold already fired for
    # the game, so the new attempt must produce ZERO new firings (v1 would have
    # re-armed and fired 3 more at 31/36/39).
    payloads2 = (
        [{"action_num": 21, "level": 1, "level_completed": True,
          "game_over": False, "board_changed": True}]
        + [{"action_num": a, "level": 2, "level_completed": False,
            "game_over": False, "board_changed": True} for a in range(22, 42)]
    )
    _install_over_stub(sen, solver_mod, ToolAgent, orig_prompt, orig_analyze,
                       payloads2)
    fire2 = []
    for p in payloads2:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            solver_mod._HarnessGameSession._execute_action(
                sess, object(), batch_index=1, batch_size=1)
        for ln in buf.getvalue().splitlines():
            if ln.startswith("SENTINEL v=2"):
                fire2.append(ln)
    acts2 = sorted(int(e.split("action_num=")[1].split()[0]) for e in fire2)
    check("M4 fresh level attempt does NOT re-arm (game-envelope unit)",
          len(fire2) == 0, str(acts2))

    # A SECOND game whose crossings span an attempt boundary: thresholds fire
    # at the correct CUMULATIVE actions (10/15/18) even though a level-up at
    # action 12 starts attempt 1 mid-envelope; the 15/18 events carry the new
    # attempt ordinal as metadata.
    state_path3 = run_dir / "tu93_p0_runtime_state.json"
    payloads3 = (
        [{"action_num": a, "level": 1, "level_completed": False,
          "game_over": False, "board_changed": True} for a in range(1, 12)]
        + [{"action_num": 12, "level": 1, "level_completed": True,
            "game_over": False, "board_changed": True}]
        + [{"action_num": a, "level": 2, "level_completed": False,
            "game_over": False, "board_changed": True} for a in range(13, 21)]
    )
    _install_over_stub(sen, solver_mod, ToolAgent, orig_prompt, orig_analyze,
                       payloads3)
    sess3 = _FakeSession(_FakeSolver(20), state_path3)
    fire3 = []
    for p in payloads3:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            solver_mod._HarnessGameSession._execute_action(
                sess3, object(), batch_index=1, batch_size=1)
        for ln in buf.getvalue().splitlines():
            if ln.startswith("SENTINEL v=2"):
                fire3.append(ln)
    acts3 = sorted(int(e.split("action_num=")[1].split()[0]) for e in fire3)
    check("M4b crossings use CUMULATIVE actions across attempt boundary",
          acts3 == [10, 15, 18], str(acts3))
    att3 = [int(e.split("attempt=")[1].split()[0]) for e in fire3]
    check("M4c post-boundary events carry the new attempt ordinal",
          att3 == [0, 1, 1], str(att3))

    # restore originals
    solver_mod._HarnessGameSession._execute_action = orig_exec
    ToolAgent._build_user_prompt = orig_prompt
    ToolAgent.analyze = orig_analyze
    os.environ.pop("SENTINEL_BUDGET", None)
    os.environ.pop("SENTINEL_THRESHOLDS", None)


def run_integration(warkit: Path) -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    structural(nb)

    src_mod = warkit / "budget_sentinel_patch.py"
    assert src_mod.is_file(), f"no budget_sentinel_patch.py under {warkit}"
    print(f"module source under test: sentinel={src_mod}")

    tmp_root = Path(tempfile.mkdtemp(prefix="sensmoke-"))
    run_dir = tmp_root / "run"
    # Mirror the kernel layout: budget_sentinel_patch.py discoverable next to cwd.
    run_dir.mkdir(parents=True)
    shutil.copy(src_mod, run_dir / "budget_sentinel_patch.py")

    cwd0 = Path.cwd()
    os.chdir(run_dir)
    try:
        ns: dict = {}
        print("I1: exec cell 2 (eval gate + sentinel seed stamp)")
        out2 = exec_cell(cell_src(nb, 2), ns)
        check("I1 RUN_HEAVY forced True by the eval line",
              ns.get("RUN_HEAVY") is True and ns.get("FORCE_OFFLINE_BENCH") is True)
        check("I1b SENTINEL_EVAL_SEED=1 stamped + banner printed",
              os.environ.get("SENTINEL_EVAL_SEED") == "1"
              and "sentinel-eval: SEED=1" in out2)
        check("I1c SENTINEL_BUDGET=150 exported + budget banner printed (C7 as amended)",
              os.environ.get("SENTINEL_BUDGET") == "150"
              and "sentinel-eval: SENTINEL_BUDGET=150" in out2)

        print("I2: exec cell 12 (sentinel graft) against the runtime module copy")
        ns["bm"] = SimpleNamespace(label="sentinel-eval-smoke")
        out12 = exec_cell(cell_src(nb, 12), ns)
        check("I2 sentinel graft applied (banner printed)",
              "sentinel v2: budget sentinel ACTIVE" in out12
              and "graft applied" in out12)
        check("I2b bm.label carries the sentinel marker",
              "-sentinel-v2" in ns["bm"].label, ns["bm"].label)
        check("I2c NO fallback-to-vanilla traceback",
              "PATCH FAILED" not in out12)
        check("I2d graft did NOT import warpack/ledger",
              "warpack_patch" not in sys.modules and "ledger_patch" not in sys.modules
              and "ledger_core" not in sys.modules)

        print("M: mechanism against the REAL bundled harness classes")
        run_mechanism(warkit, run_dir)
    finally:
        os.chdir(cwd0)
        shutil.rmtree(tmp_root, ignore_errors=True)


def run_killswitch(warkit: Path) -> None:
    """Subprocess arm: SENTINEL_DISABLE=1 -> graft no-ops, harness unpatched."""
    import inference.framework.solver as solver_mod
    from inference.agent.tool_agent import ToolAgent
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    src_mod = warkit / "budget_sentinel_patch.py"
    tmp_root = Path(tempfile.mkdtemp(prefix="senkill-"))
    run_dir = tmp_root / "run"
    run_dir.mkdir(parents=True)
    shutil.copy(src_mod, run_dir / "budget_sentinel_patch.py")
    cwd0 = Path.cwd()
    os.chdir(run_dir)
    try:
        ns: dict = {}
        exec_cell(cell_src(nb, 2), ns)
        pre_exec = solver_mod._HarnessGameSession._execute_action
        pre_prompt = ToolAgent._build_user_prompt
        ns["bm"] = SimpleNamespace(label="sentinel-kill-smoke")
        out12 = exec_cell(cell_src(nb, 12), ns)
        check("K1 kill switch: harness _execute_action left unpatched",
              solver_mod._HarnessGameSession._execute_action is pre_exec)
        check("K1b kill switch: _build_user_prompt left unpatched",
              ToolAgent._build_user_prompt is pre_prompt)
        check("K2 kill switch: sentinel banner NOT printed",
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
    assert (warkit / "budget_sentinel_patch.py").is_file(), f"bad --warkit: {warkit}"

    if "--killswitch-child" in args:
        run_killswitch(warkit)
        print(f"\nKILLSWITCH RESULT: {PASS} passed, {FAIL} failed")
        return 1 if FAIL else 0

    print(f"(a) budget-sentinel-eval integration smoke | nb={NB_PATH}")
    run_integration(warkit)

    print("K: kill-switch arm (subprocess, SENTINEL_DISABLE=1)")
    env = dict(os.environ)
    env["SENTINEL_DISABLE"] = "1"
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
