"""war-v2-eval integration smoke (panel R12 N6) -- CPU only, no GPU, no LLM.

Runtime-tests the ASSEMBLED notebook notebooks/duckwar-v2-eval/
arc3-duck-war-v2-eval.ipynb end-to-end (feedback: v38 scored 0.00 from a
missing import -- always runtime-test the exact artifact):

  S1-S6  structural: cell count, eval+seed prefix on cell 2, heavy gates on
         cells 4/6/8/10, warpack+ledger+canary in cell 12, canary summary in
         cell 14, kernel-metadata byte-parity with the ledger-OFF war-eval
         kernel (id/title/code_file excepted).
  I1     exec cell 2 (real source): RUN_HEAVY forced by the eval line,
         WAR_EVAL_SEED=1 stamped, seed banner + gate-signal record printed.
  I2     exec cell 12 (real source) against the module copies that the kernel
         will load at runtime (--warkit <dir> = downloaded canivel/arc-war-kit
         copy; defaults to the repo copies): warpack applied, ledger flags
         {ledger,escalation} ON, canary ARMED.
  I3     scripted sb26 policy through the REAL _HarnessGameSession with the
         notebook-applied grafts: still clears L2, ledger.json written with
         both level-completion FACTs (end-to-end action path).
  I4     prompt path through a real ToolAgent: patched analyze binds the
         ledger + canary game label; _build_user_prompt injects digest +
         protocol; canary counts attempts/digests/skips; a 3-refutation
         same-family ledger fires the one-shot escalation exactly once.
  I5     exec the notebook's cell-14 canary summary block: greppable
         "LEDGER CANARY game=... attempts=..." + TOTAL lines.
  I6     exec full cell 14 with RUN_HEAVY=False (fast-submit path): dummy
         submission.parquet written, canary summary correctly skipped.
  K*     kill switches (subprocess): LEDGER_FLAGS="" -> graft no-op AND
         prompt carries no injection; LEDGER_CANARY=0 -> canary DISARMED.

Run:  .venv/Scripts/python.exe duck_eval/warpack/war_v2_eval_smoke.py \
          [--warkit <dir-with-dataset-copies>]
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
import threading
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUNDLE = REPO / "duck_eval" / "taaf_bundle" / "src"
ENV_FILES = REPO / "kaggle-data" / "environment_files"
NB_PATH = REPO / "notebooks" / "duckwar-v2-eval" / "arc3-duck-war-v2-eval.ipynb"
META_PATH = REPO / "notebooks" / "duckwar-v2-eval" / "kernel-metadata.json"
BASE_META_PATH = REPO / "notebooks" / "duckwar-eval" / "kernel-metadata.json"

sys.path.insert(0, str(REPO / "duck_eval" / "ledger" / "policies"))
sys.path.insert(0, str(BUNDLE / "ARC3-Inference"))
sys.path.insert(0, str(BUNDLE / "tufa-arc-agi-framework" / "src"))

os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "war-v2-eval-smoke")
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
    """Exec a notebook cell (top-level await allowed), capturing stdout."""
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
    check("S2b cell 2 stamps WAR_EVAL_SEED=1 + seed banner",
          'os.environ["WAR_EVAL_SEED"] = "1"' in c2
          and "pairs with ledger-OFF" in c2)
    check("S2c cell 2 records the gate detection signals (war-v2)",
          "gate signals" in c2 and "KAGGLE_KERNEL_RUN_TYPE" in c2)
    gated = all(cell_src(nb, i).lstrip("#").strip().startswith("Warpack fast-submit gate")
                and "if RUN_HEAVY:" in cell_src(nb, i) for i in (4, 6, 8, 10))
    check("S3 heavy cells 4/6/8/10 gated on RUN_HEAVY", gated)
    c12 = cell_src(nb, 12)
    check("S4 cell 12 carries warpack + ledger graft + canary",
          "import warpack_patch" in c12 and "import ledger_patch" in c12
          and "LEDGER CANARY" in c12 and "ledger_canary_report" in c12)
    check("S4b ledger graft + canary gated on RUN_HEAVY",
          "if RUN_HEAVY:" in c12)
    c14 = cell_src(nb, 14)
    check("S5 cell 14 keeps fast path + adds canary summary",
          "_write_dummy_submission" in c14 and "ledger_canary_report()" in c14
          and "except NameError:" in c14)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    base = json.loads(BASE_META_PATH.read_text(encoding="utf-8"))
    delta = {k for k in set(meta) | set(base) if meta.get(k) != base.get(k)}
    check("S6 kernel-metadata matches war-eval except id/title/code_file",
          delta == {"id", "title", "code_file"}
          and meta["id"] == "canivel/arc3-duck-war-v2-eval",
          f"delta={sorted(delta)}")


def run_integration(warkit: Path | None) -> None:
    import record_policies

    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    structural(nb)

    src_warpack = (warkit / "warpack_patch.py") if warkit else (HERE / "warpack_patch.py")
    src_ledger_dir = warkit if warkit else (REPO / "duck_eval" / "ledger")
    src_continuation = ((warkit if warkit else (REPO / "duck_eval" / "continuation"))
                        / "continuation_patch.py")
    print(f"module source under test: warpack={src_warpack}\n"
          f"                          ledger dir={src_ledger_dir}\n"
          f"                          continuation={src_continuation}")

    tmp_root = Path(tempfile.mkdtemp(prefix="warv2smoke-"))
    run_dir = tmp_root / "run"
    (run_dir / "warpack").mkdir(parents=True)
    (run_dir / "ledger").mkdir()
    (run_dir / "continuation").mkdir()
    shutil.copy(src_warpack, run_dir / "warpack" / "warpack_patch.py")
    shutil.copy(src_ledger_dir / "ledger_patch.py", run_dir / "ledger" / "ledger_patch.py")
    shutil.copy(src_ledger_dir / "ledger_core.py", run_dir / "ledger" / "ledger_core.py")
    # (f) default since 2026-07-23: cell 12 now also carries the continuation
    # graft; stage its module like the kernel's arc-war-kit mount does.
    shutil.copy(src_continuation, run_dir / "continuation" / "continuation_patch.py")

    kaggle_root = Path(os.path.splitdrive(str(run_dir))[0] + "/kaggle")
    pre_existing = {p: p.is_dir() for p in (kaggle_root, kaggle_root / "working")}
    cwd0 = Path.cwd()
    os.chdir(run_dir)
    try:
        ns: dict = {}
        print("I1: exec cell 2 (eval gate + seed stamp)")
        out2 = exec_cell(cell_src(nb, 2), ns)
        check("I1 RUN_HEAVY forced True by the eval line",
              ns.get("RUN_HEAVY") is True and ns.get("FORCE_OFFLINE_BENCH") is True)
        check("I1b WAR_EVAL_SEED=1 stamped + banner printed",
              os.environ.get("WAR_EVAL_SEED") == "1"
              and "war-v2-eval: SEED=1 ledger-ON" in out2)
        check("I1c gate detection signals recorded",
              "gate signals" in out2 and "RUN_HEAVY=True" in out2)

        print("I2: exec cell 12 (warpack + ledger + canary) against runtime module copies")
        ns["bm"] = SimpleNamespace(label="war-v2-eval-smoke")
        out12 = exec_cell(cell_src(nb, 12), ns)
        check("I2 warpack patches applied", "patches applied from" in out12
              and "warpack" in out12.split("patches applied from")[0])
        check("I2b ledger applied with flags {ledger,escalation} ON",
              "'ledger': True" in out12 and "'escalation': True" in out12)
        check("I2c canary ARMED", "ledger canary: ARMED" in out12)
        check("I2d bm.label carries both graft markers",
              "-warpack-" in ns["bm"].label and "-ledger-" in ns["bm"].label,
              ns["bm"].label)
        check("I2e no fallback-to-vanilla traceback",
              "PATCH FAILED" not in out12 and "patch failed" not in out12)
        check("I2f runtime banner proves VERSION + per-game store keying",
              "ledger v2: store keying = per-game:runtime-state-stem" in out12
              and "ledger v2: patches applied" in out12)
        check("I2g (f) continuation graft applied (default since 2026-07-23)",
              "game-over-continuation graft applied" in out12
              and "-continuation-" in ns["bm"].label)

        print("I3: scripted sb26 policy through the grafted harness (real engine)")
        import arc_agi
        import inference.framework.solver as solver_mod
        from taaf.game_api import ArcadeSpec, GameAPI

        sequences = record_policies.record_all()
        arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE,
                                environments_dir=str(ENV_FILES))
        game_id = next(e.game_id for e in arcade.available_environments
                       if e.game_id.startswith("sb26"))
        api = GameAPI(env_name=game_id,
                      arcade_spec=ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE,
                                             environments_dir=str(ENV_FILES)))
        api.start_game()
        game_dir = tmp_root / "sb26"
        game_dir.mkdir()
        session = solver_mod._HarnessGameSession(
            solver=SimpleNamespace(label="war-v2-smoke", job_dir=None,
                                   max_actions_per_game=400,
                                   max_runtime_s_per_game=None,
                                   soft_time_remaining_seconds=lambda: None),
            game=api,
            analyzer=SimpleNamespace(generated_tokens=0, _timeout=5.0),
            game_index=0, pass_index=0,
            state_path=game_dir / "sb26smoke_p0_runtime_state.json",
            transcript_path=game_dir / "transcript.txt",
            analysis_html_relpath="solver_analysis/smoke.html",
            stop_event=threading.Event(),
            viewer_data_path=game_dir / "viewer_data.json",
        )
        for request in sequences["sb26"]:
            result = session.step_env(dict(request))
            assert result.get("executed"), f"sb26 action failed: {request} -> {result}"
        check("I3 sb26 still clears L2 through the notebook-grafted harness",
              int(api.game_run.levels_completed) == 2,
              f"lc={api.game_run.levels_completed}")
        # ledger v2: per-game persistence file ledger_<stem>.json
        ledger_file = game_dir / "ledger_sb26smoke_p0.json"
        led_json = (json.loads(ledger_file.read_text(encoding="utf-8"))
                    if ledger_file.is_file() else None)
        level_facts = ([f for f in led_json["facts"]
                        if f["statement"].startswith("level completed")]
                       if led_json else [])
        check("I3b per-game ledger_sb26smoke_p0.json persisted with both "
              "level-completion FACTs",
              len(level_facts) == 2, str(led_json)[:120])

        print("I4: prompt path through a real ToolAgent (canary + digest + escalation)")
        import ledger_patch as runtime_ledger_patch
        from inference.agent.tool_agent import ToolAgent

        marker = runtime_ledger_patch.PROTOCOL_LINES[:48]
        agent = ToolAgent()
        # Missing state file -> the stock analyze body returns None immediately
        # (no LLM call), but the canary label + ledger binding hooks run first;
        # the parent dir is the I3 game dir, so the persisted ledger is loaded.
        agent.analyze(game_dir / "sb26smoke_p0_runtime_state.json.MISSING", 0)
        check("I4 patched analyze binds the per-game ledger + canary label",
              getattr(agent, "_ledger_state", None) is not None
              and getattr(agent, "_canary_game", "") == "sb26smoke_p0",
              f"label={getattr(agent, '_canary_game', None)}")
        prompt = agent._build_user_prompt(0, valid_actions=["ACTION1", "ACTION2"])
        check("I4b digest + ledger protocol injected into the user prompt",
              marker in prompt and "level completed" in prompt)
        led = agent._ledger_state
        for k in range(3):
            hyp = led.add_hypothesis(
                f"execution order variant {k} drives the frame program", "ordering")
            led.refute(hyp, f"variant {k} executed fully and failed", action=10 + k)
        p_escalated = agent._build_user_prompt(1, valid_actions=["ACTION1"])
        p_after = agent._build_user_prompt(2, valid_actions=["ACTION1"])
        esc_marker = "ESCALATION -- GOAL-FAMILY CHECK"
        check("I4c escalation fires one-shot after 3 same-family refutations",
              led.escalations_fired == 1 and esc_marker in p_escalated
              and esc_marker not in p_after,
              f"fired={led.escalations_fired}")
        unbound = ToolAgent()
        unbound._build_user_prompt(0, valid_actions=["ACTION1"])
        stats = ns["LEDGER_CANARY_STATS"]
        st = stats.get("sb26smoke_p0", {})
        check("I4d canary per-game counters (attempts=3 digests=3 skips=0 aborts=0)",
              st.get("attempts") == 3 and st.get("digests") == 3
              and st.get("skips") == 0 and st.get("aborts") == 0, str(st))
        check("I4e canary counts the unbound agent as a skip",
              stats.get("unbound", {}).get("skips") == 1,
              str(stats.get("unbound")))

        print("I5: cell-14 canary summary block (RUN_HEAVY=True)")
        summary_src = cell_src(nb, 14).split("# --- war-v2 LEDGER CANARY summary")[1]
        out5 = exec_cell("# --- war-v2 LEDGER CANARY summary" + summary_src, ns)
        check("I5 greppable per-game canary line",
              "LEDGER CANARY game=sb26smoke_p0 attempts=3 digests=3 skips=0 "
              "aborts=0 escalations=1" in out5)
        check("I5b greppable TOTAL line (one real game -> stores=1)",
              "LEDGER CANARY TOTAL games=2 stores=1 escalations_total=1" in out5)

        print("I6: full cell 14 with RUN_HEAVY=False (fast-submit path intact)")
        ns14 = dict(ns)
        ns14["RUN_HEAVY"] = False
        ns14["WORKING_DIR"] = tmp_root / "working"
        ns14["WORKING_DIR"].mkdir()
        out6 = exec_cell(cell_src(nb, 14), ns14)
        check("I6 dummy submission.parquet written on the fast path",
              (ns14["WORKING_DIR"] / "submission.parquet").is_file()
              and "FAST-SUBMIT" in out6)
        check("I6b canary summary correctly skipped when not RUN_HEAVY",
              "LEDGER CANARY" not in out6)

        print("I7: two games CONCURRENT in one shared artifacts dir "
              "(live layout; per-game isolation)")
        # Fresh registry + canary stats so the TOTAL line reflects only this
        # case (dict ops only under the module lock, per the deadlock rule).
        with runtime_ledger_patch._LEDGERS_LOCK:
            runtime_ledger_patch._LEDGERS.clear()
        ns["LEDGER_CANARY_STATS"].clear()
        shared = tmp_root / "artifacts"
        shared.mkdir()
        n_turns = 25
        results: dict[str, dict] = {}

        def _play(tag: str) -> None:
            res = results[tag] = {"self": 0, "other": 0, "error": ""}
            try:
                other = "B" if tag == "A" else "A"
                agent_t = ToolAgent()
                agent_t.analyze(
                    shared / f"game{tag}_p0_runtime_state.json.MISSING", 0)
                agent_t._ledger_state.add_fact(
                    f"zebra-omega-{tag}: SPACE only decrements the timer "
                    f"in game {tag}")
                for k in range(n_turns):
                    p = agent_t._build_user_prompt(k, valid_actions=["ACTION1"])
                    res["self"] += f"zebra-omega-{tag}" in p
                    res["other"] += f"zebra-omega-{other}" in p
            except Exception as exc:  # noqa: BLE001 - surface in the check
                res["error"] = repr(exc)

        threads = [threading.Thread(target=_play, args=(t,)) for t in ("A", "B")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
        check("I7 both concurrent games ran clean",
              not results["A"]["error"] and not results["B"]["error"],
              f"{results}")
        check("I7b own FACT injected every turn in both games",
              results["A"]["self"] == n_turns and results["B"]["self"] == n_turns,
              f"A={results['A']['self']} B={results['B']['self']} (want {n_turns})")
        check("I7c ZERO cross-game digest contamination",
              results["A"]["other"] == 0 and results["B"]["other"] == 0,
              f"A saw B {results['A']['other']}x, B saw A {results['B']['other']}x")
        check("I7d per-game persistence files, no shared ledger.json",
              (shared / "ledger_gameA_p0.json").is_file()
              and (shared / "ledger_gameB_p0.json").is_file()
              and not (shared / "ledger.json").exists(),
              str(sorted(p.name for p in shared.glob("ledger*"))))
        out7 = exec_cell("# --- war-v2 LEDGER CANARY summary" + summary_src, ns)
        check("I7e canary TOTAL shows stores==2 (per-game, not shared)",
              "LEDGER CANARY TOTAL games=2 stores=2" in out7)
        check("I7f canary per-game counters exact under concurrency "
              f"(attempts={n_turns} each, no lost updates)",
              f"LEDGER CANARY game=gameA_p0 attempts={n_turns} "
              f"digests={n_turns} skips=0 aborts=0 escalations=0" in out7
              and f"LEDGER CANARY game=gameB_p0 attempts={n_turns} "
              f"digests={n_turns} skips=0 aborts=0 escalations=0" in out7)
    finally:
        os.chdir(cwd0)
        shutil.rmtree(tmp_root, ignore_errors=True)
        for p, existed in sorted(pre_existing.items(), reverse=True):
            if not existed and p.is_dir():
                with contextlib.suppress(OSError):
                    p.rmdir()  # only removes it when empty (we only mkdir'ed)


def run_killswitch(warkit: Path | None) -> None:
    """Subprocess arm: LEDGER_FLAGS='' + LEDGER_CANARY=0 -> proven no-op."""
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    src_warpack = (warkit / "warpack_patch.py") if warkit else (HERE / "warpack_patch.py")
    src_ledger_dir = warkit if warkit else (REPO / "duck_eval" / "ledger")
    src_continuation = ((warkit if warkit else (REPO / "duck_eval" / "continuation"))
                        / "continuation_patch.py")
    tmp_root = Path(tempfile.mkdtemp(prefix="warv2kill-"))
    run_dir = tmp_root / "run"
    (run_dir / "warpack").mkdir(parents=True)
    (run_dir / "ledger").mkdir()
    (run_dir / "continuation").mkdir()
    shutil.copy(src_warpack, run_dir / "warpack" / "warpack_patch.py")
    shutil.copy(src_ledger_dir / "ledger_patch.py", run_dir / "ledger" / "ledger_patch.py")
    shutil.copy(src_ledger_dir / "ledger_core.py", run_dir / "ledger" / "ledger_core.py")
    shutil.copy(src_continuation, run_dir / "continuation" / "continuation_patch.py")
    kaggle_root = Path(os.path.splitdrive(str(run_dir))[0] + "/kaggle")
    pre_existing = {p: p.is_dir() for p in (kaggle_root, kaggle_root / "working")}
    cwd0 = Path.cwd()
    os.chdir(run_dir)
    try:
        ns: dict = {}
        exec_cell(cell_src(nb, 2), ns)
        ns["bm"] = SimpleNamespace(label="killswitch-smoke")
        out12 = exec_cell(cell_src(nb, 12), ns)
        check("K1 LEDGER_FLAGS='' -> all ledger flags OFF",
              "'ledger': False" in out12 and "'escalation': False" in out12)
        check("K2 LEDGER_CANARY=0 (and flags off) -> canary DISARMED",
              "ledger canary: DISARMED" in out12)
        from inference.agent.tool_agent import ToolAgent

        import ledger_patch as runtime_ledger_patch

        agent = ToolAgent()
        prompt = agent._build_user_prompt(0, valid_actions=["ACTION1"])
        check("K3 prompt carries NO ledger injection with flags off",
              runtime_ledger_patch.PROTOCOL_LINES[:48] not in prompt)
        check("K4 no canary stats collected", "LEDGER_CANARY_STATS" not in ns)
    finally:
        os.chdir(cwd0)
        shutil.rmtree(tmp_root, ignore_errors=True)
        for p, existed in sorted(pre_existing.items(), reverse=True):
            if not existed and p.is_dir():
                with contextlib.suppress(OSError):
                    p.rmdir()


def main() -> int:
    global PASS, FAIL
    args = sys.argv[1:]
    warkit = None
    if "--warkit" in args:
        warkit = Path(args[args.index("--warkit") + 1]).resolve()
        assert (warkit / "warpack_patch.py").is_file(), f"bad --warkit: {warkit}"
    if "--killswitch-child" in args:
        run_killswitch(warkit)
        print(f"\nKILLSWITCH RESULT: {PASS} passed, {FAIL} failed")
        return 1 if FAIL else 0

    print(f"war-v2-eval integration smoke | nb={NB_PATH}")
    run_integration(warkit)

    print("K: kill-switch arm (subprocess, LEDGER_FLAGS='' LEDGER_CANARY=0)")
    env = dict(os.environ)
    env["LEDGER_FLAGS"] = ""
    env["LEDGER_CANARY"] = "0"
    env.pop("WARPACK_FORCE_OFFLINE_BENCH", None)
    child = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--killswitch-child",
         *(["--warkit", str(warkit)] if warkit else [])],
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
