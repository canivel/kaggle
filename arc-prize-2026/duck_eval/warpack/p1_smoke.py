"""P1 zero-information action suppressor smoke -- CPU only, no GPU/LLM/network.

Runtime-tests the ASSEMBLED notebook notebooks/duckp1-eval/arc3-duck-p1-eval.ipynb
AND the patched code path against the REAL offline arcengine
(feedback_test_before_submit: v38 scored 0.00 from a missing import -- always
runtime-test the exact artifact).

  S*  structural: eval + P1-flag prefix on cell 2, heavy gates, P1 graft (NOT
      warpack/ledger/sentinel/compaction/animation) in cell 12, (f) continuation
      default still riding, post-run canary in cell 14, kernel-metadata
      byte-parity with the war-eval family.
  U*  unit: P1State memo / confirm budget / decline budget / per-level scoping /
      the online latent-state detector / the memory block.
  R*  REPLAY on the three recorded runs (the pre-registered evidence):
      R1  the online detector reproduces the published 8-game latent-state set
          EXACTLY on animation_v1 (m0r0 55, re86 19, sk48 11, ka59 10, cd82 8,
          g50t 4, dc22 3, wa30 2) and flags nothing on the other 17 games
      R2  zero level-completing actions are declined or aborted, on all three
          runs, with the shipped defaults      <- the hard safety canary
      R3  zero BOARD-CHANGING actions are declined, on all three runs
      R4  the duplicate re-execution rate falls and the blind tail is drained
      R5  levels_completed is preserved on all 8 latent-state games
      R6  the published M1+M3 arithmetic is reproduced (x1.115/1.111/1.094 vs
          the published x1.11/1.11/1.09) AND is shown to delete 3 of the 17
          level-completing actions -- the finding that set the shipped defaults
  I*  integration: exec cell 2 + cell 12 (real notebook source) against the
      module copy the kernel loads at runtime, then drive the REAL patched
      solver._HarnessGameSession over a REAL offline engine.
  K*  flag gate + kill switch (subprocess): P1_SUPPRESS unset and P1_DISABLE=1
      both leave the harness byte-vanilla.

Run:  .venv/Scripts/python.exe duck_eval/warpack/p1_smoke.py [--warkit <dir>]
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
NB_PATH = REPO / "notebooks" / "duckp1-eval" / "arc3-duck-p1-eval.ipynb"
META_PATH = REPO / "notebooks" / "duckp1-eval" / "kernel-metadata.json"
BASE_META_PATH = REPO / "notebooks" / "duckwar-eval" / "kernel-metadata.json"
WARKIT_DEFAULT = REPO / "duck_eval" / "warpack" / "_kaggle_dataset"
ENV_FILES = REPO / "kaggle-data" / "environment_files"

sys.path.insert(0, str(BUNDLE / "ARC3-Inference"))
sys.path.insert(0, str(BUNDLE / "tufa-arc-agi-framework" / "src"))

os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "p1-smoke")
os.environ.setdefault("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:9/v1")

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}" + (f"  [{detail[:130]}]" if detail else ""))
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail[:400]}")


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


# --------------------------------------------------------------------------- #
# S: structural
# --------------------------------------------------------------------------- #
def structural(nb: dict) -> None:
    print("S: structural")
    c2 = cell_src(nb, 2)
    check("S1 cell 2 forces the offline bench (eval line first)",
          c2.startswith('import os; os.environ["WARPACK_FORCE_OFFLINE_BENCH"] = "1"'))
    check("S2 cell 2 sets the P1 arm flag + seed",
          'os.environ["P1_SUPPRESS"] = "1"' in c2
          and 'os.environ["P1_EVAL_SEED"] = "1"' in c2)
    check("S3 cell 2 banner names the shipped defaults",
          "memo_mode=noop" in c2 and "abort_revisit=OFF" in c2)
    c12 = cell_src(nb, 12)
    check("S4 cell 12 IS the P1 graft", "import p1_suppressor_patch" in c12
          and "p1_suppressor_patch.apply(bm)" in c12)
    check("S5 cell 12 carries NO warpack/ledger/sentinel/compaction/animation graft",
          not any(t in c12 for t in ("import warpack_patch", "import ledger_patch",
                                     "import budget_sentinel_patch",
                                     "import compaction_patch",
                                     "import animation_patch")))
    check("S6 cell 12 still carries the (f) continuation default",
          "import continuation_patch" in c12)
    check("S7 cell 12 falls back to VANILLA on failure",
          "PATCH FAILED - continuing with VANILLA duck harness" in c12)
    c14 = cell_src(nb, 14)
    check("S8 cell 14 calls the post-run canary",
          "p1_suppressor_patch as _p1" in c14 and "_p1.canary_report()" in c14)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    base = json.loads(BASE_META_PATH.read_text(encoding="utf-8"))
    delta = {k for k in set(meta) | set(base) if meta.get(k) != base.get(k)}
    check("S9 kernel-metadata differs from the war-eval family ONLY in id/title/code_file "
          "(kaggle_env_match discipline)",
          delta <= {"id", "title", "code_file"}, str(sorted(delta)))
    check("S10 kernel id is the P1 slug",
          meta["id"] == "canivel/arc3-duck-p1-eval", meta["id"])


# --------------------------------------------------------------------------- #
# U: unit
# --------------------------------------------------------------------------- #
def units(mod) -> None:
    print("U: unit (pure policy objects)")
    for k, v in {"P1_MEMO": "1", "P1_MEMO_MODE": "noop", "P1_CONFIRM": "2",
                 "P1_MAX_DECLINES": "1", "P1_ABORT": "1",
                 "P1_ABORT_CYCLE": "1", "P1_ABORT_REVISIT": "0",
                 "P1_BLOCK": "1"}.items():
        os.environ[k] = v

    st = mod.P1State("unit")
    A, B = "aaaa", "bbbb"
    check("U1 an unseen pair is never declined", st.should_decline((A, "UP")) is None)
    st.record((A, "UP"), A, True)          # first execution: a no-op
    check("U2 one observation is below the confirm budget (confirm=2)",
          st.should_decline((A, "UP")) is None)
    st.record((A, "UP"), A, True)          # second execution confirms it
    ent = st.should_decline((A, "UP"))
    check("U3 a CONFIRMED no-op pair is declined", ent is not None)
    ent.declines += 1
    check("U4 the decline budget is one per level -- a repeat request executes, "
          "so no path can be permanently blocked",
          st.should_decline((A, "UP")) is None)

    st2 = mod.P1State("unit2")
    st2.record((A, "DOWN"), B, False)      # board-changing
    st2.record((A, "DOWN"), B, False)
    check("U5 memo_mode=noop never declines a BOARD-CHANGING pair "
          "(declining one could refuse a deliberate re-traversal)",
          st2.should_decline((A, "DOWN")) is None)
    os.environ["P1_MEMO_MODE"] = "all"
    check("U6 memo_mode=all would decline it (the ablation handle exists)",
          st2.should_decline((A, "DOWN")) is not None)
    os.environ["P1_MEMO_MODE"] = "noop"

    st3 = mod.P1State("unit3")
    st3.record((A, "UP"), A, True)
    drift = st3.record((A, "UP"), B, False)   # same pair, different outcome
    check("U7 the ONLINE latent-state detector fires on a contradictory outcome",
          drift is True and st3.ambiguous is True and st3.ambiguity_events == 1)
    st3.record((A, "UP"), A, True)
    check("U8 once AMBIGUOUS, mechanism A is permanently off for that game",
          st3.should_decline((A, "UP")) is None)
    check("U9 the ambiguity flag SURVIVES a level change (game-scoped, "
          "as the safety constraint requires)",
          (st3.sync_level(1), st3.ambiguous)[1] is True)

    st4 = mod.P1State("unit4")
    st4.record((A, "UP"), A, True)
    st4.record((A, "UP"), A, True)
    st4.sync_level(1)
    check("U10 the memo is per LEVEL -- a new level starts with an empty memo, "
          "so the detector's scope always equals the memo's scope",
          st4.should_decline((A, "UP")) is None and not st4.memo)

    st5 = mod.P1State("unit5")
    st5.record((A, "LEFT"), A, True)
    st5.record((A, "RIGHT"), B, False)
    st5.last_hash = A
    block = st5.memory_block(["UP", "DOWN", "LEFT", "RIGHT", "SPACE"])
    check("U11 the memory block names the untried primitives at this board",
          "UP" in block and "DOWN" in block and "SPACE" in block
          and "NOT YET TRIED" in block)
    check("U12 the memory block names the confirmed-dead primitive",
          "CONFIRMED NO EFFECT" in block and "LEFT" in block)
    check("U13 the memory block is small (bounded schema)",
          len(block) < 900, f"{len(block)} chars")
    os.environ["P1_BLOCK"] = "0"
    check("U14 P1_BLOCK=0 removes the block entirely (ablation handle)",
          st5.memory_block(["UP"]) == "")
    os.environ["P1_BLOCK"] = "1"
    check("U15 P1_CONFIRM can never be driven below 2 (a confirm of 1 would "
          "blind the latent-state detector)",
          (os.environ.__setitem__("P1_CONFIRM", "1"), mod.CFG.confirm)[1] == 2)
    os.environ["P1_CONFIRM"] = "2"


# --------------------------------------------------------------------------- #
# R: replay on the three recorded runs (the pre-registered evidence)
# --------------------------------------------------------------------------- #
PUBLISHED_AMBIGUOUS = {"m0r0": 55, "re86": 19, "sk48": 11, "ka59": 10,
                       "cd82": 8, "g50t": 4, "dc22": 3, "wa30": 2}


def replay_tests() -> None:
    print("R: offline replay over the three recorded runs")
    sys.path.insert(0, str(HERE))
    import p1_replay_validate as V

    traces = V.load_traces()

    # -- R1 detector certification
    det = {}
    for meta, acts in traces[V.RUNS[0]]:
        if acts is None:
            continue
        seen: dict[tuple[str, str], str] = {}
        n = 0
        for a in acts:
            key = (a["prev"], a["act"])
            if key in seen and seen[key] != a["out"]:
                n += 1
            seen.setdefault(key, a["out"])
        if n:
            det[meta["game_id"][:4]] = n
    check("R1 the ONLINE detector reproduces the published latent-state set "
          "EXACTLY (8 games, exact pair counts) and flags none of the other 17",
          det == PUBLISHED_AMBIGUOUS, json.dumps(det, sort_keys=True))

    # -- R2..R5 shipped defaults
    rows = {}
    for run in V.RUNS:
        rows[run] = V.score_arm(traces, run, "shipped")
    for run, r in rows.items():
        tag = run.split("/")[-1]
        check(f"R2 [{tag}] ZERO level-completing actions declined or aborted "
              f"(the hard safety canary)", r["lc_lost"] == 0, str(r["lc_lost"]))
        check(f"R3 [{tag}] ZERO board-changing actions declined "
              f"(memo_mode=noop is board-equivalent to executing)",
              r["diverged"] == 0, str(r["diverged"]))
        check(f"R4 [{tag}] duplicate re-execution rate falls "
              f"({r['dup_rate_before']*100:.2f}% -> {r['dup_rate_after']*100:.2f}%)",
              r["dup_rate_after"] < r["dup_rate_before"] or r["dup_rate_before"] == 0)
        check(f"R4b [{tag}] the suppressor actually engages",
              (r["declined"] + r["aborted"]) > 0,
              f"declined={r['declined']} aborted={r['aborted']}")
        check(f"R5 [{tag}] score does not regress "
              f"({r['as_run']:.4f} -> {r['p1']:.4f}, x{r['multiplier']:.4f})",
              r["p1"] >= r["as_run"] - 1e-9)

    # -- R5b levels preserved on every latent-state game
    lat_ok = True
    detail = []
    for meta, acts in traces[V.RUNS[0]]:
        if acts is None or meta["game_id"][:4] not in PUBLISHED_AMBIGUOUS:
            continue
        res = V.simulate(acts, meta["levels_completed"])
        lost = sum(s["lc_lost"] for s in res["per_level"].values())
        if lost:
            lat_ok = False
            detail.append(f"{meta['game_id'][:4]}:{lost}")
    check("R5b levels_completed preserved on ALL 8 latent-state games "
          "(no level-completing action touched)", lat_ok, ",".join(detail))

    # -- R6 the published arithmetic, and why the defaults are what they are
    pub = V.published_arithmetic(traces)
    by_run = {r["run"]: r for r in pub}
    a = by_run[V.RUNS[0]]
    check("R6 the published M1+M3 arithmetic reproduces "
          "(1.6352 as-run, x1.09-1.11 across three runs)",
          abs(a["as_run"] - 1.6352194) < 1e-6
          and all(1.085 <= r["multiplier"] <= 1.12 for r in pub),
          " ".join(f"{r['run'].split('/')[-1]}:x{r['multiplier']:.4f}" for r in pub))
    check("R6b ...and it DELETES 3 level-completing actions on animation_v1 -- "
          "which is why abort_revisit and memo_mode=all ship OFF",
          a["lc_lost"] == 3, str(a["lc_lost"]))
    check("R6c the diagnosis's duplicate bucket reproduces exactly (117)",
          a["dup"] == 117, str(a["dup"]))


# --------------------------------------------------------------------------- #
# I: integration against the REAL offline engine
# --------------------------------------------------------------------------- #
class _StubSession:
    """Minimal ``_HarnessGameSession`` stand-in: exactly the attributes the
    VANILLA ``_execute_action`` / ``step_env`` touch. The game is a REAL taaf
    ``GameAPI`` on a REAL offline arcengine env."""

    def __init__(self, game) -> None:
        self.game = game
        self.analyzer = SimpleNamespace(total_tokens=0)
        self.token_baseline = 0
        self.history_entries: list = []
        self.last_engine_action = None
        self.viewer_events: list = []

    @property
    def action_count(self) -> int:
        return len(self.history_entries)

    def write_runtime_state(self) -> None:
        return None

    def timing_payload(self) -> dict:
        return {"run_elapsed_seconds": 0.0, "time_remaining_seconds": None}

    def _append_action_viewer_event(self, payload, frame) -> None:
        self.viewer_events.append(payload)

    def write_viewer_payload(self) -> None:
        return None

    def should_stop(self) -> bool:
        return False


def _make_game(env_name: str):
    import taaf.game as taaf_game
    import taaf.game_api as game_api

    spec = game_api.ArcadeSpec(environments_dir=str(ENV_FILES))
    game = game_api.GameAPI(env_name=env_name, arcade_spec=spec)
    session = taaf_game.RunSession()
    game.start_game(session)
    return game, session


def integration(warkit: Path) -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    structural(nb)

    src = warkit / "p1_suppressor_patch.py"
    assert src.is_file(), f"no p1_suppressor_patch.py under {warkit}"
    print(f"module source under test: {src}")

    tmp_root = Path(tempfile.mkdtemp(prefix="p1smoke-"))
    run_dir = tmp_root / "run"
    run_dir.mkdir(parents=True)
    shutil.copy(src, run_dir / "p1_suppressor_patch.py")
    shutil.copy(warkit / "continuation_patch.py", run_dir / "continuation_patch.py")

    cwd0 = Path.cwd()
    os.chdir(run_dir)
    try:
        ns: dict = {}
        print("I1: exec cell 2 (eval gate + arm flag)")
        out2 = exec_cell(cell_src(nb, 2), ns)
        check("I1 RUN_HEAVY forced True by the eval line",
              ns.get("RUN_HEAVY") is True and ns.get("FORCE_OFFLINE_BENCH") is True)
        check("I1b P1_SUPPRESS=1 stamped + banner printed",
              os.environ.get("P1_SUPPRESS") == "1" and "p1-eval: SEED=1" in out2)

        print("I2: exec cell 12 (P1 graft) against the runtime module copy")
        ns["bm"] = SimpleNamespace(label="p1-smoke")
        out12 = exec_cell(cell_src(nb, 12), ns)
        check("I2 P1 graft applied (banner printed)",
              "p1 v1: ACTIVE" in out12 and "graft applied" in out12)
        check("I2b banner states the safe defaults",
              "mode=noop" in out12 and "confirm=2" in out12
              and "revisit is DEFAULT OFF" in out12)
        check("I2c bm.label carries the P1 marker",
              "-p1-v1" in ns["bm"].label, ns["bm"].label)
        check("I2d NO fallback-to-vanilla traceback", "PATCH FAILED" not in out12)
        check("I2e graft did NOT import warpack/ledger/sentinel/compaction/animation",
              not {"warpack_patch", "ledger_patch", "ledger_core",
                   "budget_sentinel_patch", "compaction_patch",
                   "animation_patch"} & set(sys.modules))
        check("I2f (f) continuation still applied alongside",
              "continuation v1" in out12)

        import p1_suppressor_patch as mod
        import inference.framework.solver as solver_mod
        from inference.agent.tool_agent import ToolAgent

        units(mod)

        print("I3: REAL offline engine -- mechanism A declines a confirmed no-op")
        import arcengine
        game, sess = _make_game("ft09")
        stub = _StubSession(game)
        stub._p1_state = mod.P1State("ft09-smoke")
        # ACTION6 at a fixed dead coordinate: on ft09 a click in open space is
        # a no-op, so the same (board, action) pair is a confirmed no-op.
        act = arcengine.ActionInput(id=arcengine.GameAction.ACTION6,
                                    data={"x": 1, "y": 1})
        payloads = []
        for _ in range(4):
            payloads.append(solver_mod._HarnessGameSession._execute_action(
                stub, act, batch_index=1, batch_size=1, generated_tokens=0))
        st = stub._p1_state
        noop_run = all(p.get("board_changed") is False for p in payloads[:2])
        declined = [p for p in payloads if p.get("suppressed")]
        check("I3 the first two identical actions really are no-ops on this board",
              noop_run, str([p.get("board_changed") for p in payloads]))
        if noop_run:
            check("I3b the 3rd request is DECLINED (confirm=2) and not spent",
                  len(declined) >= 1 and declined[0]["executed"] is False,
                  f"{len(declined)} declined of 4")
            check("I3c the declined payload reports board_changed=False and "
                  "carries the note", declined
                  and declined[0]["board_changed"] is False
                  and "NO EFFECT" in declined[0]["p1_note"])
            check("I3d the 4th request EXECUTES (decline budget = 1 per pair), "
                  "so a path can never be permanently blocked",
                  len(declined) == 1, f"{len(declined)} declined")
        check("I3e zero exceptions in the patched action path",
              st.errors == 0, str(st.errors))
        game.finish_game()
        sess.close()

        print("I4: REAL offline engine -- mechanism B aborts a dead batch")
        # borrow the real (unpatched) helper methods the vanilla step_env needs
        for _name in ("_normalize_actions", "_error_payload", "_terminal_payload",
                      "_execute_action"):
            setattr(_StubSession, _name,
                    getattr(solver_mod._HarnessGameSession, _name))
        game2, sess2 = _make_game("ft09")
        stub2 = _StubSession(game2)
        stub2._p1_state = mod.P1State("ft09-batch")
        # MOUSE(row=1,col=1) is a verified no-op on this board (see I3), so the
        # batch is dead on its FIRST action and the other three are blind.
        got = solver_mod._HarnessGameSession.step_env(
            stub2, {"actions": [
                {"action": "MOUSE", "row": 1, "col": 1},
                {"action": "MOUSE", "row": 20, "col": 20},
                {"action": "MOUSE", "row": 30, "col": 30},
                {"action": "MOUSE", "row": 40, "col": 40},
            ]})
        check("I4 step_env returns a payload (no exception escaped)",
              isinstance(got, dict), str(type(got)))
        check("I4b the batch was cut after the first no-op "
              "(3 blind actions never fired)",
              isinstance(got, dict)
              and stub2._p1_state.aborted == 3
              and got.get("stop_reason") == "p1_batch_aborted"
              and got.get("stopped_early") is True
              and got.get("executed_count", 99) == 1,
              f"aborted={stub2._p1_state.aborted} "
              f"stop_reason={got.get('stop_reason')} "
              f"executed={got.get('executed_count')}")
        check("I4c the abort carries a model-facing note, not an error",
              bool(got.get("p1_note")) and "error" not in got,
              str(got.get("p1_note"))[:80])
        check("I4d zero exceptions in the batch path",
              stub2._p1_state.errors == 0, str(stub2._p1_state.errors))
        game2.finish_game()
        sess2.close()

        print("I5: ToolAgent seams (mechanism C + the compactor)")
        agent = object.__new__(ToolAgent)
        stc = mod.P1State("prompt")
        stc.record(("h1", "LEFT"), "h1", True)
        stc.last_hash = "h1"
        agent._p1_state = stc
        agent._last_step_summary = None
        agent._summarized_knowledge = {}
        text = ToolAgent._build_user_prompt(
            agent, 5, valid_actions=["UP", "DOWN", "LEFT", "RIGHT", "SPACE"],
            current_frame=None, history_entries=[], previous_step_summary=None)
        check("I5 the memory block is appended to the user prompt "
              "(rebuilt every turn -> never truncated)",
              "P1 memory (runner ground truth" in text)
        check("I5b the block names the confirmed-dead action and the untried ones",
              "CONFIRMED NO EFFECT" in text and "LEFT" in text
              and "NOT YET TRIED" in text)
        tail = text.split("P1 memory")[-1]
        check("I5c the block is a small bounded tail of the prompt",
              len(tail) < 900, f"{len(tail)} chars")
        vanilla_agent = object.__new__(ToolAgent)
        vanilla_agent._last_step_summary = None
        vanilla_agent._summarized_knowledge = {}
        text0 = ToolAgent._build_user_prompt(
            vanilla_agent, 5, valid_actions=["UP"], current_frame=None,
            history_entries=[], previous_step_summary=None)
        check("I5d an agent with no P1 state gets a byte-vanilla prompt",
              "P1 memory" not in text0)
        compact = ToolAgent._compact_action_result(
            object.__new__(ToolAgent),
            {"executed": False, "board_changed": False,
             "p1_note": "P1: not spent.", "p1_declined": 1})
        check("I5e the compactor carries the P1 note to the model",
              compact.get("p1_note") == "P1: not spent."
              and compact.get("p1_declined") == 1)

        print("I6: canary report")
        rep = mod.canary_report()
        check("I6 canary returns the shipped policy + counters",
              isinstance(rep, dict) and "dup_rate" in rep and "declined" in rep,
              json.dumps({k: rep[k] for k in ("games", "declined", "aborted")}))
    finally:
        os.chdir(cwd0)
        shutil.rmtree(tmp_root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# K: flag gate + kill switch (subprocess)
# --------------------------------------------------------------------------- #
def run_gate_child(warkit: Path) -> None:
    sys.path.insert(0, str(warkit))
    import p1_suppressor_patch as mod
    import inference.framework.solver as solver_mod
    from inference.agent.tool_agent import ToolAgent

    before_exec = solver_mod._HarnessGameSession._execute_action
    before_step = solver_mod._HarnessGameSession.step_env
    before_play = solver_mod._HarnessGameSession.play
    before_prompt = ToolAgent._build_user_prompt
    bm = SimpleNamespace(label="gate-child")
    applied = mod.apply(bm)
    check("K1 apply() returns False", applied is False)
    check("K2 solver seams untouched (vanilla)",
          solver_mod._HarnessGameSession._execute_action is before_exec
          and solver_mod._HarnessGameSession.step_env is before_step
          and solver_mod._HarnessGameSession.play is before_play)
    check("K3 ToolAgent seam untouched (vanilla)",
          ToolAgent._build_user_prompt is before_prompt)
    check("K4 bm.label unstamped", bm.label == "gate-child")


def main() -> int:
    global PASS, FAIL
    args = sys.argv[1:]
    warkit = WARKIT_DEFAULT
    if "--warkit" in args:
        warkit = Path(args[args.index("--warkit") + 1])
    warkit = warkit.resolve()
    assert (warkit / "p1_suppressor_patch.py").is_file(), f"bad --warkit: {warkit}"

    if "--gate-child" in args:
        run_gate_child(warkit)
        print(f"\nGATE RESULT: {PASS} passed, {FAIL} failed")
        return 1 if FAIL else 0

    print(f"P1 suppressor smoke | nb={NB_PATH}")
    replay_tests()
    integration(warkit)

    for label, env_delta in (
        ("flag OFF (P1_SUPPRESS unset)", {"P1_SUPPRESS": None}),
        ("kill switch (P1_DISABLE=1)", {"P1_SUPPRESS": "1", "P1_DISABLE": "1"}),
    ):
        print(f"K: {label} (subprocess)")
        env = dict(os.environ)
        for k, v in env_delta.items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = v
        env.pop("WARPACK_FORCE_OFFLINE_BENCH", None)
        child = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--gate-child",
             "--warkit", str(warkit)],
            capture_output=True, text=True, env=env, timeout=900)
        for line in child.stdout.splitlines():
            if line.strip().startswith(("PASS", "FAIL", "GATE")):
                print(f"  {line.strip()}")
        PASS += child.stdout.count("  PASS")
        FAIL += child.stdout.count("  FAIL")
        if child.returncode not in (0, 1):
            FAIL += 1
            print(child.stdout[-2000:])
            print(child.stderr[-2000:])

    print(f"\nRESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
