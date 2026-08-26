"""Warpack CPU smoke test -- no GPU, no LLM.

Run from the repo root:
    .venv/Scripts/python.exe duck_eval/warpack/smoke_test.py

Covers (task spec):
  W0  env-driven config, VERSION marker, kill switches (master + per-flag)
  W1  shortcircuit: homogeneous no-op batch stops at first confirmed no-op
  W2  recovery: no-op lock-in and GAME_OVER-loop triggers clear the analyzer
      history and write a hypothesis graveyard
  W3  prune_trace unit semantics (no-ops, reset-undone segments, trailing)
  W4-6 BANKING on the real local engines: the war-room scripted policies
      (sb26/su15/lp85) clear L1+L2 through a real _HarnessGameSession; on
      finish, banking opens a NEW play on the same card and replays the
      pruned winning trace VERBATIM; engine scorecard shows 2 plays, both
      with levels_completed=2, replay using <= the recorded actions
  W7  fast-submit dry run: gate false -> dummy submission.parquet written
      locally in seconds (the Save-Version code path), correct schema
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUNDLE = REPO / "duck_eval" / "taaf_bundle" / "src"
ENV_FILES = REPO / "kaggle-data" / "environment_files"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BUNDLE / "ARC3-Inference"))
sys.path.insert(0, str(BUNDLE / "tufa-arc-agi-framework" / "src"))

# NOTE: do NOT pre-set ONLY_RESET_LEVELS here. GameAPI._start_game sets it to
# "true" AFTER arcade.make so the make-time RESET full-resets and registers
# play 1 on the engine scorecard (mirrors the competition gateway, which
# always registers the initial play server-side).
os.environ.pop("ONLY_RESET_LEVELS", None)
os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "smoke-test-model")
os.environ.setdefault("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:9/v1")

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


# ---------------------------------------------------------------- fakes
def _playing_state():
    import arcengine

    for name in ("NOT_FINISHED", "NOT_PLAYED", "NOT_STARTED"):
        if hasattr(arcengine.GameState, name):
            return getattr(arcengine.GameState, name)
    return next(m for m in arcengine.GameState if m.name not in ("WIN", "GAME_OVER"))


class FakeGame:
    """Minimal taaf.game.Game stand-in. ``mutate_board``: board changes on
    every action. ``game_over_every_action``: each non-RESET action lands in
    GAME_OVER until a RESET."""

    number_of_levels = 5

    def __init__(self, mutate_board=False, game_over_every_action=False):
        self._mutate = mutate_board
        self._go_every = game_over_every_action
        self._grid = [[0] * 8 for _ in range(8)]
        self._n = 0
        self._state = _playing_state()
        self.game_id = "fakegame"
        self.env = None
        self.game_run = SimpleNamespace(
            game_id="fakegame", history=[], state="playing",
            levels_completed=0, solver_note=None, final_score=None,
            solver_analysis_html=None,
        )

    @property
    def current_state(self):
        return SimpleNamespace(
            frame=SimpleNamespace(data=[row[:] for row in self._grid]),
            levels_completed=0,
            raw=SimpleNamespace(state=self._state),
            just_won_level=False,
            won=False,
            available_actions=[0, 1, 2, 3, 4],
        )

    def execute_action(self, action, generated_tokens=0, uncached_input_tokens=0):
        import arcengine

        if action.id == arcengine.GameAction.RESET:
            self._state = _playing_state()
        else:
            self._n += 1
            if self._mutate:
                self._grid[0][0] = self._n
            if self._go_every:
                self._state = arcengine.GameState.GAME_OVER
        self.game_run.history.append(action)
        return self.current_state

    def finish_game(self):
        self.game_run.final_score = 0.0


class FakeSolver:
    label = "warpack-smoke"
    job_dir = None
    max_actions_per_game = 400
    max_runtime_s_per_game = None

    def soft_time_remaining_seconds(self):
        return None


class FakeAgent:
    """Analyzer stand-in with the two attributes recovery mutates."""

    generated_tokens = 0
    _timeout = 5.0

    def __init__(self):
        self._history_messages = [{"role": "user", "content": "old"}] * 5
        self._summarized_knowledge = {"recent_findings": "prior note"}


def make_session(game, tmpdir: Path, analyzer=None):
    import inference.framework.solver as solver_mod

    return solver_mod._HarnessGameSession(
        solver=FakeSolver(),
        game=game,
        analyzer=analyzer or SimpleNamespace(generated_tokens=0, _timeout=5.0),
        game_index=0,
        pass_index=0,
        state_path=tmpdir / "runtime_state.json",
        transcript_path=tmpdir / "transcript.txt",
        analysis_html_relpath="solver_analysis/smoke.html",
        stop_event=threading.Event(),
        viewer_data_path=tmpdir / "viewer_data.json",
    )


# ---------------------------------------------------------------- W0
def w0_config() -> None:
    print("W0: env-driven config + kill switches")
    import warpack_patch
    import inference.framework.solver as solver_mod

    check("VERSION marker == v1", warpack_patch.VERSION == "v1", warpack_patch.VERSION)
    os.environ["WARPACK_RECOVERY_REPEATS"] = "7"
    os.environ["WARPACK_BANK_MIN_TIME"] = "33"
    cfg = warpack_patch.WarpackConfig()
    check("env ints/floats honored",
          cfg.recovery_repeat_threshold == 7 and cfg.bank_min_time_s == 33.0,
          f"{cfg.recovery_repeat_threshold}/{cfg.bank_min_time_s}")
    os.environ.pop("WARPACK_RECOVERY_REPEATS")
    os.environ.pop("WARPACK_BANK_MIN_TIME")
    os.environ["WARPACK_BANKING"] = "0"
    check("per-flag kill switch read", warpack_patch.WarpackConfig().enable_banking is False)
    os.environ.pop("WARPACK_BANKING")

    orig_exec = solver_mod._HarnessGameSession._execute_action
    orig_finish = solver_mod._HarnessGameSession._finish_if_needed
    os.environ["WARPACK_ENABLE"] = "0"
    try:
        out = warpack_patch.apply()
        check("master kill switch: apply() returns config", out is not None and out.enable is False)
        check("master kill switch: _execute_action NOT patched",
              solver_mod._HarnessGameSession._execute_action is orig_exec)
        check("master kill switch: _finish_if_needed NOT patched",
              solver_mod._HarnessGameSession._finish_if_needed is orig_finish)
        check("master kill switch: _APPLIED still False", warpack_patch._APPLIED is False)
    finally:
        os.environ["WARPACK_ENABLE"] = "1"

    cfg = warpack_patch.apply()
    check("apply installs patches",
          solver_mod._HarnessGameSession._execute_action.__name__ == "wp_execute_action"
          and solver_mod._HarnessGameSession._finish_if_needed.__name__ == "wp_finish_if_needed")
    check("apply is idempotent", warpack_patch.apply() is not None and warpack_patch._APPLIED)
    import inference.agent.tool_agent as tool_agent_mod

    check("ToolAgent.analyze untouched",
          tool_agent_mod.ToolAgent.analyze.__name__ == "analyze")
    check("ToolAgent._build_user_prompt untouched",
          tool_agent_mod.ToolAgent._build_user_prompt.__name__ == "_build_user_prompt")


# ---------------------------------------------------------------- W1
def w1_shortcircuit() -> None:
    print("W1: shortcircuit stops homogeneous no-op batches")
    with tempfile.TemporaryDirectory() as td:
        game = FakeGame(mutate_board=False)
        session = make_session(game, Path(td))
        payload = session.step_env({"actions": [{"action": "UP"}] * 5})
        check("homogeneous no-op batch stopped after 1 execution",
              payload.get("executed_count") == 1 and len(game.game_run.history) == 1,
              f"executed={payload.get('executed_count')} n={len(game.game_run.history)}")
        check("shortcircuit event recorded",
              any(e[0] == "shortcircuit" for e in session._wp_state.events))
    with tempfile.TemporaryDirectory() as td:
        game = FakeGame(mutate_board=True)
        session = make_session(game, Path(td))
        payload = session.step_env({"actions": [{"action": "UP"}] * 5})
        check("board-changing homogeneous batch NOT shortcircuited",
              payload.get("executed_count") == 5, str(payload.get("executed_count")))
    with tempfile.TemporaryDirectory() as td:
        game = FakeGame(mutate_board=False)
        session = make_session(game, Path(td))
        payload = session.step_env(
            {"actions": [{"action": "UP"}, {"action": "DOWN"}, {"action": "UP"}]})
        check("heterogeneous no-op batch NOT shortcircuited",
              payload.get("executed_count") == 3, str(payload.get("executed_count")))


# ---------------------------------------------------------------- W2
def w2_recovery() -> None:
    print("W2: recovery refresh on lock-in and GAME_OVER loops")
    import warpack_patch

    os.environ["WARPACK_RECOVERY_REPEATS"] = "10"
    try:
        with tempfile.TemporaryDirectory() as td:
            agent = FakeAgent()
            game = FakeGame(mutate_board=False)
            session = make_session(game, Path(td), analyzer=agent)
            cfg = warpack_patch.WarpackConfig()
            # runtime config is captured at apply() time; drive via the
            # module-applied default (30) instead: send 30 identical no-ops.
            for _ in range(30):
                session.step_env({"action": "UP"})
            st = session._wp_state
            check("lock-in refresh fired",
                  any(e[0] == "refresh" for e in st.events), str(st.events[-3:]))
            check("chat history cleared", agent._history_messages == [])
            check("hypothesis graveyard written",
                  "WARPACK RECOVERY" in agent._summarized_knowledge.get("recent_findings", "")
                  and "prior note" in agent._summarized_knowledge["recent_findings"],
                  agent._summarized_knowledge.get("recent_findings", "")[:80])
    finally:
        os.environ.pop("WARPACK_RECOVERY_REPEATS")

    with tempfile.TemporaryDirectory() as td:
        agent = FakeAgent()
        game = FakeGame(mutate_board=True, game_over_every_action=True)
        session = make_session(game, Path(td), analyzer=agent)
        for _ in range(3):
            session.step_env({"action": "UP"})
            session._execute_auto_reset()
        st = session._wp_state
        check("GAME_OVER-loop refresh fired after 3 gameovers",
              any(e[0] == "refresh" and "GAME_OVER" in e[1] for e in st.events),
              str(st.events))
        check("gameover-loop history cleared", agent._history_messages == [])


# ---------------------------------------------------------------- W3
def w3_prune() -> None:
    print("W3: prune_trace unit semantics")
    from warpack_patch import TraceStep, prune_trace

    def step(name, changed, completed, lc, data=None):
        return TraceStep(name=name, data=data or {}, board_changed=changed,
                         level_completed=completed, lc_after=lc, grid_hash=None,
                         state_name="NOT_FINISHED")

    trace = [
        step("ACTION6", True, False, 0, {"x": 1, "y": 1}),   # kept
        step("ACTION5", False, False, 0),                     # no-op -> dropped
        step("ACTION6", True, False, 0, {"x": 2, "y": 2}),   # undone by RESET
        step("RESET", True, False, 0),                        # never replayed
        step("ACTION6", True, True, 1, {"x": 3, "y": 3}),    # completes L1
        step("ACTION1", False, False, 1),                     # no-op -> dropped
        step("ACTION6", True, True, 2, {"x": 4, "y": 4}),    # completes L2
        step("ACTION2", True, False, 2),                      # trailing -> dropped
    ]
    pruned = prune_trace(trace)
    got = [(s.name, s.data) for s in pruned]
    # NOTE: actions before a RESET are undone (level reset) so the whole
    # pre-RESET buffer is dropped; the winning segment is replayed alone.
    want = [("ACTION6", {"x": 3, "y": 3}), ("ACTION6", {"x": 4, "y": 4})]
    check("prune drops no-ops, reset-undone segments, trailing", got == want, str(got))
    check("empty and lc=0 traces prune to empty",
          prune_trace([]) == [] and prune_trace([step("ACTION1", True, False, 0)]) == [])


# ---------------------------------------------------------------- W4-6 banking on real engines
def _run_banked_game(prefix: str, policy_cls) -> None:
    import arc_agi
    from taaf.game_api import ArcadeSpec, GameAPI

    # A previous GameAPI in this process set ONLY_RESET_LEVELS=true; clear it
    # so THIS game's make-time RESET full-resets and registers play 1 (as the
    # competition gateway does server-side for every fresh env).
    os.environ.pop("ONLY_RESET_LEVELS", None)
    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.OFFLINE,
        environments_dir=str(ENV_FILES),
    )
    game_id = next(e.game_id for e in arcade.available_environments
                   if e.game_id.startswith(prefix))
    game = GameAPI(
        env_name=game_id,
        arcade_spec=ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE,
                               environments_dir=str(ENV_FILES)),
    )
    game.start_game()
    # Keep the engine scorecard open after play() so the test can inspect the
    # plays; reconciliation is not under test here.
    game._finish_game = lambda: None

    from warpack_patch import prune_trace

    with tempfile.TemporaryDirectory() as td:
        session = make_session(game, Path(td))
        policy = policy_cls(game, session.stop_event)
        session.analyzer = policy
        t0 = time.time()
        session.play()
        st = getattr(session, "_wp_state", None)
        run = game.game_run
        check(f"({prefix}) scripted policy cleared L1+L2 through the harness",
              run.levels_completed == 2,
              f"lc={run.levels_completed} turns={policy.turns}")
        check(f"({prefix}) trace recorded", st is not None and len(st.trace) > 0,
              "no _wp_state")
        if st is None:
            return
        pruned = prune_trace(st.trace)
        bank_events = [e for e in st.events if e[0] == "bank"]
        aborts = [e for e in st.events if e[0].startswith("bank_") ]
        check(f"({prefix}) banking fired and completed",
              bank_events == [("bank", len(pruned), 2)],
              f"events={st.events}")
        replayed = getattr(st, "replayed", [])
        check(f"({prefix}) replay is VERBATIM the pruned winning trace",
              replayed == [(s.name, dict(s.data)) for s in pruned],
              f"replayed={len(replayed)} pruned={len(pruned)} aborts={aborts}")
        # Engine-scorecard verification: 2 plays on the card, both at 2 levels,
        # replay play used <= the recorded play's actions (score = max of plays).
        engine_game_id = game.env.environment_info.game_id
        sm = game._arcade.scorecard_manager
        sc = sm.scorecards.get(game._scorecard_id)
        card = sc.cards.get(engine_game_id) if sc is not None else None
        check(f"({prefix}) engine card shows a NEW play (total_plays == 2)",
              card is not None and card.total_plays == 2,
              f"plays={getattr(card, 'total_plays', None)}")
        if card is not None:
            check(f"({prefix}) both plays completed 2 levels",
                  card.levels_completed == [2, 2], str(card.levels_completed))
            check(f"({prefix}) banked play used <= recorded actions "
                  f"(pruned replay, no extra RESET cost)",
                  card.actions[1] <= card.actions[0],
                  f"actions={card.actions} resets={card.resets}")
        print(f"    [{prefix}] {len(st.trace)} recorded / {len(pruned)} replayed "
              f"actions, plays={getattr(card, 'total_plays', '?')} "
              f"actions/play={getattr(card, 'actions', '?')} "
              f"({time.time() - t0:.1f}s)")


def w4_w6_banking() -> None:
    from policies import Lp85Policy, Sb26Policy, Su15Policy

    for prefix, cls in (("sb26", Sb26Policy), ("su15", Su15Policy), ("lp85", Lp85Policy)):
        print(f"W-bank: {prefix} winning-trace banking on the real local engine")
        try:
            _run_banked_game(prefix, cls)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            check(f"({prefix}) banking end-to-end", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------- W7
def w7_fastsubmit_dry_run() -> None:
    print("W7: fast-submit dry run (gate false -> dummy parquet)")
    import fastsubmit_cells

    with tempfile.TemporaryDirectory() as td:
        working = Path(td)
        scope: dict = {}
        t0 = time.time()
        # Exactly the code the notebook runs during an interactive Save Version.
        exec(fastsubmit_cells.FAST_PATH_SNIPPET, scope)
        scope["_write_dummy_submission"](working)
        elapsed = time.time() - t0
        out = working / "submission.parquet"
        check("submission.parquet written", out.is_file())
        check("fast path completes in seconds", elapsed < 30, f"{elapsed:.1f}s")
        import pandas as pd

        df = pd.read_parquet(out)
        check("parquet schema matches vanilla offline path",
              list(df.columns) == ["row_id", "game_id", "end_of_game", "score"]
              and len(df) == 1, str(list(df.columns)))
    # Gate logic sanity: mirrors cell 2.
    env = {"KAGGLE_IS_COMPETITION_RERUN": ""}
    true_submission = env.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
    force = env.get("WARPACK_FORCE_OFFLINE_BENCH", "").strip().lower() in {"1", "true"}
    check("gate false during Save Version (no rerun env)", (true_submission or force) is False)
    env = {"KAGGLE_IS_COMPETITION_RERUN": "true"}
    true_submission = env.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
    check("gate true inside a competition rerun", true_submission is True)


def main() -> int:
    print(f"warpack smoke test | repo={REPO}")
    w0_config()
    w1_shortcircuit()
    w2_recovery()
    w3_prune()
    w4_w6_banking()
    w7_fastsubmit_dry_run()
    print(f"\nRESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
