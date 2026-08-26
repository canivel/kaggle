"""Attempt-scheduler CPU smoke test -- no GPU, no LLM.

Run from the repo root:
    uv run python duck_eval/scheduler/smoke_test.py

Covers (task spec):
  (a) restart fires exactly at 90 actions with 0 levels
  (b) cap 2 restarts then park (bounded ~272 actions, no thrash)
  (c) no restart when levels_completed >= 1 (from start, and mid-run level-up)
  (d) kill switch SCHED_ENABLE=0 is inert (no patches installed)
  (e) taaf-bundle patch application works (real solver module + real local
      engine end-to-end play(), like phase1's T5)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import traceback
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]  # f:/kaggle/arc-prize-2026
BUNDLE = REPO / "duck_eval" / "taaf_bundle" / "src"
ENV_FILES = REPO / "kaggle-data" / "environment_files"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BUNDLE / "ARC3-Inference"))
sys.path.insert(0, str(BUNDLE / "tufa-arc-agi-framework" / "src"))

os.environ.setdefault("ONLY_RESET_LEVELS", "true")
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
def _playing_game_state_enum():
    """A non-WIN, non-GAME_OVER arcengine.GameState member."""
    import arcengine

    for name in ("NOT_FINISHED", "NOT_PLAYED", "NOT_STARTED"):
        if hasattr(arcengine.GameState, name):
            return getattr(arcengine.GameState, name)
    return next(
        m for m in arcengine.GameState if m.name not in ("WIN", "GAME_OVER")
    )


class FakeGame:
    """Minimal stand-in for taaf.game.Game driven by _HarnessGameSession.

    ``level_up_at``: levels_completed becomes 1 once this many non-RESET
    actions have executed (None = never).
    """

    number_of_levels = 5

    def __init__(self, levels_completed: int = 0, level_up_at: int | None = None):
        self._initial_lc = levels_completed
        self._level_up_at = level_up_at
        self._moves = 0
        self._grid = [[0] * 8 for _ in range(8)]
        self.game_run = SimpleNamespace(
            game_id="fakegame",
            history=[],
            state="playing",
            levels_completed=levels_completed,
            solver_note=None,
            final_score=None,
            solver_analysis_html=None,
        )
        self._playing = _playing_game_state_enum()

    def _lc(self) -> int:
        if self._level_up_at is not None and self._moves >= self._level_up_at:
            return max(self._initial_lc, 1)
        return self._initial_lc

    @property
    def current_state(self):
        lc = self._lc()
        return SimpleNamespace(
            frame=SimpleNamespace(data=[row[:] for row in self._grid]),
            levels_completed=lc,
            raw=SimpleNamespace(state=self._playing),
            just_won_level=False,
            won=False,
            available_actions=[0, 1, 2, 3, 4],
        )

    def execute_action(self, action, generated_tokens=0, uncached_input_tokens=0):
        import arcengine

        if action.id != arcengine.GameAction.RESET:
            self._moves += 1
        self.game_run.history.append(action)
        self.game_run.levels_completed = max(
            self.game_run.levels_completed, self._lc()
        )
        return self.current_state


class FakeSolver:
    label = "sched-smoke"
    job_dir = None
    max_actions_per_game = 400
    max_runtime_s_per_game = None

    def soft_time_remaining_seconds(self):
        return None


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


def reset_indices(game) -> list[int]:
    import arcengine

    return [
        i
        for i, rec in enumerate(game.game_run.history)
        if (rec.action.id if hasattr(rec, "action") else rec.id)
        == arcengine.GameAction.RESET
    ]


def drive(session, n: int) -> None:
    for _ in range(n):
        session.step_env({"action": "UP"})


# ---------------------------------------------------------------- S0 config
def s0_config() -> None:
    print("S0: env-driven config")
    import scheduler_patch

    check("VERSION marker == v1", scheduler_patch.VERSION == "v1",
          str(scheduler_patch.VERSION))
    os.environ["SCHED_RESTART_AT"] = "90"
    os.environ["SCHED_MAX_RESTARTS"] = "2"
    cfg = scheduler_patch.SchedulerConfig()
    check("SCHED_RESTART_AT=90 honored", cfg.restart_at == 90, str(cfg.restart_at))
    check("SCHED_MAX_RESTARTS=2 honored", cfg.max_restarts == 2, str(cfg.max_restarts))
    check("enabled by default", cfg.enable is True)
    os.environ["SCHED_RESTART_AT"] = "37"
    os.environ["SCHED_MAX_RESTARTS"] = "1"
    cfg2 = scheduler_patch.SchedulerConfig()
    check("custom env values honored", cfg2.restart_at == 37 and cfg2.max_restarts == 1,
          f"{cfg2.restart_at}/{cfg2.max_restarts}")
    os.environ["SCHED_RESTART_AT"] = "90"
    os.environ["SCHED_MAX_RESTARTS"] = "2"


# ---------------------------------------------------------------- S1 kill switch
def s1_kill_switch() -> None:
    print("S1: kill switch SCHED_ENABLE=0 is inert (d)")
    import scheduler_patch
    import inference.framework.solver as solver_mod

    orig_exec = solver_mod._HarnessGameSession._execute_action
    orig_stop = solver_mod._HarnessGameSession.should_stop

    os.environ["SCHED_ENABLE"] = "0"
    try:
        cfg = scheduler_patch.SchedulerConfig()
        check("(d) env kill switch read", cfg.enable is False)
        out = scheduler_patch.apply(config=cfg)
        check("(d) apply returns config", out is cfg)
        check(
            "(d) _execute_action NOT patched",
            solver_mod._HarnessGameSession._execute_action is orig_exec,
        )
        check(
            "(d) should_stop NOT patched",
            solver_mod._HarnessGameSession.should_stop is orig_stop,
        )
        check("(d) _APPLIED still False", scheduler_patch._APPLIED is False)
        # behavioral: 272 actions on a dead fake game -> zero resets, no park
        with tempfile.TemporaryDirectory() as td:
            game = FakeGame(levels_completed=0)
            session = make_session(game, Path(td))
            drive(session, 272)
            check(
                "(d) no RESET injected with kill switch",
                reset_indices(game) == [] and len(game.game_run.history) == 272,
                f"resets={reset_indices(game)} n={len(game.game_run.history)}",
            )
            check(
                "(d) no scheduler state attached",
                getattr(session, "_sched_state", None) is None,
            )
    finally:
        os.environ["SCHED_ENABLE"] = "1"


# ---------------------------------------------------------------- S2 apply
def s2_apply() -> None:
    print("S2: patch application on real taaf-bundle modules")
    import scheduler_patch
    import inference.framework.solver as solver_mod

    cfg = scheduler_patch.apply()
    check("apply returns enabled config", cfg.enable and cfg.restart_at == 90
          and cfg.max_restarts == 2, f"{cfg}")
    check(
        "_execute_action patched",
        solver_mod._HarnessGameSession._execute_action.__name__
        == "sched_execute_action",
        solver_mod._HarnessGameSession._execute_action.__name__,
    )
    check(
        "should_stop patched",
        solver_mod._HarnessGameSession.should_stop.__name__ == "sched_should_stop",
        solver_mod._HarnessGameSession.should_stop.__name__,
    )
    check("apply is idempotent", scheduler_patch.apply() is not None
          and scheduler_patch._APPLIED is True)
    # prompt/context path untouched (NO context injection)
    import inference.agent.tool_agent as tool_agent_mod

    check(
        "ToolAgent.analyze untouched",
        tool_agent_mod.ToolAgent.analyze.__name__ == "analyze",
        tool_agent_mod.ToolAgent.analyze.__name__,
    )
    check(
        "ToolAgent._build_user_prompt untouched",
        tool_agent_mod.ToolAgent._build_user_prompt.__name__ == "_build_user_prompt",
        tool_agent_mod.ToolAgent._build_user_prompt.__name__,
    )


# ---------------------------------------------------------------- S3 semantics
def s3_semantics() -> None:
    print("S3: restart/cap/park semantics on a dead fake game (a)(b)")
    with tempfile.TemporaryDirectory() as td:
        game = FakeGame(levels_completed=0)
        session = make_session(game, Path(td))
        # drive well past the park point; parked session must stop absorbing
        for _ in range(300):
            session.step_env({"action": "UP"})
        resets = reset_indices(game)
        st = getattr(session, "_sched_state", None)
        n = len(game.game_run.history)
        check("(a) first restart exactly at action 90 (history index 90)",
              resets[:1] == [90], str(resets))
        check("(a) no RESET before action 90",
              all(i >= 90 for i in resets), str(resets))
        check("(b) second restart at action 181 (90 per-attempt after reset)",
              resets == [90, 181], str(resets))
        check("(b) cumulative restarts == 2 (cap)",
              st is not None and st.restarts_done == 2,
              str(getattr(st, "restarts_done", None)))
        check("(b) parked after cap", st is not None and st.parked is True)
        check("(b) bounded at 272 actions (90+1+90+1+90), no thrash",
              n == 272, str(n))
        check("(b) parked session should_stop() is True", session.should_stop())
        check("(b) park recorded at per-attempt 90",
              st is not None and ("park", 272, 90) in st.events, str(st.events))
        check("(b) restart events at per-attempt exactly 90",
              st is not None
              and [e for e in st.events if e[0] == "restart"]
              == [("restart", 91, 90), ("restart", 182, 90)],
              str(st.events))
        payload = session.step_env({"action": "UP"})
        check("(b) post-park step_env refuses to execute",
              payload.get("executed") is False, json.dumps(payload)[:200])


def s3c_level_progress() -> None:
    print("S3c: no restart when levels_completed >= 1 (c)")
    with tempfile.TemporaryDirectory() as td:
        game = FakeGame(levels_completed=1)
        session = make_session(game, Path(td))
        drive(session, 250)
        st = getattr(session, "_sched_state", None)
        check("(c) lc=1 from start: no RESET injected",
              reset_indices(game) == [] and len(game.game_run.history) == 250,
              f"resets={reset_indices(game)}")
        check("(c) lc=1 from start: not parked, 0 restarts",
              st is not None and st.restarts_done == 0 and not st.parked)
    with tempfile.TemporaryDirectory() as td:
        game = FakeGame(levels_completed=0, level_up_at=50)
        session = make_session(game, Path(td))
        drive(session, 250)
        st = getattr(session, "_sched_state", None)
        check("(c) level-up at action 50: no RESET at 90 or ever",
              reset_indices(game) == [], str(reset_indices(game)))
        check("(c) level-up at action 50: not parked, 0 restarts",
              st is not None and st.restarts_done == 0 and not st.parked)


# ---------------------------------------------------------------- S4 real engine
class ScriptedAnalyzer:
    """Analyzer stand-in: one env action per analyze() turn, no LLM."""

    generated_tokens = 0
    _timeout = 5.0

    def __init__(self):
        self.turns = 0

    def analyze(self, state_path, action_num, valid_actions=None, step_env=None,
                **kwargs):
        from inference.agent.action_names import to_model_action

        self.turns += 1
        payload = {}
        for name in list(valid_actions or []) or ["ACTION1"]:
            request = {"action": to_model_action(name)}
            if to_model_action(name) == "MOUSE":
                request.update({"row": 32, "col": 32})
            payload = step_env(request)
            if payload.get("executed"):
                break
        return SimpleNamespace(
            retryable_failure=False,
            yielded_control=False,
            step_executed=bool(payload.get("executed")),
        )


def s4_real_engine() -> None:
    print("S4: end-to-end play() on a real local engine (e)")
    try:
        import arc_agi
        from taaf.game_api import ArcadeSpec, GameAPI

        arcade = arc_agi.Arcade(
            operation_mode=arc_agi.OperationMode.OFFLINE,
            environments_dir=str(ENV_FILES),
        )
        game_id = next(
            e.game_id
            for e in arcade.available_environments
            if e.game_id.startswith("ls20")
        )
        game = GameAPI(
            env_name=game_id,
            arcade_spec=ArcadeSpec(
                operation_mode=arc_agi.OperationMode.OFFLINE,
                environments_dir=str(ENV_FILES),
            ),
        )
        game.start_game()
        with tempfile.TemporaryDirectory() as td:
            session = make_session(game, Path(td), analyzer=ScriptedAnalyzer())
            session.play()
            run = game.game_run
            st = getattr(session, "_sched_state", None)
            restarts = [e for e in st.events if e[0] == "restart"] if st else []
            parks = [e for e in st.events if e[0] == "park"] if st else []
            check("(e) real session drove the engine",
                  len(run.history) > 0, str(len(run.history)))
            check("(e) game stayed at lc=0 (dead-game fixture)",
                  run.levels_completed == 0, str(run.levels_completed))
            check("(e) 2 restarts fired on real engine",
                  st is not None and st.restarts_done == 2,
                  str(getattr(st, "restarts_done", None)))
            check("(e) every restart at per-attempt exactly 90",
                  bool(restarts) and all(e[2] == 90 for e in restarts),
                  str(restarts))
            check("(e) parked after cap at per-attempt 90",
                  st is not None and st.parked and len(parks) == 1
                  and parks[0][2] == 90, str(parks))
            check("(e) bounded run: total actions == 272 (< wall)",
                  len(run.history) == 272, str(len(run.history)))
            check("(e) RESET rows in real history at 90 and 181",
                  reset_indices(game) == [90, 181], str(reset_indices(game)))
            check("(e) run finished (not crashed)",
                  run.state in ("gave_up", "cancelled") and run.final_score
                  is not None, f"state={run.state} score={run.final_score}")
            check("(e) analyzer turns bounded (park stops analyzer spend)",
                  ScriptedAnalyzer is not None and session.analysis_step <= 275,
                  str(session.analysis_step))
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        check("(e) real-engine end-to-end", False, f"{type(exc).__name__}: {exc}")


def main() -> int:
    print(f"scheduler smoke test | repo={REPO}")
    s0_config()
    s1_kill_switch()
    s2_apply()
    s3_semantics()
    s3c_level_progress()
    s4_real_engine()
    print(f"\nRESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
