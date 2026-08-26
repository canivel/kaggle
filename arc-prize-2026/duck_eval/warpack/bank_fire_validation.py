"""Engineered banking-fire validation (prereg amendment 2026-07-15 §A2).

Panel R11 (methodology major 2 / llm-agents N2): banking's replay mechanism has
never been observed firing outside scripted smoke tests, and the war-eval run
produced ZERO replay events (hypothesis: every run ended at its time budget, so
``_bank``'s soft-time check skipped). This script:

  1. Replays the war-eval run's OWN recorded action histories (ar25, sc25,
     m0r0, s5i5 — the games where warpack demonstrably won levels) through a
     real ``_HarnessGameSession`` on the real local engines, with unlimited
     soft time -> banking must fire: ``bank`` event, verbatim pruned replay,
     engine card shows a NEW play with actions[replay] <= actions[recorded].
  2. Reproduces the war-eval zero: same playback with a solver whose
     ``soft_time_remaining_seconds()`` returns 30 (< bank_min_time 120)
     -> expect ``("bank_skip", "time", 30)`` and NO replay.

Run from the repo root:
    .venv/Scripts/python.exe duck_eval/warpack/bank_fire_validation.py
Output: runs/war_eval_v1/bank_fire_validation.json
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import smoke_test  # noqa: E402  (sets sys.path for taaf/inference, env vars)
from smoke_test import ENV_FILES, FakeSolver, make_session  # noqa: E402

BENCH = REPO / "runs" / "kernel_pulls" / "war_eval_v1" / "benchmark.json"
OUT = REPO / "runs" / "war_eval_v1" / "bank_fire_validation.json"
GAMES = ("ar25", "sc25", "m0r0", "s5i5")


class PlaybackAnalyzer:
    """Feed the war-eval run's recorded actions through step_env verbatim."""

    generated_tokens = 0
    _timeout = 5.0

    def __init__(self, game, stop_event, actions, target_lc):
        self.game = game
        self.stop_event = stop_event
        self.actions = list(actions)
        self.i = 0
        self.target_lc = target_lc

    def analyze(self, state_path, action_num, valid_actions=None, step_env=None, **kw):
        if (int(self.game.current_state.levels_completed) >= self.target_lc
                or self.i >= len(self.actions)):
            self.stop_event.set()
            return SimpleNamespace(retryable_failure=False, yielded_control=False,
                                   step_executed=False)
        act = self.actions[self.i]
        self.i += 1
        req = {"action": act["id"]}
        if act["id"] == "ACTION6":
            req["row"] = act["data"]["y"]
            req["col"] = act["data"]["x"]
        payload = step_env(req)
        return SimpleNamespace(retryable_failure=False, yielded_control=False,
                               step_executed=bool(payload and payload.get("executed")))


class StarvedSolver(FakeSolver):
    def soft_time_remaining_seconds(self):
        return 30.0


def run_case(prefix: str, history, target_lc: int, solver=None) -> dict:
    import os

    import arc_agi
    from taaf.game_api import ArcadeSpec, GameAPI
    from warpack_patch import prune_trace

    os.environ.pop("ONLY_RESET_LEVELS", None)
    game = GameAPI(
        env_name=next(
            e.game_id for e in arc_agi.Arcade(
                operation_mode=arc_agi.OperationMode.OFFLINE,
                environments_dir=str(ENV_FILES)).available_environments
            if e.game_id.startswith(prefix)),
        arcade_spec=ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE,
                               environments_dir=str(ENV_FILES)),
    )
    game.start_game()
    game._finish_game = lambda: None  # keep engine scorecard open to inspect

    actions = [a["action"] for a in history]
    with tempfile.TemporaryDirectory() as td:
        session = make_session(game, Path(td))
        if solver is not None:
            session.solver = solver
        policy = PlaybackAnalyzer(game, session.stop_event, actions, target_lc)
        session.analyzer = policy
        t0 = time.time()
        session.play()
        st = getattr(session, "_wp_state", None)
        run = game.game_run
        out = {
            "game": prefix,
            "recorded_actions_fed": policy.i,
            "lc_reached": run.levels_completed,
            "target_lc": target_lc,
            "events": [list(e) for e in (st.events if st else [])
                       if str(e[0]).startswith("bank")],
            "replay_attempted": bool(st and any(e[0] == "bank" or
                                                str(e[0]).startswith("bank_abort")
                                                for e in st.events)),
            "replay_succeeded": bool(st and any(e[0] == "bank" for e in st.events)),
            "wallclock_s": round(time.time() - t0, 1),
        }
        if st is not None and out["replay_succeeded"]:
            pruned = prune_trace(st.trace)
            replayed = getattr(st, "replayed", [])
            out["replay_verbatim"] = replayed == [(s.name, dict(s.data)) for s in pruned]
            eng_id = game.env.environment_info.game_id
            sc = game._arcade.scorecard_manager.scorecards.get(game._scorecard_id)
            card = sc.cards.get(eng_id) if sc else None
            if card is not None:
                out["card_total_plays"] = card.total_plays
                out["card_levels_per_play"] = list(card.levels_completed)
                out["card_actions_per_play"] = list(card.actions)
                out["score_invariant"] = (card.total_plays == 2
                                          and card.levels_completed[1] >= run.levels_completed
                                          and card.actions[1] <= card.actions[0])
        return out


def main() -> int:
    import warpack_patch

    cfg = warpack_patch.apply()
    assert cfg is not None and cfg.enable and cfg.enable_banking, \
        f"warpack not active: {cfg}"
    print(f"warpack applied: enable={cfg.enable} banking={cfg.enable_banking} "
          f"bank_min_time={cfg.bank_min_time_s}")

    bench = json.loads(BENCH.read_text(encoding="utf-8"))
    runs = {r["game_id"][:4]: r for r in bench["game_runs"]}
    results = {"purpose": "prereg amendment A2 engineered banking-fire validation",
               "cases": [], "starved_case": None}

    for prefix in GAMES:
        r = runs[prefix]
        print(f"== {prefix}: replaying {len(r['history'])} war-eval actions "
              f"(target lc {r['levels_completed']}) ==")
        case = run_case(prefix, r["history"], r["levels_completed"])
        results["cases"].append(case)
        print(f"   lc {case['lc_reached']}/{case['target_lc']}  "
              f"attempted={case['replay_attempted']} succeeded={case['replay_succeeded']}  "
              f"events={case['events']}  "
              f"plays={case.get('card_total_plays')} verbatim={case.get('replay_verbatim')} "
              f"invariant={case.get('score_invariant')}")

    r = runs["ar25"]
    print("== ar25 STARVED (soft_time_remaining=30s < bank_min_time 120) ==")
    starved = run_case("ar25", r["history"], r["levels_completed"], solver=StarvedSolver())
    results["starved_case"] = starved
    print(f"   events={starved['events']} succeeded={starved['replay_succeeded']}")

    fired = [c for c in results["cases"] if c["replay_succeeded"]]
    ok_fire = len(fired) >= 1 and all(
        c.get("replay_verbatim") and c.get("score_invariant") for c in fired)
    ok_starve = (not starved["replay_succeeded"]
                 and any(e[:2] == ["bank_skip", "time"] for e in starved["events"]))
    results["verdict"] = {
        "replay_fired_on_named_games": f"{len(fired)}/{len(GAMES)}",
        "all_fired_verbatim_and_invariant": ok_fire,
        "time_starvation_reproduces_eval_zero": ok_starve,
        "PASS": ok_fire and ok_starve,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nVERDICT: {json.dumps(results['verdict'], indent=2)}")
    print(f"written: {OUT}")
    return 0 if results["verdict"]["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
