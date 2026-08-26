#!/usr/bin/env python3
"""P2 smoke -- runtime-test the reset-retry patch BEFORE any Kaggle push.

`feedback_test_before_submit.md`: v38 scored 0.00 from a missing import that no static
check caught.  This driver applies the patch to a throwaway copy of the VEHICLE bundle
and then actually executes `attempt(...)` inside the real sandbox subprocess against a
scripted fake environment, asserting the sealed contract from the prereg:

  * anchors apply exactly once and the result ast-compiles
  * attempt() runs the sequence and stops the instant the level clears
  * attempt() issues RESET when the level did NOT clear ... and NEVER after a clear
  * the post-RESET frame matches the level-start frame (the P0.1 premise, re-asserted)
  * the episode cap (40) is enforced
  * RESET inside the sequence is refused
  * action() is untouched -- the stock primitive still works

Usage:
    uv run --no-project python duck_eval/p2/p2_smoke.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VEHICLE = (
    REPO
    / "runs"
    / "harness_diff_0813"
    / "ds"
    / "jakobbrggen_taaf-kaggle-source-anim-20260807-anim"
    / "src"
    / "ARC3-Inference"
)

sys.path.insert(0, str(REPO / "duck_eval" / "p2"))
from p2_patch import EPISODE_ACTION_CAP, apply_patch  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name if not detail else f"{name} :: {detail}")


# ---------------------------------------------------------------- fake environment
class FakeGame:
    """Minimal scripted environment.

    Level clears when the cumulative action sequence on the current level hits
    `solution`.  RESET returns to the level-start board, exactly like the real engine
    under ONLY_RESET_LEVELS=true (prereg S2, case A).
    """

    def __init__(self, solution: list[str]):
        self.solution = solution
        self.level = 1
        self.progress = 0
        self.action_num = 0
        self.resets = 0
        self.log: list[str] = []

    def _board(self) -> str:
        return f"LEVEL{self.level}|PROGRESS{self.progress}"

    def handle(self, actions: list[dict]) -> dict:
        name = str(actions[0].get("action", "")).strip().upper()
        self.action_num += 1
        self.log.append(name)
        level_completed = False
        if name == "RESET":
            self.resets += 1
            self.progress = 0
        else:
            expected = (
                self.solution[self.progress] if self.progress < len(self.solution) else None
            )
            if name == expected:
                self.progress += 1
                if self.progress == len(self.solution):
                    level_completed = True
                    self.level += 1
                    self.progress = 0
            else:
                self.progress = 0
        result = {
            "executed": True,
            "action_num": self.action_num,
            "level": self.level,
            "score": float(self.level - 1),
            "reward": 1.0 if level_completed else 0.0,
            "state": "NOT_FINISHED",
            "valid_actions": ["ACTION1", "ACTION2", "RESET"],
            "board_changed": True,
            "done": False,
            "level_completed": level_completed,
            "game_over": False,
            "run_complete": False,
            "requested_count": len(actions),
            "executed_count": 1,
            "stopped_early": False,
        }
        return {
            "action_result": result,
            "state": {
                "current_frame": {"ascii": self._board(), "step": self.action_num, "level": self.level},
                "history": [],
                "valid_actions": ["ACTION1", "ACTION2", "RESET"],
                "last_action_result": result,
            },
        }


def run_snippet(mod, game: FakeGame, code: str) -> dict:
    return mod.run_sandboxed_python(
        code=code,
        initial_state={
            "current_frame": {"ascii": game._board(), "step": 0, "level": game.level},
            "history": [],
            "valid_actions": ["ACTION1", "ACTION2", "RESET"],
            "last_action_result": {},
        },
        action_handler=game.handle,
        timeout_seconds=30,
    )


def main() -> int:
    if not VEHICLE.is_dir():
        print(f"FATAL: vehicle bundle not found at {VEHICLE}")
        return 2

    tmp = Path(tempfile.mkdtemp(prefix="p2smoke_"))
    try:
        root = tmp / "ARC3-Inference"
        shutil.copytree(VEHICLE, root)

        # ---- 1. patch applies, anchors unique, ast-compiles -------------------
        try:
            info = apply_patch(root)
            check("A1 patch applies (anchors count==1, ast ok)", True)
            check(
                "A2 sandbox is the VEHICLE generation (anim-20260807)",
                bool(info["sandbox_is_vehicle_generation"]),
                f"md5={info['sandbox_md5_before']}",
            )
            check("A3 banner sealed", info["banner"] == "[p2] reset-retry armed H=4 K=5 cap=40", info["banner"])
        except Exception as exc:  # noqa: BLE001
            check("A1 patch applies", False, repr(exc))
            raise

        # ---- 2. drift really is fatal (negative control) ---------------------
        drift = tmp / "drift"
        shutil.copytree(VEHICLE, drift / "ARC3-Inference")
        sb = drift / "ARC3-Inference" / "inference" / "agent" / "python_tool_sandbox.py"
        sb.write_text(sb.read_text(encoding="utf-8").replace("    import contextlib\n", "", 1), encoding="utf-8")
        try:
            apply_patch(drift / "ARC3-Inference")
            check("A4 drift dies LOUDLY (negative control)", False, "patch silently succeeded on drifted source")
        except Exception:
            check("A4 drift dies LOUDLY (negative control)", True)

        # ---- 3. execute the patched sandbox for real -------------------------
        sys.path.insert(0, str(root))
        for mod in [m for m in list(sys.modules) if m.startswith("inference")]:
            del sys.modules[mod]
        import inference.agent.python_tool_sandbox as sandbox_mod  # noqa: E402

        # B: level does NOT clear -> RESET is issued and returns to level start
        game = FakeGame(["ACTION1", "ACTION2", "ACTION1"])
        out = run_snippet(
            sandbox_mod,
            game,
            "r = attempt(['ACTION1', 'ACTION2'])\nresult = r\nprint(r)\n",
        )
        res = out.get("result") or {}
        check("B1 attempt() returned a dict", isinstance(res, dict), str(out)[:200])
        check("B2 level not completed", res.get("level_completed") is False, str(res))
        check("B3 actions_taken == 2", res.get("actions_taken") == 2, str(res.get("actions_taken")))
        check("B4 RESET issued", res.get("reset_issued") is True, str(res))
        check("B5 returned to level start", res.get("returned_to_level_start") is True, str(res))
        check("B6 host saw exactly one RESET", game.resets == 1, f"resets={game.resets} log={game.log}")

        # C: level DOES clear -> NO reset after win, game left advanced
        game2 = FakeGame(["ACTION1", "ACTION2"])
        out2 = run_snippet(
            sandbox_mod,
            game2,
            "result = attempt(['ACTION1', 'ACTION2', 'ACTION1'])\n",
        )
        res2 = out2.get("result") or {}
        check("C1 level_completed True", res2.get("level_completed") is True, str(res2))
        check("C2 stopped at the clear (2 actions, not 3)", res2.get("actions_taken") == 2, str(res2))
        check("C3 NO RESET after win", res2.get("reset_issued") is False, str(res2))
        check("C4 host saw zero RESETs", game2.resets == 0, f"resets={game2.resets} log={game2.log}")
        check("C5 game left advanced (level 2)", game2.level == 2, f"level={game2.level}")

        # D: episode cap enforced
        game3 = FakeGame(["ACTION1"])
        out3 = run_snippet(
            sandbox_mod,
            game3,
            f"result = attempt(['ACTION2'] * {EPISODE_ACTION_CAP + 1})\n",
        )
        check(
            "D1 cap enforced (>40 refused)",
            "at most" in str(out3.get("error", "")),
            str(out3.get("error"))[:160],
        )

        # E: RESET inside the sequence is refused
        game4 = FakeGame(["ACTION1"])
        out4 = run_snippet(sandbox_mod, game4, "result = attempt(['ACTION2', 'RESET'])\n")
        check(
            "E1 RESET in sequence refused",
            "issues its own RESET" in str(out4.get("error", "")),
            str(out4.get("error"))[:160],
        )

        # F: stock action() still works (no regression)
        game5 = FakeGame(["ACTION1"])
        out5 = run_snippet(sandbox_mod, game5, "result = action(['ACTION1'])\n")
        check(
            "F1 stock action() unaffected",
            bool((out5.get("result") or {}).get("level_completed")),
            str(out5)[:200],
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n=== P2 SMOKE ===")
    for p in PASS:
        print(f"  PASS  {p}")
    for f in FAIL:
        print(f"  FAIL  {f}")
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
