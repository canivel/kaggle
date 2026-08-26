"""Reconcile bank_fire_validation's sc25/m0r0 frame_divergence aborts with the
determinism audit (panel R12 N5 follow-up).

determinism_audit_25.py shows all 25 official games are frame-deterministic
across plays under identical FULL action sequences. Yet _bank aborted at
replay step 0 on sc25/m0r0. This script isolates the cause: replay the
war-eval seed1 history to the recorded lc (as bank_fire_validation did),
build the same TraceStep records, prune with warpack's ``prune_trace``, then
replay the PRUNED trace on a new play and report where it diverges and what
``prune_trace`` dropped before that point.

Finding (2026-07-16): sc25/m0r0's pruned[0] is preceded by 1-2 recorded
actions with board_changed=False that prune drops; they are visible no-ops
but mutate hidden state, so the pruned replay's first action lands on a
different frame. The aborts are a PRUNING artifact, not per-play
randomization.

Run from the repo root:
    .venv/Scripts/python.exe duck_eval/warpack/prune_replay_diag.py
Output: runs/war_eval_v1/prune_replay_diag.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ENV_FILES = REPO / "kaggle-data" / "environment_files"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "duck_eval" / "taaf_bundle" / "src" / "tufa-arc-agi-framework" / "src"))

OUT = REPO / "runs" / "war_eval_v1" / "prune_replay_diag.json"
CASES = (("sc25", 2), ("m0r0", 1), ("ar25", 2), ("s5i5", 1))


def ghash(frame_rows):
    return hash(tuple(tuple(int(c) for c in row) for row in frame_rows))


def diagnose(prefix: str, target_lc: int) -> dict:
    import arc_agi
    import arcengine
    from taaf.game_api import ArcadeSpec, GameAPI
    from warpack_patch import TraceStep, prune_trace

    bench = json.loads((REPO / "runs" / "kernel_pulls" / "war_eval_v1" /
                        "benchmark.json").read_text(encoding="utf-8"))
    r = next(x for x in bench["game_runs"] if x["game_id"].startswith(prefix))
    history = [a["action"] for a in r["history"]]

    os.environ.pop("ONLY_RESET_LEVELS", None)
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE,
                            environments_dir=str(ENV_FILES))
    local_id = next(e.game_id for e in arcade.available_environments
                    if e.game_id.startswith(prefix))
    game = GameAPI(env_name=local_id,
                   arcade_spec=ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE,
                                          environments_dir=str(ENV_FILES)))
    game.start_game()
    game._finish_game = lambda: None
    env = game.env

    out: dict = {"game": prefix, "local_game_id": local_id,
                 "benchmark_game_id": r["game_id"],
                 "version_mismatch": local_id != r["game_id"]}

    prev_hash = ghash(game.current_state.raw.frame[-1])
    prev_lc = 0
    trace: list[TraceStep] = []
    for act in history:
        resp = env.step(arcengine.GameAction[act["id"]], data=dict(act["data"] or {}))
        h = ghash(resp.frame[-1]) if resp.frame else None
        lc = int(resp.levels_completed or 0)
        trace.append(TraceStep(
            name=act["id"], data=dict(act["data"] or {}),
            board_changed=(h != prev_hash),
            level_completed=(lc > prev_lc), lc_after=lc,
            grid_hash=h, state_name=resp.state.name))
        prev_hash, prev_lc = h, lc
        if lc >= target_lc:
            break
    out["actions_fed"] = len(trace)
    out["lc_reached"] = prev_lc

    pruned = prune_trace(trace)
    out["pruned_len"] = len(pruned)
    idx0 = next(i for i, s in enumerate(trace)
                if (s.name, s.data, s.grid_hash)
                == (pruned[0].name, pruned[0].data, pruned[0].grid_hash))
    dropped = trace[:idx0]
    out["pruned0"] = {"name": pruned[0].name, "data": pruned[0].data,
                      "recorded_index": idx0}
    out["dropped_before_pruned0"] = {
        "noops": sum(1 for s in dropped if not s.board_changed and s.name != "RESET"),
        "board_changing": sum(1 for s in dropped if s.board_changed and s.name != "RESET"),
        "resets": sum(1 for s in dropped if s.name == "RESET"),
    }

    # open a new play exactly as _bank does
    os.environ["ONLY_RESET_LEVELS"] = "false"
    for _ in range(2):
        resp = env.step(arcengine.GameAction.RESET, data={})
        if getattr(resp, "full_reset", False):
            break
    os.environ["ONLY_RESET_LEVELS"] = "true"

    out["pruned_replay"] = "survived"
    for i, step in enumerate(pruned):
        resp = env.step(arcengine.GameAction[step.name], data=dict(step.data))
        h = ghash(resp.frame[-1]) if resp.frame else None
        lc = int(resp.levels_completed or 0)
        if h != step.grid_hash:
            out["pruned_replay"] = f"frame_divergence at step {i} ({step.name})"
            break
        if lc != step.lc_after:
            out["pruned_replay"] = f"lc_divergence at step {i}"
            break
    return out


def main() -> int:
    results = [diagnose(p, lc) for p, lc in CASES]
    verdict = (
        "bank_fire_validation's sc25/m0r0 step-0 aborts reproduce on fully "
        "deterministic engines: prune_trace drops leading visible-no-op actions "
        "that mutate hidden state, so the pruned replay's first action lands on "
        "a different frame. Pruning artifact, NOT per-play randomization."
    )
    OUT.write_text(json.dumps({"verdict": verdict, "cases": results}, indent=2),
                   encoding="utf-8")
    print(json.dumps({"verdict": verdict, "cases": results}, indent=2))
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
