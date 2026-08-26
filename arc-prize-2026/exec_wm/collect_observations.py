"""Collect per-game observations for Rodionov-style executable world model generation.

For a given ARC-AGI-3 game_id, runs:
- N_random actions of pure random exploration
- BFS solution actions on each level (when BFS can solve)

Saves to: exec_wm/observations/<game_id>.json with schema:
  {
    "game_id": "bp35",
    "available_actions": [1,2,3,4,5,6,7],
    "tuples": [
      {"step": 0, "state_t": <2d list 64x64 uint8>, "action_id": int, "x": int, "y": int,
       "state_t1": <2d list>, "reward_class": int, "done": bool, "level": int}
    ],
    "summary": {"n_tuples": int, "n_levels_observed": int, "n_state_changes": int}
  }

Goal: produce small focused datasets (~200-500 tuples per game) for opus-4-8 to
study and write a Python simulator from.

Usage:
  uv run python exec_wm/collect_observations.py --game bp35 --n-random 200 --bfs-timeout 60
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Shim copied from jepa_wm/data/gen_trajectories.py to register the agents pkg
_AGENTS_DIR = ROOT / "kaggle-data" / "ARC-AGI-3-Agents"
sys.path.insert(0, str(_AGENTS_DIR))
_pkg = types.ModuleType("agents"); _pkg.__path__ = [str(_AGENTS_DIR / "agents")]
sys.modules["agents"] = _pkg
_spec = importlib.util.spec_from_file_location("agents.agent", str(_AGENTS_DIR / "agents" / "agent.py"))
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
sys.modules["agents.agent"] = _mod; _pkg.agent = _mod

from arcengine import ActionInput, GameAction  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "v39", str(ROOT / "notebooks" / "forge_agent" / "v39_agent.py")
)
v39 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v39)
BFSSolver = v39.BFSSolver
find_src = v39.find_game_source_and_class


def _instantiate(game_id: str):
    src, cls = find_src(game_id, None)
    if not src:
        raise RuntimeError(f"no source for {game_id}")
    spec_loc = importlib.util.spec_from_file_location("game_mod", src)
    mod = importlib.util.module_from_spec(spec_loc)
    spec_loc.loader.exec_module(mod)
    return getattr(mod, cls)()


def _tuple_record(step, s_t, aid, ax, ay, s_t1, level_up, level, change):
    return {
        "step": step,
        "state_t": s_t.tolist(),
        "action_id": int(aid),
        "x": int(ax),
        "y": int(ay),
        "state_t1": s_t1.tolist(),
        "reward_class": 2 if level_up else (1 if change else 0),
        "done": bool(level_up),
        "level": int(level),
    }


def collect(game_id: str, n_random: int, bfs_timeout: int, seed: int = 0) -> dict:
    rng = random.Random(seed)
    game = _instantiate(game_id)
    game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    r0 = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    cur = np.array(r0.frame[-1], dtype=np.uint8)
    avail = list(getattr(game, "_available_actions", []) or [1, 2, 3, 4, 5, 6])
    prev_levels = 0
    tuples = []
    step = 0

    for _ in range(n_random):
        aid = rng.choice(avail)
        if aid == 6:
            ax, ay = rng.randrange(64), rng.randrange(64)
            ai = ActionInput(id=GameAction.from_id(6), data={"x": ax, "y": ay})
        else:
            ax = ay = 0
            ai = ActionInput(id=GameAction.from_id(aid))
        result = game.perform_action(ai, raw=True)
        if not result or not result.frame:
            break
        s_t1 = np.array(result.frame[-1], dtype=np.uint8)
        new_levels = getattr(result, "levels_completed", 0) or 0
        level_up = new_levels > prev_levels
        change = bool((cur != s_t1).any())
        tuples.append(_tuple_record(step, cur, aid, ax, ay, s_t1, level_up, new_levels, change))
        cur = s_t1
        prev_levels = new_levels
        step += 1
        if getattr(result, "state", None) in ("GAME_OVER", "WIN"):
            game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            r2 = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            if r2 and r2.frame:
                cur = np.array(r2.frame[-1], dtype=np.uint8)
            prev_levels = 0

    n_changes = sum(1 for t in tuples if t["reward_class"] >= 1)
    n_levels = max((t["level"] for t in tuples), default=0)
    return {
        "game_id": game_id,
        "available_actions": [int(a) for a in avail],
        "tuples": tuples,
        "summary": {
            "n_tuples": len(tuples),
            "n_levels_observed": int(n_levels),
            "n_state_changes": int(n_changes),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--bfs-timeout", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(ROOT / "exec_wm" / "observations"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = collect(args.game, args.n_random, args.bfs_timeout, args.seed)
    out = out_dir / f"{args.game}.json"
    out.write_text(json.dumps(data))
    s = data["summary"]
    print(f"{args.game}: tuples={s['n_tuples']} levels={s['n_levels_observed']} state_changes={s['n_state_changes']} -> {out}")


if __name__ == "__main__":
    main()
