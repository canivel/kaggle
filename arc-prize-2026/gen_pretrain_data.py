"""
Generate (frame, action, reward) tuples by replaying BFS solutions from v22 on each game.

Strategy: import the v22 BFSSolver, run it on each game, replay solutions, log
state-action-reward triples. These are EXPERT DEMONSTRATIONS suitable for offline
training of ForgeNet.
"""

import os
import sys
import glob
import time
import copy
import logging
import importlib.util
import pickle
from collections import deque
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.WARNING)

PROJECT = Path("f:/kaggle/arc-prize-2026")
ENV_DIR = PROJECT / "kaggle-data" / "environment_files"
sys.path.insert(0, str(PROJECT / "kaggle-data" / "ARC-AGI-3-Agents"))
# Stub the agents.agent module — BFSSolver pulls from v20_agent.py which imports it
import types
_agents_pkg = types.ModuleType("agents"); _agents_pkg.__path__ = [str(PROJECT / "kaggle-data" / "ARC-AGI-3-Agents" / "agents")]
sys.modules["agents"] = _agents_pkg
_spec = importlib.util.spec_from_file_location("agents.agent", str(PROJECT / "kaggle-data" / "ARC-AGI-3-Agents" / "agents" / "agent.py"))
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
sys.modules["agents.agent"] = _mod; _agents_pkg.agent = _mod

from arcengine import ActionInput, GameAction

# Load BFSSolver and ForgeNet from v20_agent.py
v22_path = PROJECT / "notebooks" / "forge_agent" / "v20_agent.py"
spec = importlib.util.spec_from_file_location("v22_mod", str(v22_path))
v22_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v22_mod)
BFSSolver = v22_mod.BFSSolver
find_game_source = v22_mod.find_game_source_and_class


def collect_for_game(game_id):
    """Run BFS on a game, return list of (frame, action_idx, reward) tuples."""
    src, cls = find_game_source(game_id, None)
    if not src:
        print(f"  {game_id}: no source")
        return []
    bfs = BFSSolver(src, cls, scan_timeout=5, bfs_timeout=90)
    if not bfs.load():
        print(f"  {game_id}: BFS load failed")
        return []

    tuples = []
    # Solve each level sequentially
    game = bfs.game_cls()
    game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    r0 = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    if not r0.frame:
        return []
    # Track the current frame from perform_action responses (get_pixels is
    # unreliable — returns 32x32 on many games before first action).
    cur_frame = np.array(r0.frame[-1], dtype=np.int64)

    for level_idx in range(20):  # at most 20 levels
        sol = bfs.solve_level(level_idx, max_states=200000)
        if not sol:
            print(f"  {game_id}: L{level_idx} no BFS solution; stop")
            break
        n = len(sol)
        start_count = len(tuples)
        for step_idx, (act_id, data) in enumerate(sol):
            prev_frame_arr = cur_frame.copy()
            if prev_frame_arr.shape != (64, 64):
                break
            ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                  if data else ActionInput(id=GameAction.from_id(act_id)))
            result = game.perform_action(ai, raw=True)
            # Advance current frame from the response
            if result and result.frame:
                cur_frame = np.array(result.frame[-1], dtype=np.int64)
            if act_id <= 5:
                action_idx = act_id - 1
            else:
                x = int(data.get('x', 0)); y = int(data.get('y', 0))
                action_idx = 5 + y * 64 + x
            steps_to_win = n - step_idx - 1
            base_reward = 5.0 * (0.95 ** steps_to_win)  # back-labeled
            tuples.append((prev_frame_arr, action_idx, base_reward))
        print(f"  {game_id}: L{level_idx} solved in {n} actions, +{len(tuples)-start_count} tuples (total {len(tuples)})")
    return tuples


def main():
    games = sorted(p.name for p in ENV_DIR.iterdir() if p.is_dir())
    print(f"Generating expert-demo data from {len(games)} games")
    all_tuples = []
    for game_id in games:
        print(f"[{game_id}]")
        try:
            t = collect_for_game(game_id)
            all_tuples.extend(t)
        except Exception as e:
            print(f"  {game_id}: ERROR {e}")
    print(f"\nTotal tuples: {len(all_tuples)}")
    out = PROJECT / "expert_demos.pkl"
    with open(out, "wb") as f:
        pickle.dump(all_tuples, f)
    print(f"Saved {out} ({out.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
