"""Generate (frame_t, action, frame_{t+1}, reward, done) tuples by replaying
BFS solutions across all 25 public ARC-AGI-3 games — training data for the
JEPA world model.

Reward shaping:
  - +1 per step (encourages forward progress)
  - +5 when levels_completed increases (level-up)
  - 0 baseline

We save as a numpy archive: `trajectories.npz` with arrays:
  s_t       (N, 64, 64) uint8
  s_t1      (N, 64, 64) uint8
  action_id (N,) int8
  ax        (N,) int8
  ay        (N,) int8
  r_class   (N,) int8 in {0, 1, 2}
  done      (N,) int8

Usage:
  uv run python -m jepa_wm.data.gen_trajectories --out trajectories.npz
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import types
from pathlib import Path

import numpy as np

PROJECT = Path("f:/kaggle/arc-prize-2026")
ENV_DIR = PROJECT / "kaggle-data" / "environment_files"
sys.path.insert(0, str(PROJECT / "kaggle-data" / "ARC-AGI-3-Agents"))
_pkg = types.ModuleType("agents"); _pkg.__path__ = [str(PROJECT / "kaggle-data" / "ARC-AGI-3-Agents" / "agents")]
sys.modules["agents"] = _pkg
_spec = importlib.util.spec_from_file_location("agents.agent", str(PROJECT / "kaggle-data" / "ARC-AGI-3-Agents" / "agents" / "agent.py"))
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
sys.modules["agents.agent"] = _mod; _pkg.agent = _mod

from arcengine import ActionInput, GameAction

spec = importlib.util.spec_from_file_location("v39", str(PROJECT / "notebooks" / "forge_agent" / "v39_agent.py"))
v39 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v39)
BFSSolver = v39.BFSSolver
find_src = v39.find_game_source_and_class


def random_exploration(game_id: str, n_actions: int = 400, seed: int = 0) -> list[tuple]:
    """Bonus data: take random actions from a cloned game to harvest dynamics samples.
    Useful for the WM to learn action effects even on games BFS can't solve."""
    import random
    src, cls = find_src(game_id, None)
    if not src:
        return []
    spec_loc = importlib.util.spec_from_file_location("game_mod", src)
    mod = importlib.util.module_from_spec(spec_loc); spec_loc.loader.exec_module(mod)
    game_cls = getattr(mod, cls)
    rng = random.Random(seed + abs(hash(game_id)) % 1000)
    out = []
    game = game_cls()
    game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    r0 = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    if not r0 or not r0.frame:
        return []
    cur = np.array(r0.frame[-1], dtype=np.uint8)
    prev_levels = 0
    avail = list(getattr(game, '_available_actions', []) or [])
    if not avail:
        avail = [1, 2, 3, 4, 5, 6]
    for _ in range(n_actions):
        aid = rng.choice(avail)
        if aid == 6:
            ax, ay = rng.randrange(64), rng.randrange(64)
            data = {"x": ax, "y": ay}
            ai = ActionInput(id=GameAction.from_id(6), data=data)
        else:
            ax = ay = 0
            ai = ActionInput(id=GameAction.from_id(aid))
        try:
            result = game.perform_action(ai, raw=True)
        except Exception:
            break
        if not result or not result.frame:
            break
        s_t1 = np.array(result.frame[-1], dtype=np.uint8)
        new_levels = getattr(result, 'levels_completed', 0) or 0
        level_up = new_levels > prev_levels
        r_class = 2 if level_up else (1 if (cur != s_t1).any() else 0)
        done = 1 if level_up else 0
        out.append((cur.copy(), s_t1, aid, ax, ay, int(r_class), int(done)))
        cur = s_t1
        prev_levels = new_levels
        if getattr(result, 'state', None) in ('GAME_OVER', 'WIN'):
            game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            r2 = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            if r2 and r2.frame:
                cur = np.array(r2.frame[-1], dtype=np.uint8)
            prev_levels = 0
    return out


def collect_for_game(game_id: str, max_levels: int = 30, bfs_timeout: int = 180) -> list[tuple]:
    """Run BFS on a game and return list of (s_t, s_t1, action_id, x, y, r_class, done)."""
    src, cls = find_src(game_id, None)
    if not src:
        return []
    bfs = BFSSolver(src, cls, scan_timeout=5, bfs_timeout=bfs_timeout)
    if not bfs.load():
        return []
    out = []
    game = bfs.game_cls()
    game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    r0 = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    if not r0 or not r0.frame:
        return []
    cur = np.array(r0.frame[-1], dtype=np.uint8)
    prev_levels = 0
    for level_idx in range(max_levels):
        try:
            sol = bfs.solve_level(level_idx, max_states=200000)
        except Exception:
            sol = None
        if not sol:
            print(f"  {game_id}: L{level_idx} no BFS solution; stop")
            break
        n = len(sol)
        for step_idx, (act_id, data) in enumerate(sol):
            s_t = cur.copy()
            if act_id <= 5 or act_id == 7:
                ax = 0; ay = 0
            else:
                ax = int(data.get('x', 0)) if data else 0
                ay = int(data.get('y', 0)) if data else 0
            ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
            try:
                result = game.perform_action(ai, raw=True)
            except Exception:
                break
            if not result or not result.frame:
                break
            s_t1 = np.array(result.frame[-1], dtype=np.uint8)
            new_levels = getattr(result, 'levels_completed', 0) or 0
            level_up = new_levels > prev_levels
            r_class = 2 if level_up else (1 if (s_t != s_t1).any() else 0)
            done = 1 if level_up else 0
            out.append((s_t, s_t1, int(act_id), ax, ay, int(r_class), int(done)))
            cur = s_t1
            prev_levels = new_levels
        print(f"  {game_id}: L{level_idx} solved in {n} actions, tuples so far {len(out)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="jepa_wm/data/trajectories.npz")
    ap.add_argument("--games", default=None, help="comma-separated game ids; default = all")
    ap.add_argument("--bfs-timeout", type=int, default=180)
    ap.add_argument("--random-actions", type=int, default=400,
                    help="random-exploration actions per game (bulk dynamics data)")
    args = ap.parse_args()

    games = (args.games.split(",") if args.games else sorted(p.name for p in ENV_DIR.iterdir() if p.is_dir()))
    print(f"Generating from {len(games)} games (bfs_timeout={args.bfs_timeout}, random_actions={args.random_actions})")
    all_tuples = []
    t0 = time.time()
    for gid in games:
        print(f"[{gid}]")
        try:
            t = collect_for_game(gid, bfs_timeout=args.bfs_timeout)
            all_tuples.extend(t)
            print(f"  {gid}: BFS contributed {len(t)} tuples")
        except Exception as e:
            print(f"  {gid}: BFS ERROR {e}")
        if args.random_actions > 0:
            try:
                rt = random_exploration(gid, n_actions=args.random_actions)
                all_tuples.extend(rt)
                print(f"  {gid}: random-exploration contributed {len(rt)} tuples")
            except Exception as e:
                print(f"  {gid}: random ERROR {e}")
    elapsed = time.time() - t0
    print(f"\nTotal tuples: {len(all_tuples)} ({elapsed:.0f}s)")
    if not all_tuples:
        print("No data — aborting.")
        sys.exit(1)

    s_t = np.stack([t[0] for t in all_tuples]).astype(np.uint8)
    s_t1 = np.stack([t[1] for t in all_tuples]).astype(np.uint8)
    action_id = np.array([t[2] for t in all_tuples], dtype=np.int8)
    ax = np.array([t[3] for t in all_tuples], dtype=np.int8)
    ay = np.array([t[4] for t in all_tuples], dtype=np.int8)
    r_class = np.array([t[5] for t in all_tuples], dtype=np.int8)
    done = np.array([t[6] for t in all_tuples], dtype=np.int8)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, s_t=s_t, s_t1=s_t1, action_id=action_id, ax=ax, ay=ay, r_class=r_class, done=done)
    sz = out.stat().st_size / 1024 / 1024
    print(f"Saved {out} ({sz:.1f} MB, N={len(all_tuples)})")


if __name__ == "__main__":
    main()
