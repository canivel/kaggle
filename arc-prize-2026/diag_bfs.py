"""Per-game BFS failure diagnostic for the 11 zero-BFS games.

For each game: instrument BFSSolver to report
  - n_actions found by _scan_actions on L0
  - has _get_valid_actions
  - states explored before timeout
  - unique states in visited
  - max depth reached
  - whether ANY level solved
Helps classify the failure mode (action-prune / hash-explosion / depth).
"""
import sys
import time
import importlib.util
import types
from pathlib import Path

import numpy as np

PROJECT = Path("f:/kaggle/arc-prize-2026")
ENV_DIR = PROJECT / "kaggle-data" / "environment_files"
sys.path.insert(0, str(PROJECT / "kaggle-data" / "ARC-AGI-3-Agents"))
_pkg = types.ModuleType("agents"); _pkg.__path__ = [str(PROJECT / "kaggle-data" / "ARC-AGI-3-Agents" / "agents")]
sys.modules["agents"] = _pkg
_s = importlib.util.spec_from_file_location("agents.agent", str(PROJECT / "kaggle-data" / "ARC-AGI-3-Agents" / "agents" / "agent.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
sys.modules["agents.agent"] = _m; _pkg.agent = _m

from arcengine import ActionInput, GameAction

spec = importlib.util.spec_from_file_location("v22", str(PROJECT / "notebooks" / "forge_agent" / "v20_agent.py"))
v22 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v22)
BFSSolver = v22.BFSSolver
find_src = v22.find_game_source_and_class

GAMES = ["bp35", "g50t", "ka59", "lf52", "re86", "sb26", "sc25", "su15", "tn36", "tr87", "wa30"]


def diag(game_id):
    src, cls = find_src(game_id, None)
    if not src:
        return f"{game_id}: NO SOURCE"
    bfs = BFSSolver(src, cls, scan_timeout=5, bfs_timeout=60)
    if not bfs.load():
        return f"{game_id}: BFS LOAD FAIL"
    game = bfs.game_cls()
    game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    r0 = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    if not r0.frame:
        return f"{game_id}: no frame after reset"
    f0 = np.array(r0.frame[-1], dtype=np.int64)
    bg = int(np.bincount(f0.flatten(), minlength=16).argmax())
    has_gva = hasattr(game, "_get_valid_actions")
    n_gva = -1
    if has_gva:
        try:
            n_gva = len(game._get_valid_actions())
        except Exception:
            n_gva = -2
    avail = list(getattr(game, "_available_actions", []))
    t0 = time.time()
    try:
        actions = bfs._scan_actions(game, f0, bg)
    except Exception as e:
        return f"{game_id}: scan_actions ERR {e}"
    scan_t = time.time() - t0
    # Try to solve L0 with instrumentation
    t1 = time.time()
    try:
        sol = bfs.solve_level(0, max_states=300000)
    except Exception as e:
        sol = None
        solve_err = str(e)[:60]
    else:
        solve_err = ""
    solve_t = time.time() - t1
    timed_out = 0 in bfs.timed_out_levels
    return (f"{game_id}: avail={avail} has_gva={has_gva} n_gva={n_gva} "
            f"scan_actions={len(actions)} (scan {scan_t:.1f}s) "
            f"L0_solved={'YES('+str(len(sol))+')' if sol else 'NO'} "
            f"timed_out={timed_out} solve_t={solve_t:.0f}s err={solve_err}")


if __name__ == "__main__":
    for g in GAMES:
        try:
            print(diag(g), flush=True)
        except Exception as e:
            print(f"{g}: TOP-ERR {e}", flush=True)
