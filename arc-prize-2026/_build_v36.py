"""v36 = v35 + hard per-level BFS wall-clock deadline.

ka59/g50t/re86/sb26/sc25/tn36/tr87/wa30 stay 0 in v35 because BFS's chained
retry machinery (first-pass + dynamic-rescan + hidden-retry + distance-heuristic)
consumes the ENTIRE per-game budget, so _try_bfs_solve never returns and
_use_ge_level is never set -> graph-explorer never gets to run.

Fix: a single deadline attribute on BFSSolver, set when _try_bfs_solve starts,
checked in every BFS while-loop + retry gate. Past the deadline, BFS aborts
and returns None -> v35's per-level routing hands the level to graph-explorer.
HARD_CAP modest so GE gets ample time both locally and on Kaggle 8hr/game.
"""
from pathlib import Path

P = Path("f:/kaggle/arc-prize-2026/notebooks/forge_agent/v36_agent.py")
src = Path("f:/kaggle/arc-prize-2026/notebooks/forge_agent/v35_agent.py").read_text(encoding="utf-8")

# 1) Add deadline attr in BFSSolver.__init__ (after self.bfs_timeout = bfs_timeout)
old1 = "        self.bfs_timeout = bfs_timeout"
new1 = ("        self.bfs_timeout = bfs_timeout\n"
        "        self._level_deadline = float('inf')  # v36: hard per-level wall-clock")
assert src.count(old1) == 1, f"anchor1 count={src.count(old1)}"
src = src.replace(old1, new1, 1)

# 2) Gate the three BFS while-loops on the deadline.
for old in [
    "        while pq and explored < max_states and (time.time() - t0) < self.bfs_timeout:",
    "            while queue_d and explored_d < max_states * 10 and (time.time() - t0_d) < remaining_d:",
    "                while queue2 and explored2 < max_states and (time.time() - t0_2) < remaining:",
]:
    assert src.count(old) == 1, f"loop anchor missing/dup: {old[:60]}"
    src = src.replace(old, old[:-1] + " and time.time() < self._level_deadline:", 1)

# 3) Gate the retry-phase entry conditions so they don't even start past deadline.
#    dynamic rescan gate:
old_dr = "        exhausted_quickly = len(pq) == 0 and elapsed_first < self.bfs_timeout * 0.5"
new_dr = ("        exhausted_quickly = (len(pq) == 0 and elapsed_first < self.bfs_timeout * 0.5\n"
          "                             and time.time() < self._level_deadline)")
assert src.count(old_dr) == 1
src = src.replace(old_dr, new_dr, 1)

#    hidden-retry gate:
old_hr = "        if explored > 0 and (len(visited) < 200 or explored / len(visited) > 5) and elapsed_first < self.bfs_timeout * 0.8:"
new_hr = ("        if (explored > 0 and (len(visited) < 200 or explored / len(visited) > 5)\n"
          "                and elapsed_first < self.bfs_timeout * 0.8\n"
          "                and time.time() < self._level_deadline):")
assert src.count(old_hr) == 1
src = src.replace(old_hr, new_hr, 1)

# 4) Set the deadline at the top of _try_bfs_solve.
old4 = '''    def _try_bfs_solve(s, level_idx):
        """Try to solve current level. For L1+, uses A* with a goal
        heuristic derived from the previous level's win frame."""
        if s._bfs is None:
            return None'''
new4 = '''    def _try_bfs_solve(s, level_idx):
        """Try to solve current level. For L1+, uses A* with a goal
        heuristic derived from the previous level's win frame."""
        if s._bfs is None:
            return None
        # v36: hard per-level BFS wall-clock. Past this, all BFS retry
        # phases abort -> _bfs_solution stays None -> v35 routing hands
        # the level to graph-explorer instead of BFS eating the whole
        # game budget on unsolvable arm/unlock games.
        s._bfs._level_deadline = time.time() + 150.0'''
assert src.count(old4) == 1
src = src.replace(old4, new4, 1)

P.write_text(src, encoding="utf-8")
import ast
ast.parse(src)
print(f"v36 built: {src.count(chr(10))} lines, syntax OK, "
      f"deadline_attr={'_level_deadline' in src}, "
      f"gated_loops={src.count('time.time() < self._level_deadline')}")
