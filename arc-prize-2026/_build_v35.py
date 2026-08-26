"""v35 = principled per-level hybrid.

If BFS solves a level -> BFS plan (unchanged, the 14 search-games stay exact).
If BFS FAILS a level -> graph-explorer is the PRIMARY policy for that entire
level (every action), not an unproductivity-gated fallback. CNN only used if
GE returns None. This is the structurally-correct routing: search-games to
BFS, arm/unlock-games to frontier-exploration.
"""
from pathlib import Path

P = Path("f:/kaggle/arc-prize-2026/notebooks/forge_agent/v35_agent.py")
src = Path("f:/kaggle/arc-prize-2026/notebooks/forge_agent/v34_agent.py").read_text(encoding="utf-8")

# 1) After _try_bfs_solve, flag whether this level is BFS-unsolvable.
old1 = """                if s._bfs:
                    s._try_bfs_solve(lvl)"""
new1 = """                if s._bfs:
                    s._try_bfs_solve(lvl)
                # v35: BFS-unsolvable level -> graph-explorer is PRIMARY for it.
                s._use_ge_level = (s._bfs_solution is None)
                s._ge_level = -1  # force GraphExplorer re-init for this level"""
assert old1 in src, "anchor 1 not found"
src = src.replace(old1, new1, 1)

# 2) Insert GE-primary routing right before the CNN FALLBACK section.
old2 = """            # ===== CNN FALLBACK =====
            tensor = s._tensor(lf)
            raw = s._raw(lf)"""
new2 = """            # ===== v35: GRAPH-EXPLORER PRIMARY (BFS-unsolvable levels) =====
            # When BFS could not solve this level, the level is almost
            # certainly an arm/unlock game that BFS structurally cannot do
            # (verified: v29/v32/v33 all 0). Hand the WHOLE level to the
            # frontier-exploration graph-explorer instead of CNN.
            if getattr(s, '_use_ge_level', False):
                try:
                    ge_action = s._ge_pick(lf, lvl)
                except Exception as _ge_e:
                    logger.warning(f"v35 GE-primary failed: {_ge_e}")
                    ge_action = None
                if ge_action is not None:
                    raw_ge = s._raw(lf)
                    s.fhist.append(raw_ge.copy())
                    s.pr = raw_ge.copy()
                    s.la += 1
                    return ge_action
                # GE returned None -> fall through to CNN for this tick

            # ===== CNN FALLBACK =====
            tensor = s._tensor(lf)
            raw = s._raw(lf)"""
assert old2 in src, "anchor 2 not found"
src = src.replace(old2, new2, 1)

# 3) init _use_ge_level in __init__ alongside _bfs_solution = None
old3 = "        s._bfs_solution = None"
new3 = "        s._bfs_solution = None\n        s._use_ge_level = False"
assert old3 in src, "anchor 3 not found"
src = src.replace(old3, new3, 1)

P.write_text(src, encoding="utf-8")
import ast
ast.parse(src)
print(f"v35 built: {src.count(chr(10))} lines, syntax OK, "
      f"use_ge_level={'_use_ge_level' in src}, ge_primary={'GRAPH-EXPLORER PRIMARY' in src}")
