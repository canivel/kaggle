"""v40 = v39 full engine + per-level GE routing + GE-clone-replay (efficiency).

Coherent synthesis of every independently-proven win:
 (a) v39 base = MASTER BASELINE v10 (full FORGE engine: A*+IDDFS+beam(d60)+MCTS+novelty+sprite-perm)
 (b) v37's stable hashlib seed (already in v39)
 (c) v37's pickle->copy.deepcopy fallback (already in v39)
 (d) v35's per-level routing: BFS-FAILS-level -> GraphExplorer is PRIMARY for that level
 (e) NEW: GE-in-clone with minimal-path replay — GE explores a deepcopied game
     to find the win, then replays only the optimal action sequence live (kills
     the 55-836 actions/level waste that capped GE games at near-zero RHAE).

Routing:
  BFS solves level -> v39's BFS plan executes (already efficient)
  BFS fails level  -> v40 runs GE in cloned game, extracts winning path,
                      stores it as if it were a BFS solution -> executes as
                      minimal-path replay (same code path as BFS solutions)
"""
from pathlib import Path

PV40 = Path("f:/kaggle/arc-prize-2026/notebooks/forge_agent/v40_agent.py")
src = PV40.read_text(encoding="utf-8")

# Step 1: Inject GraphExplorer + FrameProcessor + NodeInfo classes from v24b
v24b = Path("f:/kaggle/arc-prize-2026/notebooks/forge_agent/v24b_agent.py").read_text(encoding="utf-8")
lines = v24b.splitlines(keepends=True)
ni_start = next(i for i, l in enumerate(lines) if l.startswith("class NodeInfo:"))
# Find anything starting after the GraphExplorer block — its end is FrameProcessor's start.
fp_start = next(i for i, l in enumerate(lines) if l.startswith("class FrameProcessor:"))
# FrameProcessor ends at next top-level class/def or class MyAgent
fp_end = None
for i in range(fp_start + 1, len(lines)):
    if lines[i].startswith("class MyAgent") or lines[i].startswith("def find_game_source") or (lines[i].startswith("class ") and not lines[i].startswith("class FrameProcessor")):
        fp_end = i; break
if fp_end is None:
    fp_end = len(lines)
ge_classes = "".join(lines[ni_start:fp_end])
print(f"GE classes block: lines {ni_start}-{fp_end} ({ge_classes.count(chr(10))} lines)")

# Inject ge_classes BEFORE class MyAgent in v40
marker_my = "class MyAgent(Agent):"
assert marker_my in src
src = src.replace(
    marker_my,
    "# ===== v40: GraphExplorer + FrameProcessor (from v24b faithful port) =====\n"
    + ge_classes + "\n\n" + marker_my,
    1,
)

# Step 2: Add GE state to MyAgent __init__ (next to existing self.x = ... in init).
# Find a stable anchor inside __init__: look for "self.start_time" or similar.
init_anchor = None
for cand in [
    "self.start_time = time.time()",
    "s.start_time = time.time()",
    "self.cl =",
    "s.cl =",
]:
    if cand in src:
        init_anchor = cand; break
assert init_anchor, "no __init__ anchor"
GE_INIT_BLOCK = """
        # v40: GraphExplorer + FrameProcessor for per-level GE-in-clone routing
        try:
            self._v40_fp = FrameProcessor()
            self._v40_ge = GraphExplorer(verbose_level=0, n_groups=5)
        except Exception:
            self._v40_fp = None; self._v40_ge = None
        self._v40_use_ge_level = False
        self._v40_ge_last_level = -1
"""
# Insert AFTER the anchor (keep its line, append our block on next line)
src = src.replace(init_anchor, init_anchor + GE_INIT_BLOCK, 1)

PV40.write_text(src, encoding="utf-8")
import ast
ast.parse(src)
print(f"v40 partial-built: {src.count(chr(10))} lines, syntax OK")
print("NOTE: GE-clone-replay routing not yet wired into choose_action — next step.")
