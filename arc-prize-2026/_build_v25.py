"""Build v25_agent.py — v22 BFS + GraphExplorer fallback hybrid.

v25 = v20_agent (= v22 BFS-only set_data, scored 0.30 on Kaggle)
       + GraphExplorer + FrameProcessor classes from v24
       + choose_action modified: BFS plan first; if no BFS plan, use GE
         (replacing the CNN fallback path).
"""

import re
from pathlib import Path

V24 = Path("f:/kaggle/arc-prize-2026/notebooks/forge_agent/v24_agent.py").read_text(encoding="utf-8")
V25 = Path("f:/kaggle/arc-prize-2026/notebooks/forge_agent/v25_agent.py").read_text(encoding="utf-8")

# Slice GE+FP from v24: lines INFINITY... up to but not including class MyAgent
v24_lines = V24.splitlines(keepends=True)
ge_start = None
my_start = None
for i, line in enumerate(v24_lines):
    if line.startswith("INFINITY = np.iinfo"):
        ge_start = i
    elif line.startswith("class MyAgent"):
        my_start = i
        break
assert ge_start is not None and my_start is not None, (ge_start, my_start)
GE_FP_BLOCK = "".join(v24_lines[ge_start:my_start])
print(f"GE+FP block: {GE_FP_BLOCK.count(chr(10))} lines")

# Inject GE+FP block before "class MyAgent(Agent):" in v25
marker = "class MyAgent(Agent):"
assert marker in V25
new_v25 = V25.replace(marker, "# ===== GraphExplorer + FrameProcessor (from v24) =====\n" + GE_FP_BLOCK + "\n\n" + marker, 1)

# Inject __init__ extension: add ge state vars right after super().__init__
init_marker = "class MyAgent(Agent):"
# Find the __init__ method body start
m = re.search(r"class MyAgent\(Agent\):.*?def __init__\(s.*?\):", new_v25, re.DOTALL)
assert m, "couldn't find MyAgent __init__"
# Append GE init lines at the end of __init__ — find the next def or end-of-method.
# Simpler: insert after the line that says s._bfs = None or right before the first def after __init__.
# Search for a stable injection point inside __init__:
init_body_anchor = "s._bfs_tried = False"
assert init_body_anchor in new_v25, "anchor not found in __init__"

GE_INIT = """
        # v25 hybrid: graph-explorer state for fallback
        s._ge_fp = FrameProcessor()
        s._ge = GraphExplorer(verbose_level=0, n_groups=5)
        s._ge_status_mask = None
        s._ge_last_hash = None
        s._ge_last_action_id = None
        s._ge_level = -1
        s._ge_action_to_action_groups = None  # cached for safety
        s._ge_failed = False
"""
new_v25 = new_v25.replace(init_body_anchor, init_body_anchor + GE_INIT, 1)

# Replace CNN FALLBACK block with GraphExplorer fallback.
# CNN block starts at "# ===== CNN FALLBACK ====="
# and ends at the "return sel" before "except Exception as e:"
cnn_start_marker = "# ===== CNN FALLBACK ====="
assert cnn_start_marker in new_v25
# Find the matching `return sel` immediately before `except Exception as e:`
# We replace everything from cnn_start_marker through that return sel with the GE block.
cnn_idx = new_v25.find(cnn_start_marker)
# Find the next occurrence of `except Exception as e:` after cnn_idx
exc_idx = new_v25.find("except Exception as e:", cnn_idx)
# Backtrack to find the "return sel" before that
return_search = new_v25.rfind("return sel", cnn_idx, exc_idx)
assert return_search != -1, "couldn't find return sel before except"
# Find end of that return sel line
end_of_return_line = new_v25.find("\n", return_search) + 1

# Build the replacement block — graph-explorer fallback in place of CNN
GE_FALLBACK = '''            # ===== GRAPH-EXPLORER FALLBACK (v25 hybrid) =====
            # Replaces CNN/WorldModel fallback path. Used when BFS has no
            # plan for current level. Graph-explorer maintains state-graph,
            # picks frontier-aware actions across 5 priority tiers.
            try:
                ge_action = s._ge_pick(lf, lvl)
                if ge_action is not None:
                    raw_now = s._raw(lf)
                    s.pr = raw_now.copy()
                    s.la += 1
                    return ge_action
            except Exception as _ge_e:
                logger.warning(f"GE fallback failed: {_ge_e}; using safe default")

            # Safe default: random valid arrow if available, else click center
            avail = list(getattr(lf, 'available_actions', []) or [])
            arrows = [a for a in avail if (a.value if hasattr(a, 'value') else int(a)) in (1, 2, 3, 4, 5)]
            if arrows:
                return random.choice(arrows) if hasattr(random.choice(arrows), 'value') else GameAction.from_id(int(random.choice(arrows)))
            sel = GameAction.ACTION6
            sel.set_data({"x": 32, "y": 32})
            return sel

'''

new_v25 = new_v25[:cnn_idx] + GE_FALLBACK + new_v25[end_of_return_line:]

# Add the _ge_pick method to MyAgent — insert before def choose_action
GE_PICK_METHOD = '''    def _ge_pick(s, lf, lvl):
        """Run one tick of the graph-explorer policy and return an action.

        Returns None on any unrecoverable issue so caller falls through.
        """
        # Reset GE on level change
        if lvl != s._ge_level:
            s._ge.reset()
            s._ge_status_mask = None
            s._ge_last_hash = None
            s._ge_last_action_id = None
            s._ge_level = lvl

        frame_np = np.array(lf.frame, dtype=np.uint8)
        if frame_np.size == 0:
            return None
        num_frames = frame_np.shape[0]
        frame_np = frame_np[-1].copy()

        level_up = (s._ge_status_mask is None) or s._ge_failed
        if level_up:
            seg, segs = s._ge_fp.segment_frame(frame_np)
            _, mask = s._ge_fp.identify_status_bars(seg, segs)
            s._ge_status_mask = mask
            s._ge_last_hash = None
            s._ge_last_action_id = None
            s._ge_failed = False

        if s._ge_status_mask is not None:
            frame_np[s._ge_status_mask] = 16
        segmented_frame, frame_segments = s._ge_fp.segment_frame(frame_np)
        avail_raw = list(getattr(lf, 'available_actions', []) or [])
        avail = [a.value if hasattr(a, 'value') else int(a) for a in avail_raw]

        SIMPLE = {1: GameAction.ACTION1, 2: GameAction.ACTION2,
                  3: GameAction.ACTION3, 4: GameAction.ACTION4,
                  5: GameAction.ACTION5}
        num_click_actions = 0
        num_actions = 0
        arrow_actions = []
        if 6 in avail:
            num_click_actions = len(frame_segments)
            num_actions = num_click_actions
            action_groups = s._ge_fp.frame_segments_to_action_groups(frame_segments, n_groups=5)
        else:
            action_groups = [set() for _ in range(5)]
        for aid in avail:
            if aid in SIMPLE:
                arrow_actions.append(SIMPLE[aid])
                action_groups[0].add(num_actions)
                num_actions += 1

        if num_actions == 0:
            return None

        frame_np[frame_np == 16] = 0
        hashed_frame = s._ge_fp.hash_frame(frame_np)

        if level_up:
            s._ge.initialize(start_node=hashed_frame, num_candidates=num_actions,
                             group2remaining_candidate_ids=action_groups)

        # Record transition from previous step
        if (not level_up) and s._ge_last_hash is not None and s._ge_last_action_id is not None:
            transition = hashed_frame != s._ge_last_hash
            try:
                s._ge.record_test(s._ge_last_hash, s._ge_last_action_id,
                                  int(transition), hashed_frame,
                                  target_num_candidates=num_actions,
                                  group2remaining_candidate_ids=action_groups,
                                  suspicious_transition=False)
            except Exception:
                # Stale graph or unknown source node — re-init from current
                s._ge.reset()
                s._ge.initialize(start_node=hashed_frame, num_candidates=num_actions,
                                 group2remaining_candidate_ids=action_groups)

        # If somehow current frame still missing, lazy add via re-init
        if hashed_frame not in s._ge._nodes:
            s._ge.reset()
            s._ge.initialize(start_node=hashed_frame, num_candidates=num_actions,
                             group2remaining_candidate_ids=action_groups)

        try:
            action_id = s._ge.choose_edge(hashed_frame, return_reasoning=False)
            action_id = int(action_id) if not isinstance(action_id, tuple) else int(action_id[0])
        except Exception:
            return None

        if action_id < num_click_actions:
            seg = frame_segments[action_id]
            seg_mask = (segmented_frame == action_id)
            pts = np.argwhere(seg_mask)
            if len(pts) == 0:
                bbox = seg.get("bbox") or seg.get("bounding_box")
                if bbox:
                    ymin, xmin, ymax, xmax = bbox
                    y, x = (ymin + ymax) // 2, (xmin + xmax) // 2
                else:
                    y, x = 32, 32
            else:
                pt = pts[random.randint(0, len(pts) - 1)]
                y, x = int(pt[0]), int(pt[1])
            action = GameAction.ACTION6
            action.set_data({"x": int(x), "y": int(y)})
        else:
            action = arrow_actions[action_id - num_click_actions]

        s._ge_last_hash = hashed_frame
        s._ge_last_action_id = action_id
        return action

'''

# Insert _ge_pick before choose_action
ca_marker = "    def choose_action(s, frames, lf):"
assert ca_marker in new_v25
new_v25 = new_v25.replace(ca_marker, GE_PICK_METHOD + ca_marker, 1)

Path("f:/kaggle/arc-prize-2026/notebooks/forge_agent/v25_agent.py").write_text(new_v25, encoding="utf-8")
print(f"Wrote {new_v25.count(chr(10))} lines to v25_agent.py")
