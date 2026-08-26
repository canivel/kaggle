"""Build v26 = v22 (BFS+CNN) + GE as tertiary fallback when CNN is unproductive.

Key difference from v25: CNN is PRESERVED. GE only fires when CNN has not
made progress for >= 100 actions (s._unproductive >= 100). This way:
- BFS-solved games: unchanged (best efficiency)
- CNN-solving games: unchanged (CNN was earning real Kaggle levels)
- Stuck games: GE takes over instead of letting CNN spin
"""

import re
from pathlib import Path

V24 = Path("f:/kaggle/arc-prize-2026/notebooks/forge_agent/v24_agent.py").read_text(encoding="utf-8")
V26 = Path("f:/kaggle/arc-prize-2026/notebooks/forge_agent/v26_agent.py").read_text(encoding="utf-8")

# 1) Inject GE+FP classes from v24 before MyAgent
v24_lines = V24.splitlines(keepends=True)
ge_start = next(i for i, l in enumerate(v24_lines) if l.startswith("INFINITY = np.iinfo"))
my_start = next(i for i, l in enumerate(v24_lines) if l.startswith("class MyAgent"))
GE_FP_BLOCK = "".join(v24_lines[ge_start:my_start])
print(f"GE+FP: {GE_FP_BLOCK.count(chr(10))} lines")

marker = "class MyAgent(Agent):"
assert marker in V26
new_v26 = V26.replace(marker, "# ===== GraphExplorer + FrameProcessor (from v24) =====\n" + GE_FP_BLOCK + "\n\n" + marker, 1)

# 2) Add dataclasses + Hashable imports
new_v26 = new_v26.replace(
    "from collections import deque\nfrom typing import Dict, List, Set, Optional, Tuple",
    "from collections import defaultdict, deque\nfrom dataclasses import dataclass, field\nfrom typing import Any, Dict, Hashable, List, Optional, Set, Tuple",
    1,
)

# 3) Add GE state vars to __init__
init_anchor = "s._bfs_tried = False"
GE_INIT = """
        # v26: graph-explorer state for tertiary fallback
        s._ge_fp = FrameProcessor()
        s._ge = GraphExplorer(verbose_level=0, n_groups=5)
        s._ge_status_mask = None
        s._ge_last_hash = None
        s._ge_last_action_id = None
        s._ge_level = -1
        s._ge_failed = False
"""
new_v26 = new_v26.replace(init_anchor, init_anchor + GE_INIT, 1)

# 4) Inject _ge_pick method before choose_action — copy from v25 build (proven working)
ca_marker = "    def choose_action(s, frames, lf):"
GE_PICK_METHOD = '''    def _ge_pick(s, lf, lvl):
        """Run one tick of the graph-explorer policy and return an action.

        Returns None on any unrecoverable issue so caller falls through.
        """
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
        avail_raw = list(getattr(lf, "available_actions", []) or [])
        avail = [a.value if hasattr(a, "value") else int(a) for a in avail_raw]

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

        if (not level_up) and s._ge_last_hash is not None and s._ge_last_action_id is not None:
            transition = hashed_frame != s._ge_last_hash
            try:
                s._ge.record_test(s._ge_last_hash, s._ge_last_action_id,
                                  int(transition), hashed_frame,
                                  target_num_candidates=num_actions,
                                  group2remaining_candidate_ids=action_groups,
                                  suspicious_transition=False)
            except Exception:
                s._ge.reset()
                s._ge.initialize(start_node=hashed_frame, num_candidates=num_actions,
                                 group2remaining_candidate_ids=action_groups)

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
new_v26 = new_v26.replace(ca_marker, GE_PICK_METHOD + ca_marker, 1)

# 5) Inject GE fallback BEFORE the CNN action selection.
# Trigger: BFS has no solution (so we're in CNN territory) AND CNN has been
# unproductive for N actions. Insert just before the "if not s._wd:" line.
trigger_anchor = "            if not s._wd:\n                if s.la<10:aidx,coords=s._heuristic(raw,avail,s.la)"
GE_TERTIARY = '''            # v26: GE tertiary fallback. Only fires when CNN has been
            # unproductive for >= 100 actions (no frame change). Preserves
            # CNN's earned levels on games where it works.
            if s._unproductive >= 100:
                try:
                    ge_action = s._ge_pick(lf, lvl)
                    if ge_action is not None:
                        s.pt = tensor
                        s.pai = 6  # treat as a click for replay-buffer indexing
                        s.pr = raw.copy()
                        s.ph = ch
                        s.la += 1
                        return ge_action
                except Exception as _ge_e:
                    logger.warning(f"v26 GE fallback failed: {_ge_e}; using CNN")

'''
new_v26 = new_v26.replace(trigger_anchor, GE_TERTIARY + trigger_anchor, 1)

Path("f:/kaggle/arc-prize-2026/notebooks/forge_agent/v26_agent.py").write_text(new_v26, encoding="utf-8")
print(f"Wrote {new_v26.count(chr(10))} lines to v26_agent.py")
