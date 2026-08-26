"""Build v24_agent.py — port of dolphin-in-a-coma graph-explorer (arXiv 2512.24156)."""

HEADER = '''# =====================================================================
# v24 — Pure Graph-Explorer port (arXiv 2512.24156, MIT licensed)
# Source: github.com/dolphin-in-a-coma/arc-agi-3-just-explore (3rd private LB)
# Adapted to Kaggle MyAgent shell + arcengine imports.
#
# Approach: NO BFS, NO CNN. Frame-by-frame state-graph exploration.
# - Connected-component segmentation (4-connected by color)
# - Status-bar masking (rule: edge + ratio + twins)
# - 5 priority tiers for click candidates
# - Level Graph Explorer: state-hash nodes, frontier-aware action choice
# =====================================================================
import logging
import time
import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional, Set, Tuple
import random

import numpy as np

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState

logger = logging.getLogger(__name__)
'''

AGENT_CODE = '''
class MyAgent(Agent):
    MAX_ACTIONS = 1000000

    SIMPLE_ACTION_ID2GAME_ACTION = {
        1: GameAction.ACTION1,
        2: GameAction.ACTION2,
        3: GameAction.ACTION3,
        4: GameAction.ACTION4,
        5: GameAction.ACTION5,
    }

    N_GROUPS = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)
        self.frame_processor = FrameProcessor()
        self.status_bar_mask = None
        self.hashed_frame2action_results = {}
        self.hashed_frame2transitions = {}
        self.last_hashed_frame = None
        self.last_action = None
        self.last_action_object = GameAction.RESET
        self.last_levels_completed = 0
        self.failed = False
        self.level_up = True
        self.last_transition_suspicious = False
        self.level_first_frame = None
        self.favor_frontier_search = True
        self.favor_new_actions = False
        self.graph_explorer = GraphExplorer(verbose_level=0, n_groups=self.N_GROUPS)

    def is_done(self, frames, latest_frame):
        try:
            return latest_frame.state is GameState.WIN
        except Exception:
            return True

    def get_frame_transition_data(self, hashed_frame, num_actions):
        if hashed_frame not in self.hashed_frame2action_results:
            self.hashed_frame2action_results[hashed_frame] = np.zeros(num_actions, dtype=np.int8)
        if hashed_frame not in self.hashed_frame2transitions:
            self.hashed_frame2transitions[hashed_frame] = [None] * num_actions
        return self.hashed_frame2action_results[hashed_frame], self.hashed_frame2transitions[hashed_frame]

    def choose_action(self, frames, latest_frame):
        if latest_frame.state in [GameState.NOT_PLAYED]:
            self.last_hashed_frame = None
            self.last_action = None
            if self.failed:
                self.level_up = True
                self.failed = False
            return GameAction.RESET
        if latest_frame.state in [GameState.GAME_OVER]:
            self.last_transition_suspicious = True
            return GameAction.RESET

        cur_levels = getattr(latest_frame, "levels_completed", 0) or 0
        if cur_levels > self.last_levels_completed:
            self.level_up = True
            self.status_bar_mask = None
        self.last_levels_completed = cur_levels

        try:
            return self._explore_choose(frames, latest_frame)
        except Exception as e:
            logger.warning(f"v24 choose_action error: {e}; re-init explorer")
            self.failed = True
            self.level_up = True
            return self.last_action_object

    def _explore_choose(self, frames, latest_frame):
        latest_frame_np = np.array(latest_frame.frame, dtype=np.uint8)
        if latest_frame_np.size == 0:
            return GameAction.RESET
        num_frames = latest_frame_np.shape[0]
        latest_frame_np = latest_frame_np[-1].copy()

        if self.level_up:
            seg_for_status, segs_for_status = self.frame_processor.segment_frame(latest_frame_np)
            _, status_mask = self.frame_processor.identify_status_bars(seg_for_status, segs_for_status)
            self.status_bar_mask = status_mask
            self.hashed_frame2action_results = {}
            self.hashed_frame2transitions = {}

        if self.status_bar_mask is not None:
            latest_frame_np[self.status_bar_mask] = 16
        segmented_frame, frame_segments = self.frame_processor.segment_frame(latest_frame_np)
        available_actions = list(getattr(latest_frame, "available_actions", []) or [])

        num_actions = 0
        num_click_actions = 0
        arrow_actions = []
        if 6 in available_actions:
            num_click_actions = len(frame_segments)
            num_actions = num_click_actions
            action_groups = self.frame_processor.frame_segments_to_action_groups(frame_segments, n_groups=self.N_GROUPS)
        else:
            action_groups = [set() for _ in range(self.N_GROUPS)]
        for action_id in available_actions:
            if action_id in self.SIMPLE_ACTION_ID2GAME_ACTION:
                arrow_actions.append(self.SIMPLE_ACTION_ID2GAME_ACTION[action_id])
                action_groups[0].add(num_actions)
                num_actions += 1

        latest_frame_np[latest_frame_np == 16] = 0
        hashed_frame = self.frame_processor.hash_frame(latest_frame_np)

        if self.level_up and self.favor_frontier_search:
            self.level_first_frame = hashed_frame
            self.graph_explorer.reset()
            self.graph_explorer.initialize(start_node=hashed_frame, num_candidates=num_actions, group2remaining_candidate_ids=action_groups)

        suspicious_transition = False
        if self.last_hashed_frame is not None and not self.level_up:
            transition = hashed_frame != self.last_hashed_frame
            suspicious_transition = (hashed_frame == self.level_first_frame and num_frames > 1)
            if self.last_transition_suspicious:
                suspicious_transition = True
                self.last_transition_suspicious = False

            if self.last_action is not None:
                prev_results, prev_trans = self.get_frame_transition_data(self.last_hashed_frame, max(num_actions, self.last_action + 1))
                if self.last_action < len(prev_results):
                    if transition:
                        prev_results[self.last_action] = 1
                        prev_trans[self.last_action] = hashed_frame
                    else:
                        prev_results[self.last_action] = -1
                        prev_trans[self.last_action] = None

            if self.favor_frontier_search and self.last_action is not None:
                self.graph_explorer.record_test(
                    self.last_hashed_frame, self.last_action, int(transition), hashed_frame,
                    target_num_candidates=num_actions,
                    group2remaining_candidate_ids=action_groups,
                    suspicious_transition=suspicious_transition,
                )

        self.level_up = False

        curr_action_results, curr_transitions = self.get_frame_transition_data(hashed_frame, num_actions)
        if self.favor_frontier_search and hashed_frame not in self.graph_explorer._nodes:
            if self.last_action is not None:
                self.graph_explorer.record_test(
                    self.last_hashed_frame, self.last_action, 1, hashed_frame,
                    target_num_candidates=num_actions,
                    group2remaining_candidate_ids=action_groups,
                    suspicious_transition=suspicious_transition,
                )

        avail_arr = np.where(curr_action_results != -1)[0]
        if len(avail_arr) == 0:
            self.last_hashed_frame = hashed_frame
            self.last_action = None
            self.last_action_object = GameAction.RESET
            return GameAction.RESET

        action_id = None
        if self.favor_frontier_search:
            try:
                result = self.graph_explorer.choose_edge(hashed_frame, return_reasoning=False)
                action_id = int(result) if not isinstance(result, tuple) else int(result[0])
            except Exception as e:
                logger.warning(f"choose_edge fail: {e}; random")
                action_id = int(random.choice(avail_arr.tolist()))
        else:
            new_actions = np.where(curr_action_results == 0)[0]
            if len(new_actions) > 0 and self.favor_new_actions:
                action_id = int(random.choice(new_actions.tolist()))
            else:
                action_id = int(random.choice(avail_arr.tolist()))

        if action_id < num_click_actions:
            seg = frame_segments[action_id]
            segment_mask = (segmented_frame == action_id)
            pts = np.argwhere(segment_mask)
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

        self.last_hashed_frame = hashed_frame
        self.last_action = action_id
        self.last_action_object = action
        return action
'''


def main():
    # graph_explorer.py — strip its imports (already in HEADER)
    ge = open(r'C:/Users/dcani/AppData/Local/Temp/arc-agi-3-just-explore/graph_explorer.py', encoding='utf-8').read()
    marker = 'INFINITY = np.iinfo(np.int32).max'
    ge_body = marker + ge.split(marker, 1)[1]

    # FrameProcessor — just slice from heuristic_agent.py
    ha = open(r'C:/Users/dcani/AppData/Local/Temp/arc-agi-3-just-explore/agents/heuristic_agent.py', encoding='utf-8').read()
    fp_body = ha.split('class FrameProcessor:', 1)[1]
    fp_body = 'class FrameProcessor:' + fp_body
    # Drop any trailing leftover beyond FrameProcessor (heuristic_agent has nothing after)

    out = HEADER + '\n\n' + ge_body + '\n\n' + fp_body + '\n\n' + AGENT_CODE
    with open(r'f:/kaggle/arc-prize-2026/notebooks/forge_agent/v24_agent.py', 'w', encoding='utf-8') as f:
        f.write(out)
    print(f'Wrote {len(out.splitlines())} lines to v24_agent.py')


if __name__ == '__main__':
    main()
