"""Improved agent v7 for local testing.
Key improvements over v6 (0.02 baseline):
1. State graph with BFS toward unexplored states (3rd place approach)
2. Early termination: stop game if stuck (no new states in N actions)
3. Smarter click targeting: prioritize small objects + edges
4. Remember productive actions across GAME_OVER resets
"""

import random
import hashlib
import time
from collections import defaultdict, deque

import numpy as np
from arcengine.enums import GameAction as GA, GameState as GS

ACTION_MAP = {a.value: a for a in GA}


class ImprovedAgent:
    """Graph-based exploration agent with BFS planning."""

    def __init__(self, avail_actions, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % (2**32 - 1))

        self.avail = avail_actions
        self.avail_vals = [a.value for a in avail_actions]
        self.has_click = 6 in self.avail_vals
        self.grid_size = 64

        # State graph
        self.state_graph = {}           # frame_hash -> {action_val -> next_hash}
        self.tried_actions = defaultdict(set)  # frame_hash -> tried action vals
        self.frame_change_actions = defaultdict(set)
        self.all_visited = set()

        # Per-level tracking
        self.prev_hash = None
        self.prev_action = None
        self.current_score = -1
        self.stuck_counter = 0
        self.actions_since_new_state = 0

        # Cross-level memory
        self.globally_productive = defaultdict(int)  # action_val -> success_count

    def _hash(self, grid):
        return hashlib.md5(grid.tobytes()).hexdigest()

    def _get_grid(self, frame):
        g = np.array(frame, dtype=np.int8)
        if g.ndim == 3: g = g[-1]
        return g

    def _get_click_targets(self, grid):
        """Find good click targets: non-background pixels, prefer edges/small objects."""
        nonzero = np.argwhere(grid != 0)
        if len(nonzero) == 0:
            return [(random.randint(0, 63), random.randint(0, 63))]

        # Group by color and find object centroids
        targets = []
        for color in range(1, 16):
            pixels = np.argwhere(grid == color)
            if len(pixels) == 0:
                continue
            # Centroid
            cy, cx = pixels.mean(axis=0).astype(int)
            targets.append((int(cy), int(cx)))
            # Also add edge pixels (more likely interactive)
            if len(pixels) > 4:
                for p in pixels[::max(1, len(pixels)//4)]:
                    targets.append((int(p[0]), int(p[1])))

        if not targets:
            idx = random.randint(0, len(nonzero) - 1)
            targets.append((int(nonzero[idx][0]), int(nonzero[idx][1])))

        return targets

    def _find_path_to_unexplored(self, current_hash):
        """BFS in state graph to find shortest path to a state with untried actions."""
        if not self.state_graph:
            return None

        queue = deque([(current_hash, [])])
        visited = {current_hash}

        while queue:
            state, path = queue.popleft()
            if len(path) > 20:  # don't plan too far ahead
                continue

            # Check if this state has untried actions
            if state != current_hash:
                tried = self.tried_actions[state]
                has_untried = any(v not in tried for v in self.avail_vals)
                if has_untried:
                    return path  # return action sequence to get here

            # Expand neighbors
            if state in self.state_graph:
                for action_val, next_state in self.state_graph[state].items():
                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append((next_state, path + [action_val]))

        return None

    def on_level_change(self, new_score):
        """Called when score changes (level complete)."""
        self.current_score = new_score
        self.all_visited.clear()
        self.tried_actions.clear()
        self.state_graph.clear()
        self.prev_hash = None
        self.prev_action = None
        self.actions_since_new_state = 0
        # Keep globally_productive and frame_change_actions!

    def choose_action(self, frame_data):
        """Returns (GameAction, data_dict_or_None)."""
        grid = self._get_grid(frame_data)
        frame_hash = self._hash(grid)

        is_new = frame_hash not in self.all_visited
        self.all_visited.add(frame_hash)

        if is_new:
            self.actions_since_new_state = 0
        else:
            self.actions_since_new_state += 1

        # Update state graph
        if self.prev_hash is not None and self.prev_action is not None:
            if self.prev_hash not in self.state_graph:
                self.state_graph[self.prev_hash] = {}
            self.state_graph[self.prev_hash][self.prev_action] = frame_hash
            if frame_hash != self.prev_hash:
                self.frame_change_actions[self.prev_hash].add(self.prev_action)
                self.globally_productive[self.prev_action] += 1

        # Strategy selection
        tried = self.tried_actions[frame_hash]
        untried_vals = [v for v in self.avail_vals if v not in tried]

        action_val = None
        data = None

        if untried_vals:
            # Priority 1: Try untried actions, prefer globally productive ones
            scored = [(v, self.globally_productive.get(v, 0)) for v in untried_vals]
            scored.sort(key=lambda x: -x[1])
            # 70% exploit best, 30% random untried
            if random.random() < 0.7 and scored[0][1] > 0:
                action_val = scored[0][0]
            else:
                action_val = random.choice(untried_vals)
        elif self.actions_since_new_state < 100:
            # Priority 2: Use BFS to find path to unexplored state
            path = self._find_path_to_unexplored(frame_hash)
            if path:
                action_val = path[0]
            else:
                # Prefer productive actions
                productive = list(self.frame_change_actions.get(frame_hash, set()))
                if productive:
                    action_val = random.choice(productive)
                else:
                    action_val = random.choice(self.avail_vals)
        else:
            # Stuck: random with emphasis on less-tried actions
            action_val = random.choice(self.avail_vals)

        # Build action
        action = ACTION_MAP[action_val]
        if action_val == 6:
            targets = self._get_click_targets(grid)
            y, x = random.choice(targets)
            data = {"x": int(x), "y": int(y)}

        self.tried_actions[frame_hash].add(action_val)
        self.prev_hash = frame_hash
        self.prev_action = action_val

        return action, data

    @property
    def is_stuck(self):
        """True if agent hasn't seen a new state in 500 actions."""
        return self.actions_since_new_state > 500
