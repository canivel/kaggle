"""Lightweight agent v5 - NO CNN, pure heuristic exploration.
110 parallel threads on CPU demands zero-overhead agents.
Uses frame hashing + systematic action exploration (graph-based, like 3rd place).
"""
import random
import time
import hashlib
from typing import Any
from collections import defaultdict

import numpy as np

from agents.structs import FrameData, GameAction, GameState
from agents.agent import Agent

MAX_TIME_PER_GAME = 300  # 5 min per game
MAX_ACTIONS_PER_GAME = 10000


class MyAgent(Agent):
    MAX_ACTIONS = MAX_ACTIONS_PER_GAME

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        self.start_time = time.time()
        print(f'Agent v5 (light): game={self.game_id}')

        self.grid_size = 64
        self.current_score = -1
        self.action_counter_local = 0
        self.action_list = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                           GameAction.ACTION4, GameAction.ACTION5]

        # Graph-based exploration state
        self.prev_frame_hash = None
        self.visited_states = set()          # set of frame hashes
        self.tried_actions = defaultdict(set) # frame_hash -> set of tried action indices
        self.frame_change_actions = defaultdict(set)  # frame_hash -> actions that changed state

    def _hash_frame(self, frame_data):
        """Fast hash of frame grid."""
        frame = np.array(frame_data.frame, dtype=np.int8)
        if frame.ndim == 3: frame = frame[-1]
        return hashlib.md5(frame.tobytes()).hexdigest()

    def _get_available_actions(self, frame_data):
        if hasattr(frame_data, 'available_actions') and frame_data.available_actions:
            return frame_data.available_actions
        return self.action_list + [GameAction.ACTION6]

    def _get_untried_actions(self, frame_hash, available):
        """Get actions not yet tried from this state."""
        tried = self.tried_actions[frame_hash]
        untried = []
        for a in available:
            v = a.value if hasattr(a, 'value') else int(a)
            if v not in tried:
                untried.append(a)
        return untried

    def _get_productive_actions(self, frame_hash, available):
        """Get actions known to cause frame changes from this state."""
        productive_vals = self.frame_change_actions[frame_hash]
        productive = []
        for a in available:
            v = a.value if hasattr(a, 'value') else int(a)
            if v in productive_vals:
                productive.append(a)
        return productive

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        elapsed = time.time() - self.start_time
        return (latest_frame.state is GameState.WIN
                or elapsed >= MAX_TIME_PER_GAME
                or self.action_counter_local >= MAX_ACTIONS_PER_GAME)

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        self.action_counter_local += 1
        score = latest_frame.score if hasattr(latest_frame, 'score') else 0

        # Level change -> partial reset (keep knowledge of action patterns)
        if score != self.current_score:
            if score != self.current_score and self.current_score >= 0:
                print(f'[{self.game_id}] Score: {self.current_score} -> {score} at action {self.action_counter_local}')
            self.visited_states.clear()
            self.tried_actions.clear()
            # Keep frame_change_actions - they transfer between levels!
            self.prev_frame_hash = None
            self.current_score = score

        # Handle resets
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self.prev_frame_hash = None
            action = GameAction.RESET
            action.reasoning = 'Reset'
            return action

        # Hash current frame
        frame_hash = self._hash_frame(latest_frame)
        self.visited_states.add(frame_hash)

        # Record if previous action changed the frame
        if self.prev_frame_hash is not None and hasattr(self, '_prev_action_val'):
            if frame_hash != self.prev_frame_hash:
                self.frame_change_actions[self.prev_frame_hash].add(self._prev_action_val)

        available = self._get_available_actions(latest_frame)

        # Priority 1: Try untried actions from this state
        untried = self._get_untried_actions(frame_hash, available)
        if untried:
            action = random.choice(untried)
        else:
            # Priority 2: Use actions known to change state
            productive = self._get_productive_actions(frame_hash, available)
            if productive:
                action = random.choice(productive)
            else:
                # Priority 3: Random from available
                action = random.choice(available)

        # Handle ACTION6 (click) - random coordinate
        a_val = action.value if hasattr(action, 'value') else int(action)
        if a_val == 6:
            # Click on random non-background pixel if possible
            frame = np.array(latest_frame.frame, dtype=np.int8)
            if frame.ndim == 3: frame = frame[-1]
            nonzero = np.argwhere(frame != 0)
            if len(nonzero) > 0:
                idx = random.randint(0, len(nonzero) - 1)
                y, x = int(nonzero[idx][0]), int(nonzero[idx][1])
            else:
                x, y = random.randint(0, 63), random.randint(0, 63)
            action.set_data({'x': x, 'y': y})
            action.reasoning = f'Click ({x},{y})'
        else:
            action.reasoning = f'{action.name}'

        # Track
        self.tried_actions[frame_hash].add(a_val)
        self.prev_frame_hash = frame_hash
        self._prev_action_val = a_val

        return action
