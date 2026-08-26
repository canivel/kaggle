"""Deep-play CNN agent: goes deep on individual games, solving multiple levels.

Combines:
- Winner's CNN frame-change prediction (BCE)
- Graph-based state tracking (3rd place)
- No model reset between levels (transfers knowledge)
- Action sequence replay on GAME_OVER
- Smart click targeting (object detection)
- Per-game time budget with early termination
"""

import hashlib
import random
import time
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from arcengine.enums import GameAction as GA, GameState as GS

ACTION_MAP = {a.value: a for a in GA}


class ActionModel(nn.Module):
    """Winner's CNN architecture."""
    def __init__(self, nc=16):
        super().__init__()
        self.conv1 = nn.Conv2d(nc, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.action_pool = nn.MaxPool2d(4, 4)
        self.action_fc = nn.Linear(256 * 16 * 16, 512)
        self.action_head = nn.Linear(512, 5)
        self.dropout = nn.Dropout(0.2)
        self.cc1 = nn.Conv2d(256, 128, 3, padding=1)
        self.cc2 = nn.Conv2d(128, 64, 3, padding=1)
        self.cc3 = nn.Conv2d(64, 32, 1)
        self.cc4 = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x)); f = F.relu(self.conv4(x))
        a = self.action_pool(f).view(f.size(0), -1)
        a = self.dropout(F.relu(self.action_fc(a)))
        al = self.action_head(a)
        c = F.relu(self.cc1(f)); c = F.relu(self.cc2(c))
        c = F.relu(self.cc3(c)); c = self.cc4(c)
        return torch.cat([al, c.view(c.size(0), -1)], dim=1)


class DeepPlayAgent:
    """CNN + graph agent that goes deep on individual games."""

    def __init__(self, avail_actions, device="cuda", seed=42):
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed % (2**32 - 1))

        self.device = device
        self.avail = avail_actions
        self.avail_vals = [a.value for a in avail_actions]
        self.has_click = 6 in self.avail_vals
        self.NC = 4096

        # CNN model (persists across levels)
        self.model = ActionModel().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

        # Experience buffer (cleared per level)
        self.buffer = deque(maxlen=200000)
        self.buffer_hashes = set()
        self.batch_size = 64
        self.train_freq = 5

        # State graph (cleared per level)
        self.state_graph = {}
        self.tried_actions = defaultdict(set)
        self.frame_change_actions = defaultdict(set)
        self.all_visited = set()

        # Tracking
        self.prev_hash = None
        self.prev_action_val = None
        self.prev_onehot = None
        self.current_level = 0
        self.actions_since_new_state = 0
        self.total_actions = 0

        # Cross-level memory
        self.globally_productive = defaultdict(int)
        self.solved_sequences = {}  # level -> [(action_val, data)]
        self.current_level_actions = []

    def _hash(self, grid):
        return hashlib.md5(grid.tobytes()).hexdigest()

    def _to_grid(self, frame_raw):
        g = np.array(frame_raw, dtype=np.int64)
        if g.ndim == 3: g = g[-1]
        return g

    def _to_onehot(self, grid):
        t = torch.zeros(16, 64, 64, dtype=torch.float32)
        t.scatter_(0, torch.from_numpy(grid).unsqueeze(0).clamp(0, 15), 1)
        return t.to(self.device)

    def _exp_hash(self, onehot_np, action_idx):
        return hashlib.md5(onehot_np.tobytes() + str(action_idx).encode()).hexdigest()

    def _get_click_targets(self, grid):
        """Find non-background pixels to click on."""
        nonzero = np.argwhere(grid != 0)
        if len(nonzero) == 0:
            return [(random.randint(0, 63), random.randint(0, 63))]
        targets = []
        for color in range(1, 16):
            pixels = np.argwhere(grid == color)
            if len(pixels) == 0: continue
            cy, cx = pixels.mean(axis=0).astype(int)
            targets.append((int(cy), int(cx)))
            # Sample some edge pixels too
            for p in pixels[::max(1, len(pixels)//3)]:
                targets.append((int(p[0]), int(p[1])))
        return targets if targets else [(int(nonzero[0][0]), int(nonzero[0][1]))]

    def _sample_cnn(self, onehot):
        """Sample action using CNN sigmoid probabilities."""
        with torch.no_grad():
            logits = self.model(onehot.unsqueeze(0)).squeeze(0)

        al, cl = logits[:5], logits[5:]
        mask = torch.full((5,), float("-inf"), device=self.device)
        a6 = False
        for v in self.avail_vals:
            if 1 <= v <= 5: mask[v - 1] = 0.
            elif v == 6: a6 = True
        al = al + mask
        if not a6: cl = cl + torch.full_like(cl, float("-inf"))

        ap = torch.sigmoid(al)
        cp = torch.sigmoid(cl) / self.NC
        probs = torch.cat([ap, cp])
        s = probs.sum()
        if s < 1e-10:
            probs = torch.zeros_like(probs)
            for v in self.avail_vals:
                if 1 <= v <= 5: probs[v-1] = 1.0
                elif v == 6: probs[5:] = 1.0 / self.NC
            s = probs.sum()
        probs = probs / s
        return np.random.choice(len(probs.cpu().numpy()), p=probs.cpu().numpy())

    def _train(self):
        if len(self.buffer) < self.batch_size: return
        bi = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in bi]
        states = torch.stack([torch.from_numpy(b["state"]).float().to(self.device) for b in batch])
        actions = torch.tensor([b["action_idx"] for b in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([b["reward"] for b in batch], dtype=torch.float32, device=self.device)
        self.optimizer.zero_grad()
        logits = self.model(states)
        sel = logits.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(sel, rewards)
        ap = torch.sigmoid(logits)
        loss = loss - 1e-4 * ap[:, :5].mean() - 1e-5 * ap[:, 5:].mean()
        loss.backward()
        self.optimizer.step()

    def on_level_change(self, new_level):
        """Level completed: clear per-level state, keep model + cross-level memory."""
        # Save solution for replay
        if self.current_level_actions:
            self.solved_sequences[self.current_level] = self.current_level_actions.copy()
        self.current_level_actions = []

        # Clear per-level state
        self.buffer.clear()
        self.buffer_hashes.clear()
        self.state_graph.clear()
        self.tried_actions.clear()
        self.all_visited.clear()
        self.prev_hash = None
        self.prev_action_val = None
        self.prev_onehot = None
        self.actions_since_new_state = 0

        # Reduce LR for fine-tuning on new level
        for pg in self.optimizer.param_groups:
            pg["lr"] = max(1e-5, pg["lr"] * 0.8)

        self.current_level = new_level

    def step(self, frame_raw):
        """Choose action for current frame. Returns (GameAction, data_dict_or_None)."""
        grid = self._to_grid(frame_raw)
        frame_hash = self._hash(grid)
        onehot = self._to_onehot(grid)
        onehot_np = onehot.cpu().numpy().astype(bool)

        is_new = frame_hash not in self.all_visited
        self.all_visited.add(frame_hash)
        self.actions_since_new_state = 0 if is_new else self.actions_since_new_state + 1

        # Update state graph
        if self.prev_hash is not None and self.prev_action_val is not None:
            if self.prev_hash not in self.state_graph:
                self.state_graph[self.prev_hash] = {}
            self.state_graph[self.prev_hash][self.prev_action_val] = frame_hash
            if frame_hash != self.prev_hash:
                self.frame_change_actions[self.prev_hash].add(self.prev_action_val)
                self.globally_productive[self.prev_action_val] += 1

        # Store experience for CNN training
        if self.prev_onehot is not None and self.prev_action_val is not None:
            exp_h = self._exp_hash(self.prev_onehot, self.prev_action_val)
            if exp_h not in self.buffer_hashes:
                changed = not np.array_equal(self.prev_onehot, onehot_np)
                self.buffer.append({
                    "state": self.prev_onehot,
                    "action_idx": self.prev_action_val if self.prev_action_val < 5 else (
                        5 + self.prev_click_idx if hasattr(self, 'prev_click_idx') else self.prev_action_val
                    ),
                    "reward": 1.0 if changed else 0.0,
                })
                self.buffer_hashes.add(exp_h)

        # Choose action strategy: blend graph exploration with CNN
        tried = self.tried_actions[frame_hash]
        untried = [v for v in self.avail_vals if v not in tried]

        action_val = None
        data = None

        if untried and random.random() < 0.5:
            # 50% chance: systematic graph exploration (try untried)
            scored = [(v, self.globally_productive.get(v, 0)) for v in untried]
            scored.sort(key=lambda x: -x[1])
            action_val = scored[0][0] if scored[0][1] > 0 else random.choice(untried)
        else:
            # 50% chance: CNN-guided exploration
            idx = self._sample_cnn(onehot)
            if idx < 5:
                action_val = idx + 1
            else:
                action_val = 6
                ci = idx - 5
                y, x = ci // 64, ci % 64
                data = {"x": int(x), "y": int(y)}

        # If action is click and no data yet, pick smart target
        if action_val == 6 and data is None:
            targets = self._get_click_targets(grid)
            y, x = random.choice(targets)
            data = {"x": int(x), "y": int(y)}

        # Track
        action = ACTION_MAP[action_val]
        self.tried_actions[frame_hash].add(action_val)
        self.prev_hash = frame_hash
        self.prev_action_val = action_val
        self.prev_onehot = onehot_np
        if action_val == 6 and data:
            self.prev_click_idx = data["y"] * 64 + data["x"]
        self.current_level_actions.append((action_val, data))
        self.total_actions += 1

        # Train CNN periodically
        if self.total_actions % self.train_freq == 0:
            self.model.train()
            self._train()
            self.model.eval()

        return action, data

    def get_replay_actions(self):
        """Get stored action sequences for solved levels."""
        return self.solved_sequences

    @property
    def is_stuck(self):
        return self.actions_since_new_state > 1000
