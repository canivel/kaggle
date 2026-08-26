import random
import time
import hashlib
from typing import Any
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent


class ActionModel(nn.Module):
    def __init__(self, n_colors=16, grid_size=64):
        super().__init__()
        self.conv1 = nn.Conv2d(n_colors, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.action_pool = nn.MaxPool2d(4, 4)
        self.action_fc = nn.Linear(256 * 16 * 16, 512)
        self.action_head = nn.Linear(512, 5)
        self.dropout = nn.Dropout(0.2)
        self.coord_conv1 = nn.Conv2d(256, 128, 3, padding=1)
        self.coord_conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.coord_conv3 = nn.Conv2d(64, 32, 1)
        self.coord_conv4 = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        features = F.relu(self.conv4(x))
        a = self.action_pool(features)
        a = a.view(a.size(0), -1)
        a = F.relu(self.action_fc(a))
        a = self.dropout(a)
        action_logits = self.action_head(a)
        c = F.relu(self.coord_conv1(features))
        c = F.relu(self.coord_conv2(c))
        c = F.relu(self.coord_conv3(c))
        c = self.coord_conv4(c)
        coord_logits = c.view(c.size(0), -1)
        return torch.cat([action_logits, coord_logits], dim=1)


class MyAgent(Agent):
    MAX_ACTIONS = float('inf')

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed % (2**32 - 1))
        self.start_time = time.time()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Agent device: {self.device}')
        self.grid_size = 64
        self.num_coordinates = self.grid_size * self.grid_size
        self.num_colours = 16
        self.action_model = ActionModel(n_colors=self.num_colours).to(self.device)
        self.optimizer = optim.Adam(self.action_model.parameters(), lr=0.0001)
        self.experience_buffer = deque(maxlen=200000)
        self.experience_hashes = set()
        self.batch_size = 64
        self.train_frequency = 5
        self.prev_frame = None
        self.prev_action_idx = None
        self.current_score = -1
        self.action_counter_local = 0
        self.action_list = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                           GameAction.ACTION4, GameAction.ACTION5]

    def _frame_to_tensor(self, frame_data: FrameData) -> torch.Tensor:
        frame = np.array(frame_data.frame, dtype=np.int64)[-1]
        tensor = torch.zeros(self.num_colours, self.grid_size, self.grid_size, dtype=torch.float32)
        tensor.scatter_(0, torch.from_numpy(frame).unsqueeze(0).clamp(0, 15), 1)
        return tensor.to(self.device)

    def _compute_hash(self, frame_np, action_idx):
        return hashlib.md5(frame_np.tobytes() + str(action_idx).encode()).hexdigest()

    def _sample_action(self, logits, available_actions):
        action_logits = logits[:5]
        coord_logits = logits[5:]
        action_mask = torch.full((5,), float('-inf'), device=self.device)
        action6_available = False
        for a in available_actions:
            a_val = a.value if hasattr(a, 'value') else a
            if 1 <= a_val <= 5:
                action_mask[a_val - 1] = 0.0
            elif a_val == 6:
                action6_available = True
        action_logits = action_logits + action_mask
        if not action6_available:
            coord_logits = coord_logits + torch.full_like(coord_logits, float('-inf'))
        action_probs = torch.sigmoid(action_logits)
        coord_probs = torch.sigmoid(coord_logits) / self.num_coordinates
        all_probs = torch.cat([action_probs, coord_probs])
        total = all_probs.sum()
        if total < 1e-10:
            all_probs = torch.zeros_like(all_probs)
            for a in available_actions:
                a_val = a.value if hasattr(a, 'value') else a
                if 1 <= a_val <= 5:
                    all_probs[a_val - 1] = 1.0
                elif a_val == 6:
                    all_probs[5:] = 1.0 / self.num_coordinates
            total = all_probs.sum()
        all_probs = all_probs / total
        return np.random.choice(len(all_probs.cpu().numpy()), p=all_probs.cpu().numpy())

    def _train(self):
        if len(self.experience_buffer) < self.batch_size:
            return
        indices = np.random.choice(len(self.experience_buffer), self.batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in indices]
        states = torch.stack([torch.from_numpy(b['state']).float().to(self.device) for b in batch])
        actions = torch.tensor([b['action_idx'] for b in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([b['reward'] for b in batch], dtype=torch.float32, device=self.device)
        self.optimizer.zero_grad()
        logits = self.action_model(states)
        selected = logits.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(selected, rewards)
        all_p = torch.sigmoid(logits)
        loss = loss - 0.0001 * all_p[:, :5].mean() - 0.00001 * all_p[:, 5:].mean()
        loss.backward()
        self.optimizer.step()

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        elapsed = time.time() - self.start_time
        return latest_frame.state is GameState.WIN or elapsed >= 5.5 * 3600

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        self.action_counter_local += 1

        if latest_frame.score != self.current_score:
            print(f'Score: {self.current_score} -> {latest_frame.score} at action {self.action_counter_local}')
            self.experience_buffer.clear()
            self.experience_hashes.clear()
            self.action_model = ActionModel(n_colors=self.num_colours).to(self.device)
            self.optimizer = optim.Adam(self.action_model.parameters(), lr=0.0001)
            self.prev_frame = None
            self.prev_action_idx = None
            self.current_score = latest_frame.score

        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self.prev_frame = None
            self.prev_action_idx = None
            action = GameAction.RESET
            action.reasoning = 'Reset'
            return action

        current_frame = self._frame_to_tensor(latest_frame)
        curr_np = current_frame.cpu().numpy().astype(bool)

        if self.prev_frame is not None and self.prev_action_idx is not None:
            h = self._compute_hash(self.prev_frame, self.prev_action_idx)
            if h not in self.experience_hashes:
                changed = not np.array_equal(self.prev_frame, curr_np)
                self.experience_buffer.append({
                    'state': self.prev_frame,
                    'action_idx': self.prev_action_idx,
                    'reward': 1.0 if changed else 0.0,
                })
                self.experience_hashes.add(h)

        with torch.no_grad():
            logits = self.action_model(current_frame.unsqueeze(0)).squeeze(0)
            idx = self._sample_action(logits, latest_frame.available_actions)

        if idx < 5:
            selected = self.action_list[idx]
            selected.reasoning = f'{selected.name}'
        else:
            coord_idx = idx - 5
            y = coord_idx // self.grid_size
            x = coord_idx % self.grid_size
            selected = GameAction.ACTION6
            selected.set_data({'x': x, 'y': y})
            selected.reasoning = f'Click ({x},{y})'

        self.prev_frame = curr_np
        self.prev_action_idx = idx if idx < 5 else (5 + (idx - 5))

        if self.action_counter_local % self.train_frequency == 0:
            self._train()

        return selected
