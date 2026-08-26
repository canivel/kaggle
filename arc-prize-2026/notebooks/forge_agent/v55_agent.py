# =====================================================================
# v55 = v54 + Phase 1e ACTION-EFFICIENCY REPLAY.
# When the agent dies and resets, replay the memoized sequence that first
# cleared each level instead of re-discovering. RHAE = (human/agent)^2 so
# this directly compounds with the warm-start gains: less wasted actions
# on already-solved levels means more squared efficiency credit per level.
# =====================================================================
# v54 = v47 SG architecture + RANDOM-POLICY pretrained warm-start.
# Phase 1d: trained on 100k random-policy tuples (natural label balance vs
# BFS data's 78.8% positive bias). Should beat cold init (v47) without the
# v36 catastrophic regression that BFS-pretrained suffered.
# =====================================================================
# v47 = StochasticGoose port (Dries Smit / Tufa Labs — 1st Preview, 12.58% RHAE)
# Source: github.com/DriesSmit/ARC3-solution/custom_agents/action.py
#
# Pure-CNN agent, no BFS, no GraphExplorer. The CNN learns
# (state, action) -> frame_changed binary classification on-the-fly during
# each game. Hash-dedup experience buffer (200K), per-level reset.
#
# Key architecture vs forge_agent v35 lineage:
#   - 16-channel one-hot 64x64 input (no embeddings, no patches)
#   - 4-layer CNN backbone (32->64->128->256, all 64x64 spatial)
#   - Separate action head (5 logits) + FULLY-CONVOLUTIONAL 64x64 coord head
#     (preserves 2D inductive bias for click locations)
#   - Hierarchical sigmoid-based sampling with /num_coords scaling
#   - BCE loss on frame_changed, light entropy bonus
#   - Reset model + buffer on level-up
# =====================================================================
import hashlib
import logging
import random
import time
import traceback
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState

logger = logging.getLogger(__name__)


class ActionModel(nn.Module):
    """SG architecture: 16-ch input, 4 conv layers, action head (5) + 64x64 coord head."""
    def __init__(self, input_channels=16, grid_size=64):
        super().__init__()
        self.grid_size = grid_size
        self.num_action_types = 5

        # Shared conv backbone (all 64x64)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Action head: maxpool to 16x16 -> flatten -> 512 -> 5
        self.action_pool = nn.MaxPool2d(4, 4)
        self.action_fc = nn.Linear(256 * 16 * 16, 512)
        self.action_head = nn.Linear(512, self.num_action_types)

        # Coord head: spatial 256 -> 128 -> 64 -> 32 -> 1, output 64x64 logits
        self.coord_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.coord_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.coord_conv3 = nn.Conv2d(64, 32, kernel_size=1)
        self.coord_conv4 = nn.Conv2d(32, 1, kernel_size=1)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        conv_features = F.relu(self.conv4(x))

        action_features = self.action_pool(conv_features)
        action_features = action_features.view(action_features.size(0), -1)
        action_features = F.relu(self.action_fc(action_features))
        action_features = self.dropout(action_features)
        action_logits = self.action_head(action_features)

        coord_features = F.relu(self.coord_conv1(conv_features))
        coord_features = F.relu(self.coord_conv2(coord_features))
        coord_features = F.relu(self.coord_conv3(coord_features))
        coord_logits = self.coord_conv4(coord_features)
        coord_logits = coord_logits.view(coord_logits.size(0), -1)

        return torch.cat([action_logits, coord_logits], dim=1)  # (B, 5+4096)


class MyAgent(Agent):
    """Pure-SG architecture. NO BFS, NO GraphExplorer. CNN learns online."""
    MAX_ACTIONS = float('inf')

    def __init__(s, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Stable hashlib seed per game (v39 fix carried forward)
        seed = int(hashlib.md5(str(s.game_id).encode()).hexdigest()[:8], 16)
        random.seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
        torch.manual_seed(seed % (2 ** 32 - 1))

        s.start_time = time.time()
        s.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        s.grid_size = 64
        s.num_coordinates = 64 * 64
        s.num_colours = 16

        s.action_model = None
        s.optimizer = None

        s.experience_buffer = deque(maxlen=200000)
        s.experience_hashes = set()
        s.batch_size = 64
        s.train_frequency = 5

        s.prev_frame = None
        s.prev_action_idx = None

        s.action_list = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                         GameAction.ACTION4, GameAction.ACTION5]
        s.current_score = -1

        # Phase 1e: per-level memoized action paths
        # level_paths[L] = list of (action_id, x, y) tuples that, executed from
        # the L0 reset state, take us from level L to level L+1.
        # Cleared at __init__ (per-game). Populated as we first reach each level.
        s.level_paths = {}
        s._current_level_actions = []  # actions taken on current level
        s._replay_queue = []  # actions queued for replay after reset
        s._max_solved_level = -1  # highest level ever solved in this game

    def _frame_to_tensor(s, fd):
        frame = np.array(fd.frame, dtype=np.int64)[-1]
        if frame.shape != (s.grid_size, s.grid_size):
            raise RuntimeError(f"unexpected frame shape {frame.shape}")
        frame = np.clip(frame, 0, s.num_colours - 1)
        tensor = torch.zeros(s.num_colours, s.grid_size, s.grid_size, dtype=torch.float32)
        tensor.scatter_(0, torch.from_numpy(frame).unsqueeze(0), 1)
        return tensor.to(s.device)

    def _experience_hash(s, frame_np, action_idx):
        return hashlib.md5(frame_np.tobytes() + str(action_idx).encode()).hexdigest()

    def _sample(s, combined_logits, available_actions):
        """Hierarchical sigmoid sampling with action-availability masking."""
        action_logits = combined_logits[:5].clone()
        coord_logits = combined_logits[5:].clone()

        action6_available = False
        action_mask = torch.full_like(action_logits, float('-inf'))
        if available_actions:
            for a in available_actions:
                av = a.value if hasattr(a, 'value') else int(a)
                if 1 <= av <= 5:
                    action_mask[av - 1] = 0.0
                elif av == 6:
                    action6_available = True
            action_logits = action_logits + action_mask
            if not action6_available:
                coord_logits = coord_logits + torch.full_like(coord_logits, float('-inf'))

        action_probs = torch.sigmoid(action_logits)
        coord_probs = torch.sigmoid(coord_logits)
        coord_probs_scaled = coord_probs / s.num_coordinates

        all_probs = torch.cat([action_probs, coord_probs_scaled])
        total = all_probs.sum()
        if not torch.isfinite(total) or total <= 0:
            # Fallback: uniform over available ACTION1-5
            valid_idx = [i for i in range(5) if action_mask[i] == 0.0]
            if not valid_idx:
                return 0, None
            return random.choice(valid_idx), None
        all_probs = all_probs / total
        idx = int(np.random.choice(len(all_probs), p=all_probs.cpu().numpy()))
        if idx < 5:
            return idx, None
        coord_idx = idx - 5
        y = coord_idx // s.grid_size
        x = coord_idx % s.grid_size
        return 5, (int(y), int(x))

    def _train(s):
        if len(s.experience_buffer) < s.batch_size:
            return
        idxs = np.random.choice(len(s.experience_buffer), s.batch_size, replace=False)
        batch = [s.experience_buffer[i] for i in idxs]
        states = torch.stack([torch.from_numpy(e['state']).float().to(s.device) for e in batch])
        action_indices = torch.tensor([e['action_idx'] for e in batch], dtype=torch.long, device=s.device)
        rewards = torch.tensor([e['reward'] for e in batch], dtype=torch.float32, device=s.device)

        s.optimizer.zero_grad()
        logits = s.action_model(states)
        selected = logits.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        main_loss = F.binary_cross_entropy_with_logits(selected, rewards)
        # Light entropy bonus
        all_probs = torch.sigmoid(logits)
        loss = main_loss - 0.0001 * all_probs[:, :5].mean() - 0.00001 * all_probs[:, 5:].mean()
        loss.backward()
        s.optimizer.step()

    def _reset_for_new_level(s):
        s.experience_buffer.clear()
        s.experience_hashes.clear()
        s.action_model = ActionModel(input_channels=s.num_colours, grid_size=s.grid_size).to(s.device)
        # Phase 1d: load weights pretrained on RANDOM-POLICY trajectories
        # (100k tuples, natural ~50/50 label balance, no BFS selection bias).
        # Silent fallback to random init if file absent.
        import os as _os
        for wp in ['/kaggle/input/sg-pretrained-weights/sg_action_model.pt',
                   'runs/sg_pretrain/sg_action_model_random.pt']:
            if _os.path.exists(wp):
                try:
                    ck = torch.load(wp, map_location=s.device, weights_only=False)
                    s.action_model.load_state_dict(ck['model'])
                    break
                except Exception:
                    pass
        s.optimizer = optim.Adam(s.action_model.parameters(), lr=0.0001)
        s.prev_frame = None
        s.prev_action_idx = None

    def is_done(s, frames, lf):
        return (lf.state is GameState.WIN
                or (time.time() - s.start_time) >= 8 * 3600 - 300)

    def choose_action(s, frames, lf):
        try:
            cur_level = lf.levels_completed or 0
            score = getattr(lf, 'score', None) or lf.levels_completed

            # Track level transitions: when we just cleared a level, record the
            # actions we took on it as the first-clear path for future replay.
            if score != s.current_score:
                if cur_level > s._max_solved_level and s._max_solved_level >= 0:
                    # We just crossed into a new level. Memoize what worked.
                    s.level_paths[s._max_solved_level + 1] = list(s._current_level_actions)
                s._max_solved_level = max(s._max_solved_level, cur_level - 1)
                s._current_level_actions = []
                s._reset_for_new_level()
                s.current_score = score

            if lf.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                s.prev_frame = None
                s.prev_action_idx = None
                # On GAME_OVER reset: queue replay of all memoized level paths
                # for levels we know how to clear, plus skip past them on the next turn.
                if lf.state is GameState.GAME_OVER:
                    s._replay_queue = []
                    for L in sorted(s.level_paths.keys()):
                        s._replay_queue.extend(s.level_paths[L])
                    s._current_level_actions = []
                a = GameAction.RESET
                a.reasoning = "reset"
                return a

            # Phase 1e: if we have queued replay actions, execute them
            if s._replay_queue:
                rep = s._replay_queue.pop(0)
                act_id, ax, ay = rep
                sel = GameAction.from_id(act_id)
                if act_id == 6:
                    sel.set_data({"x": int(ax), "y": int(ay)})
                sel.reasoning = f"replay:a{act_id}"
                # Record as part of current level's action sequence
                s._current_level_actions.append(rep)
                return sel

            cur_tensor = s._frame_to_tensor(lf)
            cur_np = cur_tensor.cpu().numpy().astype(bool)

            # Build (prev_state, action) -> frame_changed experience
            if s.prev_frame is not None and s.prev_action_idx is not None:
                eh = s._experience_hash(s.prev_frame, s.prev_action_idx)
                if eh not in s.experience_hashes:
                    frame_changed = not np.array_equal(s.prev_frame, cur_np)
                    s.experience_buffer.append({
                        'state': s.prev_frame,
                        'action_idx': s.prev_action_idx,
                        'reward': 1.0 if frame_changed else 0.0,
                    })
                    s.experience_hashes.add(eh)

            avail = getattr(lf, 'available_actions', None) or []
            with torch.no_grad():
                logits = s.action_model(cur_tensor.unsqueeze(0)).squeeze(0)
            action_idx, coords = s._sample(logits, avail)

            if action_idx < 5:
                sel = s.action_list[action_idx]
                sel.reasoning = f"sg:a{action_idx + 1}"
                unified_idx = action_idx
                # Phase 1e: record actual ARC action_id (1..5) for replay
                s._current_level_actions.append((action_idx + 1, 0, 0))
            else:
                sel = GameAction.ACTION6
                y, x = coords
                sel.set_data({"x": x, "y": y})
                sel.reasoning = f"sg:click({x},{y})"
                unified_idx = 5 + (y * s.grid_size + x)
                s._current_level_actions.append((6, int(x), int(y)))

            s.prev_frame = cur_np
            s.prev_action_idx = unified_idx

            if s.action_counter % s.train_frequency == 0:
                s._train()

            return sel
        except Exception as e:
            traceback.print_exc()
            a = random.choice(s.action_list)
            a.reasoning = f"err:{str(e)[:40]}"
            return a
