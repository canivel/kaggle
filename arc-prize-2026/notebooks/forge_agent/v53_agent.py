# =====================================================================
# v53 = v52 4-head SG + Phase 1c PER-PIXEL CHANGE MAP supervision.
# Per-pixel diff (prev_grid != cur_grid) tells us exactly which cells
# responded. The frame_changed head's 4096 coord logits get trained with
# that dense 64x64 label on EVERY transition, not just the clicked position.
# Massive sample-efficiency boost for the head where BFS/GE is blind.
# =====================================================================
# v52 = v49 SG + Phase 1b RICHER LABELS.
# Original SG: single binary head per (action) = P(frame_changes). Sparse.
# v52: 4 heads per (action) — frame_changed, score_up, novel_state, level_up.
# Sampling combines: max(frame_changed, score_up*5, novel*2, level_up*10) — heavier
# weight on score+/level+/novel actions over mere "twitch" changes. Strong supervision
# signal for the click coord head (where BFS+GE is blind).
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
    """v52 4-head SG: shared backbone + K=4 (action,coord) head pairs for
    {frame_changed, score_up, novel_state, level_up}. Output: (B, K, 5+4096)."""
    NUM_HEADS = 4  # 0=frame_changed, 1=score_up, 2=novel_state, 3=level_up

    def __init__(self, input_channels=16, grid_size=64):
        super().__init__()
        self.grid_size = grid_size
        self.num_action_types = 5

        # Shared conv backbone (all 64x64)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Action head: maxpool to 16x16 -> flatten -> 512 -> K*5
        self.action_pool = nn.MaxPool2d(4, 4)
        self.action_fc = nn.Linear(256 * 16 * 16, 512)
        self.action_head = nn.Linear(512, self.num_action_types * self.NUM_HEADS)

        # Coord head: spatial 256 -> 128 -> 64 -> 32 -> K (K logits per pixel)
        self.coord_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.coord_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.coord_conv3 = nn.Conv2d(64, 32, kernel_size=1)
        self.coord_conv4 = nn.Conv2d(32, self.NUM_HEADS, kernel_size=1)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """Returns (B, K, 5+4096) where K=NUM_HEADS."""
        B = x.size(0)
        K = self.NUM_HEADS
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        conv_features = F.relu(self.conv4(x))

        action_features = self.action_pool(conv_features)
        action_features = action_features.view(B, -1)
        action_features = F.relu(self.action_fc(action_features))
        action_features = self.dropout(action_features)
        action_logits = self.action_head(action_features)  # (B, K*5)
        action_logits = action_logits.view(B, K, self.num_action_types)  # (B, K, 5)

        coord_features = F.relu(self.coord_conv1(conv_features))
        coord_features = F.relu(self.coord_conv2(coord_features))
        coord_features = F.relu(self.coord_conv3(coord_features))
        coord_logits = self.coord_conv4(coord_features)  # (B, K, 64, 64)
        coord_logits = coord_logits.view(B, K, -1)  # (B, K, 4096)

        return torch.cat([action_logits, coord_logits], dim=2)  # (B, K, 5+4096)


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
        s.prev_score = 0
        s.prev_level = 0
        s.seen_state_hashes = set()  # for novelty signal
        # v53: keep the raw color grids for per-pixel change-map computation
        s.prev_grid = None

        # Head weights for sampling combination (heavier on score+/level+ than mere twitch)
        s.head_weights = (1.0, 5.0, 2.0, 10.0)  # frame_changed, score_up, novel_state, level_up

        s.action_list = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                         GameAction.ACTION4, GameAction.ACTION5]
        s.current_score = -1

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

    def _sample(s, all_logits, available_actions):
        """all_logits: (K, 5+4096). Combine across K heads via head_weights then
        hierarchical sigmoid sample over the 4101 unified actions."""
        K = ActionModel.NUM_HEADS
        # Weighted sum of sigmoid'd head logits → combined "interestingness" probs
        probs_by_head = torch.sigmoid(all_logits)  # (K, 4101)
        w = torch.tensor(s.head_weights, device=all_logits.device, dtype=all_logits.dtype).view(K, 1)
        combined = (probs_by_head * w).sum(dim=0)  # (4101,)
        action_probs = combined[:5].clone()
        coord_probs = combined[5:].clone()

        action6_available = False
        action_mask = torch.zeros_like(action_probs)
        if available_actions:
            valid_av = []
            for a in available_actions:
                av = a.value if hasattr(a, 'value') else int(a)
                if 1 <= av <= 5:
                    valid_av.append(av - 1)
                elif av == 6:
                    action6_available = True
            # Zero out unavailable actions
            for i in range(5):
                if i not in valid_av:
                    action_probs[i] = 0.0
            if not action6_available:
                coord_probs.zero_()

        # Coord-pixel mass is scaled down so coord head doesn't dominate via 4096 entries
        coord_probs_scaled = coord_probs / s.num_coordinates
        all_probs = torch.cat([action_probs, coord_probs_scaled])
        total = all_probs.sum()
        if not torch.isfinite(total) or total <= 0:
            # Fallback: uniform over available ACTION1-5
            valid_idx = [i for i in range(5) if action_probs[i] > 0]
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
        labels = torch.tensor([e['labels'] for e in batch], dtype=torch.float32, device=s.device)
        # v53: dense per-pixel change map (B, 4096) — the frame_changed head[0]
        # coord logits get supervised on EVERY pixel of every transition.
        change_maps = torch.tensor(
            np.stack([e['change_map'] for e in batch]),
            dtype=torch.float32, device=s.device,
        ).view(s.batch_size, -1)  # (B, 4096)

        s.optimizer.zero_grad()
        logits = s.action_model(states)  # (B, K, 4101)
        # Loss 1: gather'd logit BCE on K-dim labels (standard 4-head SG loss)
        sel = logits.gather(2, action_indices.view(-1, 1, 1).expand(-1, ActionModel.NUM_HEADS, 1)).squeeze(2)
        loss_main = F.binary_cross_entropy_with_logits(sel, labels)
        # Loss 2 (v53): dense pixel BCE on frame_changed head[0] coord logits (B, 4096)
        coord_logits_fc = logits[:, 0, 5:]  # (B, 4096)
        loss_pixel = F.binary_cross_entropy_with_logits(coord_logits_fc, change_maps)
        loss = loss_main + 0.5 * loss_pixel
        loss.backward()
        s.optimizer.step()

    def _reset_for_new_level(s):
        s.experience_buffer.clear()
        s.experience_hashes.clear()
        s.action_model = ActionModel(input_channels=s.num_colours, grid_size=s.grid_size).to(s.device)
        # Load pretrained weights from BFS-trajectory pretrain (~10k tuples,
        # 20k steps, BCE on frame_changed, val acc 0.95). Falls back to random
        # init silently if weights aren't on disk (eg. local-eval).
        import os as _os
        for wp in ['/kaggle/input/sg-pretrained-weights/sg_action_model.pt',
                   'runs/sg_pretrain/sg_action_model.pt']:
            if _os.path.exists(wp):
                try:
                    ck = torch.load(wp, map_location=s.device, weights_only=False)
                    s.action_model.load_state_dict(ck['model'])
                    logger.info(f"SG pretrained weights loaded from {wp}")
                    break
                except Exception as _e:
                    logger.warning(f"SG pretrained load failed: {_e!r}")
        s.optimizer = optim.Adam(s.action_model.parameters(), lr=0.0001)
        s.prev_frame = None
        s.prev_action_idx = None

    def is_done(s, frames, lf):
        return (lf.state is GameState.WIN
                or (time.time() - s.start_time) >= 8 * 3600 - 300)

    def choose_action(s, frames, lf):
        try:
            # Level change → reset model + buffer
            score = getattr(lf, 'score', None) or lf.levels_completed
            if score != s.current_score:
                s._reset_for_new_level()
                s.current_score = score

            if lf.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                s.prev_frame = None
                s.prev_action_idx = None
                a = GameAction.RESET
                a.reasoning = "reset"
                return a

            cur_tensor = s._frame_to_tensor(lf)
            cur_np = cur_tensor.cpu().numpy().astype(bool)
            cur_grid = np.array(lf.frame, dtype=np.int64)[-1]
            cur_score = getattr(lf, 'score', None) or lf.levels_completed
            cur_level = lf.levels_completed or 0
            cur_state_hash = hashlib.md5(cur_np.tobytes()).hexdigest()

            # Build (prev_state, action) → 4 labels + per-pixel change map
            if s.prev_frame is not None and s.prev_action_idx is not None and s.prev_grid is not None:
                eh = s._experience_hash(s.prev_frame, s.prev_action_idx)
                if eh not in s.experience_hashes:
                    # v53: per-pixel change map (64x64 bool) — dense supervision
                    change_map = (s.prev_grid != cur_grid).astype(np.float32)
                    frame_changed = float(bool(change_map.sum()))
                    score_up = float(cur_score > s.prev_score)
                    novel_state = float(cur_state_hash not in s.seen_state_hashes)
                    level_up = float(cur_level > s.prev_level)
                    labels = [frame_changed, score_up, novel_state, level_up]
                    s.experience_buffer.append({
                        'state': s.prev_frame,
                        'action_idx': s.prev_action_idx,
                        'labels': labels,
                        'change_map': change_map,
                    })
                    s.experience_hashes.add(eh)
            s.seen_state_hashes.add(cur_state_hash)
            s.prev_score = cur_score
            s.prev_level = cur_level
            s.prev_grid = cur_grid

            avail = getattr(lf, 'available_actions', None) or []
            with torch.no_grad():
                logits = s.action_model(cur_tensor.unsqueeze(0)).squeeze(0)
            action_idx, coords = s._sample(logits, avail)

            if action_idx < 5:
                sel = s.action_list[action_idx]
                sel.reasoning = f"sg:a{action_idx + 1}"
                unified_idx = action_idx
            else:
                sel = GameAction.ACTION6
                y, x = coords
                sel.set_data({"x": x, "y": y})
                sel.reasoning = f"sg:click({x},{y})"
                unified_idx = 5 + (y * s.grid_size + x)

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
