"""CNN Policy Network for ARC-AGI-3

Architecture based on the preview winner (StochasticGoose):
- Input: 64x64 grid with 16 possible colors (int8) -> learned color embedding
- 4-layer ConvNet: 32 -> 64 -> 128 -> 256 channels
- Separate heads for: action type, coordinate (x,y) for click actions, value

Key insight from live testing:
- Frame is (64, 64) int8 with values 0-15 (16-color palette)
- NOT RGB. Needs color embedding layer first.
- Action space varies per game: some have 4 actions (keyboard), some up to 7
- available_actions list tells which actions are valid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from arcengine.enums import GameAction

ACTION_MAP = {a.value: a for a in GameAction}


class ColorEmbedding(nn.Module):
    """Embed 16 discrete colors into a learned representation."""

    def __init__(self, n_colors=16, embed_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(n_colors, embed_dim)

    def forward(self, x):
        """x: [B, 64, 64] int8/long -> [B, embed_dim, 64, 64]"""
        x = x.long().clamp(0, 15)
        emb = self.embedding(x)  # [B, 64, 64, embed_dim]
        return emb.permute(0, 3, 1, 2)  # [B, embed_dim, 64, 64]


class ArcCNNPolicy(nn.Module):
    """CNN policy for ARC-AGI-3 environments.

    Input: 64x64 grid of color indices (0-15)
    Output: action logits + coordinate logits + value estimate
    """

    def __init__(self, n_colors=16, color_embed_dim=8, max_actions=8):
        super().__init__()
        self.max_actions = max_actions

        # Color embedding
        self.color_embed = ColorEmbedding(n_colors, color_embed_dim)

        # Convolutional backbone
        self.conv = nn.Sequential(
            nn.Conv2d(color_embed_dim, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 256-dim

        # Action type head (8 possible: RESET + ACTION1-7)
        self.action_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, max_actions),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Coordinate head for click actions (ACTION6)
        # Upsample back to 64x64
        self.coord_head = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),     # 32 -> 64
        )

    def forward(self, grid):
        """Forward pass.

        Args:
            grid: [B, 64, 64] int tensor (color indices 0-15)

        Returns:
            action_logits: [B, max_actions]
            coord_logits: [B, 64, 64]
            value: [B, 1]
        """
        x = self.color_embed(grid)       # [B, embed_dim, 64, 64]
        features = self.conv(x)           # [B, 256, 8, 8]

        pooled = self.global_pool(features).squeeze(-1).squeeze(-1)  # [B, 256]
        action_logits = self.action_head(pooled)   # [B, 8]
        value = self.value_head(pooled)            # [B, 1]
        coord_logits = self.coord_head(features).squeeze(1)  # [B, 64, 64]

        return action_logits, coord_logits, value

    def select_action(self, grid, available_actions, deterministic=False):
        """Select action given current grid and available actions.

        Args:
            grid: numpy array (64, 64) int8
            available_actions: list of int (valid action values)
            deterministic: if True, take argmax; else sample

        Returns:
            action: GameAction
            data: dict (with x,y for ACTION6)
            log_prob: float
        """
        x = torch.LongTensor(grid).unsqueeze(0)  # [1, 64, 64]

        with torch.no_grad():
            action_logits, coord_logits, value = self(x)

        # Mask unavailable actions
        mask = torch.full((self.max_actions,), float("-inf"))
        for a in available_actions:
            if a < self.max_actions:
                mask[a] = 0.0
        masked_logits = action_logits[0] + mask

        # Numerical stability: clamp before softmax
        masked_logits = masked_logits.clamp(min=-50, max=50)
        probs = F.softmax(masked_logits, dim=0)

        # Fallback: if NaN/all-zero, use uniform over available actions
        if torch.isnan(probs).any() or probs.sum() < 1e-8:
            probs = torch.zeros(self.max_actions)
            for a in available_actions:
                if a < self.max_actions:
                    probs[a] = 1.0
            probs = probs / probs.sum()

        if deterministic:
            action_idx = probs.argmax().item()
        else:
            action_idx = torch.multinomial(probs, 1).item()

        action = ACTION_MAP[action_idx]
        log_prob = probs[action_idx].log().item()
        data = {}

        # If ACTION6 (click), sample coordinate
        if action == GameAction.ACTION6:
            coord_probs = F.softmax(coord_logits[0].flatten(), dim=0)
            if deterministic:
                coord_idx = coord_probs.argmax().item()
            else:
                coord_idx = torch.multinomial(coord_probs, 1).item()
            data = {"x": coord_idx % 64, "y": coord_idx // 64}

        return action, data, log_prob


class WorldModel(nn.Module):
    """Predict next grid state given current grid and action.

    Learns the environment dynamics for model-based planning.
    Input: (64x64 grid, action) -> predicted next 64x64 grid
    """

    def __init__(self, n_colors=16, color_embed_dim=8, n_actions=8):
        super().__init__()
        self.color_embed = ColorEmbedding(n_colors, color_embed_dim)

        # Action embedding broadcast to spatial dims
        self.action_embed = nn.Embedding(n_actions, 4)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(color_embed_dim + 4, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 32
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 16
            nn.ReLU(),
        )

        # Decoder (predict next grid colors)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 64
            nn.ReLU(),
            nn.Conv2d(64, n_colors, 1),  # per-pixel color logits
        )

    def forward(self, grid, action_idx):
        """Predict next grid state.

        Args:
            grid: [B, 64, 64] int tensor
            action_idx: [B] int tensor (action values 0-7)

        Returns:
            next_grid_logits: [B, 16, 64, 64] (per-pixel color logits)
        """
        x = self.color_embed(grid)  # [B, embed_dim, 64, 64]

        # Broadcast action embedding spatially
        act_emb = self.action_embed(action_idx)  # [B, 4]
        act_spatial = act_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 64, 64)

        x = torch.cat([x, act_spatial], dim=1)
        features = self.encoder(x)
        return self.decoder(features)
