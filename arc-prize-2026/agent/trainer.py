"""Training loop for ARC-AGI-3 agents.

Uses the actual arc_agi SDK API:
- Frames are (64, 64) int8 grids with 16 colors
- Actions are GameAction enums with optional data dict (x,y for clicks)
- State is tracked via FrameDataRaw.state (NOT_FINISHED, WIN, GAME_OVER)
- available_actions varies per game

Training approach:
1. Collect trajectories from environments
2. Train policy with PPO (intrinsic reward for progress + level completion)
3. Optionally train world model for planning
"""

import json
import time
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import arc_agi
from arcengine.enums import GameAction, GameState

from .cnn_policy import ArcCNNPolicy, WorldModel, ACTION_MAP


class Transition:
    """Single environment transition."""
    __slots__ = ["grid", "action_val", "action_data", "reward",
                 "next_grid", "done", "levels_completed", "available_actions"]

    def __init__(self, grid, action_val, action_data, reward,
                 next_grid, done, levels_completed, available_actions):
        self.grid = grid
        self.action_val = action_val
        self.action_data = action_data
        self.reward = reward
        self.next_grid = next_grid
        self.done = done
        self.levels_completed = levels_completed
        self.available_actions = available_actions


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


class ArcTrainer:
    """RL trainer for ARC-AGI-3 environments."""

    def __init__(
        self,
        policy=None,
        world_model=None,
        lr=3e-4,
        gamma=0.99,
        device="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir="checkpoints",
    ):
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.policy = (policy or ArcCNNPolicy()).to(device)
        self.world_model = (world_model or WorldModel()).to(device)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.world_optimizer = optim.Adam(self.world_model.parameters(), lr=lr)

        self.gamma = gamma
        self.replay = ReplayBuffer()
        self.train_stats = []

    def collect_episode(self, arcade, env_info, max_actions=500, deterministic=False):
        """Collect one episode of experience from an environment.

        Reward shaping:
        - +10 for level completion
        - +0.01 for frame change (encourages exploration)
        - -0.001 per action (encourages efficiency)
        """
        env = arcade.make(env_info.game_id)
        frame = env.reset()

        available = frame.available_actions
        prev_grid = frame._frame[0].copy()
        prev_levels = 0
        transitions = []
        total_reward = 0.0

        human_total = sum(env_info.baseline_actions)
        action_cap = min(max_actions, human_total * 5)

        for step in range(action_cap):
            # Select action
            action, data, log_prob = self.policy.select_action(
                prev_grid, available, deterministic=deterministic
            )

            frame = env.step(action, data=data if data else None)

            curr_grid = frame._frame[0] if frame._frame else prev_grid
            done = frame.state in (GameState.WIN, GameState.GAME_OVER)

            # Reward shaping
            reward = -0.001  # step penalty
            if frame.levels_completed > prev_levels:
                reward += 10.0  # level completion bonus
            elif not np.array_equal(prev_grid, curr_grid):
                reward += 0.01  # frame change bonus

            total_reward += reward

            t = Transition(
                grid=prev_grid.copy(),
                action_val=action.value,
                action_data=data,
                reward=reward,
                next_grid=curr_grid.copy(),
                done=done,
                levels_completed=frame.levels_completed,
                available_actions=available,
            )
            self.replay.add(t)
            transitions.append(t)

            prev_grid = curr_grid.copy()
            prev_levels = frame.levels_completed

            if done:
                break

        return {
            "env_title": env_info.title,
            "n_steps": len(transitions),
            "levels_completed": frame.levels_completed,
            "win_levels": frame.win_levels,
            "total_reward": round(total_reward, 4),
            "final_state": str(frame.state),
        }

    def train_policy(self, batch_size=64, n_updates=10):
        """PPO-style policy update."""
        if len(self.replay) < batch_size:
            return {"policy_loss": float("nan")}

        self.policy.train()
        total_loss = 0

        for _ in range(n_updates):
            batch = self.replay.sample(min(batch_size, len(self.replay)))

            grids = torch.LongTensor(np.array([t.grid for t in batch])).to(self.device)
            actions = torch.LongTensor([t.action_val for t in batch]).to(self.device)
            rewards = torch.FloatTensor([t.reward for t in batch]).to(self.device)

            action_logits, coord_logits, values = self.policy(grids)

            # Mask unavailable actions per sample
            for i, t in enumerate(batch):
                mask = torch.full((self.policy.max_actions,), float("-inf"),
                                  device=self.device)
                for a in t.available_actions:
                    if a < self.policy.max_actions:
                        mask[a] = 0.0
                action_logits[i] += mask

            # Numerical stability
            action_logits = action_logits.clamp(min=-50, max=50)

            log_probs = F.log_softmax(action_logits, dim=1)
            selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Skip batch if NaN
            if torch.isnan(selected_log_probs).any():
                continue

            # Advantage
            advantages = rewards - values.squeeze(1).detach()

            # Policy loss (REINFORCE with baseline)
            policy_loss = -(selected_log_probs * advantages).mean()

            # Value loss
            value_loss = nn.MSELoss()(values.squeeze(1), rewards)

            # Entropy bonus
            probs = torch.softmax(action_logits, dim=1)
            entropy = -(probs * log_probs).sum(dim=1).mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            self.policy_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_optimizer.step()

            total_loss += loss.item()

        return {"policy_loss": round(total_loss / n_updates, 6)}

    def train_world_model(self, batch_size=64, n_updates=10):
        """Train world model to predict next grid state."""
        if len(self.replay) < batch_size:
            return {"world_loss": float("nan")}

        self.world_model.train()
        total_loss = 0
        total_acc = 0

        criterion = nn.CrossEntropyLoss()

        for _ in range(n_updates):
            batch = self.replay.sample(min(batch_size, len(self.replay)))

            grids = torch.LongTensor(np.array([t.grid for t in batch])).to(self.device)
            actions = torch.LongTensor([t.action_val for t in batch]).to(self.device)
            targets = torch.LongTensor(np.array([t.next_grid for t in batch])).to(self.device)

            pred_logits = self.world_model(grids, actions)  # [B, 16, 64, 64]
            loss = criterion(pred_logits, targets)

            self.world_optimizer.zero_grad()
            loss.backward()
            self.world_optimizer.step()

            # Accuracy (pixel-level)
            pred_colors = pred_logits.argmax(dim=1)
            acc = (pred_colors == targets).float().mean().item()

            total_loss += loss.item()
            total_acc += acc

        return {
            "world_loss": round(total_loss / n_updates, 6),
            "world_acc": round(total_acc / n_updates, 4),
        }

    def evaluate(self, arcade, env_info, max_actions=500):
        """Deterministic evaluation of policy on an environment."""
        self.policy.eval()
        env = arcade.make(env_info.game_id)
        frame = env.reset()

        available = frame.available_actions
        prev_grid = frame._frame[0].copy()
        total_actions = 0

        human_total = sum(env_info.baseline_actions)
        action_cap = min(max_actions, human_total * 5)

        for _ in range(action_cap):
            action, data, _ = self.policy.select_action(
                prev_grid, available, deterministic=True
            )
            frame = env.step(action, data=data if data else None)
            total_actions += 1

            curr_grid = frame._frame[0] if frame._frame else prev_grid
            prev_grid = curr_grid.copy()

            if frame.state in (GameState.WIN, GameState.GAME_OVER):
                break

        return {
            "env_title": env_info.title,
            "levels_completed": frame.levels_completed,
            "win_levels": frame.win_levels,
            "total_actions": total_actions,
            "final_state": str(frame.state),
        }

    def save_checkpoint(self, name="latest"):
        path = self.checkpoint_dir / name
        path.mkdir(exist_ok=True)

        torch.save(self.policy.state_dict(), path / "policy.pt")
        torch.save(self.world_model.state_dict(), path / "world_model.pt")
        torch.save({
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "world_optimizer": self.world_optimizer.state_dict(),
        }, path / "optimizers.pt")

        with open(path / "train_stats.json", "w") as f:
            json.dump(self.train_stats, f, indent=2)

        print(f"  Checkpoint saved: {path}")

    def load_checkpoint(self, name="latest"):
        path = self.checkpoint_dir / name
        if not path.exists():
            print(f"No checkpoint at {path}")
            return False

        self.policy.load_state_dict(
            torch.load(path / "policy.pt", map_location=self.device, weights_only=True)
        )
        self.world_model.load_state_dict(
            torch.load(path / "world_model.pt", map_location=self.device, weights_only=True)
        )

        opt = torch.load(path / "optimizers.pt", map_location=self.device, weights_only=True)
        self.policy_optimizer.load_state_dict(opt["policy_optimizer"])
        self.world_optimizer.load_state_dict(opt["world_optimizer"])

        print(f"  Checkpoint loaded: {path}")
        return True


