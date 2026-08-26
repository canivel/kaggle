"""ARC-AGI-3 Iteration 2: Faithful reproduction of 1st place solution (StochasticGoose)

Key differences from iter1:
- Supervised BCE loss (predict frame changes), NOT RL/PPO
- One-hot encoding (16 channels), not learned embeddings
- Hash-based deduplication of (state, action) pairs
- Reset model + buffer between levels
- Train inline every 5 actions
- Sigmoid + normalize sampling, not softmax
- 4-layer CNN backbone (same as winner)

This is "informed random search" - learn which actions change state, bias exploration
toward those actions.

Reference: https://github.com/DriesSmit/ARC3-solution
"""

import json
import time
import datetime
import hashlib
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import arc_agi
from arcengine.enums import GameAction, GameState

# ─── Constants ────────────────────────────────────────────────────────
ACTION_MAP = {a.value: a for a in GameAction}
RESULTS_DIR = Path("experiments")
CHECKPOINT_DIR = Path("checkpoints/iter2")
DATA_DIR = Path("data")
for d in [RESULTS_DIR, CHECKPOINT_DIR, DATA_DIR]:
    d.mkdir(exist_ok=True, parents=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
TIME_PER_GAME_SECONDS = 180  # 3 min per game (local testing; winner used 8 hours)
TRAIN_FREQUENCY = 5
BATCH_SIZE = 64
LR = 1e-4
BUFFER_SIZE = 200_000

np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"Device: {DEVICE}")


# ─── ActionModel (faithful to winner) ────────────────────────────────
class ActionModel(nn.Module):
    """CNN that predicts which actions cause frame changes.

    Input: 16-channel one-hot encoded 64x64 grid
    Output: 5 action logits + 4096 coordinate logits = 4101 total
    """

    def __init__(self, n_colors=16, grid_size=64):
        super().__init__()
        self.grid_size = grid_size
        self.n_actions = 5  # ACTION1-ACTION5

        # Shared convolutional backbone
        self.conv1 = nn.Conv2d(n_colors, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)

        # Action head: MaxPool -> flatten -> FC -> 5 logits
        self.action_pool = nn.MaxPool2d(4, 4)  # 64x64 -> 16x16
        self.action_fc = nn.Linear(256 * 16 * 16, 512)
        self.action_head = nn.Linear(512, self.n_actions)
        self.dropout = nn.Dropout(0.2)

        # Coordinate head: fully convolutional -> 64x64 spatial logits
        self.coord_conv1 = nn.Conv2d(256, 128, 3, padding=1)
        self.coord_conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.coord_conv3 = nn.Conv2d(64, 32, 1)
        self.coord_conv4 = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        # Shared backbone
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        features = F.relu(self.conv4(x))  # [B, 256, 64, 64]

        # Action head
        a = self.action_pool(features)          # [B, 256, 16, 16]
        a = a.view(a.size(0), -1)               # [B, 65536]
        a = F.relu(self.action_fc(a))           # [B, 512]
        a = self.dropout(a)
        action_logits = self.action_head(a)     # [B, 5]

        # Coordinate head
        c = F.relu(self.coord_conv1(features))  # [B, 128, 64, 64]
        c = F.relu(self.coord_conv2(c))         # [B, 64, 64, 64]
        c = F.relu(self.coord_conv3(c))         # [B, 32, 64, 64]
        c = self.coord_conv4(c)                 # [B, 1, 64, 64]
        coord_logits = c.view(c.size(0), -1)    # [B, 4096]

        return torch.cat([action_logits, coord_logits], dim=1)  # [B, 4101]


# ─── Helper functions ─────────────────────────────────────────────────
def frame_to_onehot(frame_grid, n_colors=16, device="cpu"):
    """Convert (64,64) int8 grid to (16, 64, 64) one-hot tensor."""
    frame = np.array(frame_grid, dtype=np.int64)
    if frame.ndim == 3:
        frame = frame[-1]  # take last frame if animation
    tensor = torch.zeros(n_colors, 64, 64, dtype=torch.float32)
    tensor.scatter_(0, torch.from_numpy(frame).unsqueeze(0).clamp(0, 15), 1)
    return tensor.to(device)


def compute_hash(frame_np, action_idx):
    """MD5 hash of one-hot frame + action index for deduplication."""
    h = hashlib.md5(frame_np.tobytes() + str(action_idx).encode()).hexdigest()
    return h


def sample_action(model, frame_tensor, available_actions, device="cpu"):
    """Sample action using winner's sigmoid + normalize approach.

    Returns: action_idx (0-4 for ACTION1-5, 5+ for coordinates),
             GameAction, data dict, prob
    """
    n_coords = 64 * 64

    with torch.no_grad():
        logits = model(frame_tensor.unsqueeze(0)).squeeze(0)  # [4101]

    action_logits = logits[:5]
    coord_logits = logits[5:]

    # Mask unavailable actions
    action_mask = torch.full((5,), float("-inf"), device=device)
    action6_available = False
    for a in available_actions:
        a_val = a.value if hasattr(a, "value") else a
        if 1 <= a_val <= 5:
            action_mask[a_val - 1] = 0.0
        elif a_val == 6:
            action6_available = True

    action_logits = action_logits + action_mask
    if not action6_available:
        coord_logits = coord_logits + torch.full_like(coord_logits, float("-inf"))

    # Sigmoid (not softmax!) + scale coordinates for fair sampling
    action_probs = torch.sigmoid(action_logits)
    coord_probs = torch.sigmoid(coord_logits) / n_coords

    all_probs = torch.cat([action_probs, coord_probs])
    total = all_probs.sum()
    if total < 1e-10:
        # Fallback: uniform over available
        all_probs = torch.zeros_like(all_probs)
        for a in available_actions:
            a_val = a.value if hasattr(a, "value") else a
            if 1 <= a_val <= 5:
                all_probs[a_val - 1] = 1.0
            elif a_val == 6:
                all_probs[5:] = 1.0 / n_coords
        total = all_probs.sum()

    all_probs = all_probs / total
    all_probs_np = all_probs.cpu().numpy()

    selected_idx = np.random.choice(len(all_probs_np), p=all_probs_np)

    if selected_idx < 5:
        action = ACTION_MAP[selected_idx + 1]  # ACTION1-5
        return selected_idx, action, {}, all_probs_np[selected_idx]
    else:
        coord_idx = selected_idx - 5
        y = coord_idx // 64
        x = coord_idx % 64
        action = ACTION_MAP[6]  # ACTION6
        return selected_idx, action, {"x": int(x), "y": int(y)}, all_probs_np[selected_idx]


# ─── Per-game agent loop ─────────────────────────────────────────────
def play_game(arcade, env_info, time_budget=TIME_PER_GAME_SECONDS):
    """Play one game using the winner's approach.

    - Train from scratch within each game
    - Reset model + buffer between levels
    - Hash-based dedup
    - Train every TRAIN_FREQUENCY actions
    """
    game_start = time.time()
    env = arcade.make(env_info.game_id)
    frame = env.reset()

    # Initialize model
    model = ActionModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    buffer = deque(maxlen=BUFFER_SIZE)
    seen_hashes = set()

    available = frame.available_actions
    action_space = [ACTION_MAP[a] for a in available]
    current_frame = frame_to_onehot(frame._frame[0], device=DEVICE)
    prev_frame_np = current_frame.cpu().numpy().astype(bool)
    prev_action_idx = None
    current_levels = 0
    total_actions = 0
    total_train_steps = 0

    print(f"\n  Playing {env_info.title} ({','.join(env_info.tags)})")
    print(f"    Actions: {[a.name for a in action_space]}")
    print(f"    Levels: {frame.win_levels}, Human baseline: {sum(env_info.baseline_actions)}")

    while time.time() - game_start < time_budget:
        # Check for level change -> reset everything
        if frame.levels_completed > current_levels:
            print(f"    Level {frame.levels_completed} reached! "
                  f"(action {total_actions}, {time.time()-game_start:.0f}s)")
            current_levels = frame.levels_completed
            buffer.clear()
            seen_hashes.clear()
            model = ActionModel().to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=LR)
            prev_frame_np = None
            prev_action_idx = None

        # Handle game over -> reset
        if frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            frame = env.step(GameAction.RESET)
            current_frame = frame_to_onehot(frame._frame[0], device=DEVICE)
            prev_frame_np = current_frame.cpu().numpy().astype(bool)
            prev_action_idx = None
            continue

        # Win -> done
        if frame.state == GameState.WIN:
            print(f"    WIN! All {frame.win_levels} levels in {total_actions} actions")
            break

        # Store experience from previous action
        current_frame = frame_to_onehot(frame._frame[0], device=DEVICE)
        curr_np = current_frame.cpu().numpy().astype(bool)

        if prev_frame_np is not None and prev_action_idx is not None:
            exp_hash = compute_hash(prev_frame_np, prev_action_idx)
            if exp_hash not in seen_hashes:
                frame_changed = not np.array_equal(prev_frame_np, curr_np)
                buffer.append({
                    "state": prev_frame_np,
                    "action_idx": prev_action_idx,
                    "reward": 1.0 if frame_changed else 0.0,
                })
                seen_hashes.add(exp_hash)

        # Select action
        action_idx, action, data, prob = sample_action(
            model, current_frame, action_space, device=DEVICE
        )

        # Step environment
        frame = env.step(action, data=data if data else None)
        total_actions += 1

        # Store for next experience
        prev_frame_np = curr_np
        prev_action_idx = action_idx

        # Train periodically
        if total_actions % TRAIN_FREQUENCY == 0 and len(buffer) >= BATCH_SIZE:
            model.train()
            indices = np.random.choice(len(buffer), BATCH_SIZE, replace=False)
            batch = [buffer[i] for i in indices]

            states = torch.stack([
                torch.from_numpy(b["state"]).float().to(DEVICE) for b in batch
            ])
            action_indices = torch.tensor(
                [b["action_idx"] for b in batch], dtype=torch.long, device=DEVICE
            )
            rewards = torch.tensor(
                [b["reward"] for b in batch], dtype=torch.float32, device=DEVICE
            )

            logits = model(states)  # [B, 4101]
            selected_logits = logits.gather(1, action_indices.unsqueeze(1)).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(selected_logits, rewards)

            # Light entropy regularization
            all_probs = torch.sigmoid(logits)
            action_entropy = all_probs[:, :5].mean()
            coord_entropy = all_probs[:, 5:].mean()
            loss = loss - 0.0001 * action_entropy - 0.00001 * coord_entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_steps += 1

            model.eval()

    elapsed = time.time() - game_start

    result = {
        "title": env_info.title,
        "game_id": env_info.game_id,
        "tags": env_info.tags,
        "levels_completed": frame.levels_completed if frame else 0,
        "win_levels": env_info.baseline_actions.__len__(),
        "total_actions": total_actions,
        "total_train_steps": total_train_steps,
        "buffer_size": len(buffer),
        "unique_experiences": len(seen_hashes),
        "elapsed_seconds": round(elapsed, 1),
        "human_baseline": sum(env_info.baseline_actions),
    }

    print(f"    Result: {result['levels_completed']}/{result['win_levels']} levels, "
          f"{total_actions} actions, {total_train_steps} train steps, "
          f"{elapsed:.0f}s")

    return result


# ─── Main ─────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("ARC-AGI-3 Iter 2: Winner Reproduction (StochasticGoose)")
    print("=" * 70)

    arcade = arc_agi.Arcade()
    envs = arcade.get_environments()

    # Sort by difficulty
    envs_sorted = sorted(envs, key=lambda e: sum(e.baseline_actions))

    print(f"\n{len(envs_sorted)} environments, {TIME_PER_GAME_SECONDS}s per game")
    print(f"Estimated total time: {len(envs_sorted) * TIME_PER_GAME_SECONDS / 60:.0f} min\n")

    all_results = []
    start_time = time.time()

    for i, env_info in enumerate(envs_sorted):
        print(f"[{i+1:2d}/{len(envs_sorted)}]", end="")
        result = play_game(arcade, env_info, time_budget=TIME_PER_GAME_SECONDS)
        all_results.append(result)

    total_elapsed = time.time() - start_time
    total_levels = sum(r["levels_completed"] for r in all_results)
    total_win_levels = sum(r["win_levels"] for r in all_results)
    total_actions = sum(r["total_actions"] for r in all_results)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Levels completed: {total_levels}/{total_win_levels}")
    print(f"  Total actions: {total_actions}")
    print(f"  Time: {total_elapsed/60:.1f} min")
    print(f"  Games with progress:")
    for r in all_results:
        if r["levels_completed"] > 0:
            print(f"    {r['title']:5s}: {r['levels_completed']}/{r['win_levels']} levels "
                  f"in {r['total_actions']} actions")

    # Save results
    with open(DATA_DIR / "iter2_results.json", "w") as f:
        json.dump({
            "experiment": "iter2_winner_reproduction",
            "timestamp": datetime.datetime.now().isoformat(),
            "time_per_game": TIME_PER_GAME_SECONDS,
            "total_elapsed": round(total_elapsed, 1),
            "total_levels": total_levels,
            "total_win_levels": total_win_levels,
            "per_environment": all_results,
        }, f, indent=2, default=str)

    # Log to TSV
    results_file = RESULTS_DIR / "results.tsv"
    row = "\t".join([
        "0003",
        datetime.datetime.now().isoformat(),
        "winner_repro",
        f"StochasticGoose reproduction, {TIME_PER_GAME_SECONDS}s/game",
        "N/A",
        str(total_levels),
        str(total_actions),
        "completed",
        str(round(total_elapsed, 1)),
        json.dumps({"lr": LR, "batch_size": BATCH_SIZE, "time_per_game": TIME_PER_GAME_SECONDS,
                     "device": DEVICE}),
        f"{total_levels}/{total_win_levels} levels",
    ])
    with open(results_file, "a") as f:
        f.write(row + "\n")

    print(f"\nLogged to {results_file}")


if __name__ == "__main__":
    main()
