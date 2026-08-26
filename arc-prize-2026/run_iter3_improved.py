"""ARC-AGI-3 Iteration 3: Key Improvements Over Winner Reproduction

Changes from iter2:
1. DON'T reset model between levels (winner's TODO) - keep weights, clear buffer only
2. Frame segmentation for click games - detect objects, click on them not background
3. Action sequence replay - store L1 solution, replay on GAME_OVER
4. Adaptive time allocation - more time for promising games

Target: Complete Level 2+ on at least some games
"""

import json, time, datetime, hashlib, random
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import ndimage

import arc_agi
from arcengine.enums import GameAction, GameState

ACTION_MAP = {a.value: a for a in GameAction}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)
DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("experiments"); RESULTS_DIR.mkdir(exist_ok=True)

print(f"Device: {DEVICE}")


# ─── ActionModel (same architecture as winner) ───────────────────────
class ActionModel(nn.Module):
    def __init__(self, nc=16, gs=64):
        super().__init__()
        self.conv1 = nn.Conv2d(nc, 32, 3, padding=1)
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
        f = F.relu(self.conv4(x))
        a = self.action_pool(f).view(f.size(0), -1)
        a = self.dropout(F.relu(self.action_fc(a)))
        al = self.action_head(a)
        c = F.relu(self.coord_conv1(f))
        c = F.relu(self.coord_conv2(c))
        c = F.relu(self.coord_conv3(c))
        cl = self.coord_conv4(c).view(c.size(0), -1)
        return torch.cat([al, cl], dim=1)


# ─── Frame Segmentation (NEW) ────────────────────────────────────────
def segment_frame(frame, bg_color=0, min_size=2):
    """Detect objects in the 64x64 grid. Returns list of (cy, cx, color, size)."""
    objects = []
    for color in range(16):
        if color == bg_color:
            continue
        mask = (frame == color)
        if not mask.any():
            continue
        labels, n = ndimage.label(mask)
        for i in range(1, n + 1):
            region = (labels == i)
            size = int(region.sum())
            if size < min_size:
                continue
            cy, cx = ndimage.center_of_mass(region)
            objects.append((int(cy), int(cx), color, size))
    return objects


def create_click_mask(frame, grid_size=64):
    """Create a mask that prioritizes clicking on objects, not background."""
    objects = segment_frame(frame)
    if not objects:
        return None  # no objects found, don't mask

    mask = np.zeros((grid_size, grid_size), dtype=np.float32)
    for cy, cx, color, size in objects:
        # Gaussian blob around each object centroid
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < grid_size and 0 <= nx < grid_size:
                    dist = (dy ** 2 + dx ** 2) ** 0.5
                    mask[ny, nx] = max(mask[ny, nx], np.exp(-dist / 2))
    return mask


# ─── Helper functions ─────────────────────────────────────────────────
def frame_to_onehot(grid, device="cpu"):
    f = np.array(grid, dtype=np.int64)
    if f.ndim == 3: f = f[-1]
    t = torch.zeros(16, 64, 64, dtype=torch.float32)
    t.scatter_(0, torch.from_numpy(f).unsqueeze(0).clamp(0, 15), 1)
    return t.to(device)


def sample_action(model, frame_t, avail, device, click_mask=None):
    """Sample with optional click mask for object-aware clicking."""
    NC = 4096
    with torch.no_grad():
        logits = model(frame_t.unsqueeze(0)).squeeze(0)

    al, cl = logits[:5], logits[5:]
    mask = torch.full((5,), float("-inf"), device=device)
    a6 = False
    for a in avail:
        v = a.value if hasattr(a, "value") else a
        if 1 <= v <= 5: mask[v - 1] = 0.
        elif v == 6: a6 = True
    al = al + mask
    if not a6:
        cl = cl + torch.full_like(cl, float("-inf"))

    ap = torch.sigmoid(al)
    cp = torch.sigmoid(cl) / NC

    # Apply click mask if available (boost object pixels, suppress background)
    if click_mask is not None and a6:
        cm_tensor = torch.from_numpy(click_mask.flatten()).float().to(device)
        # Blend: 70% model + 30% mask guidance
        cp = cp * (0.7 + 0.3 * cm_tensor)

    probs = torch.cat([ap, cp])
    s = probs.sum()
    if s < 1e-10:
        probs = torch.ones_like(probs)
        probs[5:] = 1. / NC
        s = probs.sum()
    probs = probs / s
    return np.random.choice(len(probs.cpu().numpy()), p=probs.cpu().numpy())


# ─── Play game with improvements ─────────────────────────────────────
def play_game(arcade, env_info, time_budget=600):
    game_start = time.time()
    env = arcade.make(env_info.game_id)
    frame = env.reset()

    avail = [ACTION_MAP[a] for a in frame.available_actions]
    has_click = any(a.value == 6 for a in avail if hasattr(a, "value"))

    # IMPROVEMENT 1: Don't reset model between levels
    model = ActionModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    buffer = deque(maxlen=200000)
    seen_hashes = set()

    cf = frame_to_onehot(frame._frame[0], DEVICE)
    prev_np = cf.cpu().numpy().astype(bool)
    prev_idx = None
    current_levels = 0
    total_actions = 0
    level_times = []

    # IMPROVEMENT 3: Store action sequences for replay
    current_level_actions = []
    solved_sequences = {}  # level -> action sequence

    print(f"\n  Playing {env_info.title} ({','.join(env_info.tags)})")
    print(f"    Actions: {[a.name for a in avail]}, Click: {has_click}")
    print(f"    Levels: {len(env_info.baseline_actions)}, Human: {sum(env_info.baseline_actions)}")

    while time.time() - game_start < time_budget:
        # Level change
        if frame.levels_completed > current_levels:
            el = time.time() - game_start
            level_times.append({"level": frame.levels_completed, "action": total_actions, "time": round(el, 1)})
            print(f"    Level {frame.levels_completed} at action {total_actions} ({el:.0f}s)")

            # Save solution sequence for this level
            solved_sequences[current_levels] = current_level_actions.copy()
            current_level_actions = []
            current_levels = frame.levels_completed

            # IMPROVEMENT 1: Only clear buffer, keep model weights!
            buffer.clear()
            seen_hashes.clear()
            # Reduce learning rate for fine-tuning on new level
            for pg in optimizer.param_groups:
                pg["lr"] = max(1e-5, pg["lr"] * 0.8)
            prev_np = None
            prev_idx = None

        # Handle game over -> replay known solutions then continue
        if frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            frame = env.step(GameAction.RESET)

            # IMPROVEMENT 3: Replay known solutions for completed levels
            for lvl in sorted(solved_sequences.keys()):
                if frame.state in (GameState.WIN, GameState.GAME_OVER):
                    break
                seq = solved_sequences[lvl]
                for action_tuple in seq:
                    if frame.state != GameState.NOT_FINISHED:
                        break
                    act, data = action_tuple
                    frame = env.step(act, data=data if data else None)
                    total_actions += 1

            cf = frame_to_onehot(frame._frame[0], DEVICE) if frame._frame else cf
            prev_np = cf.cpu().numpy().astype(bool)
            prev_idx = None
            continue

        if frame.state == GameState.WIN:
            print(f"    WIN! in {total_actions} actions")
            break

        # Get frame and optionally segment for click guidance
        raw_grid = np.array(frame._frame[0], dtype=np.int64)
        if raw_grid.ndim == 3: raw_grid = raw_grid[-1]

        cf = frame_to_onehot(frame._frame[0], DEVICE)
        cnp = cf.cpu().numpy().astype(bool)

        # IMPROVEMENT 2: Frame segmentation for click games
        click_mask = None
        if has_click and total_actions % 10 == 0:  # recompute every 10 steps
            click_mask = create_click_mask(raw_grid)

        # Store experience
        if prev_np is not None and prev_idx is not None:
            h = hashlib.md5(prev_np.tobytes() + str(prev_idx).encode()).hexdigest()
            if h not in seen_hashes:
                changed = not np.array_equal(prev_np, cnp)
                buffer.append({"state": prev_np, "action_idx": prev_idx,
                              "reward": 1.0 if changed else 0.0})
                seen_hashes.add(h)

        # Select action
        idx = sample_action(model, cf, avail, DEVICE, click_mask)
        if idx < 5:
            action = ACTION_MAP[idx + 1]
            data = None
            frame = env.step(action)
        else:
            ci = idx - 5
            y, x = ci // 64, ci % 64
            action = ACTION_MAP[6]
            data = {"x": int(x), "y": int(y)}
            frame = env.step(action, data=data)

        # Record for replay
        current_level_actions.append((action, data))
        total_actions += 1
        prev_np = cnp
        prev_idx = idx

        # Train
        if total_actions % 5 == 0 and len(buffer) >= 64:
            model.train()
            bi = np.random.choice(len(buffer), 64, replace=False)
            batch = [buffer[i] for i in bi]
            s = torch.stack([torch.from_numpy(b["state"]).float().to(DEVICE) for b in batch])
            ai = torch.tensor([b["action_idx"] for b in batch], dtype=torch.long, device=DEVICE)
            r = torch.tensor([b["reward"] for b in batch], dtype=torch.float32, device=DEVICE)
            optimizer.zero_grad()
            lg = model(s)
            sel = lg.gather(1, ai.unsqueeze(1)).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(sel, r)
            ap = torch.sigmoid(lg)
            loss = loss - 1e-4 * ap[:, :5].mean() - 1e-5 * ap[:, 5:].mean()
            loss.backward()
            optimizer.step()
            model.eval()

        if total_actions % 2000 == 0:
            el = time.time() - game_start
            print(f"      {total_actions} actions, {len(buffer)} exp, "
                  f"lvls={frame.levels_completed}, {total_actions/el:.0f} act/s")

    elapsed = time.time() - game_start
    return {
        "title": env_info.title, "game_id": env_info.game_id,
        "tags": env_info.tags, "has_click": has_click,
        "levels_completed": frame.levels_completed if frame else 0,
        "win_levels": len(env_info.baseline_actions),
        "total_actions": total_actions,
        "elapsed": round(elapsed, 1),
        "acts_per_sec": round(total_actions / max(elapsed, 1), 1),
        "level_times": level_times,
        "human_baseline": sum(env_info.baseline_actions),
        "solved_sequences_count": len(solved_sequences),
    }


# ─── Main with adaptive time allocation ──────────────────────────────
def main():
    print("=" * 70)
    print("ARC-AGI-3 Iter 3: No-Reset + Segmentation + Replay")
    print("=" * 70)

    arcade = arc_agi.Arcade()
    envs = arcade.get_environments()
    envs_sorted = sorted(envs, key=lambda e: sum(e.baseline_actions))

    # IMPROVEMENT 4: Adaptive time allocation
    # Phase 1: Quick scan (60s each) to identify promising games
    SCAN_TIME = 60
    DEEP_TIME = 600  # 10 min for promising games

    print(f"\n--- Phase 1: Quick scan ({SCAN_TIME}s each) ---")
    scan_results = []
    for i, env_info in enumerate(envs_sorted):
        print(f"[{i+1:2d}/{len(envs_sorted)}]", end="")
        result = play_game(arcade, env_info, time_budget=SCAN_TIME)
        scan_results.append(result)

    # Identify promising games (completed L1 or high frame-change engagement)
    promising = [r for r in scan_results if r["levels_completed"] > 0]
    print(f"\n--- Phase 1 Results: {len(promising)}/{len(scan_results)} games reached L1 ---")
    for r in promising:
        print(f"  {r['title']:5s}: L{r['levels_completed']}, {r['total_actions']} actions")

    # Phase 2: Deep dive on promising games
    if promising:
        print(f"\n--- Phase 2: Deep dive ({DEEP_TIME}s each) on {len(promising)} games ---")
        deep_results = []
        for r in promising:
            env_info = [e for e in envs if e.game_id == r["game_id"]][0]
            print(f"[Deep]", end="")
            result = play_game(arcade, env_info, time_budget=DEEP_TIME)
            deep_results.append(result)

        # Summary
        total_levels = sum(r["levels_completed"] for r in deep_results)
        print(f"\n--- Phase 2 Results ---")
        for r in deep_results:
            lt = f"  L{[l['level'] for l in r['level_times']]}" if r["level_times"] else ""
            print(f"  {r['title']:5s}: {r['levels_completed']}/{r['win_levels']} lvls, "
                  f"{r['total_actions']} acts{lt}")
    else:
        deep_results = scan_results
        total_levels = sum(r["levels_completed"] for r in scan_results)

    # Save results
    all_results = {"scan": scan_results, "deep": deep_results if promising else [],
                   "total_levels": total_levels}
    with open(DATA_DIR / "iter3_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Log
    results_file = RESULTS_DIR / "results.tsv"
    row = "\t".join([
        "0004", datetime.datetime.now().isoformat(), "iter3_improved",
        "No-reset + segmentation + replay + adaptive time",
        "N/A", str(total_levels), str(sum(r["total_actions"] for r in (deep_results if promising else scan_results))),
        "completed", str(round(sum(r["elapsed"] for r in scan_results + (deep_results if promising else [])), 1)),
        json.dumps({"scan_time": SCAN_TIME, "deep_time": DEEP_TIME, "device": DEVICE}),
        f"{total_levels} levels, {len(promising)} promising games",
    ])
    with open(results_file, "a") as f:
        f.write(row + "\n")

    print(f"\n{'='*70}")
    print(f"TOTAL: {total_levels} levels")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
