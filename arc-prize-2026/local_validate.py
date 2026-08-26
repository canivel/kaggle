"""Local validation harness - mimics Kaggle's gateway+Swarm but locally.
Runs the same agent code against all 25 public games and computes RHAE.
Usage: uv run python local_validate.py
"""

import json, time, hashlib, random, importlib.util, sys, os
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import arc_agi
from arcengine.enums import GameAction as GA, GameState as GS

ACTION_MAP = {a.value: a for a in GA}

# ─── RHAE Scoring (matches competition formula) ──────────────────────
def compute_rhae(levels_actions, baseline_actions):
    """Compute RHAE for one game.
    levels_actions: dict of {level_idx: agent_action_count}
    baseline_actions: list of human baseline per level
    """
    if not levels_actions:
        return 0.0

    total_weight = 0
    total_score = 0
    n_levels = len(baseline_actions)

    for l in range(n_levels):
        w = l + 1  # 1-indexed weight
        total_weight += w
        if l in levels_actions:
            h = baseline_actions[l]
            a = levels_actions[l]
            s = min(1.0, h / max(a, 1)) ** 2
            total_score += w * s

    return total_score / total_weight if total_weight > 0 else 0.0


# ─── Agent adapter (bridges arc-agi SDK <-> our agent logic) ─────────
class LocalAgentRunner:
    """Runs agent logic against local arc-agi SDK environments."""

    def __init__(self, agent_module_path="my_agent_local.py"):
        self.agent_path = agent_module_path

    def run_game(self, arcade, env_info, time_budget=300, max_actions=10000):
        """Run agent on one game, return metrics."""
        env = arcade.make(env_info.game_id)
        frame = env.reset()

        # Agent state (same as Kaggle agent but adapted for SDK)
        grid_size = 64
        prev_frame_hash = None
        prev_action_val = None
        current_score = 0
        visited_states = set()
        tried_actions = defaultdict(set)
        frame_change_actions = defaultdict(set)

        total_actions = 0
        levels_completed = 0
        level_actions = {}  # level_idx -> action_count
        current_level_start = 0
        level_times = []

        avail = [ACTION_MAP[a] for a in frame.available_actions]
        t0 = time.time()

        while time.time() - t0 < time_budget and total_actions < max_actions:
            # Handle game over
            if frame.state in (GS.NOT_PLAYED, GS.GAME_OVER):
                frame = env.step(GA.RESET)
                prev_frame_hash = None
                prev_action_val = None
                continue

            if frame.state == GS.WIN:
                break

            # Check level change
            if frame.levels_completed > levels_completed:
                actions_this_level = total_actions - current_level_start
                level_actions[levels_completed] = actions_this_level
                level_times.append({
                    "level": frame.levels_completed,
                    "actions": actions_this_level,
                    "time": round(time.time() - t0, 1)
                })
                levels_completed = frame.levels_completed
                current_level_start = total_actions
                visited_states.clear()
                tried_actions.clear()
                prev_frame_hash = None
                prev_action_val = None

            # Hash current frame
            raw = frame._frame[0]
            grid = np.array(raw, dtype=np.int8)
            if grid.ndim == 3: grid = grid[-1]
            frame_hash = hashlib.md5(grid.tobytes()).hexdigest()
            visited_states.add(frame_hash)

            # Record frame change from previous action
            if prev_frame_hash is not None and prev_action_val is not None:
                if frame_hash != prev_frame_hash:
                    frame_change_actions[prev_frame_hash].add(prev_action_val)

            # Choose action (same logic as Kaggle agent)
            tried = tried_actions[frame_hash]
            untried = [a for a in avail if a.value not in tried]

            if untried:
                action = random.choice(untried)
            else:
                productive_vals = frame_change_actions.get(frame_hash, set())
                productive = [a for a in avail if a.value in productive_vals]
                if productive:
                    action = random.choice(productive)
                else:
                    action = random.choice(avail)

            # Execute
            data = None
            if action == GA.ACTION6:
                nonzero = np.argwhere(grid != 0)
                if len(nonzero) > 0:
                    idx = random.randint(0, len(nonzero) - 1)
                    y, x = int(nonzero[idx][0]), int(nonzero[idx][1])
                else:
                    x, y = random.randint(0, 63), random.randint(0, 63)
                data = {"x": x, "y": y}

            frame = env.step(action, data=data)
            tried_actions[frame_hash].add(action.value)
            prev_frame_hash = frame_hash
            prev_action_val = action.value
            total_actions += 1

        elapsed = time.time() - t0

        # Record last level if in progress
        if levels_completed > 0 and levels_completed not in level_actions:
            level_actions[levels_completed] = total_actions - current_level_start

        rhae = compute_rhae(level_actions, env_info.baseline_actions)

        return {
            "title": env_info.title,
            "game_id": env_info.game_id,
            "tags": env_info.tags,
            "levels_completed": frame.levels_completed if frame else levels_completed,
            "win_levels": len(env_info.baseline_actions),
            "total_actions": total_actions,
            "level_actions": level_actions,
            "level_times": level_times,
            "rhae": round(rhae, 6),
            "elapsed": round(elapsed, 1),
            "acts_per_sec": round(total_actions / max(elapsed, 0.1), 1),
            "human_baseline": sum(env_info.baseline_actions),
            "visited_states": len(visited_states),
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=int, default=300, help="Seconds per game")
    parser.add_argument("--actions", type=int, default=10000, help="Max actions per game")
    parser.add_argument("--games", type=int, default=25, help="Number of games to test")
    parser.add_argument("--parallel", type=int, default=1, help="Parallel games (1=sequential)")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Local Validation: {args.games} games, {args.time}s/game, {args.actions} max actions")
    print("=" * 70)

    arcade = arc_agi.Arcade()
    envs = sorted(arcade.get_environments(), key=lambda e: sum(e.baseline_actions))
    envs = envs[:args.games]

    runner = LocalAgentRunner()
    results = []
    t0 = time.time()

    if args.parallel > 1:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(runner.run_game, arcade, ei, args.time, args.actions): ei
                for ei in envs
            }
            for i, future in enumerate(as_completed(futures)):
                r = future.result()
                results.append(r)
                status = f"L{r['levels_completed']}" if r['levels_completed'] > 0 else "---"
                print(f"[{i+1:2d}/{len(envs)}] {r['title']:5s} {status:4s} "
                      f"RHAE={r['rhae']:.4f} acts={r['total_actions']:5d} "
                      f"states={r['visited_states']:5d} {r['elapsed']:.0f}s")
    else:
        for i, env_info in enumerate(envs):
            r = runner.run_game(arcade, env_info, args.time, args.actions)
            results.append(r)
            status = f"L{r['levels_completed']}" if r['levels_completed'] > 0 else "---"
            print(f"[{i+1:2d}/{len(envs)}] {r['title']:5s} {status:4s} "
                  f"RHAE={r['rhae']:.4f} acts={r['total_actions']:5d} "
                  f"states={r['visited_states']:5d} {r['elapsed']:.0f}s")

    elapsed = time.time() - t0

    # Summary
    total_levels = sum(r["levels_completed"] for r in results)
    total_rhae = np.mean([r["rhae"] for r in results])

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"  Total levels: {total_levels}")
    print(f"  Mean RHAE: {total_rhae:.6f}")
    print(f"  Time: {elapsed:.0f}s")

    print(f"\n  Games with levels:")
    for r in sorted(results, key=lambda x: -x["rhae"]):
        if r["levels_completed"] > 0:
            print(f"    {r['title']:5s}: L{r['levels_completed']}/{r['win_levels']} "
                  f"RHAE={r['rhae']:.4f} acts={r['total_actions']}")

    print(f"\n  By tag:")
    by_tag = defaultdict(list)
    for r in results:
        tag = ",".join(r["tags"]) or "none"
        by_tag[tag].append(r)
    for tag, rs in sorted(by_tag.items()):
        avg_rhae = np.mean([r["rhae"] for r in rs])
        lvls = sum(r["levels_completed"] for r in rs)
        print(f"    {tag:20s}: {len(rs)} games, {lvls} levels, RHAE={avg_rhae:.4f}")

    # Save
    output = {
        "config": {"time_per_game": args.time, "max_actions": args.actions},
        "summary": {"total_levels": total_levels, "mean_rhae": round(total_rhae, 6),
                     "elapsed": round(elapsed, 1)},
        "results": results,
    }
    Path("data").mkdir(exist_ok=True)
    with open("data/local_validation.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to data/local_validation.json")


if __name__ == "__main__":
    main()
