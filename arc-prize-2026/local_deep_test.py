"""Deep play test: CNN agent goes deep on promising games.
Sequential processing, 30 min per game, GPU.
"""

import json, time, hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import arc_agi
from arcengine.enums import GameAction as GA, GameState as GS
from agent_deep import DeepPlayAgent, ACTION_MAP


def compute_rhae(level_actions, baseline_actions):
    """Compute RHAE for one game."""
    if not level_actions:
        return 0.0
    n = len(baseline_actions)
    total_w = n * (n + 1) / 2
    score = 0.0
    for l in range(n):
        w = l + 1
        if l in level_actions:
            h = baseline_actions[l]
            a = level_actions[l]
            s = min(1.0, h / max(a, 1)) ** 2
            score += w * s
    return score / total_w


def play_deep(arcade, env_info, time_budget=1800, max_actions=100000, device="cuda"):
    """Play one game deeply with CNN agent."""
    env = arcade.make(env_info.game_id)
    frame = env.reset()

    avail = [ACTION_MAP[a] for a in frame.available_actions]
    agent = DeepPlayAgent(avail, device=device, seed=42 + hash(env_info.game_id))

    total_actions = 0
    levels_completed = 0
    level_actions = {}
    current_level_start = 0
    level_times = []
    t0 = time.time()

    while time.time() - t0 < time_budget and total_actions < max_actions:
        # Handle resets
        if frame.state in (GS.NOT_PLAYED, GS.GAME_OVER):
            frame = env.step(GA.RESET)
            agent.prev_hash = None
            agent.prev_action_val = None
            agent.prev_onehot = None

            # Replay solved levels
            replays = agent.get_replay_actions()
            for lvl in sorted(replays.keys()):
                if frame.state != GS.NOT_FINISHED:
                    break
                for action_val, data in replays[lvl]:
                    if frame.state != GS.NOT_FINISHED:
                        break
                    action = ACTION_MAP[action_val]
                    frame = env.step(action, data=data)
                    total_actions += 1
            continue

        if frame.state == GS.WIN:
            break

        # Level change
        if frame.levels_completed > levels_completed:
            acts = total_actions - current_level_start
            level_actions[levels_completed] = acts
            el = time.time() - t0
            level_times.append({"level": frame.levels_completed, "actions": acts, "time": round(el, 1)})
            print(f"    Level {frame.levels_completed} in {acts} actions ({el:.0f}s)")
            levels_completed = frame.levels_completed
            current_level_start = total_actions
            agent.on_level_change(levels_completed)

        # Early termination if stuck too long
        if agent.is_stuck:
            print(f"    Stuck at level {levels_completed + 1}, stopping early")
            break

        # Agent step
        action, data = agent.step(frame._frame[0])
        frame = env.step(action, data=data)
        total_actions += 1

        # Progress
        if total_actions % 5000 == 0:
            el = time.time() - t0
            print(f"      {total_actions} acts, L{levels_completed}, "
                  f"{len(agent.all_visited)} states, {el:.0f}s")

    elapsed = time.time() - t0
    rhae = compute_rhae(level_actions, env_info.baseline_actions)

    return {
        "title": env_info.title,
        "tags": env_info.tags,
        "levels_completed": levels_completed,
        "win_levels": len(env_info.baseline_actions),
        "total_actions": total_actions,
        "level_actions": {str(k): v for k, v in level_actions.items()},
        "level_times": level_times,
        "rhae": round(rhae, 6),
        "elapsed": round(elapsed, 1),
        "human_baseline": env_info.baseline_actions,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=int, default=1800, help="Seconds per game (default 30min)")
    parser.add_argument("--actions", type=int, default=100000)
    parser.add_argument("--games", type=int, default=5, help="Number of easiest games")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Deep Play Test: {args.games} games, {args.time}s/game, device={device}")
    print("=" * 70)

    arcade = arc_agi.Arcade()
    envs = sorted(arcade.get_environments(), key=lambda e: sum(e.baseline_actions))

    # Pick games that we know complete L1 quickly
    priority_titles = ["LP85", "R11L", "AR25", "TN36", "VC33", "SP80", "CN04", "M0R0", "CD82"]
    priority_envs = []
    for title in priority_titles[:args.games]:
        for e in envs:
            if e.title == title:
                priority_envs.append(e)
                break

    results = []
    t0 = time.time()

    for i, env_info in enumerate(priority_envs):
        print(f"\n[{i+1}/{len(priority_envs)}] {env_info.title} ({','.join(env_info.tags)})")
        print(f"  Levels: {len(env_info.baseline_actions)}, Human: {env_info.baseline_actions}")
        r = play_deep(arcade, env_info, args.time, args.actions, device)
        results.append(r)
        print(f"  Result: L{r['levels_completed']}/{r['win_levels']} "
              f"RHAE={r['rhae']:.4f} acts={r['total_actions']}")

    total_elapsed = time.time() - t0
    total_levels = sum(r["levels_completed"] for r in results)
    mean_rhae = np.mean([r["rhae"] for r in results])

    # Full RHAE as if 25 games (15 zeros for unsolved)
    full_rhae = sum(r["rhae"] for r in results) / 25

    print(f"\n{'='*70}")
    print(f"RESULTS: {total_levels} levels, {total_elapsed:.0f}s")
    print(f"  Mean RHAE ({len(results)} games): {mean_rhae:.6f}")
    print(f"  Full RHAE (as 25 games): {full_rhae:.6f}")
    print(f"  Target: 0.1")
    print(f"{'='*70}")
    for r in sorted(results, key=lambda x: -x["rhae"]):
        print(f"  {r['title']:5s}: L{r['levels_completed']}/{r['win_levels']} "
              f"RHAE={r['rhae']:.6f} acts={r['total_actions']} "
              f"levels: {r.get('level_times', [])}")

    Path("data").mkdir(exist_ok=True)
    with open("data/deep_play_results.json", "w") as f:
        json.dump({"results": results, "total_levels": total_levels,
                   "mean_rhae": mean_rhae, "full_rhae": full_rhae}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
