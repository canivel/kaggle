"""Local validation v2 - uses improved agent with BFS graph search."""

import json, time, hashlib, random
from pathlib import Path
from collections import defaultdict

import numpy as np
import arc_agi
from arcengine.enums import GameAction as GA, GameState as GS
from agent_local import ImprovedAgent, ACTION_MAP


def compute_rhae(level_actions, baseline_actions):
    if not level_actions:
        return 0.0
    total_weight = 0
    total_score = 0
    for l in range(len(baseline_actions)):
        w = l + 1
        total_weight += w
        if l in level_actions:
            h = baseline_actions[l]
            a = level_actions[l]
            s = min(1.0, h / max(a, 1)) ** 2
            total_score += w * s
    return total_score / total_weight if total_weight > 0 else 0.0


def run_game(arcade, env_info, time_budget=300, max_actions=10000):
    env = arcade.make(env_info.game_id)
    frame = env.reset()

    avail = [ACTION_MAP[a] for a in frame.available_actions]
    agent = ImprovedAgent(avail, seed=42 + hash(env_info.game_id))

    total_actions = 0
    levels_completed = 0
    level_actions = {}
    current_level_start = 0
    level_times = []
    t0 = time.time()

    while time.time() - t0 < time_budget and total_actions < max_actions:
        if frame.state in (GS.NOT_PLAYED, GS.GAME_OVER):
            frame = env.step(GA.RESET)
            agent.prev_hash = None
            agent.prev_action = None
            continue

        if frame.state == GS.WIN:
            break

        # Level change
        if frame.levels_completed > levels_completed:
            actions_this_level = total_actions - current_level_start
            level_actions[levels_completed] = actions_this_level
            level_times.append({
                "level": frame.levels_completed,
                "actions": actions_this_level,
                "time": round(time.time() - t0, 1)
            })
            print(f"    Level {frame.levels_completed} in {actions_this_level} actions ({time.time()-t0:.0f}s)")
            levels_completed = frame.levels_completed
            current_level_start = total_actions
            agent.on_level_change(levels_completed)

        # Early termination if stuck
        if agent.is_stuck:
            break

        # Choose action
        action, data = agent.choose_action(frame._frame[0])
        frame = env.step(action, data=data)
        total_actions += 1

    elapsed = time.time() - t0
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
        "visited_states": len(agent.all_visited),
        "graph_edges": sum(len(v) for v in agent.state_graph.values()),
        "human_baseline": sum(env_info.baseline_actions),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=int, default=300)
    parser.add_argument("--actions", type=int, default=10000)
    parser.add_argument("--games", type=int, default=25)
    args = parser.parse_args()

    print("=" * 70)
    print(f"Local Validation v2 (BFS agent): {args.games} games, {args.time}s, {args.actions} max")
    print("=" * 70)

    arcade = arc_agi.Arcade()
    envs = sorted(arcade.get_environments(), key=lambda e: sum(e.baseline_actions))[:args.games]

    results = []
    t0 = time.time()

    for i, env_info in enumerate(envs):
        print(f"[{i+1:2d}/{len(envs)}] {env_info.title:5s} ({','.join(env_info.tags):15s})", end=" ", flush=True)
        r = run_game(arcade, env_info, args.time, args.actions)
        status = f"L{r['levels_completed']}" if r['levels_completed'] > 0 else "---"
        print(f"{status:4s} RHAE={r['rhae']:.4f} acts={r['total_actions']:5d} "
              f"states={r['visited_states']:5d} edges={r['graph_edges']:5d}")
        results.append(r)

    elapsed = time.time() - t0
    total_levels = sum(r["levels_completed"] for r in results)
    mean_rhae = np.mean([r["rhae"] for r in results])

    print(f"\n{'='*70}")
    print(f"RESULTS: {total_levels} levels, RHAE={mean_rhae:.6f}, {elapsed:.0f}s")
    print(f"{'='*70}")
    for r in sorted(results, key=lambda x: -x["rhae"]):
        if r["levels_completed"] > 0 or r["rhae"] > 0:
            print(f"  {r['title']:5s}: L{r['levels_completed']}/{r['win_levels']} "
                  f"RHAE={r['rhae']:.4f} acts={r['total_actions']}")

    Path("data").mkdir(exist_ok=True)
    with open("data/local_validation_v2.json", "w") as f:
        json.dump({"summary": {"levels": total_levels, "rhae": mean_rhae, "elapsed": elapsed},
                   "results": results}, f, indent=2, default=str)
    print(f"\nSaved to data/local_validation_v2.json")
    print(f"\nComparison: v1 baseline = 8 levels, RHAE=0.001744")
    print(f"            v2 improved = {total_levels} levels, RHAE={mean_rhae:.6f}")


if __name__ == "__main__":
    main()
