"""ARC-AGI-3 Baseline Agent - Iteration 0

Approach: Random exploration across all 25 public environments
- Measure frame-change rates per environment/control type
- Track levels completed with random actions
- Establish RHAE floor for comparison
- Save trajectory data for analysis

Target: Establish baseline metrics (random is the floor)
"""

import json
import time
import datetime
from pathlib import Path

import numpy as np

import arc_agi
from arcengine.enums import GameAction, GameState

# ─── Constants ────────────────────────────────────────────────────────
ACTION_MAP = {a.value: a for a in GameAction}
RESULTS_DIR = Path("experiments")
DATA_DIR = Path("data")
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
SEED = 42
np.random.seed(SEED)


# ─── Environment Exploration ──────────────────────────────────────────
def explore_environment(arc, env_info, max_actions=500):
    """Random exploration of a single environment.

    Returns dict with trajectory data and metrics.
    """
    env = arc.make(env_info.game_id)
    frame = env.reset()

    available = frame.available_actions
    action_space = [ACTION_MAP[a] for a in available]

    trajectory = []
    total_actions = 0
    frame_changes = 0
    prev_grid = frame._frame[0].copy() if frame._frame else None
    start_time = time.time()

    # Cap at 5x human baseline total actions
    human_total = sum(env_info.baseline_actions)
    action_cap = min(max_actions, human_total * 5)

    while total_actions < action_cap:
        # Random action from available actions
        action_val = int(np.random.choice(available))
        action = ACTION_MAP[action_val]
        data = {}

        # ACTION6 needs x,y coordinates
        if action == GameAction.ACTION6:
            data = {"x": int(np.random.randint(0, 64)), "y": int(np.random.randint(0, 64))}

        frame = env.step(action, data=data)
        total_actions += 1

        # Check frame change
        curr_grid = frame._frame[0] if frame._frame else prev_grid
        changed = not np.array_equal(prev_grid, curr_grid)
        if changed:
            frame_changes += 1
        prev_grid = curr_grid.copy()

        trajectory.append({
            "step": total_actions,
            "action": action.name,
            "data": data,
            "frame_changed": changed,
            "levels_completed": frame.levels_completed,
            "state": str(frame.state),
        })

        if frame.state in (GameState.WIN, GameState.GAME_OVER):
            break

    elapsed = time.time() - start_time

    return {
        "game_id": env_info.game_id,
        "title": env_info.title,
        "tags": env_info.tags,
        "action_space": [a.name for a in action_space],
        "n_actions": len(action_space),
        "total_actions": total_actions,
        "action_cap": action_cap,
        "human_baseline_total": human_total,
        "frame_changes": frame_changes,
        "frame_change_rate": frame_changes / total_actions if total_actions > 0 else 0,
        "levels_completed": frame.levels_completed,
        "win_levels": frame.win_levels,
        "final_state": str(frame.state),
        "elapsed_seconds": round(elapsed, 2),
        "trajectory_length": len(trajectory),
    }


# ─── RHAE Calculation ─────────────────────────────────────────────────
def compute_rhae(result, env_info):
    """Compute RHAE score for an environment run.

    S(l,e) = min(1.0, h(l,e) / a(l,e))^2
    Environment score: sum(w_l * S(l,e)) / sum(w_l) where w_l = l (1-indexed)
    """
    levels_completed = result["levels_completed"]
    if levels_completed == 0:
        return 0.0

    # We don't have per-level action counts from random agent, so approximate:
    # Assume actions were spread evenly across completed levels
    # This is a rough approximation - real scoring needs per-level tracking
    total_agent_actions = result["total_actions"]
    baseline = env_info.baseline_actions

    # For completed levels, compute score
    score_sum = 0.0
    weight_sum = 0.0
    actions_per_level = total_agent_actions / levels_completed if levels_completed > 0 else total_agent_actions

    for l in range(levels_completed):
        w = l + 1  # 1-indexed weight
        h = baseline[l] if l < len(baseline) else baseline[-1]
        a = actions_per_level  # rough approximation
        s = min(1.0, h / a) ** 2
        score_sum += w * s
        weight_sum += w

    return score_sum / weight_sum if weight_sum > 0 else 0.0


# ─── Main ─────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("ARC-AGI-3 Baseline: Random Agent on 25 Public Environments")
    print("=" * 70)

    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    print(f"\n{len(envs)} environments loaded\n")

    all_results = []
    start_time = time.time()

    for i, env_info in enumerate(envs):
        print(f"[{i+1:2d}/25] {env_info.title:5s} ({','.join(env_info.tags):15s}) "
              f"| {len(env_info.baseline_actions)} levels, "
              f"human={sum(env_info.baseline_actions)} actions ... ", end="", flush=True)

        result = explore_environment(arc, env_info, max_actions=2000)
        rhae = compute_rhae(result, env_info)
        result["rhae_estimate"] = round(rhae, 6)

        all_results.append(result)

        print(f"done: {result['levels_completed']}/{result['win_levels']} levels, "
              f"{result['total_actions']} actions, "
              f"change_rate={result['frame_change_rate']:.0%}, "
              f"RHAE~{rhae:.4f}")

    total_elapsed = time.time() - start_time

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_levels = sum(r["levels_completed"] for r in all_results)
    total_win_levels = sum(r["win_levels"] for r in all_results)
    total_actions = sum(r["total_actions"] for r in all_results)
    avg_change_rate = np.mean([r["frame_change_rate"] for r in all_results])
    avg_rhae = np.mean([r["rhae_estimate"] for r in all_results])

    print(f"  Environments tested: {len(all_results)}")
    print(f"  Levels completed: {total_levels}/{total_win_levels}")
    print(f"  Total actions: {total_actions}")
    print(f"  Avg frame change rate: {avg_change_rate:.1%}")
    print(f"  Avg RHAE estimate: {avg_rhae:.6f}")
    print(f"  Time: {total_elapsed:.1f}s")

    # By control type
    print("\n  By control type:")
    by_tag = {}
    for r in all_results:
        tag = ",".join(r["tags"]) or "none"
        by_tag.setdefault(tag, []).append(r)
    for tag, results in sorted(by_tag.items()):
        avg_cr = np.mean([r["frame_change_rate"] for r in results])
        avg_rh = np.mean([r["rhae_estimate"] for r in results])
        lvls = sum(r["levels_completed"] for r in results)
        print(f"    {tag:20s}: {len(results)} envs, "
              f"change_rate={avg_cr:.0%}, levels={lvls}, RHAE~{avg_rh:.6f}")

    # ── Save results ──────────────────────────────────────────────
    # Trajectories summary (not full trajectories - too large)
    output = {
        "experiment": "random_baseline",
        "timestamp": datetime.datetime.now().isoformat(),
        "seed": SEED,
        "total_elapsed": round(total_elapsed, 1),
        "summary": {
            "total_levels": total_levels,
            "total_win_levels": total_win_levels,
            "total_actions": total_actions,
            "avg_frame_change_rate": round(avg_change_rate, 4),
            "avg_rhae": round(avg_rhae, 6),
        },
        "per_environment": all_results,
    }

    with open(DATA_DIR / "baseline_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved results to data/baseline_results.json")

    # Log to TSV
    results_file = RESULTS_DIR / "results.tsv"
    if not results_file.exists():
        header = "\t".join([
            "experiment_id", "timestamp", "agent_type", "description",
            "rhae_score", "levels_completed", "total_actions",
            "status", "duration_seconds", "params", "notes"
        ])
        results_file.write_text(header + "\n")

    row = "\t".join([
        "0001",
        datetime.datetime.now().isoformat(),
        "random",
        "Random baseline on all 25 public envs",
        str(round(avg_rhae, 6)),
        str(total_levels),
        str(total_actions),
        "completed",
        str(round(total_elapsed, 1)),
        json.dumps({"seed": SEED, "max_actions": 2000}),
        f"change_rate={avg_change_rate:.2%}",
    ])
    with open(results_file, "a") as f:
        f.write(row + "\n")
    print(f"Logged to experiments/results.tsv")

    print("\n" + "=" * 70)
    print("Next: run_iter1_cnn.py (train CNN policy)")
    print("=" * 70)


if __name__ == "__main__":
    main()
