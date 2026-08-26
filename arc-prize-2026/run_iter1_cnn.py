"""ARC-AGI-3 Iteration 1: CNN Agent Training

Approach: Train CNN policy with RL on all 25 public environments
  - Color embedding (16 colors -> learned features)
  - 4-layer ConvNet backbone
  - PPO-style updates with reward shaping
  - Also train world model (predict next grid state)

Reward shaping:
  +10 for level completion
  +0.01 for frame change (exploration)
  -0.001 per action (efficiency pressure)

Target: Complete at least some levels, beat random baseline
"""

import sys
import json
import time
import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

import arc_agi
from arcengine.enums import GameAction, GameState
from agent.cnn_policy import ArcCNNPolicy, WorldModel
from agent.trainer import ArcTrainer


# ─── Configuration ────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
COLLECT_STEPS = 300         # max actions per episode during collection
N_COLLECTION_ROUNDS = 5     # rounds of collect + train
N_POLICY_UPDATES = 20       # policy gradient updates per round
N_WORLD_UPDATES = 20        # world model updates per round
BATCH_SIZE = 64
LR = 3e-4

RESULTS_DIR = Path("experiments")
CHECKPOINT_DIR = Path("checkpoints/iter1")
RESULTS_DIR.mkdir(exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")


def main():
    print("=" * 70)
    print("ARC-AGI-3 Iteration 1: CNN Policy Training")
    print("=" * 70)

    arcade = arc_agi.Arcade()
    envs = arc_agi.Arcade().get_environments()
    print(f"\n{len(envs)} public environments available")

    # Sort by difficulty (human baseline actions)
    envs_sorted = sorted(envs, key=lambda e: sum(e.baseline_actions))
    easy_envs = envs_sorted[:10]  # easiest 10 for training
    eval_envs = envs_sorted[:5]   # easiest 5 for evaluation

    print(f"Training on {len(easy_envs)} easiest environments:")
    for e in easy_envs:
        print(f"  {e.title:5s} ({','.join(e.tags):15s}): "
              f"{len(e.baseline_actions)} levels, human={sum(e.baseline_actions)} actions")

    # Initialize trainer
    trainer = ArcTrainer(
        lr=LR,
        device=DEVICE,
        checkpoint_dir=str(CHECKPOINT_DIR),
    )

    start_time = time.time()
    best_levels = 0

    # ── Training Loop ─────────────────────────────────────────────
    for round_idx in range(N_COLLECTION_ROUNDS):
        print(f"\n{'='*60}")
        print(f"Round {round_idx + 1}/{N_COLLECTION_ROUNDS}")
        print(f"{'='*60}")

        # Collect experience from each training environment
        round_levels = 0
        round_reward = 0.0
        for env_info in easy_envs:
            stats = trainer.collect_episode(
                arcade, env_info, max_actions=COLLECT_STEPS
            )
            round_levels += stats["levels_completed"]
            round_reward += stats["total_reward"]

        print(f"  Collection: {round_levels} levels, "
              f"reward={round_reward:.2f}, "
              f"buffer={len(trainer.replay)}")

        # Train policy
        for update in range(N_POLICY_UPDATES):
            pol_stats = trainer.train_policy(batch_size=BATCH_SIZE)
            if (update + 1) % 5 == 0:
                print(f"  Policy update {update+1}: loss={pol_stats['policy_loss']}")

        # Train world model
        for update in range(N_WORLD_UPDATES):
            wm_stats = trainer.train_world_model(batch_size=BATCH_SIZE)
            if (update + 1) % 5 == 0:
                print(f"  World model update {update+1}: "
                      f"loss={wm_stats['world_loss']}, acc={wm_stats['world_acc']}")

        # Evaluate
        print(f"\n  Evaluation:")
        eval_levels = 0
        eval_actions = 0
        for env_info in eval_envs:
            result = trainer.evaluate(arcade, env_info, max_actions=500)
            eval_levels += result["levels_completed"]
            eval_actions += result["total_actions"]
            if result["levels_completed"] > 0:
                print(f"    {result['env_title']:5s}: "
                      f"{result['levels_completed']}/{result['win_levels']} levels "
                      f"in {result['total_actions']} actions")

        print(f"  Total: {eval_levels} levels in {eval_actions} actions")

        # Save if improved
        if eval_levels > best_levels:
            best_levels = eval_levels
            trainer.save_checkpoint("best")
            print(f"  NEW BEST: {best_levels} levels!")

        trainer.train_stats.append({
            "round": round_idx + 1,
            "collect_levels": round_levels,
            "collect_reward": round_reward,
            "eval_levels": eval_levels,
            "eval_actions": eval_actions,
            "buffer_size": len(trainer.replay),
        })

    trainer.save_checkpoint("final")

    # ── Final Evaluation on ALL envs ──────────────────────────────
    print(f"\n{'='*60}")
    print("Final Evaluation: All 25 Environments")
    print(f"{'='*60}")

    trainer.load_checkpoint("best")
    total_levels = 0
    total_actions = 0

    for env_info in envs:
        result = trainer.evaluate(arcade, env_info, max_actions=1000)
        total_levels += result["levels_completed"]
        total_actions += result["total_actions"]
        status = ""
        if result["levels_completed"] > 0:
            status = f" *** {result['levels_completed']}/{result['win_levels']}"
        print(f"  {result['env_title']:5s}: {result['total_actions']:4d} actions, "
              f"levels={result['levels_completed']}/{result['win_levels']}{status}")

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Total levels completed: {total_levels}")
    print(f"  Total actions: {total_actions}")
    print(f"  Training time: {elapsed/60:.1f} minutes")
    print(f"  Best eval levels: {best_levels}")

    # Log experiment
    results_file = RESULTS_DIR / "results.tsv"
    if not results_file.exists():
        header = "\t".join([
            "experiment_id", "timestamp", "agent_type", "description",
            "rhae_score", "levels_completed", "total_actions",
            "status", "duration_seconds", "params", "notes"
        ])
        results_file.write_text(header + "\n")

    row = "\t".join([
        "0002",
        datetime.datetime.now().isoformat(),
        "cnn_ppo",
        "CNN policy + world model, PPO, 5 rounds",
        "N/A",
        str(total_levels),
        str(total_actions),
        "completed",
        str(round(elapsed, 1)),
        json.dumps({"lr": LR, "batch_size": BATCH_SIZE, "rounds": N_COLLECTION_ROUNDS,
                     "device": DEVICE}),
        f"best_eval_levels={best_levels}",
    ])
    with open(results_file, "a") as f:
        f.write(row + "\n")

    print(f"\nLogged to {results_file}")
    print("\nNext steps:")
    print("  - If levels > 0: iterate on reward shaping (run_iter2_reward.py)")
    print("  - If levels = 0: try per-environment specialization")
    print("  - Train world model longer for model-based planning")


if __name__ == "__main__":
    main()
