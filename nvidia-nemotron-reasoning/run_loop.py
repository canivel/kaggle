"""
Autoresearch-style experiment loop for the NVIDIA Nemotron Reasoning Challenge.

Usage:
    uv run python run_loop.py                         # interactive
    uv run python run_loop.py --auto --max-rounds 10  # non-interactive
    uv run python run_loop.py --rank 32 --epochs 5    # single run
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
BEST_ADAPTER_DIR = BASE_DIR / "best_adapter"
EXPERIMENTS_TSV = BASE_DIR / "experiments.tsv"

TARGET_ACCURACY = 0.683

TSV_FIELDS = ["id", "description", "val_accuracy", "status", "timestamp", "notes"]

# uv run env — keep cache on F: to avoid filling C:
UV_ENV = {
    **os.environ,
    "UV_CACHE_DIR": "F:/.uv-cache",
    "HF_HOME": "F:/hf-cache",
    "PYTHONIOENCODING": "utf-8",
}
UV_RUN = ["uv", "run", "python"]


# ---------------------------------------------------------------------------
# Experiment log
# ---------------------------------------------------------------------------
def read_experiments() -> list[dict]:
    if not EXPERIMENTS_TSV.exists():
        return []
    with open(EXPERIMENTS_TSV, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_experiment(row: dict):
    exists = EXPERIMENTS_TSV.exists()
    with open(EXPERIMENTS_TSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TSV_FIELDS, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"  -> Logged exp {row['id']}: {row['status']}  (val_acc={row['val_accuracy']})")


def next_exp_id(experiments: list[dict]) -> str:
    if not experiments:
        return "001"
    last = max(int(e["id"]) for e in experiments if e.get("id", "").isdigit())
    return f"{last + 1:03d}"


def best_accuracy(experiments: list[dict]) -> float:
    kept = [e for e in experiments if e.get("status") == "kept"]
    if not kept:
        return 0.0
    return max(float(e["val_accuracy"]) for e in kept if e.get("val_accuracy") not in ("N/A", ""))


# ---------------------------------------------------------------------------
# Subprocess runners
# ---------------------------------------------------------------------------
def run_training(config: dict) -> str | None:
    cmd = UV_RUN + [
        "train.py",
        "--run-name", config["run_name"],
        "--rank", str(config["rank"]),
        "--alpha", str(config["alpha"]),
        "--epochs", str(config["epochs"]),
        "--lr", str(config["lr"]),
        "--batch-size", str(config["batch_size"]),
        "--grad-accum", str(config["grad_accum"]),
        "--max-len", str(config["max_len"]),
    ]
    if config.get("model_id"):
        cmd += ["--model-id", config["model_id"]]
    if config.get("use_qlora"):
        cmd += ["--qlora"]

    print(f"\n$ {' '.join(cmd)}")
    rc = subprocess.run(cmd, cwd=str(BASE_DIR), env=UV_ENV).returncode
    if rc != 0:
        print(f"Training failed (exit {rc})")
        return None

    adapter_path = CHECKPOINT_DIR / config["run_name"] / "adapter"
    if not adapter_path.exists():
        print(f"Adapter missing at {adapter_path}")
        return None
    return str(adapter_path)


def run_evaluation(adapter_path: str | None, model_id: str | None = None) -> float | None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_file = BASE_DIR / f"preds_{ts}.csv"

    cmd = UV_RUN + ["evaluate.py", "--split", "val", "--output", str(pred_file)]
    if adapter_path:
        cmd += ["--adapter", adapter_path]
    else:
        cmd += ["--no-adapter"]
    if model_id:
        cmd += ["--model-id", model_id]

    print(f"\n$ {' '.join(cmd)}")
    rc = subprocess.run(cmd, cwd=str(BASE_DIR), env=UV_ENV).returncode
    if rc != 0:
        print(f"Evaluation failed (exit {rc})")
        return None

    try:
        import pandas as pd
        df = pd.read_csv(pred_file)
        if "correct" in df.columns:
            return float(df["correct"].mean())
    except Exception as e:
        print(f"Could not parse predictions: {e}")
    return None


# ---------------------------------------------------------------------------
# Strategy presets for --auto mode
# ---------------------------------------------------------------------------
AUTO_STRATEGIES = [
    {
        "description": "QLoRA r=16 alpha=32 3ep — baseline",
        "rank": 16, "alpha": 32, "epochs": 3, "lr": 2e-4,
        "batch_size": 2, "grad_accum": 8, "max_len": 2048, "use_qlora": True,
    },
    {
        "description": "QLoRA r=32 alpha=64 3ep",
        "rank": 32, "alpha": 64, "epochs": 3, "lr": 2e-4,
        "batch_size": 2, "grad_accum": 8, "max_len": 2048, "use_qlora": True,
    },
    {
        "description": "QLoRA r=16 alpha=32 5ep lower lr",
        "rank": 16, "alpha": 32, "epochs": 5, "lr": 5e-5,
        "batch_size": 2, "grad_accum": 8, "max_len": 2048, "use_qlora": True,
    },
    {
        "description": "QLoRA r=32 alpha=32 3ep longer seq",
        "rank": 32, "alpha": 32, "epochs": 3, "lr": 1e-4,
        "batch_size": 1, "grad_accum": 16, "max_len": 3072, "use_qlora": True,
    },
    {
        "description": "QLoRA r=8 alpha=16 3ep small rank",
        "rank": 8, "alpha": 16, "epochs": 3, "lr": 2e-4,
        "batch_size": 2, "grad_accum": 8, "max_len": 2048, "use_qlora": True,
    },
]


def get_strategy_interactive(exp_id: str, experiments: list[dict]) -> dict | None:
    print(f"\n{'='*60}")
    print(f"Exp {exp_id} | Best so far: {best_accuracy(experiments):.4f} | Target: {TARGET_ACCURACY}")
    print("Enter config (or 'quit'):")
    print("  rank=<n> alpha=<n> epochs=<n> lr=<f> desc=<text> qlora")
    user_input = input("> ").strip()
    if user_input.lower() in ("quit", "q", "exit"):
        return None

    config = {
        "run_name": f"exp_{exp_id}",
        "rank": 16, "alpha": 32, "epochs": 3, "lr": 2e-4,
        "batch_size": 2, "grad_accum": 8, "max_len": 2048,
        "use_qlora": True,
        "description": f"manual experiment {exp_id}",
    }
    for token in user_input.split():
        if "=" in token:
            k, v = token.split("=", 1)
            k = k.strip().replace("-", "_")
            if k in ("rank", "alpha", "epochs", "batch_size", "grad_accum", "max_len"):
                config[k] = int(v)
            elif k == "lr":
                config[k] = float(v)
            elif k in ("desc", "description"):
                config["description"] = v
        elif token.lower() == "qlora":
            config["use_qlora"] = True
    return config


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run_loop(args):
    experiments = read_experiments()
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    round_num = 0
    while True:
        round_num += 1
        if args.max_rounds and round_num > args.max_rounds:
            print(f"\nReached max rounds ({args.max_rounds}). Stopping.")
            break

        exp_id = next_exp_id(experiments)
        current_best = best_accuracy(experiments)

        if args.auto:
            strategy = AUTO_STRATEGIES[(round_num - 1) % len(AUTO_STRATEGIES)].copy()
            strategy["run_name"] = f"exp_{exp_id}"
            config = strategy
            print(f"\n[Auto] Round {round_num} | Exp {exp_id} | {config['description']}")
        elif args.rank is not None:
            config = {
                "run_name": f"exp_{exp_id}",
                "description": f"CLI r={args.rank} a={args.alpha} ep={args.epochs}",
                "rank": args.rank, "alpha": args.alpha, "epochs": args.epochs,
                "lr": args.lr, "batch_size": args.batch_size,
                "grad_accum": args.grad_accum, "max_len": args.max_len,
                "use_qlora": not args.no_qlora,
            }
        else:
            config = get_strategy_interactive(exp_id, experiments)
            if config is None:
                print("Stopping loop.")
                break

        print(f"\n--- Training {exp_id}: {config['description']} ---")
        adapter_path = run_training(config)

        if adapter_path:
            print(f"\n--- Evaluating {exp_id} ---")
            val_acc = run_evaluation(adapter_path)
        else:
            val_acc = None

        if val_acc is not None and val_acc > current_best:
            status = "kept"
            if BEST_ADAPTER_DIR.exists():
                shutil.rmtree(BEST_ADAPTER_DIR)
            shutil.copytree(adapter_path, str(BEST_ADAPTER_DIR))
            print(f"\n*** New best! {val_acc:.4f} > {current_best:.4f} -> saved to best_adapter/ ***")
        else:
            status = "discarded"
            msg = f"{val_acc:.4f} <= {current_best:.4f}" if val_acc is not None else "eval failed"
            print(f"\nNo improvement ({msg}). Discarding.")

        row = {
            "id": exp_id,
            "description": config.get("description", ""),
            "val_accuracy": f"{val_acc:.4f}" if val_acc is not None else "N/A",
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "notes": f"r={config['rank']} a={config['alpha']} ep={config['epochs']} lr={config['lr']} qlora={config.get('use_qlora',False)}",
        }
        write_experiment(row)
        experiments.append(row)

        if val_acc is not None and val_acc > TARGET_ACCURACY:
            print(f"\n{'='*60}")
            print(f"TARGET REACHED! {val_acc:.4f} > {TARGET_ACCURACY}")
            print("Best adapter: best_adapter/")
            print(f"{'='*60}")
            break

        if args.rank is not None:
            break  # single run


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--auto", action="store_true")
    p.add_argument("--max-rounds", type=int, default=None)
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--no-qlora", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run_loop(parse_args())
