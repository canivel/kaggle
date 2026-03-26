"""
Autoresearch Orchestrator — runs LOCALLY on your machine (CPU only).
Dispatches GPU work to a RunPod worker via SSH/rsync.

Architecture:
    Local (Windows/CPU)              RunPod (A100 80GB)
    ─────────────────                ──────────────────
    ┌─────────────────┐              ┌─────────────────┐
    │  Orchestrator    │──rsync───>  │  gpu_worker.py   │
    │  - strategy gen  │  config     │  - load model    │
    │  - experiment log│<─rsync────  │  - train LoRA    │
    │  - keep/discard  │  adapter+   │  - evaluate      │
    │  - analysis      │  results    │  - save adapter   │
    └─────────────────┘              └─────────────────┘

Usage:
    # First: set up RunPod and note the SSH connection
    python run_autoresearch.py --runpod-host <IP> --runpod-port <PORT> --auto
    python run_autoresearch.py --runpod-host <IP> --auto --max-rounds 10

    # Or manual mode
    python run_autoresearch.py --runpod-host <IP>

    # Local-only mode (if you have a Linux GPU locally)
    python run_autoresearch.py --local
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
BEST_ADAPTER_DIR = BASE_DIR / "best_adapter"
EXPERIMENTS_TSV = BASE_DIR / "experiments.tsv"
DATA_DIR = BASE_DIR / "data"

TARGET_ACCURACY = 0.683
TSV_FIELDS = ["id", "description", "val_accuracy", "train_loss", "status",
              "timestamp", "train_time_min", "notes"]


# ---------------------------------------------------------------------------
# Experiment log (local, persistent)
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
    print(f"  -> Logged exp {row['id']}: {row['status']}  acc={row['val_accuracy']}")


def next_exp_id(experiments: list[dict]) -> str:
    if not experiments:
        return "001"
    nums = [int(e["id"]) for e in experiments if e.get("id", "").isdigit()]
    return f"{max(nums) + 1:03d}" if nums else "001"


def best_accuracy(experiments: list[dict]) -> float:
    kept = [e for e in experiments if e.get("status") == "kept"]
    if not kept:
        return 0.0
    vals = [float(e["val_accuracy"]) for e in kept
            if e.get("val_accuracy") not in ("N/A", "", None)]
    return max(vals) if vals else 0.0


# ---------------------------------------------------------------------------
# RunPod SSH/rsync helpers
# ---------------------------------------------------------------------------
class RunPodWorker:
    """Dispatches GPU work to RunPod via SSH."""

    def __init__(self, host, port=22, user="root", key=None):
        self.host = host
        self.port = port
        self.user = user
        self.ssh_opts = ["-o", "StrictHostKeyChecking=no", "-p", str(port)]
        if key:
            self.ssh_opts += ["-i", key]
        self.remote = f"{user}@{host}"

    def ssh_cmd(self, cmd):
        full = ["ssh"] + self.ssh_opts + [self.remote, cmd]
        print(f"  [SSH] {cmd}")
        return subprocess.run(full, capture_output=True, text=True, timeout=7200)

    def rsync_up(self, local_path, remote_path):
        """Upload file/dir to RunPod."""
        cmd = ["rsync", "-avz", "-e", f"ssh {' '.join(self.ssh_opts)}",
               str(local_path), f"{self.remote}:{remote_path}"]
        print(f"  [UPLOAD] {local_path} -> {remote_path}")
        return subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    def rsync_down(self, remote_path, local_path):
        """Download file/dir from RunPod."""
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        cmd = ["rsync", "-avz", "-e", f"ssh {' '.join(self.ssh_opts)}",
               f"{self.remote}:{remote_path}", str(local_path)]
        print(f"  [DOWNLOAD] {remote_path} -> {local_path}")
        return subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    def check_ready(self) -> bool:
        """Check if worker is set up and model is downloaded."""
        r = self.ssh_cmd("test -f /workspace/model/config.json && echo READY || echo NOTREADY")
        return "READY" in r.stdout

    def setup(self):
        """Initial setup: upload scripts and data."""
        print("\n--- Setting up RunPod worker ---")
        self.rsync_up(BASE_DIR / "gpu_worker.py", "/workspace/")
        self.rsync_up(DATA_DIR / "train.csv", "/workspace/data/")

        if not self.check_ready():
            print("Model not found on RunPod. Running setup...")
            self.rsync_up(BASE_DIR / "runpod_setup.sh", "/workspace/")
            r = self.ssh_cmd("bash /workspace/runpod_setup.sh")
            print(r.stdout[-500:] if r.stdout else "")
            if r.returncode != 0:
                print(f"Setup failed: {r.stderr[-500:]}")
                return False
        else:
            print("RunPod worker ready (model already downloaded)")
        return True

    def run_experiment(self, config: dict) -> dict | None:
        """Upload config, run training on GPU, download results."""
        exp_id = config["exp_id"]
        config_path = f"/workspace/data/config_{exp_id}.json"
        adapter_remote = f"/workspace/adapters/exp_{exp_id}"
        adapter_local = CHECKPOINT_DIR / f"exp_{exp_id}"

        # Upload config
        local_config = BASE_DIR / f"_tmp_config_{exp_id}.json"
        local_config.write_text(json.dumps(config, indent=2))
        self.rsync_up(local_config, config_path)
        local_config.unlink()

        # Run training on GPU
        print(f"\n  [GPU] Training exp {exp_id}...")
        r = self.ssh_cmd(f"cd /workspace && python gpu_worker.py --config {config_path}")
        print(r.stdout[-2000:] if r.stdout else "")
        if r.returncode != 0:
            print(f"  [GPU] FAILED: {r.stderr[-500:]}")
            return None

        # Download adapter + results
        self.rsync_down(f"{adapter_remote}/", str(adapter_local) + "/")

        # Read result
        result_file = adapter_local / "result.json"
        if result_file.exists():
            return json.loads(result_file.read_text())
        return None


# ---------------------------------------------------------------------------
# Local worker (for Linux machines with GPU)
# ---------------------------------------------------------------------------
class LocalWorker:
    """Runs GPU work locally (requires mamba_ssm + CUDA GPU)."""

    def check_ready(self):
        return True

    def setup(self):
        return True

    def run_experiment(self, config: dict) -> dict | None:
        config_path = BASE_DIR / f"_tmp_config_{config['exp_id']}.json"
        config_path.write_text(json.dumps(config, indent=2))
        try:
            r = subprocess.run(
                [sys.executable, str(BASE_DIR / "gpu_worker.py"), "--config", str(config_path)],
                cwd=str(BASE_DIR), timeout=14400,
            )
            config_path.unlink(missing_ok=True)
            result_file = Path(config.get("output_dir", "adapters")) / f"exp_{config['exp_id']}" / "result.json"
            if result_file.exists():
                return json.loads(result_file.read_text())
        except Exception as e:
            print(f"Local training failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Autoresearch strategies — the experiment search space
# ---------------------------------------------------------------------------
AUTO_STRATEGIES = [
    # Round 1: Baseline with competition-specified settings
    {
        "description": "baseline r=32 a=64 3ep full data",
        "rank": 32, "alpha": 64, "epochs": 3, "lr": 2e-4,
        "batch_size": 4, "grad_accum": 4, "max_len": 4096,
    },
    # Round 2: Lower learning rate for stability
    {
        "description": "r=32 a=64 3ep lower lr=5e-5",
        "rank": 32, "alpha": 64, "epochs": 3, "lr": 5e-5,
        "batch_size": 4, "grad_accum": 4, "max_len": 4096,
    },
    # Round 3: Higher alpha ratio
    {
        "description": "r=32 a=128 3ep higher alpha",
        "rank": 32, "alpha": 128, "epochs": 3, "lr": 1e-4,
        "batch_size": 4, "grad_accum": 4, "max_len": 4096,
    },
    # Round 4: More epochs
    {
        "description": "r=32 a=64 5ep more training",
        "rank": 32, "alpha": 64, "epochs": 5, "lr": 1e-4,
        "batch_size": 4, "grad_accum": 4, "max_len": 4096,
    },
    # Round 5: Longer sequences
    {
        "description": "r=32 a=64 3ep max_len=8192",
        "rank": 32, "alpha": 64, "epochs": 3, "lr": 2e-4,
        "batch_size": 1, "grad_accum": 16, "max_len": 8192,
    },
    # Round 6: Smaller rank, faster iteration
    {
        "description": "r=16 a=32 5ep fast iteration",
        "rank": 16, "alpha": 32, "epochs": 5, "lr": 2e-4,
        "batch_size": 4, "grad_accum": 4, "max_len": 4096,
    },
    # Round 7: Very low lr, long training
    {
        "description": "r=32 a=64 8ep slow cook lr=3e-5",
        "rank": 32, "alpha": 64, "epochs": 8, "lr": 3e-5,
        "batch_size": 4, "grad_accum": 4, "max_len": 4096,
    },
    # Round 8: Max rank with warmup
    {
        "description": "r=32 a=64 3ep warmup=0.1",
        "rank": 32, "alpha": 64, "epochs": 3, "lr": 2e-4,
        "batch_size": 4, "grad_accum": 4, "max_len": 4096,
        "warmup_ratio": 0.1,
    },
]


# ---------------------------------------------------------------------------
# Main autoresearch loop
# ---------------------------------------------------------------------------
def run_autoresearch(args):
    experiments = read_experiments()
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    # Set up worker
    if args.local:
        worker = LocalWorker()
    else:
        if not args.runpod_host:
            print("ERROR: Provide --runpod-host or use --local")
            sys.exit(1)
        worker = RunPodWorker(
            host=args.runpod_host,
            port=args.runpod_port,
            key=args.ssh_key,
        )

    print("\n=== Autoresearch: Nemotron Reasoning Challenge ===")
    print(f"Target accuracy: {TARGET_ACCURACY}")
    print(f"Best so far: {best_accuracy(experiments):.4f}")
    print(f"Experiments completed: {len(experiments)}")

    if not worker.setup():
        print("Worker setup failed. Exiting.")
        sys.exit(1)

    round_num = 0
    while True:
        round_num += 1
        if args.max_rounds and round_num > args.max_rounds:
            print(f"\nReached max rounds ({args.max_rounds}).")
            break

        exp_id = next_exp_id(experiments)
        current_best = best_accuracy(experiments)

        # Pick strategy
        if args.auto:
            strategy = AUTO_STRATEGIES[(round_num - 1) % len(AUTO_STRATEGIES)].copy()
            strategy["exp_id"] = exp_id
            config = strategy
        else:
            config = get_interactive_config(exp_id, experiments)
            if config is None:
                break

        # Set defaults
        config.setdefault("exp_id", exp_id)
        config.setdefault("data_path", "/workspace/data/train.csv")
        config.setdefault("model_path", "/workspace/model")
        config.setdefault("output_dir", "/workspace/adapters")
        config.setdefault("n_eval", 200)

        print(f"\n{'='*60}")
        print(f"Round {round_num} | Exp {exp_id} | Best: {current_best:.4f} | Target: {TARGET_ACCURACY}")
        print(f"Config: {config['description']}")
        print(f"{'='*60}")

        # Run on GPU
        t0 = time.time()
        result = worker.run_experiment(config)
        wall_time = (time.time() - t0) / 60

        # Process result
        if result and "val_accuracy" in result:
            val_acc = result["val_accuracy"]
            train_loss = result.get("train_loss", "N/A")

            if val_acc > current_best:
                status = "kept"
                # Download and save as best adapter
                adapter_local = CHECKPOINT_DIR / f"exp_{exp_id}"
                if BEST_ADAPTER_DIR.exists():
                    shutil.rmtree(BEST_ADAPTER_DIR)
                if adapter_local.exists():
                    shutil.copytree(str(adapter_local), str(BEST_ADAPTER_DIR))
                print(f"\n*** NEW BEST! {val_acc:.4f} > {current_best:.4f} ***")
                print(f"*** Saved to {BEST_ADAPTER_DIR} ***")
            else:
                status = "discarded"
                print(f"\nNo improvement: {val_acc:.4f} <= {current_best:.4f}")
        else:
            val_acc = None
            train_loss = "N/A"
            status = "failed"
            print("\nExperiment failed — no result returned")

        # Log
        row = {
            "id": exp_id,
            "description": config.get("description", ""),
            "val_accuracy": f"{val_acc:.4f}" if val_acc else "N/A",
            "train_loss": str(train_loss),
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "train_time_min": f"{wall_time:.1f}",
            "notes": f"r={config.get('rank')} a={config.get('alpha')} ep={config.get('epochs')} lr={config.get('lr')} len={config.get('max_len')}",
        }
        write_experiment(row)
        experiments.append(row)

        # Check if we beat the target
        if val_acc and val_acc >= TARGET_ACCURACY:
            print(f"\n{'='*60}")
            print(f"TARGET REACHED! {val_acc:.4f} >= {TARGET_ACCURACY}")
            print(f"Best adapter: {BEST_ADAPTER_DIR}")
            print(f"{'='*60}")
            break

        # Print leaderboard
        print(f"\n--- Experiment Log ---")
        for e in experiments[-5:]:
            mark = ">>>" if e.get("status") == "kept" else "   "
            print(f"  {mark} {e['id']}: acc={e.get('val_accuracy','N/A'):>7s}  {e.get('description','')[:50]}")


def get_interactive_config(exp_id, experiments):
    print(f"\n{'='*60}")
    print(f"Exp {exp_id} | Best: {best_accuracy(experiments):.4f} | Target: {TARGET_ACCURACY}")
    print("Enter: rank=N alpha=N epochs=N lr=F max_len=N desc=TEXT")
    print("  (or 'quit' to stop)")
    user = input("> ").strip()
    if user.lower() in ("quit", "q", "exit"):
        return None

    config = {
        "exp_id": exp_id, "rank": 32, "alpha": 64, "epochs": 3,
        "lr": 2e-4, "batch_size": 4, "grad_accum": 4, "max_len": 4096,
        "description": f"manual {exp_id}",
    }
    for token in user.split():
        if "=" in token:
            k, v = token.split("=", 1)
            k = k.replace("-", "_")
            if k in ("rank", "alpha", "epochs", "batch_size", "grad_accum", "max_len", "n_train", "n_eval"):
                config[k] = int(v)
            elif k == "lr":
                config[k] = float(v)
            elif k in ("desc", "description"):
                config["description"] = v
    return config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Autoresearch orchestrator for Nemotron Reasoning")
    p.add_argument("--runpod-host", type=str, help="RunPod SSH hostname or IP")
    p.add_argument("--runpod-port", type=int, default=22, help="RunPod SSH port")
    p.add_argument("--ssh-key", type=str, default=None, help="Path to SSH private key")
    p.add_argument("--local", action="store_true", help="Run GPU work locally instead of RunPod")
    p.add_argument("--auto", action="store_true", help="Auto-cycle through strategies")
    p.add_argument("--max-rounds", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    run_autoresearch(parse_args())
