# NVIDIA Nemotron Reasoning Challenge

LoRA fine-tuning pipeline for the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge).

## Goal
Train a LoRA adapter (rank ≤ 32) for `nvidia/Nemotron-H-8B-Base-8K` to solve logical reasoning puzzles.
Target: beat **0.683** accuracy (current 1st place as of March 2026).

## Project Structure

```
nvidia-nemotron-reasoning/
├── COMPETITION.md       # Competition details, timeline, leaderboard
├── program.md           # Strategy guide for the experiment loop
├── README.md            # This file
├── requirements.txt
│
├── train.py             # SFT fine-tuning with PEFT/LoRA
├── evaluate.py          # Inference + accuracy evaluation
├── run_loop.py          # Autoresearch-style experiment loop
│
├── experiments.tsv      # Experiment log (auto-generated)
├── best_adapter/        # Current best LoRA adapter (auto-generated)
├── checkpoints/         # All experiment checkpoints (auto-generated)
│
└── data/
    ├── train.csv        # Training puzzles + answers
    └── test.csv         # Test puzzles (for submission)
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download competition data (requires Kaggle API credentials)
kaggle competitions download -c nvidia-nemotron-model-reasoning-challenge -p data/
cd data && unzip nvidia-nemotron-model-reasoning-challenge.zip && cd ..
```

## Quick Start

### 1. Zero-shot baseline
```bash
python evaluate.py --no-adapter --split val
```

### 2. Train a LoRA adapter
```bash
python train.py --rank 16 --alpha 32 --epochs 3
```

### 3. Evaluate adapter
```bash
python evaluate.py --adapter checkpoints/<run_name>/adapter --split val
```

### 4. Run the experiment loop (interactive)
```bash
python run_loop.py
```

### 5. Run the experiment loop (automated, cycles through preset strategies)
```bash
python run_loop.py --auto --max-rounds 20
```

## Submission

Once you have a good adapter in `best_adapter/`:
```bash
cd best_adapter && zip -r ../submission.zip . && cd ..
# Upload submission.zip to Kaggle
```

## Experiment Tracking

All experiments are logged in `experiments.tsv`:

| Column | Description |
|---|---|
| `id` | Sequential ID (001, 002, ...) |
| `description` | Human-readable description |
| `val_accuracy` | Validation accuracy |
| `status` | `kept` (new best) or `discarded` |
| `timestamp` | ISO timestamp |
| `notes` | LoRA config details |

## Strategies

See `program.md` for the full list of strategies to try, ordered by expected impact:
1. Prompt engineering (zero-shot / few-shot baseline)
2. Baseline LoRA SFT
3. Chain-of-thought / reasoning trace training
4. Synthetic data augmentation
5. GRPO / reinforcement learning from accuracy reward
6. LoRA config search (rank, alpha, target modules)
7. Data filtering and oversampling

## Puzzle Types

- **Bit manipulation** — 8-bit binary number transformation rules
- **Algebraic equations** — solve for variables
- **Text encryption** — decode/encode cipher text

## Key Constraints

- LoRA rank ≤ 32 (hard competition limit)
- Submission must include `adapter_config.json`
- vLLM inference: `temperature=0.0`, `max_tokens=7680`, `max_lora_rank=32`
- Top 3 finishers must publish a public notebook + writeup
