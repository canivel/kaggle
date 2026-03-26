# Agent Program: NVIDIA Nemotron Reasoning Challenge

## Goal
Fine-tune a LoRA adapter (rank ≤ 32) for `nvidia/Nemotron-3-Nano-30B` (Kaggle model: `metric/nemotron-3-nano-30b-a3b-bf16`)
to solve logical reasoning puzzles and achieve accuracy > 0.683 on the Kaggle leaderboard.

## Compute
- **Kaggle GPU:** RTX PRO 6000 Blackwell (96GB VRAM) — free for this competition
- **Training must run on Kaggle** — the 30B model requires `mamba_ssm` (Linux-only)
- Local dev on Windows/RTX 3080 is for code development only

## Data
- `data/train.csv` — columns: `id`, `problem`, `answer`
- `data/test.csv` — columns: `id`, `problem`
- Use 90/10 train/validation split (stratify by puzzle type if detectable)
- Validation accuracy is your proxy metric; target > 0.70 on held-out val before submitting

## Experiment Loop

For each experiment:
1. Modify config/strategy (see strategies below)
2. Run `python train.py --config <config_name>`
3. Run `python evaluate.py --adapter checkpoints/<run_name> --split val`
4. Log result to `experiments.tsv`
5. If val accuracy improves over best, keep adapter in `best_adapter/`; otherwise discard
6. Repeat

## Strategies to Try (in order of expected impact)

### 1. Prompt Engineering (no training needed)
- Try chain-of-thought prompting: "Think step by step, then give your final answer in \\boxed{}"
- Try few-shot examples in system prompt (1-3 examples per puzzle type)
- Evaluate zero-shot vs. few-shot on val set using `evaluate.py --no-adapter`
- This gives you a baseline before any fine-tuning

### 2. Baseline LoRA Fine-tuning
- `rank=32, alpha=64, target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$"`
- NOTE: Nemotron-3-Nano-30B uses non-standard module names (Mamba+Transformer hybrid)
- Train on full train set with standard SFT (format: problem → answer in \\boxed{})
- 3 epochs, lr=2e-4, batch_size=4, gradient_accumulation=4
- Use `kaggle_train.py` on Kaggle notebook with RTX PRO 6000

### 3. Response Format Training
- Format training data as: `<problem>\n\nLet me think step by step.\n<chain-of-thought reasoning>\n\nThe answer is \\boxed{<answer>}`
- Generate synthetic chain-of-thought rationales for training examples using a capable model (GPT-4o, Claude 3.7, etc.)
- Fine-tune on (problem, reasoning+answer) pairs

### 4. Synthetic Data Augmentation
- Generate additional training examples of each puzzle type
- For bit manipulation: write a rule generator and solver to create unlimited (puzzle, answer) pairs
- For algebraic equations: generate random equation systems and solve symbolically with sympy
- For text encryption: generate random cipher keys and encrypt/decrypt pairs
- Fine-tune on original + synthetic data (try 1x, 2x, 5x augmentation ratios)

### 5. GRPO / Reinforcement Learning
- Use TRL's GRPOTrainer or PPOTrainer
- Reward function: +1 if extracted \\boxed{} answer matches ground truth, 0 otherwise
- Start from the best SFT checkpoint
- Low lr (1e-5), KL penalty to prevent collapse

### 6. LoRA Config Search
- Try different ranks: 8, 16, 32 (max)
- Try different target modules: q/v only vs. q/k/v/o vs. all linear layers
- Try different alpha values: alpha = rank vs. alpha = 2*rank

### 7. Data Filtering
- Identify which puzzle types the model struggles with (per-type accuracy)
- Oversample hard puzzle types in training
- Remove potential label errors (examples where multiple models disagree)

## Tracking

All experiments logged in `experiments.tsv`:
```
id  description  val_accuracy  status  timestamp  notes
```

- `status`: `kept` (new best) or `discarded`
- Increment `id` sequentially starting from 001
- Always fill in all fields before moving to next experiment

## Constraints
- LoRA rank ≤ 32 (hard constraint from competition rules)
- Adapter must load with PEFT and be compatible with vLLM LoRA serving
- `adapter_config.json` must be present in submission zip
- Submission: `zip -r submission.zip best_adapter/`

## Success Criteria
- Val accuracy > 0.70
- Leaderboard score > 0.683
- Public notebook published explaining the approach

## File Structure
```
nvidia-nemotron-reasoning/
├── program.md          # this file
├── train.py            # SFT fine-tuning script
├── evaluate.py         # inference + accuracy evaluation
├── run_loop.py         # orchestration loop
├── experiments.tsv     # experiment log
├── best_adapter/       # current best LoRA adapter
├── checkpoints/        # all experiment checkpoints
├── data/
│   ├── train.csv
│   └── test.csv
└── configs/            # YAML configs per experiment
```
