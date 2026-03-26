# NVIDIA Nemotron Model Reasoning Challenge

## Overview
- **Competition:** NVIDIA Nemotron Model Reasoning Challenge
- **URL:** https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
- **Task:** Create a LoRA adapter (rank ≤ 32) for Nemotron-3-Nano-30B model to solve logical reasoning puzzles
- **Metric:** Accuracy (answers extracted from `\boxed{}` in model output)

## Inference Parameters
- Engine: vLLM
- `max_lora_rank=32`
- `max_tokens=7680`
- `top_p=1.0`
- `temperature=0.0`
- `max_num_seqs=64`
- `gpu_memory_utilization=0.85`
- `max_model_len=8192`

## Dataset
- `train.csv` — puzzles with answers
- `test.csv` — puzzles only (~3 sample problems)

## Submission
- `submission.zip` containing LoRA adapter files including `adapter_config.json`

## Timeline
| Milestone | Date |
|---|---|
| Competition Start | March 16, 2026 |
| Midpoint Cutoff | April 9, 2026 |
| Entry Deadline | June 8, 2026 |
| Final Deadline | June 15, 2026 |

## Prizes
| Place | Cash | Hardware |
|---|---|---|
| 1st | $25,000 | 5 DGX Sparks |
| 2nd | $15,000 | 2 DGX Sparks |
| 3rd | $5,000 | 1 DGX Spark |

**Note:** Top 3 must publish a public notebook + writeup for prize eligibility.

## Leaderboard (as of March 18, 2026)
| Rank | Team | Score |
|---|---|---|
| 1st | Claude or Claw | 0.683 |
| 2nd | CausalLM.org | 0.687 |
| 3rd | Dennis | 0.681 |
| 4th | Jack | 0.674 |
| 5th | Team Wookie | 0.668 |

**Target:** Beat 0.683 (current 1st place)

## Puzzle Types
1. **Bit manipulation** — rules applied to 8-bit binary numbers
2. **Algebraic equations** — solve for variables
3. **Text encryption** — decode/encode cipher text

## Compute (Kaggle-Provided)
- **GPU:** NVIDIA RTX PRO 6000 Blackwell (96GB VRAM) — free on Kaggle for this competition
- **Environment:** `mamba_ssm`, `peft`, `transformers`, `torch` pre-installed
- **Google Cloud G4 VMs** powered by Blackwell GPUs

## Model
- **Base:** `nvidia/Nemotron-3-Nano-30B` (Kaggle model ID: `metric/nemotron-3-nano-30b-a3b-bf16`)
- **Download:** `kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")`
- **Architecture:** Mamba+Transformer hybrid (requires `mamba_ssm` — Linux only, pre-installed on Kaggle)
- **Adapter:** LoRA, rank ≤ 32
- **LoRA target modules:** `r".*\.(in_proj|out_proj|up_proj|down_proj)$"` (NOT standard q/k/v/o_proj)
- **Trainable params:** ~880M / 32.4B total (2.7%) at rank=32
- **Format:** PEFT-compatible `adapter_config.json` + weights
- **Untrained LoRA baseline score:** 0.50 (random init)
