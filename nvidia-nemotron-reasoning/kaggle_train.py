"""
Kaggle Training Notebook for NVIDIA Nemotron Reasoning Challenge.

Run on Kaggle with RTX PRO 6000 Blackwell GPU (96GB VRAM).
Trains a LoRA adapter on the Nemotron-3-Nano-30B model.

To use: Copy this into a Kaggle notebook cell, or upload as a script.
Requires: GPU RTX Pro 6000 accelerator selected in Kaggle notebook settings.
"""

import os
import re
import gc
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

# ── Configuration ─────────────────────────────────────────────────────────────
# Model
MODEL_ID = "metric/nemotron-3-nano-30b-a3b-bf16"  # Kaggle model ID
LORA_RANK = 32          # max allowed by competition
LORA_ALPHA = 64         # 2x rank is a common default
LORA_DROPOUT = 0.05
TARGET_MODULES = r".*\.(in_proj|out_proj|up_proj|down_proj)$"

# Training
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
GRAD_ACCUM = 4
MAX_SEQ_LEN = 4096      # Nemotron supports 8192, use 4096 for training efficiency
WARMUP_RATIO = 0.05

# Paths
DATA_DIR = Path("/kaggle/input/nvidia-nemotron-3-reasoning-challenge")
OUTPUT_DIR = Path("/kaggle/working")
ADAPTER_DIR = OUTPUT_DIR / "adapter"

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert at solving logical reasoning puzzles including "
    "bit manipulation, algebra, and text encryption. "
    "Solve the problem step by step, then give your final answer inside \\boxed{}."
)


def format_example(prompt: str, answer: str | None = None) -> str:
    """Format a training example as chat-style text."""
    text = f"<extra_id_0>System\n{SYSTEM_PROMPT}\n<extra_id_1>User\n{prompt}\n<extra_id_1>Assistant\n"
    if answer is not None:
        text += f"Let me work through this step by step.\n\nThe answer is \\boxed{{{answer}}}"
    return text


def extract_answer(text: str) -> str | None:
    """Extract last \\boxed{...} from model output."""
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else None


# ── Data Loading ──────────────────────────────────────────────────────────────
def load_data(val_frac: float = 0.1, seed: int = 42):
    df = pd.read_csv(DATA_DIR / "train.csv")
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.dropna(subset=["prompt", "answer"]).reset_index(drop=True)
    df["answer"] = df["answer"].astype(str).str.strip()

    rng = np.random.default_rng(seed)
    n_val = max(50, int(len(df) * val_frac))
    val_idx = set(rng.choice(len(df), size=n_val, replace=False))
    mask = np.array([i in val_idx for i in range(len(df))])

    train_df = df[~mask].reset_index(drop=True)
    val_df = df[mask].reset_index(drop=True)
    print(f"Data split: train={len(train_df)}, val={len(val_df)}")
    return train_df, val_df


# ── Model Setup ───────────────────────────────────────────────────────────────
def setup_model():
    import kagglehub
    import mamba_ssm  # noqa: F401 — required for Nemotron-H architecture
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    print("Downloading model...")
    model_path = kagglehub.model_download(f"{MODEL_ID}/transformers/default")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading model (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ── Training ──────────────────────────────────────────────────────────────────
def train(model, tokenizer, train_df, val_df):
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    train_texts = [format_example(r["prompt"], r["answer"]) for _, r in train_df.iterrows()]
    val_texts = [format_example(r["prompt"], r["answer"]) for _, r in val_df.iterrows()]

    train_ds = Dataset.from_dict({"text": train_texts})
    val_ds = Dataset.from_dict({"text": val_texts})

    run_name = f"nemotron_r{LORA_RANK}_a{LORA_ALPHA}_ep{NUM_EPOCHS}"
    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        run_name=run_name,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_seq_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        report_to="none",
        seed=42,
        gradient_checkpointing=True,
        optim="adamw_torch",
        dataloader_pin_memory=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    print(f"\nTraining: {run_name}")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"Training complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    return trainer


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.inference_mode()
def evaluate(model, tokenizer, val_df, n_samples: int = 100, max_new_tokens: int = 512):
    """Quick accuracy check on val set."""
    model.eval()
    df = val_df.head(n_samples)
    prompts = df["prompt"].tolist()
    gt_answers = df["answer"].astype(str).str.strip().tolist()

    correct = 0
    no_answer = 0
    t0 = time.time()

    for i, (prompt, gt) in enumerate(zip(prompts, gt_answers)):
        text = format_example(prompt)
        enc = tokenizer(text, return_tensors="pt").to(model.device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
        output = tokenizer.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_answer(output)

        if pred is None:
            no_answer += 1
        elif pred == gt:
            correct += 1

        if (i + 1) % 20 == 0 or i == 0:
            acc_so_far = correct / (i + 1)
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(prompts)}] acc={acc_so_far:.3f} no_ans={no_answer} ({elapsed:.0f}s)")

    accuracy = correct / len(prompts)
    elapsed = time.time() - t0
    print(f"\nVal accuracy: {accuracy:.4f} ({correct}/{len(prompts)})")
    print(f"No-answer: {no_answer}/{len(prompts)}")
    print(f"Time: {elapsed:.0f}s ({elapsed/len(prompts):.1f}s/sample)")
    return accuracy


# ── Save & Package ────────────────────────────────────────────────────────────
def save_submission(trainer, tokenizer):
    """Save adapter and create submission.zip."""
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    print(f"Adapter saved to {ADAPTER_DIR}")

    # Create submission.zip in /kaggle/working
    os.chdir(str(ADAPTER_DIR))
    subprocess.run("zip -r /kaggle/working/submission.zip .", shell=True, check=True)
    os.chdir(str(OUTPUT_DIR))
    print("Created submission.zip")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("NVIDIA Nemotron Reasoning — SFT Fine-Tuning")
    print(f"LoRA: r={LORA_RANK} alpha={LORA_ALPHA}")
    print(f"Training: {NUM_EPOCHS} epochs, lr={LEARNING_RATE}, bs={BATCH_SIZE}x{GRAD_ACCUM}")
    print("=" * 60)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.total_memory / 1e9:.1f} GB)")

    # Load data
    train_df, val_df = load_data()

    # Setup model
    model, tokenizer = setup_model()

    # Train
    trainer = train(model, tokenizer, train_df, val_df)

    # Quick eval on 100 val samples
    print("\n--- Evaluation ---")
    accuracy = evaluate(model, tokenizer, val_df, n_samples=100)

    # Save
    save_submission(trainer, tokenizer)

    print(f"\nFinal val accuracy: {accuracy:.4f}")
    print(f"Target: 0.683")
    print(f"Gap: {0.683 - accuracy:+.4f}")

    return accuracy


if __name__ == "__main__":
    main()
