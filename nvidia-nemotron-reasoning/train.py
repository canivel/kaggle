"""
QLoRA fine-tuning for the NVIDIA Nemotron Reasoning Challenge.

Usage:
    uv run python train.py [options]

GPU notes:
    RTX 3080 (10GB): use --qlora (4-bit base + LoRA). Max --max-len 2048.
    A100/H100 (40-80GB): can drop --qlora, increase --max-len 4096+.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import torch

DATA_DIR = Path(__file__).parent / "data"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"

# RTX 3080 default: 8B model
DEFAULT_MODEL_ID = "nvidia/Nemotron-Mini-4B-Instruct"

# ── Prompt ──────────────────────────────────────────────────────────────────
SYSTEM = (
    "You are an expert at solving logical reasoning puzzles including bit manipulation, "
    "algebra, and text encryption. Solve the problem step by step, then give your final "
    "answer inside \\boxed{}."
)

def fmt(prompt: str, answer: str | None = None) -> str:
    text = f"<|system|>\n{SYSTEM}\n<|user|>\n{prompt}\n<|assistant|>\n"
    if answer is not None:
        text += f"\\boxed{{{answer}}}<|endoftext|>"
    return text


# ── Data ─────────────────────────────────────────────────────────────────────
def load_splits(data_path: Path, val_frac: float = 0.1, seed: int = 42):
    df = pd.read_csv(data_path / "train.csv")
    df.columns = [c.strip().lower() for c in df.columns]
    assert "prompt" in df.columns, f"Missing 'prompt' column. Got: {list(df.columns)}"
    assert "answer" in df.columns, f"Missing 'answer' column. Got: {list(df.columns)}"
    df = df.dropna(subset=["prompt", "answer"]).reset_index(drop=True)
    df["answer"] = df["answer"].astype(str).str.strip()

    rng = np.random.default_rng(seed)
    val_idx = set(rng.choice(len(df), size=max(10, int(len(df) * val_frac)), replace=False))
    mask = np.array([i in val_idx for i in range(len(df))])
    train_df = df[~mask].reset_index(drop=True)
    val_df = df[mask].reset_index(drop=True)
    print(f"Split -> train: {len(train_df)}  val: {len(val_df)}")
    return train_df, val_df


# ── Model ─────────────────────────────────────────────────────────────────────
def build(model_id: str, rank: int, alpha: int, use_qlora: bool):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    load_kw: dict = dict(
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if use_qlora:
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    print(f"Loading {'4-bit QLoRA' if use_qlora else 'bf16'} model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kw)

    if use_qlora:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


# ── Training ─────────────────────────────────────────────────────────────────
def train(args):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"lora_r{args.rank}_a{args.alpha}_{ts}"
    out_dir = CHECKPOINT_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df = load_splits(Path(args.data_path))
    val_df.to_csv(out_dir / "val_split.csv", index=False)

    from datasets import Dataset
    train_ds = Dataset.from_dict({"text": [fmt(r["prompt"], r["answer"]) for _, r in train_df.iterrows()]})
    val_ds   = Dataset.from_dict({"text": [fmt(r["prompt"], r["answer"]) for _, r in val_df.iterrows()]})

    model, tokenizer = build(args.model_id, args.rank, args.alpha, args.qlora)

    from trl import SFTConfig, SFTTrainer

    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
        run_name=run_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=not args.qlora,
        fp16=False,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_length=args.max_len,
        dataset_text_field="text",
        report_to="none",
        seed=42,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if args.qlora else "adamw_torch",
        dataloader_pin_memory=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    print(f"\n>> Training: {run_name}")
    trainer.train()

    adapter_path = out_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\nOK Adapter saved -> {adapter_path}")

    meta = {
        "run_name": run_name, "model_id": args.model_id,
        "rank": args.rank, "alpha": args.alpha, "qlora": args.qlora,
        "epochs": args.epochs, "lr": args.lr, "timestamp": ts,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return str(adapter_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default=None)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--data-path", default=str(DATA_DIR))
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--qlora", action="store_true", default=True,
                   help="Use 4-bit QLoRA (required for RTX 3080 10GB)")
    p.add_argument("--no-qlora", action="store_false", dest="qlora")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    adapter_path = train(args)
    print(f"\nDone -> {adapter_path}")
