"""
GPU Worker — runs on RunPod (A100 80GB).
Receives experiment configs, trains LoRA, evaluates, returns results.

Usage:
    python gpu_worker.py --config config.json          # single experiment
    python gpu_worker.py --config batch_configs.json   # batch of experiments
    python gpu_worker.py --server --port 8000          # HTTP server mode

Config JSON format:
{
    "exp_id": "001",
    "description": "r=32 a=64 3ep baseline",
    "rank": 32, "alpha": 64, "epochs": 3,
    "lr": 2e-4, "batch_size": 4, "grad_accum": 4,
    "max_len": 4096, "warmup_ratio": 0.05,
    "n_train": null, "n_eval": 200,
    "data_path": "/workspace/data/train.csv",
    "model_path": "/workspace/model",
    "output_dir": "/workspace/adapters"
}
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Constants (matching competition spec)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert at solving logical reasoning puzzles including "
    "bit manipulation, algebra, and text encryption. "
    "Solve the problem step by step, then give your final answer inside \\boxed{}."
)

# Competition-specified LoRA targets
LORA_TARGET_MODULES = r".*\.(in_proj|out_proj|up_proj|down_proj)$"

CHAT_TEMPLATE = "<extra_id_0>System\n{system}\n<extra_id_1>User\n{user}\n<extra_id_1>Assistant\n"


def format_example(prompt, answer=None):
    text = CHAT_TEMPLATE.format(system=SYSTEM_PROMPT, user=prompt)
    if answer is not None:
        text += f"Let me work through this step by step.\n\nThe answer is \\boxed{{{answer}}}"
    return text


def extract_answer(text):
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    return matches[-1].strip() if matches else None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_splits(data_path, n_train=None, val_frac=0.1, seed=42):
    df = pd.read_csv(data_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.dropna(subset=["prompt", "answer"]).reset_index(drop=True)
    df["answer"] = df["answer"].astype(str).str.strip()

    rng = np.random.default_rng(seed)
    n_val = max(100, int(len(df) * val_frac))
    val_idx = set(rng.choice(len(df), size=n_val, replace=False))
    mask = np.array([i in val_idx for i in range(len(df))])
    train_df = df[~mask].reset_index(drop=True)
    val_df = df[mask].reset_index(drop=True)

    if n_train and len(train_df) > n_train:
        train_df = train_df.sample(n=n_train, random_state=seed).reset_index(drop=True)

    return train_df, val_df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_adapter(config: dict) -> dict:
    """Train a LoRA adapter. Returns result dict with metrics."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    from transformers import DataCollatorForLanguageModeling
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset

    t_start = time.time()
    exp_id = config["exp_id"]
    model_path = config.get("model_path", "/workspace/model")
    data_path = config.get("data_path", "/workspace/data/train.csv")
    output_base = Path(config.get("output_dir", "/workspace/adapters"))
    adapter_dir = output_base / f"exp_{exp_id}"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    rank = config.get("rank", 32)
    alpha = config.get("alpha", 64)
    epochs = config.get("epochs", 3)
    lr = config.get("lr", 2e-4)
    batch_size = config.get("batch_size", 4)
    grad_accum = config.get("grad_accum", 4)
    max_len = config.get("max_len", 4096)
    warmup_ratio = config.get("warmup_ratio", 0.05)
    n_train = config.get("n_train")
    n_eval = config.get("n_eval", 200)

    print(f"\n{'='*60}")
    print(f"Experiment {exp_id}: {config.get('description', '')}")
    print(f"  rank={rank} alpha={alpha} epochs={epochs} lr={lr}")
    print(f"  batch={batch_size} grad_accum={grad_accum} max_len={max_len}")
    print(f"{'='*60}")

    # Load data
    train_df, val_df = load_splits(data_path, n_train=n_train)
    print(f"Data: train={len(train_df)} val={len(val_df)}")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    t_load = time.time() - t_start
    print(f"Model loaded in {t_load:.0f}s — GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Apply LoRA
    lora_config = LoraConfig(
        r=rank, lora_alpha=alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenize
    train_texts = [format_example(r["prompt"], r["answer"]) for _, r in train_df.iterrows()]
    val_texts = [format_example(r["prompt"], r["answer"]) for _, r in val_df.head(500).iterrows()]

    train_tok = tokenizer(train_texts, truncation=True, max_length=max_len, padding=False)
    val_tok = tokenizer(val_texts, truncation=True, max_length=max_len, padding=False)

    train_ds = Dataset.from_dict({"input_ids": train_tok["input_ids"], "attention_mask": train_tok["attention_mask"]})
    val_ds = Dataset.from_dict({"input_ids": val_tok["input_ids"], "attention_mask": val_tok["attention_mask"]})

    # Train
    training_args = TrainingArguments(
        output_dir=str(adapter_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        bf16=True,
        logging_steps=20,
        save_strategy="no",
        eval_strategy="epoch",
        report_to="none",
        seed=42,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        max_grad_norm=1.0,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=data_collator,
    )

    print(f"\nTraining ({len(train_ds)} samples, {epochs} epochs)...")
    t_train_start = time.time()
    train_result = trainer.train()
    t_train = time.time() - t_train_start
    train_loss = train_result.training_loss
    print(f"Training done in {t_train/60:.1f} min — loss: {train_loss:.4f}")

    # Save adapter
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # Evaluate accuracy on val set
    print(f"\nEvaluating on {n_eval} val samples...")
    accuracy, no_answer = evaluate_adapter(model, tokenizer, val_df.head(n_eval), max_len)

    t_total = time.time() - t_start

    result = {
        "exp_id": exp_id,
        "description": config.get("description", ""),
        "rank": rank, "alpha": alpha, "epochs": epochs, "lr": lr,
        "batch_size": batch_size, "grad_accum": grad_accum, "max_len": max_len,
        "n_train": len(train_ds), "n_eval": n_eval,
        "train_loss": round(train_loss, 6),
        "val_accuracy": round(accuracy, 4),
        "no_answer_rate": round(no_answer / max(n_eval, 1), 4),
        "train_time_min": round(t_train / 60, 1),
        "total_time_min": round(t_total / 60, 1),
        "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
        "adapter_path": str(adapter_dir),
        "timestamp": datetime.now().isoformat(),
    }

    # Save result alongside adapter
    (adapter_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"\nResult: acc={accuracy:.4f} loss={train_loss:.4f} time={t_total/60:.1f}min")
    print(f"Adapter saved: {adapter_dir}")

    return result


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.inference_mode()
def evaluate_adapter(model, tokenizer, val_df, max_len, max_new_tokens=512):
    model.eval()
    correct = 0
    no_answer = 0
    t0 = time.time()

    for i, (_, row) in enumerate(val_df.iterrows()):
        prompt = format_example(row["prompt"])
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).to("cuda:0")
        gen = model.generate(
            **enc, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
        output = tokenizer.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_answer(output)
        gt = str(row["answer"]).strip()

        if pred is None:
            no_answer += 1
        elif pred == gt:
            correct += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(val_df)}] acc={correct/(i+1):.3f} no_ans={no_answer} ({elapsed:.0f}s)")

    accuracy = correct / len(val_df)
    print(f"  Final: {accuracy:.4f} ({correct}/{len(val_df)}) no_ans={no_answer}")
    return accuracy, no_answer


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------
def run_batch(configs: list[dict]):
    """Run multiple experiments, reusing the loaded model."""
    results = []
    for i, config in enumerate(configs):
        print(f"\n{'#'*60}")
        print(f"# Batch {i+1}/{len(configs)}: {config.get('description', config['exp_id'])}")
        print(f"{'#'*60}")

        try:
            result = train_adapter(config)
            results.append(result)
        except Exception as e:
            print(f"ERROR in exp {config['exp_id']}: {e}")
            results.append({"exp_id": config["exp_id"], "error": str(e)})

        # Free GPU memory between experiments
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # Save batch results
    out_path = Path(configs[0].get("output_dir", "/workspace/adapters")) / "batch_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nBatch results saved: {out_path}")
    return results


# ---------------------------------------------------------------------------
# HTTP server mode (for orchestrator integration)
# ---------------------------------------------------------------------------
def run_server(port=8000):
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import threading

    class WorkerHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == "/train":
                length = int(self.headers["Content-Length"])
                body = json.loads(self.rfile.read(length))

                # Run in thread to not block
                configs = body if isinstance(body, list) else [body]
                results = run_batch(configs)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(results).encode())

            elif self.path == "/health":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
            else:
                self.send_response(404)
                self.end_headers()

        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.end_headers()
                gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                self.wfile.write(json.dumps({"status": "ready", "gpu_mem_gb": gpu_mem}).encode())
            else:
                self.send_response(404)
                self.end_headers()

    server = HTTPServer(("0.0.0.0", port), WorkerHandler)
    print(f"GPU Worker server running on port {port}")
    print(f"  POST /train — submit experiment config(s)")
    print(f"  GET  /health — check worker status")
    server.serve_forever()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="GPU Worker for Nemotron LoRA training")
    p.add_argument("--config", type=str, help="Path to config JSON (single or array)")
    p.add_argument("--server", action="store_true", help="Run as HTTP server")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    if args.server:
        run_server(args.port)
    elif args.config:
        with open(args.config) as f:
            data = json.load(f)
        if isinstance(data, list):
            run_batch(data)
        else:
            train_adapter(data)
    else:
        p.print_help()
