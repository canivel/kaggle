"""
Local training on RTX 3080 (10GB VRAM).
Uses fake mamba_ssm + 4-bit quantization + CPU offloading.

Usage: uv run python local_train.py
"""
import os, sys, re, json, time, gc
from pathlib import Path

# Add fake mamba_ssm to path BEFORE any imports
sys.path.insert(0, str(Path(__file__).parent))

os.environ["HF_HOME"] = "F:/hf-cache"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pandas as pd
import torch

# Verify fake mamba_ssm loads
import mamba_ssm
print(f"mamba_ssm (fake): {mamba_ssm.__version__}")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# ── Config ────────────────────────────────────────────────────────────────
MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
ADAPTER_DIR = BASE_DIR / "local_adapter"

LORA_RANK = 32
LORA_ALPHA = 64
NUM_EPOCHS = 1
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
GRAD_ACCUM = 16
MAX_SEQ_LEN = 512  # all prompts are <510 chars

SYSTEM_PROMPT = (
    "You are an expert at solving logical reasoning puzzles including "
    "bit manipulation, algebra, and text encryption. "
    "Solve the problem step by step, then give your final answer inside \\boxed{}."
)


def format_example(prompt, answer=None):
    text = f"<extra_id_0>System\n{SYSTEM_PROMPT}\n<extra_id_1>User\n{prompt}\n<extra_id_1>Assistant\n"
    if answer is not None:
        text += f"\\boxed{{{answer}}}"
    return text


def extract_answer(text):
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    return matches[-1].strip() if matches else None


# ── Load Data ─────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_DIR / "train.csv")
df.columns = [c.strip().lower() for c in df.columns]
df = df.dropna(subset=["prompt", "answer"]).reset_index(drop=True)
df["answer"] = df["answer"].astype(str).str.strip()

rng = np.random.default_rng(42)
n_val = 200
val_idx = set(rng.choice(len(df), size=n_val, replace=False))
mask = np.array([i in val_idx for i in range(len(df))])
train_df = df[~mask].reset_index(drop=True)
val_df = df[mask].reset_index(drop=True)
print(f"Train: {len(train_df)} | Val: {len(val_df)}")

# ── Load Model (4-bit + CPU offload) ─────────────────────────────────────
print(f"Loading tokenizer from {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# No quantization — fp16 + CPU offload (137GB RAM can hold the 60GB model)
print("Loading model (fp16, GPU+CPU split, 137GB RAM available)...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    max_memory={0: "7GiB", "cpu": "80GiB"},
    offload_folder=str(BASE_DIR / "offload_weights"),
)

# Force pure PyTorch Mamba path
for mod_name, mod in sys.modules.items():
    if "modeling_nemotron_h" in mod_name:
        if hasattr(mod, "is_fast_path_available"):
            mod.is_fast_path_available = False
        if hasattr(mod, "rmsnorm_fn"):
            from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
            mod.rmsnorm_fn = rmsnorm_fn
        print(f"Patched {mod_name}: pure PyTorch mode")
        break

gpu_gb = torch.cuda.memory_allocated() / 1e9
print(f"Model loaded: {gpu_gb:.1f} GB GPU")

# ── LoRA ──────────────────────────────────────────────────────────────────
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

lora_config = LoraConfig(
    r=LORA_RANK, lora_alpha=LORA_ALPHA,
    target_modules=["in_proj", "out_proj", "up_proj", "down_proj"],
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Sanity check
print("Sanity forward pass...")
test_ids = tokenizer("Test", return_tensors="pt").to(model.device)
with torch.no_grad():
    _ = model(**test_ids)
print(f"Forward pass OK! GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# ── Tokenize ─────────────────────────────────────────────────────────────
print("Tokenizing...")
train_texts = [format_example(r["prompt"], r["answer"]) for _, r in train_df.iterrows()]
val_texts = [format_example(r["prompt"], r["answer"]) for _, r in val_df.iterrows()]

train_tok = tokenizer(train_texts, truncation=True, max_length=MAX_SEQ_LEN, padding=False, return_tensors=None)
val_tok = tokenizer(val_texts, truncation=True, max_length=MAX_SEQ_LEN, padding=False, return_tensors=None)

train_ds = Dataset.from_dict({"input_ids": train_tok["input_ids"], "attention_mask": train_tok["attention_mask"]})
val_ds = Dataset.from_dict({"input_ids": val_tok["input_ids"], "attention_mask": val_tok["attention_mask"]})
print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Avg tokens: {np.mean([len(ids) for ids in train_tok['input_ids']]):.0f}")

# ── Train ─────────────────────────────────────────────────────────────────
class AutoSaveCallback:
    """Saves submission-ready adapter every N steps."""
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step > 0 and state.global_step % 100 == 0:
            self._save(model, state.global_step)
    def on_train_end(self, args, state, control, model=None, **kwargs):
        self._save(model, state.global_step)
    def _save(self, model, step):
        ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ADAPTER_DIR))
        tokenizer.save_pretrained(str(ADAPTER_DIR))
        cfg_path = ADAPTER_DIR / "adapter_config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["base_model_name_or_path"] = MODEL_ID
        cfg_path.write_text(json.dumps(cfg, indent=2))
        print(f"\n>> Adapter saved at step {step} -> {ADAPTER_DIR}")


from transformers import TrainerCallback
class SaveCB(TrainerCallback, AutoSaveCallback):
    pass

run_name = f"local_r{LORA_RANK}_a{LORA_ALPHA}_ep{NUM_EPOCHS}"
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=str(BASE_DIR / "local_checkpoints"),
    run_name=run_name,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    fp16=False,
    bf16=False,
    logging_steps=5,
    save_strategy="no",
    eval_strategy="no",
    report_to="none",
    seed=42,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_torch",
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    max_grad_norm=1.0,
)

trainer = Trainer(
    model=model, args=training_args,
    train_dataset=train_ds, eval_dataset=val_ds,
    data_collator=data_collator,
    callbacks=[SaveCB()],
)

steps = len(train_ds) // (BATCH_SIZE * GRAD_ACCUM)
print(f"\nTraining: {run_name}")
print(f"Samples: {len(train_ds)} | Batch: {BATCH_SIZE}x{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}")
print(f"Steps/epoch: ~{steps} | Auto-saves every 100 steps")
t0 = time.time()
trainer.train()
elapsed = (time.time() - t0) / 60
print(f"Done in {elapsed:.1f} min ({elapsed/60:.1f}h)")

# ── Submit ────────────────────────────────────────────────────────────────
import subprocess
zip_path = BASE_DIR / "submission.zip"
subprocess.run(
    f"cd {ADAPTER_DIR} && zip -r {zip_path} adapter_config.json adapter_model.safetensors",
    shell=True
)
print(f"\nsubmission.zip: {zip_path.stat().st_size / 1e6:.1f} MB")

result = subprocess.run(
    ["kaggle", "competitions", "submit",
     "-c", "nvidia-nemotron-model-reasoning-challenge",
     "-f", str(zip_path),
     "-m", f"local {run_name}"],
    capture_output=True, text=True
)
print(f"Kaggle: {result.stdout.strip() or result.stderr.strip()}")
