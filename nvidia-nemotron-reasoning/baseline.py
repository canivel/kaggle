"""
Zero-shot baseline evaluation on 50 val samples.
Uses 4-bit quantization so it fits in RTX 3080 10GB VRAM.

Usage:
    uv run python baseline.py
    uv run python baseline.py --n-samples 20 --max-new-tokens 128
"""

import argparse
import re
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import torch

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_MODEL_ID = "nvidia/Nemotron-Mini-4B-Instruct"

# Competition inference params (temperature=0, no sampling)
SYSTEM = (
    "You are an expert at solving logical reasoning puzzles including bit manipulation, "
    "algebra, and text encryption. Solve the problem step by step, then give your final "
    "answer inside \\boxed{}."
)


def build_prompt(tokenizer, problem: str) -> str:
    """Use the model's own chat template if available, else a generic fallback."""
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": problem},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # Generic fallback (shouldn't be needed for Nemotron-Mini)
    return f"<extra_id_0>System\n{SYSTEM}\n<extra_id_1>User\n{problem}\n<extra_id_1>Assistant\n"


def extract_answer(text: str) -> str | None:
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else None


def main(args):
    print(f"\n{'='*60}")
    print("NVIDIA Nemotron Reasoning — Zero-Shot Baseline")
    print(f"Model: {args.model_id}")
    print(f"{'='*60}\n")

    # ── CUDA check ───────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Cannot run 8B model without GPU.")
        sys.exit(1)
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1e9
    print(f"GPU: {props.name}  VRAM: {vram_gb:.1f} GB")
    if vram_gb < 6:
        print("WARNING: Less than 6GB VRAM — may OOM even with 4-bit quant")

    # ── Load data ────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_DIR / "train.csv")
    df.columns = [c.strip().lower() for c in df.columns]

    # Reproducible val sample (last 10% = 950 rows, we eval first N)
    rng = np.random.default_rng(42)
    n_total = len(df)
    val_idx = rng.choice(n_total, size=max(50, int(n_total * 0.1)), replace=False)
    val_df = df.iloc[sorted(val_idx)].reset_index(drop=True)
    val_df = val_df.head(args.n_samples)
    print(f"Evaluating on {len(val_df)} samples\n")

    # ── Load model ───────────────────────────────────────────────────────────
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    used_gb = torch.cuda.memory_allocated() / 1e9
    print(f"Model loaded. GPU memory: {used_gb:.1f} GB\n")

    # ── Inference ────────────────────────────────────────────────────────────
    prompts = val_df["prompt"].tolist()
    gt_answers = val_df["answer"].astype(str).str.strip().tolist()

    results = []
    t0 = datetime.now()

    with torch.inference_mode():
        for i, (prompt, gt) in enumerate(zip(prompts, gt_answers), 1):
            enc = tokenizer(build_prompt(tokenizer, prompt), return_tensors="pt").to(model.device)
            gen = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_ids = gen[0][enc["input_ids"].shape[1]:]
            output = tokenizer.decode(new_ids, skip_special_tokens=True)
            pred = extract_answer(output)
            correct = pred == gt

            results.append({
                "id": val_df.iloc[i-1].get("id", i),
                "prompt": prompt[:80] + "...",
                "gt": gt,
                "pred": pred,
                "correct": correct,
                "output": output,
            })

            elapsed = (datetime.now() - t0).total_seconds()
            print(f"[{i:3d}/{len(prompts)}] {'OK' if correct else '--'}  GT={gt!r:20s}  Pred={pred!r}  ({elapsed:.0f}s)")

    # ── Results ──────────────────────────────────────────────────────────────
    n = len(results)
    n_correct = sum(r["correct"] for r in results)
    n_noans = sum(r["pred"] is None for r in results)
    accuracy = n_correct / n

    elapsed_total = (datetime.now() - t0).total_seconds()

    print(f"\n{'='*60}")
    print(f"  Model:       {args.model_id}")
    print(f"  Mode:        zero-shot (no fine-tuning)")
    print(f"  Samples:     {n}")
    print(f"  Correct:     {n_correct}")
    print(f"  No-answer:   {n_noans}")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  Time:        {elapsed_total:.0f}s  ({elapsed_total/n:.1f}s/sample)")
    print(f"  Target:      0.683 (1st place)")
    print(f"  Gap:         {0.683 - accuracy:+.4f}")
    print(f"{'='*60}\n")

    # Save
    out_path = Path(__file__).parent / "baseline_results.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Results saved -> {out_path}")

    return accuracy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--n-samples", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=256)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
