"""
Inference + accuracy evaluation for the NVIDIA Nemotron Reasoning Challenge.

Usage:
    uv run python evaluate.py --adapter checkpoints/my_run/adapter --split val
    uv run python evaluate.py --no-adapter --split val   # zero-shot
    uv run python evaluate.py --adapter best_adapter/ --split test --output preds.csv
"""

import argparse
import re
from pathlib import Path
from datetime import datetime

import pandas as pd
import torch

DEFAULT_MODEL_ID = "nvidia/Nemotron-Mini-4B-Instruct"
DATA_DIR = Path(__file__).parent / "data"

SYSTEM = (
    "You are an expert at solving logical reasoning puzzles including bit manipulation, "
    "algebra, and text encryption. Solve the problem step by step, then give your final "
    "answer inside \\boxed{}."
)

def build_prompt(problem: str) -> str:
    return f"<|system|>\n{SYSTEM}\n<|user|>\n{problem}\n<|assistant|>\n"


def extract_answer(text: str) -> str | None:
    """Extract last \\boxed{...} from model output, handling nested braces."""
    # Match \boxed{ ... } with one level of nesting
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None


def load_model(model_id: str, adapter_path: str | None, use_4bit: bool = True):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kw: dict = dict(device_map="auto", trust_remote_code=True)
    if use_4bit:
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        load_kw["torch_dtype"] = torch.bfloat16

    print(f"Loading base model: {model_id} ({'4-bit' if use_4bit else 'bf16'})")
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kw)

    if adapter_path:
        from peft import PeftModel
        print(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


@torch.inference_mode()
def run_inference(model, tokenizer, prompts: list[str], max_new_tokens: int = 256) -> list[str]:
    outputs = []
    for i, prompt in enumerate(prompts, 1):
        if i % 10 == 0 or i == 1:
            print(f"  [{i}/{len(prompts)}]", flush=True)
        enc = tokenizer(build_prompt(prompt), return_tensors="pt").to(model.device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
        new = gen[0][enc["input_ids"].shape[1]:]
        outputs.append(tokenizer.decode(new, skip_special_tokens=True))
    return outputs


def evaluate(args):
    # ── Load data ────────────────────────────────────────────────────────────
    if args.split == "val":
        if args.adapter:
            val_csv = Path(args.adapter).parent / "val_split.csv"
            if val_csv.exists():
                df = pd.read_csv(val_csv)
                print(f"Val split from adapter dir: {len(df)} examples")
            else:
                df = pd.read_csv(DATA_DIR / "train.csv")
                df.columns = [c.strip().lower() for c in df.columns]
                df = df.tail(max(50, int(len(df) * 0.1))).reset_index(drop=True)
        else:
            df = pd.read_csv(DATA_DIR / "train.csv")
            df.columns = [c.strip().lower() for c in df.columns]
            df = df.tail(max(50, int(len(df) * 0.1))).reset_index(drop=True)
            print(f"Using last {len(df)} rows as val")
        if args.n_samples:
            df = df.head(args.n_samples)
            print(f"Subsampled to {len(df)} examples (--n-samples)")
    elif args.split == "test":
        df = pd.read_csv(DATA_DIR / "test.csv")
        df.columns = [c.strip().lower() for c in df.columns]
    else:
        raise ValueError(f"Unknown split: {args.split}")

    prompts = df["prompt"].tolist()
    has_gt = "answer" in df.columns

    # ── Load model ───────────────────────────────────────────────────────────
    adapter_path = None if args.no_adapter else args.adapter
    model, tokenizer = load_model(args.model_id, adapter_path, use_4bit=not args.no_4bit)

    # ── Inference ────────────────────────────────────────────────────────────
    print(f"\nRunning inference on {len(prompts)} examples (max_new_tokens={args.max_new_tokens}) ...")
    raw_outputs = run_inference(model, tokenizer, prompts, args.max_new_tokens)
    extracted = [extract_answer(o) for o in raw_outputs]

    # ── Accuracy ─────────────────────────────────────────────────────────────
    accuracy = None
    if has_gt:
        gt = df["answer"].astype(str).str.strip().tolist()
        correct_flags = [e == g for e, g in zip(extracted, gt)]
        correct = sum(correct_flags)
        accuracy = correct / len(gt)
        no_ans = sum(1 for e in extracted if e is None)

        print(f"\n{'='*52}")
        print(f"  Accuracy:    {accuracy:.4f}  ({correct}/{len(gt)})")
        print(f"  No-answer:   {no_ans}/{len(gt)}")
        print(f"{'='*52}")

        print("\nSample (first 20):")
        for i, (p, o, g, e) in enumerate(zip(prompts[:20], raw_outputs[:20], gt[:20], extracted[:20])):
            mark = "OK" if e == g else "XX"
            print(f"  [{i+1:3d}] {mark} GT={g!r:20s}  Pred={e!r}")
    else:
        print(f"\nGenerated {len(extracted)} predictions (no ground truth)")

    # ── Save ─────────────────────────────────────────────────────────────────
    if args.output:
        out = pd.DataFrame({
            "id": df.get("id", pd.Series(range(len(df)))),
            "prompt": prompts,
            "raw_output": raw_outputs,
            "extracted_answer": extracted,
        })
        if has_gt:
            out["ground_truth"] = df["answer"].astype(str).str.strip()
            out["correct"] = [e == g for e, g in zip(extracted, out["ground_truth"])]
        out.to_csv(args.output, index=False)
        print(f"\nSaved predictions -> {args.output}")

    return accuracy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default=None)
    p.add_argument("--no-adapter", action="store_true")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--output", default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--n-samples", type=int, default=None,
                   help="Evaluate on first N samples only (for quick checks)")
    p.add_argument("--no-4bit", action="store_true",
                   help="Load in bf16 instead of 4-bit (needs ~16GB VRAM for 8B)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.no_adapter and args.adapter is None:
        args.no_adapter = True
    acc = evaluate(args)
    if acc is not None:
        print(f"\nFinal accuracy: {acc:.4f}")
