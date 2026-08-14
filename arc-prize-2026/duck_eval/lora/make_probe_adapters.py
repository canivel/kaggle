"""Build two randomly-initialized LoRA adapters for the SERVE smoke. CPU only.

Why two, and why random: the serve path has never been observed working in this
competition (auxentr specified it and their kernel ERRORed at t=12 s). We want
certainty about *serving* before spending a training token, and a random adapter
buys it for one GPU-hour. But one adapter is not enough:

  * `noop`  -- standard LoRA init, `B = 0`, so the delta is exactly zero. Its
    outputs MUST be token-identical to the base. This measures the *cost* of
    having LoRA enabled (throughput, memory) with zero confound.
  * `probe` -- `B` small but non-zero, so the delta MUST reach the logits. Its
    outputs must DIFFER from the base.

Run both against the same server and the failure modes separate cleanly:

  | noop == base | probe != base | verdict                                     |
  |--------------|---------------|---------------------------------------------|
  | yes          | yes           | LoRA is loaded AND applied. PASS.           |
  | yes          | **no**        | adapter silently ignored -- the exact failure `vllm_runtime_lora_guard` exists to catch |
  | **no**       | yes           | a zero-delta adapter changed output => the LoRA path is numerically unsound |

Shapes and key names are GROUND TRUTH, taken from auxentr's shipped
`adapter_model.safetensors` (`iseesmth/duck-harness-nca-qwen36-adapter-20260811`),
not inferred:

    base_model.model.model.language_model.layers.{i}.self_attn.{q,k,v,o}_proj.lora_{A,B}.weight
    i in {3,7,11,...,63}   (the 16 full_attention layers, interval 4)
    q: A[16,5120]  B[12288,16]   <- 12288, NOT 6144: Qwen3.5 gated attention
    k: A[16,5120]  B[ 1024,16]
    v: A[16,5120]  B[ 1024,16]
    o: A[16,6144]  B[ 5120,16]
    dtype F32, 128 tensors, 10,485,760 params, 41,943,040 B + header

    ../../.venv/Scripts/python.exe make_probe_adapters.py --out ../../runs/lora_lane/probe_adapters
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FULL_ATTENTION_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]
RANK = 16
ALPHA = 32
# (module, in_features, out_features)
PROJECTIONS = [("q_proj", 5120, 12288), ("k_proj", 5120, 1024),
               ("v_proj", 5120, 1024), ("o_proj", 6144, 5120)]
PREFIX = "base_model.model.model.language_model.layers"

# auxentr's regex, verbatim -- it is what vLLM's peft_helper accepts and what
# their (statically valid) config used. Reusing it removes one variable.
TARGET_MODULES = (
    r"(?:.*?(?:language|text).*?(?:self_attn|attention|attn|mixer).*?(?:q_proj|k_proj|v_proj|o_proj))"
    r"|(?:\bmodel\.layers\.[\d]{1,}\.(?:self_attn|attention|attn|mixer)\.(?:(?:q_proj|k_proj|v_proj|o_proj)))"
)

EXPECTED_PARAMS = 10_485_760
EXPECTED_BYTES = EXPECTED_PARAMS * 4


def adapter_config() -> dict:
    return {
        "auto_mapping": {
            "base_model_class": "Qwen3_5ForConditionalGeneration",
            "parent_library": "transformers.models.qwen3_5.modeling_qwen3_5",
        },
        "base_model_name_or_path": "vrfai/Qwen3.6-27B-FP8",
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": ALPHA,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": RANK,
        "target_modules": TARGET_MODULES,
        "task_type": "CAUSAL_LM",
        # rsLoRA => scaling alpha/sqrt(r) = 8.0, not alpha/r = 2.0. auxentr set it
        # and vLLM 0.19's peft_helper honours it; matching removes a variable.
        "use_rslora": True,
    }


def save_safetensors(tensors: dict, path: Path) -> None:
    """Minimal safetensors writer (F32 only), so this script needs nothing but
    numpy. Format: <u64 header_len><json header><contiguous little-endian data>,
    data offsets relative to the end of the header. Verified byte-compatible by
    re-reading auxentr's file with the same parser."""
    import numpy as np

    header: dict = {}
    offset = 0
    blobs = []
    for name in sorted(tensors):
        arr = np.ascontiguousarray(tensors[name], dtype="<f4")
        raw = arr.tobytes()
        header[name] = {"dtype": "F32", "shape": list(arr.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        blobs.append(raw)
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-len(header_json)) % 8  # safetensors requires 8-byte alignment
    header_json += b" " * pad
    with path.open("wb") as handle:
        handle.write(len(header_json).to_bytes(8, "little"))
        handle.write(header_json)
        for raw in blobs:
            handle.write(raw)


def build(kind: str, seed: int) -> dict:
    import numpy as np

    rng = np.random.default_rng(seed)
    tensors: dict = {}
    for layer in FULL_ATTENTION_LAYERS:
        for name, fan_in, fan_out in PROJECTIONS:
            stem = f"{PREFIX}.{layer}.self_attn.{name}"
            # Kaiming-uniform on A, exactly as PEFT initializes it.
            bound = (3.0 / fan_in) ** 0.5
            a = rng.uniform(-bound, bound, size=(RANK, fan_in)).astype("float32")
            if kind == "noop":
                b = np.zeros((fan_out, RANK), dtype="float32")
            else:
                # Small but unmistakable: with rsLoRA scaling 8.0 and |A| ~ 1e-2,
                # a 1e-3 B puts the delta well above bf16 noise yet far from
                # destroying the model, so a bad output cannot be blamed on garbage.
                b = (rng.standard_normal((fan_out, RANK)) * 1e-3).astype("float32")
            tensors[f"{stem}.lora_A.weight"] = a
            tensors[f"{stem}.lora_B.weight"] = b
    return tensors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.out)
    report: dict[str, dict] = {}
    for kind, seed in (("noop", 20260813), ("probe", 20260814)):
        target = root / f"lora-{kind}"
        target.mkdir(parents=True, exist_ok=True)
        tensors = build(kind, seed)
        n_params = sum(int(t.size) for t in tensors.values())
        if n_params != EXPECTED_PARAMS:
            raise SystemExit(f"FATAL: {kind} has {n_params:,} params, want {EXPECTED_PARAMS:,}")
        save_safetensors(tensors, target / "adapter_model.safetensors")
        (target / "adapter_config.json").write_text(
            json.dumps(adapter_config(), indent=2), encoding="utf-8"
        )
        blob = (target / "adapter_model.safetensors").read_bytes()
        nonzero_b = sum(int(bool(t.any())) for k, t in tensors.items() if k.endswith("lora_B.weight"))
        report[kind] = {
            "tensors": len(tensors),
            "params": n_params,
            "bytes": len(blob),
            "sha256_16": hashlib.sha256(blob).hexdigest()[:16],
            "nonzero_lora_B_modules": nonzero_b,
            "expected_delta": "exactly zero" if kind == "noop" else "non-zero",
        }
        if kind == "noop" and nonzero_b != 0:
            raise SystemExit("FATAL: noop adapter has a non-zero B")
        if kind == "probe" and nonzero_b != 64:
            raise SystemExit(f"FATAL: probe adapter has {nonzero_b}/64 non-zero B")
        print(f"{kind}: {len(tensors)} tensors, {n_params:,} params, {len(blob):,} B "
              f"(payload {EXPECTED_BYTES:,} + header {len(blob) - EXPECTED_BYTES})")

    (root / "PROBE_ADAPTERS.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
