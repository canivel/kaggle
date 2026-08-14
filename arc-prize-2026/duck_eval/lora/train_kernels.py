"""The hand-rolled training pieces stage T needs, isolated so they can be tested.

Neither wheelhouse ships `peft`, `accelerate`, or a fused cross-entropy
(`learnings/war_room/lora_lane_2026-08-13.md` §1.4), so LoRA, the model loader
and the loss all have to be written by hand. Hand-rolled numerics are where
silent bugs live (R-12), so every piece here is validated against a reference
implementation on a small model on the local 3080 BEFORE any Kaggle GPU-hour is
spent -- see `test_train_kernels.py`.

This module is deliberately dependency-light: `torch` only. It is meant to be
embedded verbatim into the stage-T notebook.

Pieces:
  LoRALinear              frozen base + B@A delta, rsLoRA-aware
  attach_lora             wrap matching nn.Linear modules in-place
  lora_state_dict         PEFT-format state dict (key names are ground truth)
  save_peft_adapter       adapter_config.json + adapter_model.safetensors
  chunked_cross_entropy   never materializes [S, vocab] logits
  dequant_fp8             compressed-tensors per-tensor static FP8 -> bf16
  stream_assign           meta-init + per-tensor assign from a safetensors file
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _ckpt


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """`y = base(x) + scale * (x @ A^T) @ B^T`, base frozen.

    Scaling follows PEFT: `alpha / r` normally, `alpha / sqrt(r)` when
    `use_rslora`. auxentr ships `use_rslora: true` with alpha=32, r=16, so the
    scale is 8.0 and NOT 2.0 -- a 4x error if you assume the classic formula,
    and one vLLM would faithfully reproduce at serve time.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, use_rslora: bool = True) -> None:
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.r = int(r)
        self.alpha = float(alpha)
        self.use_rslora = bool(use_rslora)
        self.scaling = self.alpha / (math.sqrt(self.r) if use_rslora else self.r)
        device = base.weight.device
        dtype = torch.float32
        self.lora_A = nn.Parameter(torch.empty(self.r, base.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.r, device=device, dtype=dtype))
        # PEFT's init: kaiming-uniform on A, zeros on B => the adapter starts as
        # an exact no-op, which is also what makes the `noop` serve probe valid.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        lora = F.linear(F.linear(x.to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return out + self.scaling * lora.to(out.dtype)


def attach_lora(
    model: nn.Module,
    target_pattern: str,
    *,
    r: int = 16,
    alpha: float = 32.0,
    use_rslora: bool = True,
) -> list[str]:
    """Wrap every `nn.Linear` whose dotted name matches `target_pattern`.

    Returns the wrapped names. Everything else in the model is frozen, so the
    caller can assert `trainable == 2*r*sum(in+out)` exactly.
    """
    regex = re.compile(target_pattern)
    targets = [
        name for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and regex.search(name)
    ]
    for param in model.parameters():
        param.requires_grad_(False)
    for name in targets:
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        base = getattr(parent, parts[-1])
        setattr(parent, parts[-1], LoRALinear(base, r=r, alpha=alpha, use_rslora=use_rslora))
    return targets


def lora_state_dict(model: nn.Module, *, prefix: str = "base_model.model.") -> dict:
    """PEFT-format state dict. Key convention verified against auxentr's shipped
    `adapter_model.safetensors`:
        base_model.model.<module path>.lora_{A,B}.weight
    """
    out: dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            out[f"{prefix}{name}.lora_A.weight"] = module.lora_A.detach().to(torch.float32)
            out[f"{prefix}{name}.lora_B.weight"] = module.lora_B.detach().to(torch.float32)
    return out


def save_peft_adapter(
    model: nn.Module,
    path: Path,
    *,
    target_modules: str,
    r: int = 16,
    alpha: float = 32.0,
    use_rslora: bool = True,
    base_model: str = "vrfai/Qwen3.6-27B-FP8",
) -> dict:
    from make_probe_adapters import save_safetensors  # same minimal writer

    path.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.cpu().numpy() for k, v in lora_state_dict(model).items()}
    save_safetensors(tensors, path / "adapter_model.safetensors")
    config = {
        "base_model_name_or_path": base_model,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": r,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
        "use_rslora": use_rslora,
    }
    (path / "adapter_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def _chunk_sum_ce(piece: torch.Tensor, weight: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(F.linear(piece, weight).float(), target, reduction="sum")


def chunked_cross_entropy(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk: int = 1024,
    ignore_index: int = -100,
    recompute: bool = True,
) -> torch.Tensor:
    """Mean CE over non-ignored positions, WITHOUT ever holding the full
    `[tokens, vocab]` logit tensor.

    This is not an optimization. With `vocab = 248,320` the logits alone are
    0.497 MB/token; at 32 k context, with the fp32 upcast and its gradient, that
    is 48.8 GB and the card is 95.6 GiB total (`lora_lane_2026-08-13.md` §1.3).

    **`recompute` is the whole point, and leaving it off is a trap.** A plain
    Python loop over chunks does NOT save memory: autograd keeps every chunk's
    logits alive until backward runs, so peak memory is the same as the naive
    version plus loop overhead. Measured on the local 3080 at
    tokens=4096/vocab=32000, the non-recompute loop gave only a **2.3x**
    reduction -- against the ~20x the feasibility table assumes. Checkpointing
    each chunk drops the logits immediately and recomputes them in backward, at
    the cost of one extra `lm_head` matmul per chunk (a few percent of step
    time, against tens of GB).
    """
    hidden = hidden.reshape(-1, hidden.shape[-1])
    labels = labels.reshape(-1)
    keep = labels != ignore_index
    n_keep = int(keep.sum())
    if n_keep == 0:
        return hidden.sum() * 0.0
    hidden = hidden[keep]
    labels = labels[keep]

    total = hidden.new_zeros((), dtype=torch.float32)
    use_ckpt = recompute and torch.is_grad_enabled() and (
        hidden.requires_grad or lm_head_weight.requires_grad
    )
    for start in range(0, hidden.shape[0], chunk):
        piece = hidden[start : start + chunk]
        target = labels[start : start + chunk]
        if use_ckpt:
            total = total + _ckpt.checkpoint(
                _chunk_sum_ce, piece, lm_head_weight, target, use_reentrant=False
            )
        else:
            total = total + _chunk_sum_ce(piece, lm_head_weight, target)
    return total / n_keep


# ---------------------------------------------------------------------------
# FP8 -> BF16 streaming load (no accelerate, no host-RAM spike)
# ---------------------------------------------------------------------------
def dequant_fp8(weight: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    """compressed-tensors `float-quantized`, strategy `tensor`, symmetric:
    the stored value is the fp8 code and the scale is a single scalar."""
    out = weight.to(torch.bfloat16)
    if scale is not None:
        out = out * scale.to(torch.float32).reshape(()).to(torch.bfloat16)
    return out


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as handle:
        length = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return header, 8 + length


def stream_assign(
    model: nn.Module,
    path: Path,
    *,
    device: str = "cuda",
    loader=None,
) -> dict:
    """Assign parameters onto a meta-initialized model one tensor at a time.

    `accelerate` is absent from both wheelhouses, so `device_map=` and
    `low_cpu_mem_usage=` are unavailable and a CPU-then-`.cuda()` load of a
    54.7 GB model is not on the table. This walks the checkpoint instead,
    dequantizing FP8 inline and writing straight to the GPU, so host RAM never
    holds more than one tensor.

    `loader(name) -> Tensor | None` supplies raw tensors; the default reads them
    out of the safetensors file. Returns a report.
    """
    header, _ = read_safetensors_header(path)
    if loader is None:
        raise ValueError("stream_assign needs a loader(name) -> Tensor")

    params = dict(model.named_parameters())
    params.update(dict(model.named_buffers()))
    assigned, dequantized, missing = 0, 0, []
    for name, param in params.items():
        raw = loader(name)
        if raw is None:
            missing.append(name)
            continue
        scale = loader(name + "_scale")
        value = dequant_fp8(raw, scale) if raw.dtype in (torch.float8_e4m3fn,) else raw
        if scale is not None or raw.dtype in (torch.float8_e4m3fn,):
            dequantized += 1
        value = value.to(device=device, dtype=param.dtype if param.dtype.is_floating_point else value.dtype)
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        leaf = parts[-1]
        if isinstance(getattr(parent, leaf, None), nn.Parameter):
            setattr(parent, leaf, nn.Parameter(value, requires_grad=False))
        else:
            parent.register_buffer(leaf, value)
        assigned += 1
    return {"in_checkpoint": len(header), "assigned": assigned,
            "dequantized": dequantized, "missing": missing}


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
