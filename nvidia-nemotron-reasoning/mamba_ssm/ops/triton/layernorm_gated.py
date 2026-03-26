"""Pure PyTorch replacement for mamba_ssm Triton layernorm kernels."""
import torch
import torch.nn.functional as F


def rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-6, group_size=None, norm_before_gate=True):
    x_shape = x.shape
    if group_size is not None:
        x = x.view(-1, x.shape[-1] // group_size, group_size)
    x_float = x.float()
    variance = x_float.pow(2).mean(-1, keepdim=True)
    x_normed = (x_float * torch.rsqrt(variance + eps)).to(x.dtype)
    if group_size is not None:
        x_normed = x_normed.view(x_shape)
    if weight is not None:
        x_normed = x_normed * weight
    if bias is not None:
        x_normed = x_normed + bias
    if z is not None:
        if norm_before_gate:
            x_normed = x_normed * F.silu(z)
    return x_normed


def layernorm_fn(x, weight, bias=None, z=None, eps=1e-6, group_size=None, norm_before_gate=True):
    x_shape = x.shape
    if group_size is not None:
        x = x.view(-1, x.shape[-1] // group_size, group_size)
    x_float = x.float()
    mean = x_float.mean(-1, keepdim=True)
    variance = (x_float - mean).pow(2).mean(-1, keepdim=True)
    x_normed = ((x_float - mean) * torch.rsqrt(variance + eps)).to(x.dtype)
    if group_size is not None:
        x_normed = x_normed.view(x_shape)
    if weight is not None:
        x_normed = x_normed * weight
    if bias is not None:
        x_normed = x_normed + bias
    if z is not None:
        if norm_before_gate:
            x_normed = x_normed * F.silu(z)
    return x_normed
