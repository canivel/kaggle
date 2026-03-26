"""Stub — not used when is_fast_path_available=False."""

def mamba_chunk_scan_combined(*args, **kwargs):
    raise NotImplementedError("Use pure PyTorch path (is_fast_path_available=False)")

def mamba_split_conv1d_scan_combined(*args, **kwargs):
    raise NotImplementedError("Use pure PyTorch path (is_fast_path_available=False)")
