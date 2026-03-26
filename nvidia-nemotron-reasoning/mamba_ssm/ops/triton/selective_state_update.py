"""Stub — not used when is_fast_path_available=False."""

def selective_state_update(*args, **kwargs):
    raise NotImplementedError("Use pure PyTorch path (is_fast_path_available=False)")
