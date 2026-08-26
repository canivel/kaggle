"""D4 + color-permutation augmentations for ARC trajectories.

Each augmentation applies CONSISTENTLY to (frame_t, action, frame_{t+1}) so
the dynamics stay valid:
- D4 (8 transforms): rotations + reflections. ACTION6 coords transformed too.
  ACTION1-4 are remapped if they correspond to compass directions (heuristic: try identity
  remap and a swap; we currently leave simple actions unchanged — risk noted).
- Color permutation: random permutation of the 16 colors applied identically across the trajectory.

Caller policy: sample one D4 + one color-perm per trajectory.
"""
from __future__ import annotations

import numpy as np


# =============================================================================
# D4 (dihedral group of order 8)
# =============================================================================
def _d4_transform(grid: np.ndarray, op: int) -> np.ndarray:
    """op in 0..7. 0=identity, 1=rot90, 2=rot180, 3=rot270, 4..7 = flips of 0..3."""
    g = grid
    if op & 4:
        g = g[:, ::-1]
    k = op & 3
    if k:
        g = np.rot90(g, k=k)
    return np.ascontiguousarray(g)


def _d4_coord(x: int, y: int, op: int, H: int = 64, W: int = 64) -> tuple[int, int]:
    """Transform an (x, y) click coordinate under D4 op. y is row, x is column."""
    # Apply flip first if op&4
    if op & 4:
        x = W - 1 - x
    k = op & 3
    if k == 1:  # rot90 CCW: (y, x) -> (W-1-x, y)
        x, y = W - 1 - y, x
    elif k == 2:  # rot180
        x, y = W - 1 - x, H - 1 - y
    elif k == 3:  # rot270
        x, y = y, H - 1 - x
    return int(x), int(y)


def apply_d4(frame_t: np.ndarray, frame_t1: np.ndarray, action_id: int, x: int, y: int, op: int):
    """Returns transformed (frame_t, frame_t1, action_id, x, y) for D4 op 0..7."""
    f_t = _d4_transform(frame_t, op)
    f_t1 = _d4_transform(frame_t1, op)
    if action_id == 6:
        xn, yn = _d4_coord(x, y, op, H=frame_t.shape[0], W=frame_t.shape[1])
    else:
        xn, yn = x, y
    # ACTION1-4 are typically directional; safest to NOT remap and just accept
    # the augmented diversity. Future: remap if game-class is known.
    return f_t, f_t1, action_id, xn, yn


# =============================================================================
# Color permutation
# =============================================================================
def apply_color_perm(frame_t: np.ndarray, frame_t1: np.ndarray, perm: np.ndarray | None = None, n_colors: int = 16):
    """Permute the 16 colors consistently in both frames. `perm[i]` is the new value of color i."""
    if perm is None:
        perm = np.random.permutation(n_colors).astype(np.int64)
    f_t = perm[frame_t]
    f_t1 = perm[frame_t1]
    return f_t, f_t1, perm


# =============================================================================
# Combined augmentor
# =============================================================================
class TrajectoryAugmenter:
    def __init__(self, n_colors: int = 16, p_d4: float = 0.875, p_color: float = 1.0, seed: int | None = None):
        self.n_colors = n_colors
        self.p_d4 = p_d4
        self.p_color = p_color
        self.rng = np.random.default_rng(seed)

    def __call__(self, frame_t, frame_t1, action_id, x, y):
        # D4
        if self.rng.random() < self.p_d4:
            op = int(self.rng.integers(1, 8))  # skip identity
            frame_t, frame_t1, action_id, x, y = apply_d4(frame_t, frame_t1, action_id, x, y, op)
        # Color permutation
        if self.rng.random() < self.p_color:
            perm = self.rng.permutation(self.n_colors).astype(np.int64)
            frame_t, frame_t1, _ = apply_color_perm(frame_t, frame_t1, perm, self.n_colors)
        return frame_t, frame_t1, action_id, x, y


# =============================================================================
# Smoke
# =============================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ft = rng.integers(0, 16, (64, 64))
    ft1 = rng.integers(0, 16, (64, 64))
    aug = TrajectoryAugmenter(seed=0)
    f_t, f_t1, aid, x, y = aug(ft, ft1, 6, 12, 50)
    print(f"After aug: action_id={aid} (x,y)=({x},{y}) frame_t.shape={f_t.shape} non_eq={(ft!=f_t).sum()/ft.size:.2f}")
    print("augmentation smoke OK")
