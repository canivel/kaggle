"""Hook for plugging JEPA-MCTS into the v39-class agent.

The host agent calls `jepa_pick_action(s, lf, lvl)` after BFS has failed
the level (s._bfs_solution is None). If the JEPA-WM is initialized with
pretrained weights, MCTS returns an action; otherwise None.

Design constraints:
  - Pretrained weights loaded once at agent init from a Kaggle dataset path
    (e.g. /kaggle/input/jepa-wm-weights/jepa_wm_final.pt).
  - Must run on the agent's chosen device (CUDA when available).
  - Must NEVER hijack a BFS-solvable level (gated on s._bfs_solution is None).
  - Must be robust: any exception falls through silently.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    from jepa_wm.models.jepa import JEPAConfig, JEPAWorldModel
    from jepa_wm.inference.mcts import LatentMCTS, MCTSConfig, ActionSpec
    _JEPA_AVAILABLE = True
except Exception as _e:
    _JEPA_AVAILABLE = False
    _IMPORT_ERR = repr(_e)


# Candidate Kaggle paths where weights might be shipped
WEIGHT_CANDIDATES = [
    "/kaggle/input/jepa-wm-weights/jepa_wm_final.pt",
    "/kaggle/input/jepa-wm-weights/jepa_wm.pt",
    "jepa_wm/checkpoints/jepa_wm_final.pt",
    "jepa_wm_final.pt",
]


class JEPAHook:
    def __init__(self, device: Optional[str] = None, n_simulations: int = 32, max_depth: int = 8):
        self.available = False
        self.wm = None
        self.mcts = None
        if not _JEPA_AVAILABLE:
            return
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weight_path = None
        for cand in WEIGHT_CANDIDATES:
            if os.path.exists(cand):
                weight_path = cand
                break
        if weight_path is None:
            return  # No weights -> hook disabled silently
        try:
            ckpt = torch.load(weight_path, map_location=device, weights_only=False)
            cfg_dict = ckpt.get("cfg") or {}
            cfg = JEPAConfig(**{k: v for k, v in cfg_dict.items() if hasattr(JEPAConfig, k)})
            self.wm = JEPAWorldModel(cfg).to(device).eval()
            self.wm.load_state_dict(ckpt["wm"])
            self.mcts = LatentMCTS(self.wm, MCTSConfig(n_simulations=n_simulations, max_depth=max_depth), device=device)
            self.device = device
            self.available = True
        except Exception:
            self.available = False

    def pick_action(self, frame: np.ndarray, available_actions: list[int], click_candidates: list[tuple[int, int]] | None = None) -> Optional[tuple[int, int, int]]:
        """frame: (64,64) uint8/int. available_actions: list of action_ids in [1..7].
        click_candidates: optional list of (x, y) coord pairs for ACTION6.
        Returns (action_id, x, y) or None.
        """
        if not self.available:
            return None
        try:
            grid = torch.from_numpy(np.asarray(frame, dtype=np.int64))
            if grid.shape != (64, 64):
                return None
            grid = grid.unsqueeze(0).to(self.device)
            with torch.no_grad():
                z = self.wm.encoder(grid)
            # Map raw ACTION1-7 -> 0-indexed for embedding lookup.
            actions: list[ActionSpec] = []
            for aid in available_actions:
                if aid == 6 and click_candidates:
                    for (cx, cy) in click_candidates[:8]:
                        actions.append(ActionSpec(action_id=5, x=int(cx) % 64, y=int(cy) % 64))  # ACTION6 -> idx 5
                elif 1 <= aid <= 5:
                    actions.append(ActionSpec(action_id=int(aid) - 1))  # ACTION1->0..ACTION5->4
                elif aid == 7:
                    actions.append(ActionSpec(action_id=6))  # ACTION7 -> idx 6
            if not actions:
                return None
            best_idx, _stats = self.mcts.search(z, actions, add_root_dirichlet=False)
            a = actions[best_idx]
            # Map 0-indexed internal id back to raw ARC action_id (1..7).
            if a.action_id == 5:
                return (6, a.x, a.y)  # ACTION6 click
            if a.action_id == 6:
                return (7, 0, 0)  # ACTION7 undo
            return (a.action_id + 1, 0, 0)  # ACTION1..ACTION5
        except Exception:
            return None


# Smoke
if __name__ == "__main__":
    hook = JEPAHook()
    print("JEPA hook available:", hook.available)
    if hook.available:
        import numpy as np
        frame = np.random.randint(0, 16, (64, 64), dtype=np.int64)
        out = hook.pick_action(frame, [1, 2, 3, 4, 6], click_candidates=[(32, 32), (16, 48)])
        print("Pick:", out)
