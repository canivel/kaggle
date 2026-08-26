"""ExecWMHook — pluggable hook for v64-class agents that uses Rodionov-style
per-game executable simulators to plan actions on BFS-failed levels.

Each game has a Python module exec_wm/sims/<game_type>_sim.py that exports:
  simulate(state, action_id, x, y) -> (next_state, reward_class:int, done:bool)

The hook:
- At init: scans known paths for sim files, builds a registry by game_type.
- At pick_action: does N-step beam search through the executable sim, scoring
  rollouts by predicted reward (level-up > frame-change > no-change), picks
  the action with the best score.

Design constraints (mirrors JEPAHook from jepa_wm.inference.agent_hooks):
- Silent no-op if no sim is registered for the current game_type
- Never crashes the agent
- Inference is single-process Python — no neural network, no GPU
"""
from __future__ import annotations

import glob
import importlib.util
import logging
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Candidate paths where sim files may live (local dev + Kaggle dataset)
SIM_PATH_PATTERNS = [
    "/kaggle/input/exec-wm-sims/*_sim.py",
    "/kaggle/input/datasets/canivel/exec-wm-sims/*_sim.py",
    "/kaggle/working/exec_wm/sims/*_sim.py",
    "exec_wm/sims/*_sim.py",
    "/kaggle/input/**/exec_wm/sims/*_sim.py",
]


def _discover_sim_files() -> list[str]:
    seen = set()
    out = []
    for pat in SIM_PATH_PATTERNS:
        for p in glob.glob(pat, recursive=True):
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def _load_sim(path: str) -> Optional[Callable]:
    try:
        name = Path(path).stem
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not callable(getattr(mod, "simulate", None)):
            return None
        return mod.simulate
    except Exception as e:
        logger.warning(f"ExecWMHook: failed to load {path}: {e}")
        return None


# Click candidates probed by the planner on action 6
CLICK_CANDIDATES = [
    (32, 32), (16, 16), (48, 48), (16, 48), (48, 16),
    (32, 16), (32, 48), (16, 32), (48, 32),
    (24, 24), (24, 40), (40, 24), (40, 40),
]


class ExecWMHook:
    def __init__(self, beam_width: int = 4, lookahead: int = 3):
        self.registry: dict[str, Callable] = {}
        self.beam_width = beam_width
        self.lookahead = lookahead
        for p in _discover_sim_files():
            game_type = Path(p).stem.removesuffix("_sim")
            sim = _load_sim(p)
            if sim is not None:
                self.registry[game_type] = sim
        self.available = bool(self.registry)
        if self.available:
            logger.info(f"ExecWMHook: loaded {len(self.registry)} sims: {sorted(self.registry)}")

    def has_sim(self, game_type: str) -> bool:
        return game_type in self.registry

    def pick_action(
        self,
        frame: np.ndarray,
        available_actions: list[int],
        game_type: str,
    ) -> Optional[tuple[int, int, int]]:
        """Beam search through executable sim. Returns (action_id, x, y) or None.

        Scoring per rollout step:
          reward_class==2 (level-up) -> +100  and TERMINATES that rollout
          reward_class==1 (frame change) -> +1
          reward_class==0 (no change) -> 0
        Lookahead totals are summed with a 0.95 discount.
        """
        sim = self.registry.get(game_type)
        if sim is None:
            return None
        try:
            grid = np.asarray(frame, dtype=np.uint8)
            if grid.shape != (64, 64):
                return None

            # Build root action candidates: every available action_id, with
            # action 6 expanded over CLICK_CANDIDATES.
            candidates: list[tuple[int, int, int]] = []
            for aid in available_actions:
                if aid == 6:
                    for cx, cy in CLICK_CANDIDATES:
                        candidates.append((6, int(cx), int(cy)))
                elif 1 <= aid <= 7:
                    candidates.append((int(aid), 0, 0))
            if not candidates:
                return None

            best_action = None
            best_score = -float("inf")
            for root in candidates:
                aid, ax, ay = root
                try:
                    ns, rc, dn = sim(grid.tolist(), aid, ax, ay)
                except Exception:
                    continue
                ns_arr = np.asarray(ns, dtype=np.uint8)
                if ns_arr.shape != (64, 64):
                    continue
                base = 100.0 if rc == 2 else (1.0 if rc == 1 else 0.0)
                if dn or rc == 2 or self.lookahead <= 1:
                    score = base
                else:
                    # 1-step lookahead: from the new state, find the BEST follow-up
                    # over the same action set (cheap proxy for short beam search).
                    discount = 0.95
                    best_followup = 0.0
                    for follow_aid in available_actions:
                        if follow_aid == 6:
                            for fcx, fcy in CLICK_CANDIDATES[:4]:
                                try:
                                    _, frc, _ = sim(ns_arr.tolist(), 6, fcx, fcy)
                                except Exception:
                                    continue
                                fscore = 100.0 if frc == 2 else (1.0 if frc == 1 else 0.0)
                                if fscore > best_followup:
                                    best_followup = fscore
                        elif 1 <= follow_aid <= 7:
                            try:
                                _, frc, _ = sim(ns_arr.tolist(), int(follow_aid), 0, 0)
                            except Exception:
                                continue
                            fscore = 100.0 if frc == 2 else (1.0 if frc == 1 else 0.0)
                            if fscore > best_followup:
                                best_followup = fscore
                    score = base + discount * best_followup

                if score > best_score:
                    best_score = score
                    best_action = root

            if best_action is None or best_score <= 0:
                return None
            return best_action
        except Exception as e:
            logger.warning(f"ExecWMHook pick_action failed: {e}")
            return None


# ============================================================================
# Smoke test
# ============================================================================
if __name__ == "__main__":
    hook = ExecWMHook()
    print(f"Available: {hook.available}")
    print(f"Registered: {sorted(hook.registry)}")
    if hook.has_sim("bp35"):
        frame = np.random.randint(0, 16, (64, 64), dtype=np.uint8)
        pick = hook.pick_action(frame, [1, 2, 3, 4, 5, 6, 7], "bp35")
        print(f"bp35 pick: {pick}")
