"""Validate a generated executable world model against recorded observations.

Loads exec_wm/sims/<game_id>_sim.py and runs its `simulate(state, action_id, x, y)`
against the observations in exec_wm/observations/<game_id>.json.

WARNING (added 2026-08-10, R24 minutes §5.2 vi): the DEFAULT `--split all` is
**in-sample** — it scores every tuple, including the ones the authoring model
read while writing the sim. `exec_wm/scale_summary.md` was produced at
`--split all` and its numbers must never be described as "held out". Use
`--split test` for an out-of-sample read, and note that even that is only
honest if the sim was authored on `--split train` alone.

The simulator must implement:
  simulate(state: list[list[int]] | np.ndarray, action_id: int, x: int, y: int)
    -> (next_state: list/ndarray, reward_class: int, done: bool)

Reports:
- next-state exact-match %
- next-state mean per-pixel match %
- reward_class accuracy
- done accuracy
- breakdown by action_id

Usage:
  uv run python exec_wm/validate_sim.py --game bp35
  uv run python exec_wm/validate_sim.py --game bp35 --split test --train-frac 0.7
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_sim(game_id: str):
    p = ROOT / "exec_wm" / "sims" / f"{game_id}_sim.py"
    if not p.exists():
        raise FileNotFoundError(p)
    spec = importlib.util.spec_from_file_location(f"{game_id}_sim", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "simulate"):
        raise AttributeError(f"{p} must define simulate(state, action_id, x, y)")
    return mod.simulate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--split", default="all", choices=["all", "train", "test"])
    ap.add_argument("--train-frac", type=float, default=0.7)
    args = ap.parse_args()

    obs = json.loads((ROOT / "exec_wm" / "observations" / f"{args.game}.json").read_text())
    tuples = obs["tuples"]
    if args.split != "all":
        split_at = int(len(tuples) * args.train_frac)
        tuples = tuples[:split_at] if args.split == "train" else tuples[split_at:]

    sim = load_sim(args.game)

    by_action = {}
    n_total = 0
    n_state_exact = 0
    pixel_match_sum = 0.0
    n_reward = 0
    n_done = 0
    n_errors = 0

    for t in tuples:
        n_total += 1
        s_t = np.asarray(t["state_t"], dtype=np.uint8)
        s_t1_truth = np.asarray(t["state_t1"], dtype=np.uint8)
        try:
            pred = sim(s_t.tolist(), int(t["action_id"]), int(t["x"]), int(t["y"]))
        except Exception as e:
            n_errors += 1
            by_action.setdefault(t["action_id"], {"n": 0, "exact": 0, "errors": 0})
            by_action[t["action_id"]]["n"] += 1
            by_action[t["action_id"]]["errors"] += 1
            continue
        try:
            ns, rc, dn = pred
            ns_arr = np.asarray(ns, dtype=np.uint8)
            if ns_arr.shape != (64, 64):
                raise ValueError(f"sim returned shape {ns_arr.shape}")
        except Exception:
            n_errors += 1
            continue

        exact = bool(np.array_equal(ns_arr, s_t1_truth))
        pm = float((ns_arr == s_t1_truth).mean())
        if exact:
            n_state_exact += 1
        pixel_match_sum += pm
        if int(rc) == int(t["reward_class"]):
            n_reward += 1
        if bool(dn) == bool(t["done"]):
            n_done += 1

        by_action.setdefault(t["action_id"], {"n": 0, "exact": 0, "errors": 0, "pm": 0.0})
        by_action[t["action_id"]]["n"] += 1
        by_action[t["action_id"]]["exact"] += 1 if exact else 0
        by_action[t["action_id"]]["pm"] = by_action[t["action_id"]].get("pm", 0.0) + pm

    out = {
        "game": args.game,
        "split": args.split,
        "n": n_total,
        "errors": n_errors,
        "state_exact_pct": (n_state_exact / max(1, n_total - n_errors)) * 100,
        "pixel_match_pct": (pixel_match_sum / max(1, n_total - n_errors)) * 100,
        "reward_acc_pct": (n_reward / max(1, n_total - n_errors)) * 100,
        "done_acc_pct": (n_done / max(1, n_total - n_errors)) * 100,
        "by_action": {
            int(k): {
                "n": v["n"],
                "exact_pct": (v["exact"] / max(1, v["n"])) * 100,
                "pixel_pct": (v.get("pm", 0.0) / max(1, v["n"])) * 100,
                "errors": v["errors"],
            }
            for k, v in by_action.items()
        },
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
