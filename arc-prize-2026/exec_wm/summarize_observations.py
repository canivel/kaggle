"""Compress per-game observations into a digestible summary for an LLM.

Reads exec_wm/observations/<game_id>.json and produces:
  exec_wm/observations/<game_id>.summary.json

The summary keeps:
- Distribution: action_id -> count, reward_class -> count
- 6 fully-rendered exemplar tuples (sampled to cover action diversity)
- For ALL tuples: a compact diff representation
    (action_id, x, y, reward_class, done, level, n_changed_pixels,
     sample_changes=[(i,j,old,new), ...up to 12])
- Grid shape + value range

Avoids dumping 200x64x64 raw arrays into the LLM context.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def compact_diff(s_t, s_t1, max_samples: int = 12):
    a = np.asarray(s_t, dtype=np.int16)
    b = np.asarray(s_t1, dtype=np.int16)
    mask = a != b
    coords = np.argwhere(mask)
    n_changed = int(coords.shape[0])
    sample = []
    if n_changed > 0:
        idx = np.linspace(0, n_changed - 1, num=min(max_samples, n_changed)).astype(int)
        for k in idx:
            i, j = coords[k]
            sample.append([int(i), int(j), int(a[i, j]), int(b[i, j])])
    return n_changed, sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    args = ap.parse_args()
    obs = json.loads((ROOT / "exec_wm" / "observations" / f"{args.game}.json").read_text())
    tuples = obs["tuples"]

    by_action = Counter(t["action_id"] for t in tuples)
    by_reward = Counter(t["reward_class"] for t in tuples)

    # Diff every tuple compactly
    compact = []
    arrs_t = [np.asarray(t["state_t"], dtype=np.uint8) for t in tuples]
    arrs_t1 = [np.asarray(t["state_t1"], dtype=np.uint8) for t in tuples]
    for t, a, b in zip(tuples, arrs_t, arrs_t1):
        n, sample = compact_diff(a, b)
        compact.append({
            "step": t["step"],
            "action_id": t["action_id"],
            "x": t["x"],
            "y": t["y"],
            "reward_class": t["reward_class"],
            "done": t["done"],
            "level": t["level"],
            "n_changed": n,
            "sample_changes": sample,
        })

    # Pick exemplars: try to cover each action_id at least once
    seen_actions = set()
    exemplar_idxs = []
    for i, t in enumerate(tuples):
        if t["action_id"] not in seen_actions:
            seen_actions.add(t["action_id"])
            exemplar_idxs.append(i)
        if len(exemplar_idxs) >= 6:
            break
    # Pad with high-change examples
    if len(exemplar_idxs) < 6:
        rest = sorted(
            [i for i in range(len(tuples)) if i not in exemplar_idxs],
            key=lambda i: -compact[i]["n_changed"],
        )
        for i in rest:
            exemplar_idxs.append(i)
            if len(exemplar_idxs) >= 6:
                break

    exemplars = [
        {
            "step": tuples[i]["step"],
            "action_id": tuples[i]["action_id"],
            "x": tuples[i]["x"],
            "y": tuples[i]["y"],
            "reward_class": tuples[i]["reward_class"],
            "done": tuples[i]["done"],
            "level": tuples[i]["level"],
            "state_t": tuples[i]["state_t"],
            "state_t1": tuples[i]["state_t1"],
        }
        for i in exemplar_idxs
    ]

    val_range = (int(min(a.min() for a in arrs_t + arrs_t1)),
                 int(max(a.max() for a in arrs_t + arrs_t1)))

    summary = {
        "game_id": obs["game_id"],
        "available_actions": obs["available_actions"],
        "grid_shape": [64, 64],
        "value_range": val_range,
        "n_tuples": len(tuples),
        "action_distribution": dict(by_action),
        "reward_distribution": dict(by_reward),
        "exemplars": exemplars,
        "compact_tuples": compact,
    }

    out = ROOT / "exec_wm" / "observations" / f"{args.game}.summary.json"
    out.write_text(json.dumps(summary, indent=1))
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"{args.game}: summary {size_mb:.2f}MB -> {out}")
    print(f"  actions: {dict(by_action)}")
    print(f"  rewards: {dict(by_reward)}")
    print(f"  exemplars: {len(exemplars)}, compact_tuples: {len(compact)}")


if __name__ == "__main__":
    main()
