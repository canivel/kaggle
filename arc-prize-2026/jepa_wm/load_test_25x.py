"""Simulate the Kaggle competition rerun: 25 parallel CPU agents each loading
the JEPA hook and calling pick_action a few times. Measures total RAM,
per-worker peak RAM, time-to-load, time-per-pick_action, and counts failures.

Goal: figure out a JEPA budget (n_simulations, max_depth, and whether to load
the full 25M-param model at all) that fits inside Kaggle's parallel rerun env.

Kaggle competition rerun: ALL 25 ARC-AGI-3 public games run concurrently on
CPU; total memory budget effectively ~13GB per the standard env. If we want
JEPA to be safe, sum of (per-worker peak RAM) must stay well under that.

Usage:
  uv run python -m jepa_wm.load_test_25x --workers 25 --n-sims 32 --max-depth 8
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def worker(idx: int, n_sims: int, max_depth: int, n_picks: int, out_q) -> None:
    """Run a single 'agent': load JEPA hook, do a few picks, report metrics."""
    rec = {"idx": idx, "ok": False, "err": None}
    try:
        import psutil  # type: ignore
        import torch
        torch.set_num_threads(1)  # mirror Kaggle parallel-on-CPU constraint
        proc = psutil.Process()
        t0 = time.time()
        from jepa_wm.inference.agent_hooks import JEPAHook
        import jepa_wm.inference.agent_hooks as ah
        ah.WEIGHT_CANDIDATES = [str(PROJECT / "jepa_wm" / "checkpoints" / "jepa_wm_final.pt")] + ah.WEIGHT_CANDIDATES
        hook = JEPAHook(n_simulations=n_sims, max_depth=max_depth)
        rec["load_s"] = time.time() - t0
        rec["available"] = hook.available
        rec["mem_after_load_mb"] = proc.memory_info().rss / 1024 / 1024
        if not hook.available:
            rec["err"] = "hook unavailable (weights?)"
            out_q.put(rec)
            return
        # Warmup pick + measured picks
        rng = np.random.default_rng(idx + 1)
        frame = rng.integers(0, 16, (64, 64), dtype=np.int64)
        hook.pick_action(frame, [1, 2, 3, 4, 6], click_candidates=[(32, 32)])  # warmup
        pick_times = []
        for k in range(n_picks):
            frame = rng.integers(0, 16, (64, 64), dtype=np.int64)
            t1 = time.time()
            out = hook.pick_action(frame, [1, 2, 3, 4, 5, 6], click_candidates=[(rng.integers(0, 64), rng.integers(0, 64)) for _ in range(4)])
            pick_times.append(time.time() - t1)
            if out is None:
                rec["err"] = f"pick_action returned None on iter {k}"
                break
        rec["pick_times"] = pick_times
        rec["mem_peak_mb"] = proc.memory_info().rss / 1024 / 1024
        rec["ok"] = rec["err"] is None
    except Exception as e:
        rec["err"] = repr(e)[:300]
    out_q.put(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=25)
    ap.add_argument("--n-sims", type=int, default=32)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--n-picks", type=int, default=3)
    args = ap.parse_args()

    print(f"=== 25x JEPA hook load test ===")
    print(f"workers={args.workers}  n_sims={args.n_sims}  max_depth={args.max_depth}  n_picks={args.n_picks}")

    mp.set_start_method("spawn", force=True)
    q = mp.Queue()
    t0 = time.time()
    procs = []
    for i in range(args.workers):
        p = mp.Process(target=worker, args=(i, args.n_sims, args.max_depth, args.n_picks, q))
        p.start()
        procs.append(p)
    results = []
    for _ in procs:
        results.append(q.get(timeout=600))
    for p in procs:
        p.join(timeout=10)
    wall = time.time() - t0

    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    print(f"\n=== RESULTS ===")
    print(f"wall={wall:.1f}s  ok={len(ok)}/{len(results)}  failed={len(bad)}")
    if ok:
        loads = [r["load_s"] for r in ok]
        mems_load = [r["mem_after_load_mb"] for r in ok]
        mems_peak = [r["mem_peak_mb"] for r in ok]
        picks = [pt for r in ok for pt in r["pick_times"]]
        print(f"  load_s: min={min(loads):.2f} max={max(loads):.2f} mean={sum(loads)/len(loads):.2f}")
        print(f"  mem_after_load_mb: min={min(mems_load):.0f} max={max(mems_load):.0f} mean={sum(mems_load)/len(mems_load):.0f}  TOTAL={sum(mems_load):.0f}")
        print(f"  mem_peak_mb:        min={min(mems_peak):.0f} max={max(mems_peak):.0f} mean={sum(mems_peak)/len(mems_peak):.0f}  TOTAL={sum(mems_peak):.0f}")
        if picks:
            print(f"  pick_s: min={min(picks):.3f} max={max(picks):.3f} mean={sum(picks)/len(picks):.3f}")
    if bad:
        print(f"\n=== FAILURES ===")
        for r in bad[:5]:
            print(f"  worker {r['idx']}: {r.get('err')!r}")


if __name__ == "__main__":
    main()
