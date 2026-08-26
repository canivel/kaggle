"""Test all seed harnesses from the meta-harness benchmark.
Quick local validation before running the full evolution loop.
Superseded by: uv run python arc_metaharness.py --mode eval
"""

import sys
sys.path.insert(0, "f:/kaggle/kaos")

# Register the benchmark so imports work
from kaos.metaharness.benchmarks import arc_agi3 as arc_mod  # noqa: F401 registers arc-agi-3

import json, time
from pathlib import Path
import numpy as np

ArcAGI3Benchmark = arc_mod.ArcAGI3Benchmark
SEED_RANDOM = arc_mod.SEED_RANDOM
SEED_SYSTEMATIC = arc_mod.SEED_SYSTEMATIC
SEED_PRODUCTIVE_FIRST = arc_mod.SEED_PRODUCTIVE_FIRST
SEED_CLICK_OBJECTS = arc_mod.SEED_CLICK_OBJECTS

SEEDS = {
    "random": SEED_RANDOM,
    "systematic": SEED_SYSTEMATIC,
    "productive": SEED_PRODUCTIVE_FIRST,
    "click_obj": SEED_CLICK_OBJECTS,
}

def main():
    print("=" * 70)
    print("Seed Harness Evaluation on ARC-AGI-3")
    print("=" * 70)

    bench = ArcAGI3Benchmark(time_per_game=60, max_actions=3000, n_search_games=10)
    problems = bench.get_search_set()
    print(f"\n{len(problems)} search games loaded\n")

    all_results = {}

    for name, code in SEEDS.items():
        print(f"\n--- Seed: {name} ---")
        results = bench.evaluate_harness(code, problems)
        total_levels = sum(r.get("levels", 0) for r in results)
        total_actions = sum(r.get("actions", 0) for r in results)
        mean_rhae = np.mean([r.get("rhae", 0) for r in results])

        print(f"  Levels: {total_levels}, Actions: {total_actions}, RHAE: {mean_rhae:.6f}")
        for r, p in zip(results, problems):
            if r.get("levels", 0) > 0:
                print(f"    {p.input['title']:5s}: L{r['levels']} RHAE={r['rhae']:.4f} acts={r['actions']}")

        all_results[name] = {
            "total_levels": total_levels,
            "mean_rhae": mean_rhae,
            "per_game": results,
        }

    # Summary
    print(f"\n{'='*70}")
    print(f"{'Seed':15s} {'Levels':>8s} {'RHAE':>10s}")
    print(f"{'-'*35}")
    for name, res in sorted(all_results.items(), key=lambda x: -x[1]["mean_rhae"]):
        print(f"{name:15s} {res['total_levels']:8d} {res['mean_rhae']:10.6f}")

    Path("data").mkdir(exist_ok=True)
    with open("data/seed_test_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to data/seed_test_results.json")


if __name__ == "__main__":
    main()
