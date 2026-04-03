"""ARC-AGI-3 Meta-Harness Runner.

Uses the KAOS MetaHarnessSearch to automatically evolve agent strategies
via an LLM proposer that reads execution traces and proposes improvements.

Two modes:
  --mode eval   Direct evaluation of seeds (no LLM, no vLLM needed)
  --mode search Full KAOS metaharness with LLM proposer (needs vLLM or kaos.yaml)

Usage:
    # Evaluate all seeds locally (fast, no LLM):
    uv run python arc_metaharness.py --mode eval --games 5 --time 30

    # Full LLM-driven search (needs vLLM running per kaos.yaml):
    uv run python arc_metaharness.py --mode search --games 8 --time 60 --iters 10

    # Export best harness to Kaggle submission:
    uv run python arc_metaharness.py --mode eval --export submission_best.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Register the benchmark
sys.path.insert(0, str(Path(__file__).parent.parent / "kaos"))

from kaos.metaharness.benchmarks import arc_agi3  # noqa: F401 — registers arc-agi-3
from kaos.metaharness.benchmarks import get_benchmark
from kaos.metaharness.benchmarks.arc_agi3 import ArcAGI3Benchmark


# ─── Direct Evaluation (no LLM) ──────────────────────────────────────

def run_eval(
    n_games: int,
    time_per_game: int,
    export_path: str | None,
    extra_harness: str | None,
) -> None:
    """Evaluate all seed harnesses + any extra harness, pick best, optionally export."""
    bench = ArcAGI3Benchmark(
        time_per_game=time_per_game,
        n_search_games=n_games,
        n_test_games=max(n_games, 10),
    )
    problems = bench.get_search_set()
    print(f"\n=== ARC-AGI-3 Direct Evaluation ===")
    print(f"Games: {len(problems)} | Time per game: {time_per_game}s")
    print(f"Games: {[p.problem_id for p in problems]}\n")

    harnesses = {
        "random": bench.get_seed_harnesses()[0],
        "systematic": bench.get_seed_harnesses()[1],
        "productive_first": bench.get_seed_harnesses()[2],
        "click_objects": bench.get_seed_harnesses()[3],
    }
    if extra_harness:
        code = Path(extra_harness).read_text()
        harnesses["extra"] = code

    results: dict[str, dict] = {}
    best_name, best_rhae = "", 0.0

    for name, code in harnesses.items():
        t0 = time.time()
        per_game = bench.evaluate_harness(code, problems)
        elapsed = time.time() - t0

        avg_rhae = sum(r["rhae"] for r in per_game) / len(per_game)
        avg_levels = sum(r["levels"] for r in per_game) / len(per_game)
        avg_actions = sum(r["actions"] for r in per_game) / len(per_game)

        results[name] = {
            "avg_rhae": avg_rhae,
            "avg_levels": avg_levels,
            "avg_actions": avg_actions,
            "per_game": per_game,
            "elapsed": elapsed,
        }

        print(f"{name:20s}  RHAE={avg_rhae:.4f}  levels={avg_levels:.1f}  "
              f"actions={avg_actions:.0f}  ({elapsed:.0f}s)")

        if avg_rhae > best_rhae:
            best_rhae = avg_rhae
            best_name = name

    print(f"\nBest: {best_name} (RHAE={best_rhae:.4f})")

    # Save results
    out_path = Path("experiments") / f"metaharness_eval_{int(time.time())}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({
        "timestamp": int(time.time()),
        "n_games": n_games,
        "time_per_game": time_per_game,
        "best": best_name,
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "per_game"}
                    for k, v in results.items()},
    }, indent=2))
    print(f"Results saved: {out_path}")

    if export_path:
        best_code = harnesses[best_name]
        _export_kaggle(best_code, export_path, best_name, best_rhae)


def _export_kaggle(harness_code: str, path: str, name: str, rhae: float) -> None:
    """Wrap a KAOS harness into a minimal Kaggle Agent class."""
    # Extract choose_action from harness code
    lines = harness_code.split("\n")
    ca_start = next((i for i, l in enumerate(lines) if l.startswith("def choose_action")), None)

    if ca_start is None:
        print(f"WARNING: No choose_action found in {name}, exporting raw harness.")
        Path(path).write_text(harness_code)
        return

    choose_action_src = "\n".join(lines[ca_start:])

    kaggle_agent = f'''"""
Kaggle ARC-AGI-3 agent — exported from KAOS meta-harness.
Strategy: {name}  |  Local RHAE: {rhae:.4f}
"""
import random, hashlib, time, numpy as np
from collections import defaultdict
from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent

GA_MAP = {{a.value: a for a in GameAction}}

# ─── Strategy (evolved by KAOS meta-harness) ─────────────────────────
{choose_action_src}
# ─────────────────────────────────────────────────────────────────────


class MyAgent(Agent):
    MAX_ACTIONS = 50000

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._state = {{
            "prev_hash": None, "prev_action": None, "prev_grid": None,
            "visited_hashes": set(),
            "frame_change_actions": {{}},
            "tried_actions": {{}},
            "globally_productive": {{}},
            "level": 0, "total_actions": 0, "actions_this_level": 0,
        }}
        self._level = 0
        self._total = 0
        self._lvl_start = 0

    def act(self, frames: list[FrameData]) -> GameAction:
        latest = frames[-1]
        if latest.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        # Level change detection
        lvl = latest.levels_completed
        if lvl > self._level:
            self._level = lvl
            self._lvl_start = self._total
            self._state.update(
                level=lvl, actions_this_level=0,
                visited_hashes=set(), tried_actions={{}},
            )

        grid = np.array(latest.frame, dtype=np.int8)
        if grid.ndim == 3: grid = grid[-1]
        fh = hashlib.md5(grid.tobytes()).hexdigest()
        self._state["visited_hashes"].add(fh)

        prev_h = self._state["prev_hash"]
        prev_a = self._state["prev_action"]
        if prev_h and prev_a is not None and fh != prev_h:
            self._state["frame_change_actions"].setdefault(prev_h, set()).add(prev_a)
            self._state["globally_productive"][prev_a] = \\
                self._state["globally_productive"].get(prev_a, 0) + 1

        self._state["total_actions"] = self._total
        self._state["actions_this_level"] = self._total - self._lvl_start

        avail = [a.value for a in GameAction if a is not GameAction.RESET]

        try:
            action_val, data = choose_action(grid, avail, self._state)
        except Exception:
            action_val = random.choice(avail)
            data = None

        self._state["tried_actions"].setdefault(fh, set()).add(action_val)
        self._state["prev_hash"] = fh
        self._state["prev_action"] = action_val
        self._state["prev_grid"] = grid
        self._total += 1

        action = GA_MAP.get(action_val, GameAction.ACTION1)
        if action_val == 6:
            if data is None:
                nz = np.argwhere(grid != 0)
                if len(nz):
                    i = random.randint(0, len(nz)-1)
                    data = {{"x": int(nz[i][1]), "y": int(nz[i][0])}}
                else:
                    data = {{"x": random.randint(0,63), "y": random.randint(0,63)}}
            action.set_data(data)
        return action
'''
    Path(path).write_text(kaggle_agent)
    print(f"Exported Kaggle agent: {path}")


# ─── Full KAOS MetaHarnessSearch (requires vLLM) ─────────────────────

async def run_search(
    n_games: int,
    time_per_game: int,
    n_iterations: int,
    kaos_yaml: str,
    db_path: str,
) -> None:
    """Run full KAOS MetaHarnessSearch with LLM-based proposer."""
    from kaos import Kaos
    from kaos.metaharness.harness import SearchConfig
    from kaos.metaharness.search import MetaHarnessSearch
    from kaos.router.gepa import GEPARouter

    bench = ArcAGI3Benchmark(
        time_per_game=time_per_game,
        n_search_games=n_games,
    )

    config = SearchConfig(
        benchmark="arc-agi-3",
        max_iterations=n_iterations,
        candidates_per_iteration=2,
        # Match ArcAGI3Benchmark.objectives exactly so Pareto uses all three
        objectives=["+rhae", "+levels", "-actions"],
        max_parallel_evals=2,
        harness_timeout_seconds=time_per_game * n_games * 2,
    )

    db = Kaos(db_path)
    router = GEPARouter.from_config(kaos_yaml)

    print(f"\n=== ARC-AGI-3 MetaHarness Search ===")
    print(f"Games: {n_games} | Time/game: {time_per_game}s | Iterations: {n_iterations}")
    print(f"DB: {db_path} | Config: {kaos_yaml}\n")

    search = MetaHarnessSearch(db, router, bench, config)
    result = await search.run()

    print("\n" + "=" * 60)
    print(result.summary())
    print("=" * 60)

    if result.frontier.points:
        best = result.frontier.points[0]
        print(f"\nBest: rhae={best.scores.get('rhae', 0):.4f}")
        try:
            source = db.read(result.search_agent_id,
                             f"/harnesses/{best.harness_id}/source.py").decode()
            best_path = f"experiments/best_harness_{int(time.time())}.py"
            Path(best_path).write_text(source)
            print(f"Best harness saved: {best_path}")
            _export_kaggle(source, best_path.replace(".py", "_kaggle.py"),
                           "kaos_evolved", best.scores.get("rhae", 0))
        except FileNotFoundError:
            pass

    db.close()
    print(f"\nFull archive: {db_path}")


# ─── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="ARC-AGI-3 KAOS Meta-Harness")
    p.add_argument("--mode", choices=["eval", "search"], default="eval",
                   help="eval: direct seed eval (no LLM); search: full KAOS search (needs vLLM)")
    p.add_argument("--games", type=int, default=5, help="Games in search set")
    p.add_argument("--time", type=int, default=30, help="Seconds per game")
    p.add_argument("--iters", type=int, default=10, help="(search only) LLM iterations")
    p.add_argument("--kaos-yaml", default="kaos.yaml", help="(search only) KAOS config file")
    p.add_argument("--db", default="arc-search.db", help="(search only) KAOS database path")
    p.add_argument("--export", default=None, help="(eval only) Export best harness to file")
    p.add_argument("--harness", default=None, help="(eval only) Extra harness .py file to include")
    args = p.parse_args()

    if args.mode == "eval":
        run_eval(
            n_games=args.games,
            time_per_game=args.time,
            export_path=args.export,
            extra_harness=args.harness,
        )
    else:
        asyncio.run(run_search(
            n_games=args.games,
            time_per_game=args.time,
            n_iterations=args.iters,
            kaos_yaml=args.kaos_yaml,
            db_path=args.db,
        ))


if __name__ == "__main__":
    main()
