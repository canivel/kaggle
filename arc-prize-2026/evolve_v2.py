"""Programmatic agent evolution v2 - no LLM needed.
Generates strategy variants by mutating parameters and combining approaches.
Tests each variant, keeps the best, breeds new variants.
"""

import json, time, hashlib, random, copy
from pathlib import Path
from collections import defaultdict

import numpy as np
import arc_agi
from arcengine.enums import GameAction as GA, GameState as GS

ACTION_MAP = {a.value: a for a in GA}
DATA_DIR = Path("data/evolution_v2")
DATA_DIR.mkdir(exist_ok=True, parents=True)


def compute_rhae(level_actions, baseline):
    if not level_actions: return 0.0
    n = len(baseline); tw = n*(n+1)/2; s = 0.0
    for l in range(n):
        w = l+1
        if l in level_actions:
            s += w * min(1.0, baseline[l]/max(level_actions[l],1))**2
    return s/tw


# ─── Strategy as a parameterized function ─────────────────────────────
class Strategy:
    """Parameterized agent strategy that can be mutated."""

    def __init__(self, params=None):
        self.params = params or {
            # Exploration
            "untried_first": True,
            "productive_bias": 0.7,     # probability of choosing productive action
            "random_explore": 0.1,      # probability of random action

            # Click targeting
            "click_mode": "centroid",    # centroid, edge, random_nonzero, systematic
            "click_jitter": 2,           # pixels of randomness around target

            # State management
            "stuck_threshold": 200,      # actions without new state = stuck
            "stuck_action": "random",    # random, undo, reset

            # Action ordering
            "action_order": "productive_first",  # productive_first, sequential, random
            "prefer_keyboard": True,     # try keyboard before click when both available
        }

    def mutate(self):
        """Create a mutated copy."""
        new_params = copy.deepcopy(self.params)
        # Pick 1-3 params to mutate
        n_mutations = random.randint(1, 3)
        keys = random.sample(list(new_params.keys()), min(n_mutations, len(new_params)))

        for key in keys:
            val = new_params[key]
            if isinstance(val, bool):
                new_params[key] = not val
            elif isinstance(val, float):
                new_params[key] = max(0.0, min(1.0, val + random.gauss(0, 0.15)))
            elif isinstance(val, int):
                new_params[key] = max(1, val + random.randint(-50, 50))
            elif isinstance(val, str):
                if key == "click_mode":
                    new_params[key] = random.choice(["centroid", "edge", "random_nonzero", "systematic"])
                elif key == "stuck_action":
                    new_params[key] = random.choice(["random", "undo", "reset"])
                elif key == "action_order":
                    new_params[key] = random.choice(["productive_first", "sequential", "random"])

        return Strategy(new_params)

    @staticmethod
    def crossover(a, b):
        """Combine two strategies."""
        new_params = {}
        for key in a.params:
            new_params[key] = a.params[key] if random.random() < 0.5 else b.params[key]
        return Strategy(new_params)

    def choose_action(self, grid, available_actions, state):
        """The actual action selection logic, driven by parameters."""
        p = self.params
        frame_hash = hashlib.md5(grid.tobytes()).hexdigest()
        tried = state["tried_actions"].get(frame_hash, set())
        untried = [a for a in available_actions if a not in tried]

        has_click = 6 in available_actions
        keyboard_actions = [a for a in available_actions if a != 6]

        # Check if stuck
        actions_no_new = state.get("_actions_no_new", 0)
        if frame_hash not in state["visited_hashes"]:
            state["_actions_no_new"] = 0
        else:
            state["_actions_no_new"] = actions_no_new + 1

        if state.get("_actions_no_new", 0) > p["stuck_threshold"]:
            if p["stuck_action"] == "random":
                action = random.choice(available_actions)
                return self._finalize(action, grid, available_actions, state)
            elif p["stuck_action"] == "undo" and 7 in available_actions:
                return 7, None

        # Random exploration chance
        if random.random() < p["random_explore"]:
            action = random.choice(available_actions)
            return self._finalize(action, grid, available_actions, state)

        # Untried first
        if p["untried_first"] and untried:
            if p["prefer_keyboard"] and has_click:
                kb_untried = [a for a in untried if a != 6]
                if kb_untried:
                    return self._select_from(kb_untried, state, grid)

            return self._select_from(untried, state, grid)

        # All tried - use productive actions
        productive = state["frame_change_actions"].get(frame_hash, set())
        prod_avail = [a for a in available_actions if a in productive]

        if prod_avail and random.random() < p["productive_bias"]:
            if p["action_order"] == "productive_first":
                # Sort by global productivity
                scored = [(a, state["globally_productive"].get(a, 0)) for a in prod_avail]
                scored.sort(key=lambda x: -x[1])
                action = scored[0][0]
            elif p["action_order"] == "sequential":
                action = prod_avail[state.get("_seq_idx", 0) % len(prod_avail)]
                state["_seq_idx"] = state.get("_seq_idx", 0) + 1
            else:
                action = random.choice(prod_avail)
            return self._finalize(action, grid, available_actions, state)

        # Fallback: random
        action = random.choice(available_actions)
        return self._finalize(action, grid, available_actions, state)

    def _select_from(self, actions, state, grid):
        """Select from a list, preferring productive ones."""
        p = self.params
        if p["action_order"] == "productive_first":
            scored = [(a, state["globally_productive"].get(a, 0)) for a in actions]
            scored.sort(key=lambda x: -x[1])
            # Top action with some randomness
            if len(scored) > 1 and random.random() < 0.3:
                action = scored[1][0]
            else:
                action = scored[0][0]
        elif p["action_order"] == "sequential":
            action = actions[state.get("_seq_idx", 0) % len(actions)]
            state["_seq_idx"] = state.get("_seq_idx", 0) + 1
        else:
            action = random.choice(actions)
        return self._finalize(action, grid, actions, state)

    def _finalize(self, action, grid, available_actions, state):
        """Generate click data if needed."""
        p = self.params
        if action != 6:
            return action, None

        # Click targeting
        nonzero = np.argwhere(grid != 0)
        if len(nonzero) == 0:
            return 6, {"x": random.randint(0, 63), "y": random.randint(0, 63)}

        if p["click_mode"] == "centroid":
            colors = np.unique(grid[grid != 0])
            if len(colors) > 0:
                color = colors[state.get("_color_idx", 0) % len(colors)]
                state["_color_idx"] = state.get("_color_idx", 0) + 1
                pixels = np.argwhere(grid == color)
                cy, cx = pixels.mean(axis=0).astype(int)
                jitter = p["click_jitter"]
                cy = min(63, max(0, cy + random.randint(-jitter, jitter)))
                cx = min(63, max(0, cx + random.randint(-jitter, jitter)))
                return 6, {"x": int(cx), "y": int(cy)}

        elif p["click_mode"] == "edge":
            colors = np.unique(grid[grid != 0])
            if len(colors) > 0:
                color = random.choice(colors)
                pixels = np.argwhere(grid == color)
                # Pick edge pixel
                if len(pixels) > 2:
                    idx = random.choice([0, len(pixels)-1, len(pixels)//2])
                else:
                    idx = 0
                return 6, {"x": int(pixels[idx][1]), "y": int(pixels[idx][0])}

        elif p["click_mode"] == "systematic":
            # Systematic grid scan
            step = state.get("_scan_step", 0)
            x = (step * 7) % 64  # prime step for coverage
            y = (step * 11) % 64
            state["_scan_step"] = step + 1
            return 6, {"x": x, "y": y}

        # random_nonzero
        idx = random.randint(0, len(nonzero) - 1)
        return 6, {"x": int(nonzero[idx][1]), "y": int(nonzero[idx][0])}


# ─── Evaluation ───────────────────────────────────────────────────────
def run_strategy(strategy, env_info, arcade, time_budget=120, max_actions=5000):
    env = arcade.make(env_info.game_id)
    frame = env.reset()
    avail = frame.available_actions

    state = {"prev_hash": None, "prev_action": None, "visited_hashes": set(),
             "frame_change_actions": {}, "tried_actions": {},
             "globally_productive": {}, "level": 0, "total_actions": 0,
             "actions_this_level": 0}

    total = 0; levels = 0; level_actions = {}; lvl_start = 0; t0 = time.time()

    while time.time()-t0 < time_budget and total < max_actions:
        if frame.state in (GS.NOT_PLAYED, GS.GAME_OVER):
            frame = env.step(GA.RESET)
            state["prev_hash"] = None; state["prev_action"] = None
            continue
        if frame.state == GS.WIN: break

        if frame.levels_completed > levels:
            level_actions[levels] = total - lvl_start
            levels = frame.levels_completed; lvl_start = total
            state["level"] = levels; state["actions_this_level"] = 0
            state["visited_hashes"] = set(); state["tried_actions"] = {}

        grid = np.array(frame._frame[0], dtype=np.int8)
        if grid.ndim == 3: grid = grid[-1]
        fh = hashlib.md5(grid.tobytes()).hexdigest()
        state["visited_hashes"].add(fh)

        if state["prev_hash"] and state["prev_action"] is not None:
            if fh != state["prev_hash"]:
                state["frame_change_actions"].setdefault(state["prev_hash"], set()).add(state["prev_action"])
                state["globally_productive"][state["prev_action"]] = \
                    state["globally_productive"].get(state["prev_action"], 0) + 1

        state["total_actions"] = total; state["actions_this_level"] = total - lvl_start

        action_val, data = strategy.choose_action(grid, avail, state)

        state["tried_actions"].setdefault(fh, set()).add(action_val)
        state["prev_hash"] = fh; state["prev_action"] = action_val

        frame = env.step(ACTION_MAP[action_val], data=data)
        total += 1

    return {"rhae": compute_rhae(level_actions, env_info.baseline_actions),
            "levels": levels, "actions": total,
            "level_actions": {str(k):v for k,v in level_actions.items()}}


def evaluate_strategy(strategy, envs, arcade, time_budget=120, max_actions=5000):
    results = []
    for ei in envs:
        try:
            r = run_strategy(strategy, ei, arcade, time_budget, max_actions)
        except Exception as e:
            r = {"rhae": 0.0, "levels": 0, "actions": 0, "error": str(e)}
        r["title"] = ei.title
        results.append(r)
    mean_rhae = np.mean([r["rhae"] for r in results])
    total_levels = sum(r["levels"] for r in results)
    return {"mean_rhae": mean_rhae, "total_levels": total_levels, "per_game": results}


# ─── Evolution ────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--time", type=int, default=120)
    parser.add_argument("--actions", type=int, default=5000)
    parser.add_argument("--games", type=int, default=10)
    args = parser.parse_args()

    print("=" * 70)
    print(f"Strategy Evolution: {args.generations} gens, pop={args.population}")
    print(f"  {args.games} games, {args.time}s/game, {args.actions} max")
    print("=" * 70)

    arcade = arc_agi.Arcade()
    envs = sorted(arcade.get_environments(), key=lambda e: sum(e.baseline_actions))[:args.games]

    # Initial population
    population = [Strategy() for _ in range(args.population)]
    # Add some diverse seeds
    for i in range(min(3, args.population)):
        s = Strategy()
        s.params["click_mode"] = ["centroid", "edge", "systematic"][i]
        s.params["productive_bias"] = [0.5, 0.8, 0.3][i]
        s.params["random_explore"] = [0.05, 0.15, 0.3][i]
        population[i] = s

    best_ever_rhae = 0.0
    best_ever_strategy = None
    history = []

    for gen in range(args.generations):
        print(f"\n--- Gen {gen+1}/{args.generations} (pop={len(population)}) ---")

        # Evaluate all
        scored = []
        for i, strat in enumerate(population):
            ev = evaluate_strategy(strat, envs, arcade, args.time, args.actions)
            scored.append((strat, ev))
            lvl_info = " ".join(f"{g['title']}:L{g['levels']}" for g in ev["per_game"] if g["levels"]>0)
            print(f"  [{i}] RHAE={ev['mean_rhae']:.6f} lvls={ev['total_levels']} {lvl_info}")

        # Sort by RHAE
        scored.sort(key=lambda x: -x[1]["mean_rhae"])
        best_strat, best_ev = scored[0]

        if best_ev["mean_rhae"] > best_ever_rhae:
            best_ever_rhae = best_ev["mean_rhae"]
            best_ever_strategy = copy.deepcopy(best_strat)
            print(f"  *** NEW BEST: RHAE={best_ever_rhae:.6f} ***")

        history.append({
            "generation": gen+1,
            "best_rhae": best_ev["mean_rhae"],
            "best_levels": best_ev["total_levels"],
            "best_params": best_strat.params,
            "population_rhae": [s[1]["mean_rhae"] for s in scored],
        })

        # Selection: keep top 50%
        survivors = [s[0] for s in scored[:len(scored)//2]]

        # Breed new population
        new_pop = list(survivors)  # keep survivors
        while len(new_pop) < args.population:
            if random.random() < 0.7:
                # Mutate a survivor
                parent = random.choice(survivors)
                new_pop.append(parent.mutate())
            else:
                # Crossover two survivors
                p1, p2 = random.sample(survivors, 2)
                new_pop.append(Strategy.crossover(p1, p2))

        population = new_pop

    # Final
    print(f"\n{'='*70}")
    print(f"EVOLUTION COMPLETE")
    print(f"  Best RHAE: {best_ever_rhae:.6f}")
    print(f"  Best params: {json.dumps(best_ever_strategy.params, indent=2)}")
    print(f"  Target: 0.1")
    print(f"{'='*70}")

    with open(DATA_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)
    with open(DATA_DIR / "best_params.json", "w") as f:
        json.dump(best_ever_strategy.params, f, indent=2)

    print(f"\nSaved to {DATA_DIR}/")


if __name__ == "__main__":
    main()
