"""Evolution v3: Structural mutations + BFS replay + pattern detection.
Key insight: parameter tuning plateaus at 0.008. Need new strategy ARCHITECTURES.

New strategies this version can discover:
1. BFS shortest path replay (find level solution, replay it optimally)
2. Pattern-based click (detect grid patterns, click systematically)
3. Action sequence memory (remember what worked, repeat it)
4. Undo-based backtracking (try action, undo if no progress)
5. Diff-based targeting (compare frames, click on changed pixels)
"""

import json, time, hashlib, random, copy
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import arc_agi
from arcengine.enums import GameAction as GA, GameState as GS

ACTION_MAP = {a.value: a for a in GA}
DATA_DIR = Path("data/evolution_v3")
DATA_DIR.mkdir(exist_ok=True, parents=True)


def compute_rhae(level_actions, baseline):
    if not level_actions: return 0.0
    n = len(baseline); tw = n*(n+1)/2; s = 0.0
    for l in range(n):
        w = l+1
        if l in level_actions:
            s += w * min(1.0, baseline[l]/max(level_actions[l],1))**2
    return s/tw


class Strategy:
    """Strategy with structural variation - different exploration architectures."""

    ARCHITECTURES = [
        "explore_productive",   # Original: untried first, then productive
        "bfs_replay",          # BFS to find path, replay shortest
        "diff_chase",          # Compare frames, act on changes
        "action_memory",       # Remember action sequences that caused progress
        "systematic_sweep",    # Systematic grid sweep for clicks
    ]

    def __init__(self, params=None):
        self.params = params or {
            "architecture": "explore_productive",
            "productive_bias": 0.7,
            "random_explore": 0.1,
            "click_mode": "centroid",
            "click_jitter": 2,
            "stuck_threshold": 200,
            "prefer_keyboard": True,
            "bfs_depth": 10,
            "memory_window": 50,
            "sweep_step": 4,
        }

    def mutate(self):
        new = copy.deepcopy(self.params)
        n = random.randint(1, 3)
        keys = random.sample(list(new.keys()), min(n, len(new)))
        for k in keys:
            v = new[k]
            if k == "architecture":
                new[k] = random.choice(self.ARCHITECTURES)
            elif isinstance(v, bool):
                new[k] = not v
            elif isinstance(v, float):
                new[k] = max(0.0, min(1.0, v + random.gauss(0, 0.2)))
            elif isinstance(v, int):
                new[k] = max(1, v + random.randint(-30, 30))
            elif isinstance(v, str):
                if k == "click_mode":
                    new[k] = random.choice(["centroid", "edge", "random_nonzero", "systematic", "diff"])
        return Strategy(new)

    @staticmethod
    def crossover(a, b):
        new = {}
        for k in a.params:
            new[k] = a.params[k] if random.random() < 0.5 else b.params[k]
        return Strategy(new)

    def choose_action(self, grid, available_actions, state):
        arch = self.params["architecture"]
        try:
            if arch == "explore_productive":
                return self._explore_productive(grid, available_actions, state)
            elif arch == "bfs_replay":
                return self._bfs_replay(grid, available_actions, state)
            elif arch == "diff_chase":
                return self._diff_chase(grid, available_actions, state)
            elif arch == "action_memory":
                return self._action_memory(grid, available_actions, state)
            elif arch == "systematic_sweep":
                return self._systematic_sweep(grid, available_actions, state)
        except Exception:
            pass
        return random.choice(available_actions), None

    def _explore_productive(self, grid, available_actions, state):
        """Original best: try untried, prefer productive."""
        p = self.params
        fh = hashlib.md5(grid.tobytes()).hexdigest()
        tried = state["tried_actions"].get(fh, set())
        untried = [a for a in available_actions if a not in tried]

        if random.random() < p["random_explore"]:
            action = random.choice(available_actions)
        elif untried:
            if p["prefer_keyboard"]:
                kb = [a for a in untried if a != 6]
                if kb:
                    scored = [(a, state["globally_productive"].get(a, 0)) for a in kb]
                    scored.sort(key=lambda x: -x[1])
                    action = scored[0][0]
                else:
                    action = untried[0]
            else:
                scored = [(a, state["globally_productive"].get(a, 0)) for a in untried]
                scored.sort(key=lambda x: -x[1])
                action = scored[0][0]
        else:
            prod = state["frame_change_actions"].get(fh, set())
            pa = [a for a in available_actions if a in prod]
            if pa and random.random() < p["productive_bias"]:
                action = random.choice(pa)
            else:
                action = random.choice(available_actions)

        return action, self._click_data(action, grid, state)

    def _bfs_replay(self, grid, available_actions, state):
        """BFS in state graph to find shortest path to unexplored states."""
        p = self.params
        fh = hashlib.md5(grid.tobytes()).hexdigest()
        graph = state.get("_graph", {})
        state["_graph"] = graph

        # Update graph
        if state.get("_bfs_prev_hash") and state.get("_bfs_prev_action") is not None:
            ph = state["_bfs_prev_hash"]
            if ph not in graph: graph[ph] = {}
            graph[ph][state["_bfs_prev_action"]] = fh

        # Check if we have a replay path
        replay = state.get("_replay_path", [])
        if replay:
            action = replay.pop(0)
            state["_replay_path"] = replay
            state["_bfs_prev_hash"] = fh
            state["_bfs_prev_action"] = action
            return action, self._click_data(action, grid, state)

        # BFS to find path to state with untried actions
        tried = state["tried_actions"].get(fh, set())
        untried = [a for a in available_actions if a not in tried]

        if untried:
            action = untried[0]
        else:
            # BFS
            path = self._bfs_find_unexplored(fh, graph, available_actions, state, p["bfs_depth"])
            if path and len(path) > 1:
                state["_replay_path"] = path[1:]  # skip first (we'll take it now)
                action = path[0]
            else:
                prod = state["frame_change_actions"].get(fh, set())
                pa = [a for a in available_actions if a in prod]
                action = random.choice(pa) if pa else random.choice(available_actions)

        state["_bfs_prev_hash"] = fh
        state["_bfs_prev_action"] = action
        return action, self._click_data(action, grid, state)

    def _bfs_find_unexplored(self, start, graph, available, state, max_depth):
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            node, path = queue.popleft()
            if len(path) > max_depth: break
            if node != start:
                tried = state["tried_actions"].get(node, set())
                if any(a not in tried for a in available):
                    return path
            if node in graph:
                for action, next_node in graph[node].items():
                    if next_node not in visited:
                        visited.add(next_node)
                        queue.append((next_node, path + [action]))
        return None

    def _diff_chase(self, grid, available_actions, state):
        """Compare current grid with previous, focus actions on changed areas."""
        prev_grid = state.get("prev_grid")
        if prev_grid is not None and grid.shape == prev_grid.shape:
            diff = (grid != prev_grid)
            changed_pixels = np.argwhere(diff)
            if len(changed_pixels) > 0 and 6 in available_actions:
                # Click on a changed pixel
                idx = random.randint(0, len(changed_pixels) - 1)
                y, x = int(changed_pixels[idx][0]), int(changed_pixels[idx][1])
                return 6, {"x": x, "y": y}

        # Fallback to productive exploration
        return self._explore_productive(grid, available_actions, state)

    def _action_memory(self, grid, available_actions, state):
        """Remember sequences of actions that led to frame changes, repeat them."""
        p = self.params
        memory = state.get("_action_memory", deque(maxlen=p["memory_window"]))
        state["_action_memory"] = memory

        fh = hashlib.md5(grid.tobytes()).hexdigest()

        # If previous action changed the frame, record it
        if state.get("prev_hash") and fh != state["prev_hash"]:
            memory.append(state.get("prev_action"))

        # If we have a memory of productive actions, prefer them
        if memory and random.random() < p["productive_bias"]:
            # Pick the most common productive action from memory
            from collections import Counter
            counts = Counter(a for a in memory if a in available_actions)
            if counts:
                action = counts.most_common(1)[0][0]
                return action, self._click_data(action, grid, state)

        return self._explore_productive(grid, available_actions, state)

    def _systematic_sweep(self, grid, available_actions, state):
        """Systematic grid sweep for click-based games."""
        p = self.params
        step = p["sweep_step"]

        if 6 not in available_actions:
            return self._explore_productive(grid, available_actions, state)

        scan_idx = state.get("_scan_idx", 0)
        # Sweep non-zero pixels in order
        nonzero = np.argwhere(grid != 0)
        if len(nonzero) > 0:
            sorted_pixels = nonzero[np.lexsort((nonzero[:, 1], nonzero[:, 0]))]
            # Step through with stride
            pixel_idx = (scan_idx * step) % len(sorted_pixels)
            y, x = int(sorted_pixels[pixel_idx][0]), int(sorted_pixels[pixel_idx][1])
            state["_scan_idx"] = scan_idx + 1
            return 6, {"x": x, "y": y}

        state["_scan_idx"] = scan_idx + 1
        return self._explore_productive(grid, available_actions, state)

    def _click_data(self, action, grid, state):
        if action != 6: return None
        p = self.params
        mode = p["click_mode"]
        nonzero = np.argwhere(grid != 0)

        if mode == "centroid" and len(nonzero) > 0:
            colors = np.unique(grid[grid != 0])
            if len(colors) > 0:
                ci = state.get("_color_idx", 0)
                color = colors[ci % len(colors)]
                state["_color_idx"] = ci + 1
                pixels = np.argwhere(grid == color)
                cy, cx = pixels.mean(axis=0).astype(int)
                j = p["click_jitter"]
                return {"x": min(63,max(0,int(cx)+random.randint(-j,j))),
                        "y": min(63,max(0,int(cy)+random.randint(-j,j)))}

        elif mode == "edge" and len(nonzero) > 0:
            colors = np.unique(grid[grid != 0])
            if len(colors) > 0:
                color = random.choice(colors)
                pixels = np.argwhere(grid == color)
                idx = random.choice([0, -1, len(pixels)//2]) if len(pixels)>2 else 0
                return {"x": int(pixels[idx][1]), "y": int(pixels[idx][0])}

        elif mode == "diff":
            prev = state.get("prev_grid")
            if prev is not None and grid.shape == prev.shape:
                diff = np.argwhere(grid != prev)
                if len(diff) > 0:
                    idx = random.randint(0, len(diff)-1)
                    return {"x": int(diff[idx][1]), "y": int(diff[idx][0])}

        elif mode == "systematic":
            step = state.get("_click_step", 0)
            x = (step * 7) % 64
            y = (step * 11) % 64
            state["_click_step"] = step + 1
            return {"x": x, "y": y}

        if len(nonzero) > 0:
            idx = random.randint(0, len(nonzero)-1)
            return {"x": int(nonzero[idx][1]), "y": int(nonzero[idx][0])}
        return {"x": random.randint(0,63), "y": random.randint(0,63)}


# ─── Evaluation (same as v2) ─────────────────────────────────────────
def run_strategy(strategy, env_info, arcade, time_budget=120, max_actions=5000):
    env = arcade.make(env_info.game_id)
    frame = env.reset()
    avail = frame.available_actions

    state = {"prev_hash": None, "prev_action": None, "prev_grid": None,
             "visited_hashes": set(), "frame_change_actions": {},
             "tried_actions": {}, "globally_productive": {},
             "level": 0, "total_actions": 0, "actions_this_level": 0}

    total = 0; levels = 0; level_actions = {}; lvl_start = 0; t0 = time.time()

    while time.time()-t0 < time_budget and total < max_actions:
        if frame.state in (GS.NOT_PLAYED, GS.GAME_OVER):
            frame = env.step(GA.RESET)
            state["prev_hash"] = None; state["prev_action"] = None; state["prev_grid"] = None
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
        state["prev_hash"] = fh; state["prev_action"] = action_val; state["prev_grid"] = grid.copy()

        frame = env.step(ACTION_MAP[action_val], data=data)
        total += 1

    return {"rhae": compute_rhae(level_actions, env_info.baseline_actions),
            "levels": levels, "actions": total,
            "level_actions": {str(k):v for k,v in level_actions.items()}}


def evaluate(strategy, envs, arcade, tb, ma):
    results = []
    for ei in envs:
        try:
            r = run_strategy(strategy, ei, arcade, tb, ma)
        except Exception as e:
            r = {"rhae": 0.0, "levels": 0, "actions": 0, "error": str(e)}
        r["title"] = ei.title
        results.append(r)
    return {"mean_rhae": np.mean([r["rhae"] for r in results]),
            "total_levels": sum(r["levels"] for r in results), "per_game": results}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--time", type=int, default=120)
    parser.add_argument("--actions", type=int, default=5000)
    parser.add_argument("--games", type=int, default=10)
    args = parser.parse_args()

    print("=" * 70)
    print(f"Evolution v3 (structural): {args.generations} gens, pop={args.population}")
    print("=" * 70)

    arcade = arc_agi.Arcade()
    envs = sorted(arcade.get_environments(), key=lambda e: sum(e.baseline_actions))[:args.games]

    # Diverse initial population: one per architecture
    population = []
    for arch in Strategy.ARCHITECTURES:
        s = Strategy()
        s.params["architecture"] = arch
        population.append(s)
    while len(population) < args.population:
        population.append(Strategy().mutate())

    best_rhae = 0.0; best_strategy = None; history = []

    for gen in range(args.generations):
        print(f"\n--- Gen {gen+1}/{args.generations} ---")
        scored = []
        for i, strat in enumerate(population):
            ev = evaluate(strat, envs, arcade, args.time, args.actions)
            scored.append((strat, ev))
            lvls = " ".join(f"{g['title']}:L{g['levels']}" for g in ev["per_game"] if g["levels"]>0)
            arch = strat.params["architecture"][:12]
            print(f"  [{i}] {arch:12s} RHAE={ev['mean_rhae']:.6f} lvls={ev['total_levels']} {lvls}")

        scored.sort(key=lambda x: -x[1]["mean_rhae"])
        bs, be = scored[0]

        if be["mean_rhae"] > best_rhae:
            best_rhae = be["mean_rhae"]; best_strategy = copy.deepcopy(bs)
            print(f"  *** NEW BEST: RHAE={best_rhae:.6f} arch={bs.params['architecture']} ***")

        history.append({"gen": gen+1, "best_rhae": be["mean_rhae"],
                        "best_arch": bs.params["architecture"],
                        "best_levels": be["total_levels"]})

        # Select top 50% + breed
        survivors = [s[0] for s in scored[:len(scored)//2]]
        new_pop = list(survivors)
        while len(new_pop) < args.population:
            if random.random() < 0.6:
                new_pop.append(random.choice(survivors).mutate())
            elif random.random() < 0.8:
                p1, p2 = random.sample(survivors, 2)
                new_pop.append(Strategy.crossover(p1, p2))
            else:
                # Inject fresh random strategy
                new_pop.append(Strategy().mutate())
        population = new_pop

    print(f"\n{'='*70}")
    print(f"BEST: RHAE={best_rhae:.6f}")
    print(f"Params: {json.dumps(best_strategy.params, indent=2)}")
    print(f"{'='*70}")

    with open(DATA_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(DATA_DIR / "best_params.json", "w") as f:
        json.dump(best_strategy.params, f, indent=2)

if __name__ == "__main__":
    main()
