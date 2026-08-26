"""Continuous evolution loop - runs indefinitely, saves best agents.
Builds on the v3 evolution that achieved 0.10 on Kaggle.

Improvements over v3:
- Longer time budgets per game (more actions = more exploration)
- More games tested (all 25 public)
- Larger population + more diverse mutations
- Saves checkpoints every N generations
- Logs progress to file for monitoring
"""

import json, time, hashlib, random, copy, sys, logging
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime

import numpy as np
import arc_agi
from arcengine.enums import GameAction as GA, GameState as GS

ACTION_MAP = {a.value: a for a in GA}
DATA_DIR = Path("data/continuous_evolution")
DATA_DIR.mkdir(exist_ok=True, parents=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(DATA_DIR / "evolution.log"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("evolve")


def compute_rhae(level_actions, baseline):
    if not level_actions: return 0.0
    n = len(baseline); tw = n*(n+1)/2; s = 0.0
    for l in range(n):
        w = l+1
        if l in level_actions:
            s += w * min(1.0, baseline[l]/max(level_actions[l],1))**2
    return s/tw


class Strategy:
    """Same architecture as v3 but with more mutation options."""

    ARCHITECTURES = [
        "explore_productive",
        "bfs_replay",
        "diff_chase",
        "action_memory",
        "systematic_sweep",
        "hybrid_bfs_productive",
        "color_hunter",           # NEW: click every unique color object
        "symmetry_exploit",       # NEW: detect symmetry, act on pattern
        "greedy_novelty",         # NEW: always pick action leading to most new states
    ]

    def __init__(self, params=None):
        self.params = params or {
            "architecture": "explore_productive",
            "productive_bias": 0.7,
            "random_explore": 0.186,    # evolved winner
            "click_mode": "centroid",
            "click_jitter": 2,
            "stuck_threshold": 200,
            "prefer_keyboard": False,   # evolved winner
            "bfs_depth": 10,
            "memory_window": 50,
            "sweep_step": 4,
            "explore_decay": 0.995,     # NEW: reduce randomness over time
            "novelty_bonus": 0.3,       # NEW: prefer actions leading to new states
        }
        self.fitness = 0.0
        self.id = hashlib.md5(str(random.random()).encode()).hexdigest()[:8]

    def mutate(self):
        new = copy.deepcopy(self.params)
        n = random.randint(1, 4)
        keys = random.sample(list(new.keys()), min(n, len(new)))
        for k in keys:
            v = new[k]
            if k == "architecture":
                new[k] = random.choice(self.ARCHITECTURES)
            elif isinstance(v, bool):
                new[k] = not v
            elif isinstance(v, float):
                # Wider mutations for more exploration
                delta = random.gauss(0, 0.25)
                new[k] = max(0.001, min(0.999, v + delta))
            elif isinstance(v, int):
                new[k] = max(1, v + random.randint(-50, 50))
            elif isinstance(v, str):
                if k == "click_mode":
                    new[k] = random.choice(["centroid", "edge", "random_nonzero", "systematic", "diff"])
        s = Strategy(new)
        return s

    @staticmethod
    def crossover(a, b):
        new = {}
        for k in a.params:
            new[k] = a.params[k] if random.random() < 0.5 else b.params[k]
        return Strategy(new)

    def choose_action(self, grid, available_actions, state):
        arch = self.params["architecture"]
        p = self.params

        # Decay exploration over time
        decay = p.get("explore_decay", 1.0)
        effective_explore = p["random_explore"] * (decay ** state.get("total_actions", 0))

        try:
            if arch == "explore_productive":
                return self._explore_productive(grid, available_actions, state, effective_explore)
            elif arch == "bfs_replay":
                return self._bfs_replay(grid, available_actions, state, effective_explore)
            elif arch == "diff_chase":
                return self._diff_chase(grid, available_actions, state, effective_explore)
            elif arch == "action_memory":
                return self._action_memory(grid, available_actions, state, effective_explore)
            elif arch == "systematic_sweep":
                return self._systematic_sweep(grid, available_actions, state, effective_explore)
            elif arch == "hybrid_bfs_productive":
                return self._hybrid(grid, available_actions, state, effective_explore)
            elif arch == "color_hunter":
                return self._color_hunter(grid, available_actions, state, effective_explore)
            elif arch == "symmetry_exploit":
                return self._symmetry_exploit(grid, available_actions, state, effective_explore)
            elif arch == "greedy_novelty":
                return self._greedy_novelty(grid, available_actions, state, effective_explore)
        except Exception:
            pass
        return random.choice(available_actions), None

    def _explore_productive(self, grid, available_actions, state, explore_rate):
        p = self.params
        fh = hashlib.md5(grid.tobytes()).hexdigest()
        tried = state["tried_actions"].get(fh, set())
        untried = [a for a in available_actions if a not in tried]

        if random.random() < explore_rate:
            return random.choice(available_actions), self._click_data(random.choice(available_actions), grid, state)

        if untried:
            scored = [(a, state["globally_productive"].get(a, 0)) for a in untried]
            scored.sort(key=lambda x: -x[1])
            if len(scored) > 1 and random.random() < p.get("novelty_bonus", 0.3):
                action = scored[random.randint(0, min(2, len(scored)-1))][0]
            else:
                action = scored[0][0]
        else:
            prod = state["frame_change_actions"].get(fh, set())
            pa = [a for a in available_actions if a in prod]
            if pa and random.random() < p["productive_bias"]:
                action = random.choice(pa)
            else:
                action = random.choice(available_actions)

        return action, self._click_data(action, grid, state)

    def _bfs_replay(self, grid, available_actions, state, explore_rate):
        p = self.params
        fh = hashlib.md5(grid.tobytes()).hexdigest()
        graph = state.get("_graph", {})
        state["_graph"] = graph

        if state.get("_bfs_prev_hash") and state.get("_bfs_prev_action") is not None:
            ph = state["_bfs_prev_hash"]
            if ph not in graph: graph[ph] = {}
            graph[ph][state["_bfs_prev_action"]] = fh

        replay = state.get("_replay_path", [])
        if replay:
            action = replay.pop(0)
            state["_replay_path"] = replay
            state["_bfs_prev_hash"] = fh
            state["_bfs_prev_action"] = action
            return action, self._click_data(action, grid, state)

        tried = state["tried_actions"].get(fh, set())
        untried = [a for a in available_actions if a not in tried]

        if untried:
            action = untried[0]
        else:
            path = self._bfs_find(fh, graph, available_actions, state, p["bfs_depth"])
            if path and len(path) > 1:
                state["_replay_path"] = path[1:]
                action = path[0]
            else:
                return self._explore_productive(grid, available_actions, state, explore_rate)

        state["_bfs_prev_hash"] = fh
        state["_bfs_prev_action"] = action
        return action, self._click_data(action, grid, state)

    def _bfs_find(self, start, graph, available, state, max_depth):
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
                for action, nxt in graph[node].items():
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append((nxt, path + [action]))
        return None

    def _diff_chase(self, grid, available_actions, state, explore_rate):
        prev = state.get("prev_grid")
        if prev is not None and grid.shape == prev.shape:
            diff = np.argwhere(grid != prev)
            if len(diff) > 0 and 6 in available_actions:
                idx = random.randint(0, len(diff)-1)
                return 6, {"x": int(diff[idx][1]), "y": int(diff[idx][0])}
        return self._explore_productive(grid, available_actions, state, explore_rate)

    def _action_memory(self, grid, available_actions, state, explore_rate):
        p = self.params
        mem = state.get("_amem", deque(maxlen=p["memory_window"]))
        state["_amem"] = mem
        fh = hashlib.md5(grid.tobytes()).hexdigest()
        if state.get("prev_hash") and fh != state["prev_hash"]:
            mem.append(state.get("prev_action"))
        if mem and random.random() < p["productive_bias"]:
            from collections import Counter
            counts = Counter(a for a in mem if a in available_actions)
            if counts:
                return counts.most_common(1)[0][0], self._click_data(counts.most_common(1)[0][0], grid, state)
        return self._explore_productive(grid, available_actions, state, explore_rate)

    def _systematic_sweep(self, grid, available_actions, state, explore_rate):
        if 6 not in available_actions:
            return self._explore_productive(grid, available_actions, state, explore_rate)
        step = self.params["sweep_step"]
        si = state.get("_si", 0)
        nz = np.argwhere(grid != 0)
        if len(nz) > 0:
            srt = nz[np.lexsort((nz[:,1], nz[:,0]))]
            pi = (si * step) % len(srt)
            y, x = int(srt[pi][0]), int(srt[pi][1])
            state["_si"] = si + 1
            return 6, {"x": x, "y": y}
        return self._explore_productive(grid, available_actions, state, explore_rate)

    def _hybrid(self, grid, available_actions, state, explore_rate):
        """Hybrid: BFS when state graph is dense, productive otherwise."""
        graph = state.get("_graph", {})
        if len(graph) > 20:
            return self._bfs_replay(grid, available_actions, state, explore_rate)
        return self._explore_productive(grid, available_actions, state, explore_rate)

    def _color_hunter(self, grid, available_actions, state, explore_rate):
        """Systematically click every unique color object in order."""
        if 6 not in available_actions:
            return self._explore_productive(grid, available_actions, state, explore_rate)

        colors = np.unique(grid[grid != 0])
        if len(colors) == 0:
            return self._explore_productive(grid, available_actions, state, explore_rate)

        # Track which colors we've clicked on this frame
        fh = hashlib.md5(grid.tobytes()).hexdigest()
        clicked_colors = state.get("_clicked_colors", {})
        state["_clicked_colors"] = clicked_colors
        frame_clicked = clicked_colors.get(fh, set())

        # Find first unclicked color
        unclicked = [c for c in colors if int(c) not in frame_clicked]
        if not unclicked:
            # All colors clicked, try keyboard actions
            tried = state["tried_actions"].get(fh, set())
            kb_untried = [a for a in available_actions if a != 6 and a not in tried]
            if kb_untried:
                return kb_untried[0], None
            # Reset clicked tracking and try again with different positions
            frame_clicked.clear()
            unclicked = list(colors)

        target_color = int(unclicked[0])
        pixels = np.argwhere(grid == target_color)
        cy, cx = pixels.mean(axis=0).astype(int)
        frame_clicked.add(target_color)
        clicked_colors[fh] = frame_clicked
        return 6, {"x": int(cx), "y": int(cy)}

    def _symmetry_exploit(self, grid, available_actions, state, explore_rate):
        """Detect grid symmetry and patterns, exploit them."""
        # Check horizontal symmetry
        left = grid[:, :32]
        right = grid[:, 32:][:, ::-1]
        h_sym = np.mean(left == right)

        # Check vertical symmetry
        top = grid[:32, :]
        bottom = grid[32:, :][::-1, :]
        v_sym = np.mean(top == bottom)

        # If high symmetry, try actions on the asymmetric part
        if 6 in available_actions:
            if h_sym > 0.8:
                # Click on asymmetric pixels (likely interactive elements)
                diff = (left != right)
                asym_pixels = np.argwhere(diff)
                if len(asym_pixels) > 0:
                    idx = random.randint(0, len(asym_pixels)-1)
                    return 6, {"x": int(asym_pixels[idx][1]), "y": int(asym_pixels[idx][0])}

            if v_sym > 0.8:
                diff = (top != bottom)
                asym_pixels = np.argwhere(diff)
                if len(asym_pixels) > 0:
                    idx = random.randint(0, len(asym_pixels)-1)
                    return 6, {"x": int(asym_pixels[idx][1]), "y": int(asym_pixels[idx][0])}

        # Fallback
        return self._explore_productive(grid, available_actions, state, explore_rate)

    def _greedy_novelty(self, grid, available_actions, state, explore_rate):
        """Always prefer actions that lead to states we haven't seen before."""
        fh = hashlib.md5(grid.tobytes()).hexdigest()
        graph = state.get("_graph", {})
        state["_graph"] = graph

        # Update graph
        if state.get("_gn_prev_hash") and state.get("_gn_prev_action") is not None:
            ph = state["_gn_prev_hash"]
            if ph not in graph: graph[ph] = {}
            graph[ph][state["_gn_prev_action"]] = fh

        # Find actions that lead to unvisited states
        if fh in graph:
            novel_actions = []
            known_actions = []
            for a in available_actions:
                if a in graph[fh]:
                    next_state = graph[fh][a]
                    if next_state not in state["visited_hashes"]:
                        novel_actions.append(a)
                    else:
                        known_actions.append(a)
                else:
                    novel_actions.append(a)  # unknown = potentially novel

            if novel_actions:
                action = random.choice(novel_actions)
                state["_gn_prev_hash"] = fh
                state["_gn_prev_action"] = action
                return action, self._click_data(action, grid, state)

        # Fallback
        result = self._explore_productive(grid, available_actions, state, explore_rate)
        action_val = result[0]
        state["_gn_prev_hash"] = fh
        state["_gn_prev_action"] = action_val
        return result

    def _click_data(self, action, grid, state):
        if action != 6: return None
        p = self.params
        nz = np.argwhere(grid != 0)
        if len(nz) == 0:
            return {"x": random.randint(0,63), "y": random.randint(0,63)}

        mode = p.get("click_mode", "centroid")
        if mode == "centroid":
            colors = np.unique(grid[grid != 0])
            if len(colors) > 0:
                ci = state.get("_ci", 0)
                c = colors[ci % len(colors)]
                state["_ci"] = ci + 1
                px = np.argwhere(grid == c)
                cy, cx = px.mean(axis=0).astype(int)
                j = p.get("click_jitter", 2)
                return {"x": min(63,max(0,int(cx)+random.randint(-j,j))),
                        "y": min(63,max(0,int(cy)+random.randint(-j,j)))}
        elif mode == "edge":
            colors = np.unique(grid[grid != 0])
            if len(colors) > 0:
                c = random.choice(colors)
                px = np.argwhere(grid == c)
                idx = random.choice([0,-1,len(px)//2]) if len(px)>2 else 0
                return {"x": int(px[idx][1]), "y": int(px[idx][0])}
        elif mode == "diff":
            prev = state.get("prev_grid")
            if prev is not None and grid.shape == prev.shape:
                d = np.argwhere(grid != prev)
                if len(d) > 0:
                    idx = random.randint(0,len(d)-1)
                    return {"x": int(d[idx][1]), "y": int(d[idx][0])}

        idx = random.randint(0, len(nz)-1)
        return {"x": int(nz[idx][1]), "y": int(nz[idx][0])}


# ─── Evaluation ───────────────────────────────────────────────────────
def run_strategy(strategy, env_info, arcade, time_budget, max_actions):
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

        # Safety: ensure ACTION6 always has valid data
        if action_val == 6 and (data is None or "x" not in data or "y" not in data):
            nz = np.argwhere(grid != 0)
            if len(nz) > 0:
                idx = random.randint(0, len(nz)-1)
                data = {"x": int(nz[idx][1]), "y": int(nz[idx][0])}
            else:
                data = {"x": random.randint(0,63), "y": random.randint(0,63)}

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
    mean_rhae = np.mean([r["rhae"] for r in results])
    total_levels = sum(r["levels"] for r in results)
    # Composite fitness: rewards both coverage (levels) AND efficiency (RHAE)
    # levels_bonus: each level is worth 0.001 base RHAE
    fitness = mean_rhae + total_levels * 0.001
    return {"mean_rhae": mean_rhae, "total_levels": total_levels,
            "fitness": fitness, "per_game": results}


# ─── Continuous Loop ──────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--time", type=int, default=120, help="Seconds per game")
    parser.add_argument("--actions", type=int, default=5000)
    parser.add_argument("--games", type=int, default=25, help="Number of games")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    args = parser.parse_args()

    log.info("=" * 70)
    log.info(f"Continuous Evolution: pop={args.population}, {args.games} games, {args.time}s/game")
    log.info("=" * 70)

    arcade = arc_agi.Arcade()
    envs = sorted(arcade.get_environments(), key=lambda e: sum(e.baseline_actions))[:args.games]

    # Start from the winning params
    population = []
    base = Strategy()  # has winning params as defaults
    population.append(base)
    for arch in Strategy.ARCHITECTURES:
        s = Strategy()
        s.params["architecture"] = arch
        population.append(s)
    while len(population) < args.population:
        population.append(base.mutate())

    best_rhae = 0.0; best_strategy = None; gen = 0; history = []

    try:
        while True:  # infinite loop
            gen += 1
            log.info(f"\n--- Gen {gen} (pop={len(population)}) ---")

            scored = []
            for i, strat in enumerate(population):
                ev = evaluate(strat, envs, arcade, args.time, args.actions)
                scored.append((strat, ev))
                lvls = " ".join(f"{g['title']}:L{g['levels']}" for g in ev["per_game"] if g["levels"]>0)
                arch = strat.params["architecture"][:15]
                log.info(f"  [{i:2d}] {arch:15s} RHAE={ev['mean_rhae']:.6f} lvls={ev['total_levels']:2d} {lvls}")

            scored.sort(key=lambda x: -x[1]["fitness"])
            bs, be = scored[0]

            if be["fitness"] > best_rhae:
                best_rhae = be["fitness"]; best_strategy = copy.deepcopy(bs)
                log.info(f"  *** NEW BEST: fitness={best_rhae:.6f} RHAE={be['mean_rhae']:.6f} lvls={be['total_levels']} arch={bs.params['architecture']} ***")
                with open(DATA_DIR / "best_params.json", "w") as f:
                    json.dump(best_strategy.params, f, indent=2)

            history.append({
                "gen": gen, "time": datetime.now().isoformat(),
                "best_rhae": be["mean_rhae"],
                "best_fitness": be["fitness"],
                "best_ever": best_rhae,
                "best_arch": bs.params["architecture"],
                "best_levels": be["total_levels"],
            })

            # Checkpoint
            if gen % args.checkpoint_every == 0:
                with open(DATA_DIR / "history.json", "w") as f:
                    json.dump(history, f, indent=2)
                log.info(f"  Checkpoint saved (gen {gen}, best={best_rhae:.6f})")

            # Selection + breeding
            survivors = [s[0] for s in scored[:len(scored)//2]]
            new_pop = list(survivors)
            while len(new_pop) < args.population:
                r = random.random()
                if r < 0.5:
                    new_pop.append(random.choice(survivors).mutate())
                elif r < 0.8:
                    p1, p2 = random.sample(survivors, 2)
                    new_pop.append(Strategy.crossover(p1, p2))
                else:
                    # Fresh injection
                    new_pop.append(Strategy().mutate())
            population = new_pop

    except KeyboardInterrupt:
        log.info(f"\nStopped at gen {gen}. Best RHAE={best_rhae:.6f}")
        with open(DATA_DIR / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        with open(DATA_DIR / "best_params.json", "w") as f:
            json.dump(best_strategy.params if best_strategy else {}, f, indent=2)


if __name__ == "__main__":
    main()
