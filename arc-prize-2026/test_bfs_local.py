"""Test BFS solver locally on public games. Measures if our improvements help."""
import importlib.util, hashlib, copy, time, sys, os, json
from collections import deque
from pathlib import Path
import numpy as np
from arcengine.enums import GameAction, ActionInput, GameState

ACTION_MAP = {a.value: a for a in GameAction}

def load_game(env_dir):
    """Load game class from environment directory."""
    for root, dirs, files in os.walk(env_dir):
        for f in files:
            if f.endswith('.py') and not f.startswith('__'):
                path = os.path.join(root, f)
                name = f[:-3]
                cls_name = name[0].upper() + name[1:]
                spec = importlib.util.spec_from_file_location('gm', path)
                mod = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(mod)
                    cls = getattr(mod, cls_name, None)
                    if cls and hasattr(cls, 'perform_action'):
                        return cls
                except:
                    pass
    return None

def bfs_solve(game_cls, level_idx=0, timeout=60, max_states=50000):
    """BFS solve a level. Returns (solution, n_states, elapsed)."""
    game = game_cls()
    frame = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    if not frame._frame: return None, 0, 0

    f0 = np.array(frame._frame[-1])
    avail = frame.available_actions

    def state_hash(g, f):
        h = hashlib.md5(np.array(f._frame[-1]).tobytes()).hexdigest()
        for k, v in sorted(g.__dict__.items()):
            if k.startswith('_'): continue
            if isinstance(v, (int, float, bool)):
                h += f"|{k}={v}"
        return h

    # Scan effective actions
    actions = []
    for a in avail:
        if a == 6:
            # Object centroid scan
            from scipy import ndimage
            for c in range(1, 16):
                mask = (f0 == c)
                if not mask.any(): continue
                labeled, n = ndimage.label(mask)
                for i in range(1, min(n+1, 5)):
                    region = (labeled == i)
                    cy, cx = ndimage.center_of_mass(region)
                    actions.append((6, {'x': int(cx), 'y': int(cy), 'game_id': ''}))
        else:
            g = copy.deepcopy(game)
            try:
                r = g.perform_action(ActionInput(id=GameAction(a)), raw=True)
                if r._frame and np.any(f0 != np.array(r._frame[-1])):
                    actions.append((a, None))
            except: pass

    if not actions:
        return None, 0, 0

    t0 = time.time()
    ih = state_hash(game, frame)
    queue = deque([(copy.deepcopy(game), frame, [], ih)])
    visited = {ih}
    best_level = 0
    best_solution = None

    while queue and time.time() - t0 < timeout and len(visited) < max_states:
        g, f, path, _ = queue.popleft()
        if len(path) > 30: continue

        for act_id, data in actions:
            g2 = copy.deepcopy(g)
            ai = ActionInput(id=GameAction(act_id) if act_id <= 5 else GameAction.ACTION6)
            if data: ai = ActionInput(id=GameAction.ACTION6, data=data)

            try:
                f2 = g2.perform_action(ai, raw=True)
            except: continue

            if f2.levels_completed > best_level:
                best_level = f2.levels_completed
                best_solution = path + [(act_id, data)]
                print(f"    L{best_level} in {len(best_solution)} actions!")

            h2 = state_hash(g2, f2)
            if h2 not in visited:
                visited.add(h2)
                queue.append((g2, f2, path + [(act_id, data)], h2))

    elapsed = time.time() - t0
    return best_solution, len(visited), elapsed

def main():
    print("BFS Local Test")
    print("=" * 60)

    env_base = "environment_files"
    games = sorted(os.listdir(env_base))

    results = []
    for game_name in games:
        env_dir = os.path.join(env_base, game_name)
        if not os.path.isdir(env_dir): continue

        game_cls = load_game(env_dir)
        if not game_cls:
            print(f"  {game_name}: SKIP (no game class)")
            continue

        print(f"\n  {game_name} ({game_cls.__name__}):")
        sol, states, elapsed = bfs_solve(game_cls, level_idx=0, timeout=30, max_states=20000)

        r = {"game": game_name, "solved": sol is not None,
             "actions": len(sol) if sol else 0, "states": states, "time": round(elapsed, 1)}
        results.append(r)

        if sol:
            print(f"    SOLVED L1 in {len(sol)} actions ({states} states, {elapsed:.1f}s)")
        else:
            print(f"    FAILED ({states} states, {elapsed:.1f}s)")

    print(f"\n{'='*60}")
    solved = [r for r in results if r["solved"]]
    print(f"Solved {len(solved)}/{len(results)} games at L1")
    for r in sorted(solved, key=lambda x: x["actions"]):
        print(f"  {r['game']:6s}: {r['actions']:3d} actions ({r['states']} states, {r['time']}s)")

    Path("data").mkdir(exist_ok=True)
    with open("data/bfs_local_test.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
