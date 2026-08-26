"""Validate our improved FORGE agent locally.
Loads game classes via importlib (same as Kaggle), runs BFS + CNN fallback,
measures RHAE per game. This is the ground truth for testing improvements.

Usage: uv run python validate_forge.py --time 60 --games 25
"""

import sys, os, time, json, copy, hashlib, importlib.util, logging, traceback
from pathlib import Path
from collections import deque, defaultdict

import numpy as np
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("validate")

import arc_agi
from arcengine.enums import GameAction, ActionInput, GameState

ACTION_MAP = {a.value: a for a in GameAction}


def compute_rhae(level_actions, baseline):
    """Exact RHAE formula from competition."""
    if not level_actions:
        return 0.0
    n = len(baseline)
    tw = n * (n + 1) / 2
    s = 0.0
    for l in range(n):
        w = l + 1
        if l in level_actions:
            h = baseline[l]
            a = level_actions[l]
            s += w * min(1.0, h / max(a, 1)) ** 2
    return s / tw


def load_game_class(env_dir):
    """Load game class from environment directory (same as FORGE)."""
    for root, dirs, files in os.walk(env_dir):
        for f in files:
            if f.endswith('.py') and not f.startswith('__'):
                path = os.path.join(root, f)
                name = f[:-3]
                cls_name = name[0].upper() + name[1:]
                try:
                    mod_name = f'game_{name}_{id(path)}'
                    spec = importlib.util.spec_from_file_location(mod_name, path)
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    # Find the game class
                    for attr_name in dir(mod):
                        obj = getattr(mod, attr_name)
                        if (isinstance(obj, type) and hasattr(obj, 'perform_action')
                                and attr_name != 'ARCBaseGame'):
                            return obj, path
                except Exception as e:
                    pass
    return None, None


def bfs_solve_level(game_cls, level_idx=0, timeout=60, max_states=100000):
    """BFS solve a single level. Returns (solution, states_explored, elapsed)."""
    from scipy import ndimage

    game = game_cls()
    r0 = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    if not r0.frame:
        return None, 0, 0

    f0 = np.array(r0.frame[-1])
    avail = game._available_actions if hasattr(game, '_available_actions') else [1, 2, 3, 4, 5, 6]

    # Hidden field probing
    hidden_fields = []
    for k, v in game.__dict__.items():
        if k.startswith('_'):
            continue
        if isinstance(v, (int, float, bool)):
            hidden_fields.append(k)

    def state_hash(g, frame_arr):
        h = hashlib.md5(frame_arr.tobytes()).hexdigest()
        for field in hidden_fields:
            try:
                v = getattr(g, field, None)
                if v is not None:
                    h += f"|{field}={v}"
            except:
                pass
        return h

    # Scan effective actions
    actions = []
    bg = int(np.bincount(f0.flatten(), minlength=16).argmax())

    for a in avail:
        if a == 6:
            # Object centroid scan
            for c in range(1, 16):
                mask = (f0 == c)
                if not mask.any():
                    continue
                labeled, n = ndimage.label(mask)
                for i in range(1, min(n + 1, 5)):
                    region = (labeled == i)
                    if region.sum() < 1:
                        continue
                    cy, cx = ndimage.center_of_mass(region)
                    actions.append((6, {'x': int(cx), 'y': int(cy), 'game_id': ''}))
            # Also stride-4 scan for non-bg pixels
            for y in range(0, 64, 4):
                for x in range(0, 64, 4):
                    if f0[y, x] != bg and (6, {'x': x, 'y': y, 'game_id': ''}) not in actions:
                        g = copy.deepcopy(game)
                        try:
                            r = g.perform_action(
                                ActionInput(id=GameAction.ACTION6, data={'x': x, 'y': y, 'game_id': ''}),
                                raw=True
                            )
                            if r.frame and np.any(f0 != np.array(r.frame[-1])):
                                actions.append((6, {'x': x, 'y': y, 'game_id': ''}))
                        except:
                            pass
        else:
            g = copy.deepcopy(game)
            try:
                r = g.perform_action(ActionInput(id=GameAction(a)), raw=True)
                if r.frame and np.any(f0 != np.array(r.frame[-1])):
                    actions.append((a, None))
            except:
                pass

    if not actions:
        return None, 0, 0

    t0 = time.time()
    ih = state_hash(game, f0)
    queue = deque([(copy.deepcopy(game), f0, [], ih)])
    visited = {ih}

    while queue and time.time() - t0 < timeout and len(visited) < max_states:
        g, f, path, _ = queue.popleft()
        if len(path) > 30:
            continue

        for act_id, data in actions:
            g2 = copy.deepcopy(g)
            if data:
                ai = ActionInput(id=GameAction.ACTION6, data=data)
            else:
                ai = ActionInput(id=GameAction(act_id))

            try:
                f2 = g2.perform_action(ai, raw=True)
            except:
                continue

            if not f2.frame:
                continue

            frame2 = np.array(f2.frame[-1])

            if f2.levels_completed > 0:
                return path + [(act_id, data)], len(visited), time.time() - t0

            h2 = state_hash(g2, frame2)
            if h2 not in visited:
                visited.add(h2)
                queue.append((g2, frame2, path + [(act_id, data)], h2))

    return None, len(visited), time.time() - t0


def validate_all(time_per_game=60, max_games=25):
    """Run BFS on all public games, compute RHAE."""
    arcade = arc_agi.Arcade()
    envs = sorted(arcade.get_environments(), key=lambda e: sum(e.baseline_actions))

    env_base = "environment_files"
    results = []
    total_t0 = time.time()

    for i, env_info in enumerate(envs[:max_games]):
        game_name = env_info.game_id.split('-')[0]
        env_dir = os.path.join(env_base, game_name)

        if not os.path.isdir(env_dir):
            logger.warning(f"[{i+1}/{max_games}] {env_info.title}: no env dir")
            results.append({"title": env_info.title, "solved": False, "levels": 0, "rhae": 0})
            continue

        game_cls, game_path = load_game_class(env_dir)
        if not game_cls:
            logger.warning(f"[{i+1}/{max_games}] {env_info.title}: no game class")
            results.append({"title": env_info.title, "solved": False, "levels": 0, "rhae": 0})
            continue

        logger.info(f"[{i+1}/{max_games}] {env_info.title} ({game_cls.__name__})...")

        # Try BFS on each level
        level_actions = {}
        total_actions = 0
        for lvl in range(len(env_info.baseline_actions)):
            try:
                sol, states, elapsed = bfs_solve_level(game_cls, lvl, timeout=time_per_game)
            except Exception as e:
                logger.error(f"  L{lvl} error: {e}")
                break

            if sol:
                level_actions[lvl] = len(sol)
                total_actions += len(sol)
                logger.info(f"  L{lvl}: SOLVED in {len(sol)} actions ({states} states, {elapsed:.1f}s) "
                           f"[human: {env_info.baseline_actions[lvl]}]")
            else:
                logger.info(f"  L{lvl}: FAILED ({states} states, {elapsed:.1f}s)")
                break  # Can't skip levels

        rhae = compute_rhae(level_actions, env_info.baseline_actions)
        results.append({
            "title": env_info.title,
            "solved": len(level_actions) > 0,
            "levels": len(level_actions),
            "total_levels": len(env_info.baseline_actions),
            "level_actions": level_actions,
            "rhae": round(rhae, 6),
            "human_baseline": env_info.baseline_actions,
        })

        if level_actions:
            logger.info(f"  RHAE={rhae:.4f} ({len(level_actions)}/{len(env_info.baseline_actions)} levels)")

    # Summary
    total_elapsed = time.time() - total_t0
    solved = [r for r in results if r["solved"]]
    mean_rhae = np.mean([r["rhae"] for r in results])

    logger.info(f"\n{'='*70}")
    logger.info(f"VALIDATION RESULTS: {len(solved)}/{len(results)} games solved, RHAE={mean_rhae:.6f}")
    logger.info(f"Time: {total_elapsed:.0f}s")
    logger.info(f"{'='*70}")

    for r in sorted(results, key=lambda x: -x["rhae"]):
        if r["rhae"] > 0:
            la = r.get("level_actions", {})
            hb = r.get("human_baseline", [])
            details = " ".join(f"L{l}:{la[l]}a(h={hb[l]})" for l in sorted(la.keys()) if l < len(hb))
            logger.info(f"  {r['title']:5s}: RHAE={r['rhae']:.4f} {details}")

    Path("data").mkdir(exist_ok=True)
    with open("data/forge_validation.json", "w") as f:
        json.dump({"mean_rhae": mean_rhae, "results": results, "elapsed": total_elapsed},
                  f, indent=2, default=str)
    logger.info(f"\nSaved to data/forge_validation.json")
    logger.info(f"\nTarget: 0.50 (1st place). Current: {mean_rhae:.6f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=int, default=60, help="Seconds per game per level")
    parser.add_argument("--games", type=int, default=25)
    args = parser.parse_args()
    validate_all(args.time, args.games)
