"""Local validation of FORGE v19 agent against all 25 ARC-AGI-3 games.

Runs the REAL BFS solver + CNN fallback (same code as Kaggle submission)
and computes RHAE that should match the leaderboard.

Usage:
    uv run python local_forge_eval.py                    # all 25 games, 120s each
    uv run python local_forge_eval.py --games 10 --time 60   # quick test
    uv run python local_forge_eval.py --time 300             # deep eval (like Kaggle)
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

# ARC-AGI SDK
import arc_agi
from arcengine.enums import GameAction as GA, GameState as GS, ActionInput


def compute_rhae(level_actions, baseline_actions):
    """RHAE metric matching Kaggle competition formula."""
    if not level_actions:
        return 0.0
    n = len(baseline_actions)
    total_w = n * (n + 1) / 2
    score = 0.0
    for l in range(n):
        w = l + 1
        if l in level_actions:
            h = baseline_actions[l]
            a = level_actions[l]
            s = min(1.0, h / max(a, 1)) ** 2
            score += w * s
    return score / total_w


class ForgeBFSEvaluator:
    """Runs FORGE BFS solver directly against game source (offline solving)."""

    def __init__(self, env_files_dir="environment_files"):
        self.env_dir = Path(env_files_dir)

    def _find_game_source(self, game_id):
        """Find game .py file and extract class name."""
        gid = game_id.split('-')[0]
        cls_name = gid[0].upper() + gid[1:]

        # Search in environment_files/{gid}/*/
        game_dir = self.env_dir / gid
        if game_dir.exists():
            for py_file in game_dir.rglob(f"{gid}.py"):
                import re
                content = py_file.read_text()[:2000]
                m = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', content)
                if m:
                    cls_name = m.group(1)
                return str(py_file), cls_name

        return None, None

    def _load_game_class(self, game_path, class_name):
        """Load game class via importlib."""
        try:
            spec = importlib.util.spec_from_file_location('game_mod', game_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, class_name)
        except Exception as e:
            print(f"  Load failed: {e}")
            return None

    def _state_hash(self, game, frame, hidden_fields=None):
        fh = hashlib.md5(frame.tobytes()).hexdigest()[:16]
        if hidden_fields:
            extras = []
            for field_name in hidden_fields:
                try:
                    v = getattr(game, field_name, None)
                    if v is not None:
                        extras.append(f"{field_name}={v}")
                except:
                    pass
            if extras:
                return fh + "|" + "|".join(extras)
        return fh

    def _probe_hidden_fields(self, game, actions):
        initial = {}
        for k, v in game.__dict__.items():
            if isinstance(v, (int, float, bool)) and not k.startswith('__'):
                initial[k] = v
        changing = set()
        frame0 = game.get_pixels(0, 0, 64, 64)
        for act_id, data in actions[:10]:
            g = copy.deepcopy(game)
            try:
                ai = ActionInput(id=GA.from_id(act_id), data=data) if data else ActionInput(id=GA.from_id(act_id))
                g.perform_action(ai, raw=True)
            except:
                continue
            for k, v in g.__dict__.items():
                if isinstance(v, (int, float, bool)) and not k.startswith('__'):
                    if k in initial and v != initial[k]:
                        if k not in ('_action_count', '_full_reset', '_action_complete'):
                            changing.add(k)
        return sorted([f for f in changing if not f.startswith('_') or f in ('_current_level_index', '_score')])

    def _scan_actions(self, game, f0, bg, timeout=5):
        avail = game._available_actions
        actions = []
        # Keyboard
        for a in [a for a in avail if a <= 5]:
            g = copy.deepcopy(game)
            try:
                r = g.perform_action(ActionInput(id=GA.from_id(a)), raw=True)
                if r.frame and np.sum(f0 != np.array(r.frame[-1])) > 0:
                    actions.append((a, None))
            except:
                pass
        # Click
        if 6 in avail:
            t0 = time.time()
            seen = set()
            hits = []
            for y in range(0, 64, 2):
                if time.time() - t0 > timeout:
                    break
                for x in range(0, 64, 2):
                    if f0[y, x] == bg:
                        continue
                    g = copy.deepcopy(game)
                    try:
                        r = g.perform_action(
                            ActionInput(id=GA.ACTION6, data={'x': x, 'y': y, 'game_id': 'bfs'}),
                            raw=True)
                        if not r.frame:
                            continue
                        f = np.array(r.frame[-1])
                        if np.sum(f0 != f) > 0:
                            eh = hashlib.md5(f.tobytes()).hexdigest()[:12]
                            if eh not in seen:
                                seen.add(eh)
                                actions.append((6, {'x': x, 'y': y, 'game_id': 'bfs'}))
                                hits.append((x, y))
                    except:
                        pass
            # Neighbor probe
            for hx, hy in hits:
                if time.time() - t0 > timeout + 2:
                    break
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = hx+dx, hy+dy
                    if 0 <= nx < 64 and 0 <= ny < 64 and f0[ny, nx] != bg:
                        g = copy.deepcopy(game)
                        try:
                            r = g.perform_action(
                                ActionInput(id=GA.ACTION6, data={'x': nx, 'y': ny, 'game_id': 'bfs'}),
                                raw=True)
                            if r.frame:
                                f = np.array(r.frame[-1])
                                if np.sum(f0 != f) > 0:
                                    eh = hashlib.md5(f.tobytes()).hexdigest()[:12]
                                    if eh not in seen:
                                        seen.add(eh)
                                        actions.append((6, {'x': nx, 'y': ny, 'game_id': 'bfs'}))
                        except:
                            pass
        return actions

    def _replay_to_level(self, game_cls, level_idx, prior_solutions):
        """Create a fresh game and replay all prior level solutions to reach level_idx.

        The game framework advances levels via next_level() when a level is solved,
        which increments _score and transitions _current_level_index. Simply calling
        set_level(N) followed by RESET triggers full_reset() (because _action_count==0),
        which resets back to L0 with _score=0.

        Instead, we must naturally play through L0..L_{idx-1} so the game state
        (score, level data, on_set_level callbacks) is correct when we start BFS on level_idx.

        Returns (game_instance, last_frame) or (None, None) on failure.
        """
        game = game_cls()
        # Full reset to start clean at L0
        game.perform_action(ActionInput(id=GA.RESET), raw=True)
        game.perform_action(ActionInput(id=GA.RESET), raw=True)

        for lvl in range(level_idx):
            if lvl not in prior_solutions:
                return None, None
            sol = prior_solutions[lvl]
            for act_id, data in sol:
                try:
                    ai = ActionInput(id=GA.from_id(act_id), data=data) if data else ActionInput(id=GA.from_id(act_id))
                    r = game.perform_action(ai, raw=True)
                except Exception:
                    return None, None
            # After replaying level solution, the game should have advanced
            if game._current_level_index <= lvl:
                # The solution didn't actually complete the level
                return None, None

        # Now we should be at the target level
        if game._current_level_index != level_idx:
            return None, None

        # DO NOT call RESET here! After auto-advancing, _action_count == 0,
        # so RESET would trigger full_reset() which destroys _score and goes back to L0.
        # Instead, just get the current frame — the level is already freshly initialized
        # by _really_set_next_level() -> set_level() -> on_set_level().
        f0 = game.get_pixels(0, 0, 64, 64)
        if f0 is None:
            return None, None

        return game, np.array(f0)

    def solve_level(self, game_cls, level_idx, bfs_timeout=120, max_states=100000,
                    prev_solution=None, prior_solutions=None):
        """BFS solve one level. Returns action list or None.

        For level_idx > 0, replays all prior solutions to naturally reach the target level
        before running BFS. This ensures _score and game state are correct.

        Args:
            game_cls: The game class to instantiate.
            level_idx: Which level to solve.
            bfs_timeout: Time budget in seconds for BFS.
            max_states: Max BFS states to explore.
            prev_solution: Solution from previous level (for transfer heuristic).
            prior_solutions: Dict of {level_idx: action_list} for ALL previously solved levels.
        """
        from collections import deque

        if level_idx == 0:
            # L0: simple fresh game + full reset
            game = game_cls()
            game.perform_action(ActionInput(id=GA.RESET), raw=True)
            r0 = game.perform_action(ActionInput(id=GA.RESET), raw=True)
            if not r0.frame:
                return None
            f0 = np.array(r0.frame[-1])
        else:
            # L1+: replay prior solutions to reach this level naturally
            if not prior_solutions:
                return None
            game, f0 = self._replay_to_level(game_cls, level_idx, prior_solutions)
            if game is None:
                return None

        bg = int(np.bincount(f0.flatten(), minlength=16).argmax())

        # Transfer from prev level solution (try it on the correctly-initialized game)
        if prev_solution and level_idx > 0:
            g = copy.deepcopy(game)
            for i, (act_id, data) in enumerate(prev_solution):
                try:
                    ai = ActionInput(id=GA.from_id(act_id), data=data) if data else ActionInput(id=GA.from_id(act_id))
                    r = g.perform_action(ai, raw=True)
                    if r.levels_completed > level_idx or g._current_level_index > level_idx:
                        return prev_solution[:i+1]
                except:
                    break

        # Scan actions
        actions = self._scan_actions(game, f0, bg)

        # Warmup unlock
        if not actions:
            for warmup_id in [a for a in game._available_actions if a <= 4]:
                g_w = copy.deepcopy(game)
                try:
                    g_w.perform_action(ActionInput(id=GA.from_id(warmup_id)), raw=True)
                    f_after = np.array(g_w.get_pixels(0, 0, 64, 64))
                    warmup_actions = self._scan_actions(g_w, f_after, bg)
                    if warmup_actions:
                        game = g_w; f0 = f_after; actions = warmup_actions
                        break
                except:
                    pass

        if not actions:
            return None

        # Adaptive depth
        max_depth = 30
        if len(actions) <= 4: max_depth = 50
        elif len(actions) <= 8: max_depth = 40

        # BFS phase 1: frame hash only
        visited = set()
        queue = deque()
        h0 = self._state_hash(game, f0)
        visited.add(h0)
        queue.append((copy.deepcopy(game), [], 0))
        t0 = time.time()
        explored = 0

        while queue and explored < max_states and (time.time() - t0) < bfs_timeout:
            g, hist, depth = queue.popleft()
            for act_id, data in actions:
                g2 = copy.deepcopy(g)
                try:
                    ai = ActionInput(id=GA.from_id(act_id), data=data) if data else ActionInput(id=GA.from_id(act_id))
                    r = g2.perform_action(ai, raw=True)
                except:
                    continue
                explored += 1
                if not r.frame:
                    continue
                f = np.array(r.frame[-1])
                h = self._state_hash(g2, f)
                if h in visited:
                    continue
                visited.add(h)
                new_hist = hist + [(act_id, data)]
                if r.levels_completed > level_idx or g2._current_level_index > level_idx:
                    return new_hist
                if depth < max_depth:
                    queue.append((g2, new_hist, depth + 1))

        elapsed_first = time.time() - t0

        # BFS phase 2: hidden state retry
        if len(visited) < 50 and elapsed_first < bfs_timeout * 0.8:
            hidden = self._probe_hidden_fields(game, actions)
            if hidden:
                # Re-create game at correct level (replay for L1+)
                if level_idx == 0:
                    game2 = game_cls()
                    game2.perform_action(ActionInput(id=GA.RESET), raw=True)
                    r2 = game2.perform_action(ActionInput(id=GA.RESET), raw=True)
                    f0_2 = np.array(r2.frame[-1]) if r2.frame else f0
                else:
                    game2, f0_2 = self._replay_to_level(game_cls, level_idx, prior_solutions or {})
                    if game2 is None:
                        return None
                visited2 = {self._state_hash(game2, f0_2, hidden)}
                queue2 = deque([(copy.deepcopy(game2), [], 0)])
                t0_2 = time.time()
                remaining = max(30, bfs_timeout - elapsed_first)
                explored2 = 0
                while queue2 and explored2 < max_states and (time.time() - t0_2) < remaining:
                    g, hist, depth = queue2.popleft()
                    for act_id, data in actions:
                        g2 = copy.deepcopy(g)
                        try:
                            ai = ActionInput(id=GA.from_id(act_id), data=data) if data else ActionInput(id=GA.from_id(act_id))
                            r = g2.perform_action(ai, raw=True)
                        except: continue
                        explored2 += 1
                        if not r.frame: continue
                        f = np.array(r.frame[-1])
                        h = self._state_hash(g2, f, hidden)
                        if h in visited2: continue
                        visited2.add(h)
                        if r.levels_completed > level_idx or g2._current_level_index > level_idx:
                            return hist + [(act_id, data)]
                        if depth < max_depth:
                            queue2.append((g2, hist + [(act_id, data)], depth + 1))

        return None

    def evaluate_game(self, env_info, time_budget=120):
        """Evaluate one game with BFS solver. Returns metrics dict."""
        gid = env_info.game_id.split('-')[0]
        game_path, cls_name = self._find_game_source(gid)
        baseline = env_info.baseline_actions
        n_levels = len(baseline)

        result = {
            "game_id": env_info.game_id,
            "title": env_info.title or gid.upper(),
            "tags": env_info.tags or [],
            "n_levels": n_levels,
            "human_baseline": sum(baseline),
            "baseline_per_level": baseline,
        }

        if not game_path:
            result.update({"levels_solved": 0, "rhae": 0.0, "total_actions": 0,
                           "error": "game source not found", "level_details": []})
            return result

        game_cls = self._load_game_class(game_path, cls_name)
        if not game_cls:
            result.update({"levels_solved": 0, "rhae": 0.0, "total_actions": 0,
                           "error": "game class load failed", "level_details": []})
            return result

        t0 = time.time()
        solutions = {}
        level_details = []
        total_actions = 0

        for level_idx in range(n_levels):
            elapsed = time.time() - t0
            remaining = max(10, time_budget - elapsed)
            if remaining < 5:
                break

            # Budget: distribute remaining time across unsolved levels
            levels_left = n_levels - level_idx
            if level_idx == 0:
                # L0 gets up to 40% of total budget
                budget = min(remaining * 0.4, 600)
            else:
                # L1+ share remaining time equally, with a floor
                budget = min(remaining / max(levels_left, 1), 300)
            budget = max(10, budget)

            lt0 = time.time()
            prev_sol = solutions.get(level_idx - 1) if level_idx > 0 else None

            try:
                sol = self.solve_level(game_cls, level_idx, bfs_timeout=budget,
                                       prev_solution=prev_sol,
                                       prior_solutions=solutions)
            except Exception as e:
                sol = None

            lt_elapsed = time.time() - lt0

            if sol:
                solutions[level_idx] = sol
                act_count = len(sol)
                total_actions += act_count
                level_details.append({
                    "level": level_idx,
                    "solved": True,
                    "actions": act_count,
                    "human_baseline": baseline[level_idx],
                    "efficiency": round(baseline[level_idx] / max(act_count, 1), 3),
                    "time": round(lt_elapsed, 1),
                })
            else:
                level_details.append({
                    "level": level_idx,
                    "solved": False,
                    "time": round(lt_elapsed, 1),
                })
                break  # Can't proceed to next level without solving this one

        # Compute RHAE
        level_actions = {d["level"]: d["actions"] for d in level_details if d.get("solved")}
        rhae = compute_rhae(level_actions, baseline)
        levels_solved = len(level_actions)

        result.update({
            "levels_solved": levels_solved,
            "rhae": round(rhae, 6),
            "total_actions": total_actions,
            "elapsed": round(time.time() - t0, 1),
            "level_details": level_details,
        })
        return result


def main():
    parser = argparse.ArgumentParser(description="Local FORGE BFS evaluation")
    parser.add_argument("--games", type=int, default=25, help="Number of games")
    parser.add_argument("--time", type=int, default=120, help="Seconds per game")
    parser.add_argument("--sort", default="difficulty",
                        choices=["difficulty", "alpha"],
                        help="Game ordering")
    args = parser.parse_args()

    print("=" * 70)
    print(f"FORGE BFS Local Evaluation: {args.games} games, {args.time}s/game")
    print("=" * 70)

    arcade = arc_agi.Arcade()
    envs = sorted(arcade.get_environments(), key=lambda e: sum(e.baseline_actions))
    envs = envs[:args.games]

    evaluator = ForgeBFSEvaluator()
    results = []
    t0 = time.time()

    for i, env_info in enumerate(envs):
        gid = env_info.game_id.split('-')[0]
        print(f"\n[{i+1:2d}/{len(envs)}] {(env_info.title or gid.upper()):5s} "
              f"({len(env_info.baseline_actions)} levels, "
              f"human={sum(env_info.baseline_actions)} acts) ...", end=" ", flush=True)

        r = evaluator.evaluate_game(env_info, time_budget=args.time)
        results.append(r)

        if r["levels_solved"] > 0:
            print(f"L{r['levels_solved']}/{r['n_levels']} "
                  f"RHAE={r['rhae']:.4f} "
                  f"acts={r['total_actions']} "
                  f"({r['elapsed']:.0f}s)")
        else:
            err = r.get("error", "unsolved")
            print(f"--- ({err}, {r['elapsed']:.0f}s)")

    elapsed = time.time() - t0

    # Summary
    total_levels = sum(r["levels_solved"] for r in results)
    total_possible = sum(r["n_levels"] for r in results)
    mean_rhae = np.mean([r["rhae"] for r in results])

    print(f"\n{'=' * 70}")
    print(f"RESULTS (approx Kaggle LB)")
    print(f"{'=' * 70}")
    print(f"  Mean RHAE:  {mean_rhae:.6f}  <-- This should match Kaggle LB")
    print(f"  Levels:     {total_levels} / {total_possible}")
    print(f"  Time:       {elapsed:.0f}s")

    print(f"\n  Solved games:")
    for r in sorted(results, key=lambda x: -x["rhae"]):
        if r["levels_solved"] > 0:
            print(f"    {r['title']:5s}: L{r['levels_solved']}/{r['n_levels']} "
                  f"RHAE={r['rhae']:.4f} "
                  f"acts={r['total_actions']} human={r['human_baseline']}")
            for ld in r["level_details"]:
                if ld.get("solved"):
                    eff = ld.get("efficiency", 0)
                    print(f"           L{ld['level']}: {ld['actions']} acts "
                          f"(human={ld['human_baseline']}, eff={eff:.2f}x) "
                          f"{ld['time']:.1f}s")

    print(f"\n  Unsolved games:")
    for r in results:
        if r["levels_solved"] == 0:
            print(f"    {r['title']:5s}: {r.get('error', 'BFS timeout')} "
                  f"({r['n_levels']} levels, human={r['human_baseline']})")

    # Save
    output = {
        "config": {"time_per_game": args.time, "n_games": args.games,
                   "timestamp": int(time.time())},
        "summary": {
            "mean_rhae": round(mean_rhae, 6),
            "total_levels": total_levels,
            "total_possible": total_possible,
            "elapsed": round(elapsed, 1),
        },
        "results": results,
    }
    Path("data").mkdir(exist_ok=True)
    outfile = "data/forge_eval.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
