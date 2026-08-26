"""Meta-harness evolution loop for ARC-AGI-3.
Uses Claude to propose new agent strategies, evaluates them on games,
keeps the best, repeats. Inspired by KAOS meta-harness architecture.

Usage: uv run python evolve_agents.py --generations 10 --time 120
"""

import sys
sys.path.insert(0, "f:/kaggle/kaos")

import json, time, hashlib, random, os, traceback
from pathlib import Path
from collections import defaultdict

import numpy as np
import subprocess

import arc_agi
from arcengine.enums import GameAction as GA, GameState as GS

ACTION_MAP = {a.value: a for a in GA}
DATA_DIR = Path("data/evolution")
DATA_DIR.mkdir(exist_ok=True, parents=True)


# ─── RHAE Scoring ────────────────────────────────────────────────────
def compute_rhae(level_actions, baseline_actions):
    if not level_actions: return 0.0
    n = len(baseline_actions)
    tw = n * (n + 1) / 2
    s = 0.0
    for l in range(n):
        w = l + 1
        if l in level_actions:
            h = baseline_actions[l]
            a = level_actions[l]
            s += w * min(1.0, h / max(a, 1)) ** 2
    return s / tw


# ─── Run Agent Code on Game ──────────────────────────────────────────
def run_agent(agent_code, env_info, arcade, time_budget=120, max_actions=5000):
    """Execute agent code on one game. Returns metrics dict."""
    ns = {}
    try:
        exec(agent_code, {"__builtins__": __builtins__, "np": np, "random": random,
                           "hashlib": hashlib, "defaultdict": defaultdict,
                           "deque": __import__("collections").deque}, ns)
    except Exception as e:
        return {"rhae": 0.0, "levels": 0, "actions": 0, "error": f"Compile: {e}"}

    if "choose_action" not in ns:
        return {"rhae": 0.0, "levels": 0, "actions": 0, "error": "No choose_action"}

    choose_fn = ns["choose_action"]
    env = arcade.make(env_info.game_id)
    frame = env.reset()
    avail = frame.available_actions

    state = {
        "prev_grid": None, "prev_action": None, "prev_hash": None,
        "visited_hashes": set(), "frame_change_actions": {},
        "tried_actions": {}, "level": 0, "total_actions": 0,
        "actions_this_level": 0, "globally_productive": {},
    }

    total = 0; levels = 0; level_actions = {}; lvl_start = 0; t0 = time.time()

    while time.time() - t0 < time_budget and total < max_actions:
        if frame.state in (GS.NOT_PLAYED, GS.GAME_OVER):
            frame = env.step(GA.RESET)
            state["prev_grid"] = None; state["prev_action"] = None; state["prev_hash"] = None
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

        state["total_actions"] = total
        state["actions_this_level"] = total - lvl_start

        try:
            action_val, data = choose_fn(grid, avail, state)
        except Exception:
            action_val = random.choice(avail); data = None

        if action_val == 6 and data is None:
            nz = np.argwhere(grid != 0)
            if len(nz) > 0:
                i = random.randint(0, len(nz)-1)
                data = {"x": int(nz[i][1]), "y": int(nz[i][0])}
            else:
                data = {"x": random.randint(0,63), "y": random.randint(0,63)}

        state["tried_actions"].setdefault(fh, set()).add(action_val)
        state["prev_grid"] = grid; state["prev_action"] = action_val; state["prev_hash"] = fh

        frame = env.step(ACTION_MAP[action_val], data=data)
        total += 1

    return {
        "rhae": compute_rhae(level_actions, env_info.baseline_actions),
        "levels": levels, "actions": total,
        "level_actions": {str(k): v for k, v in level_actions.items()},
    }


# ─── Evaluate on All Games ───────────────────────────────────────────
def evaluate(agent_code, envs, arcade, time_budget=120, max_actions=5000):
    results = []
    for env_info in envs:
        try:
            r = run_agent(agent_code, env_info, arcade, time_budget, max_actions)
        except Exception as e:
            r = {"rhae": 0.0, "levels": 0, "actions": 0, "error": str(e)}
        r["title"] = env_info.title
        results.append(r)
    mean_rhae = np.mean([r["rhae"] for r in results])
    total_levels = sum(r["levels"] for r in results)
    return {"mean_rhae": mean_rhae, "total_levels": total_levels, "per_game": results}


# ─── Propose New Agent with Claude ───────────────────────────────────
def propose_agent(archive, generation):
    """Use Claude CLI to propose a new agent strategy based on past results."""

    archive_text = ""
    for entry in archive[-10:]:
        archive_text += f"\n--- Agent gen={entry['generation']} RHAE={entry['eval']['mean_rhae']:.6f} levels={entry['eval']['total_levels']} ---\n"
        archive_text += f"Per-game: "
        for g in entry["eval"]["per_game"]:
            if g["levels"] > 0:
                archive_text += f"{g['title']}:L{g['levels']}(RHAE={g['rhae']:.4f},acts={g['actions']}) "
        archive_text += f"\nCode:\n```python\n{entry['code'][:2000]}\n```\n"

    prompt = f"""You are evolving agent strategies for ARC-AGI-3, an interactive grid-based game benchmark.

GAME RULES:
- 64x64 grid, 16 colors (0=background)
- Actions: integers in available_actions list (1-5 = keyboard, 6 = click at x,y)
- Goal: complete levels efficiently (fewer actions = higher RHAE score)
- RHAE = min(1, human_actions/agent_actions)^2, averaged across levels and games
- Agent sees: grid (64x64 int8), available_actions (list of ints), state dict

STATE DICT contains:
- prev_hash, visited_hashes, tried_actions[hash] -> set of tried action vals
- frame_change_actions[hash] -> set of actions that changed frame from that state
- globally_productive[action_val] -> count of times this action changed any frame
- level, total_actions, actions_this_level

PREVIOUS RESULTS (generation {generation}):
{archive_text}

KEY INSIGHTS:
- The best agents try untried actions first (systematic exploration)
- Actions that changed the frame before are more likely to be useful
- For click games (action 6), clicking on colored objects beats random
- R11L can be solved L1 in ~9 actions, LP85 in ~27 actions
- Later levels are harder, human baselines: LP85=[33,22,31,23,33,34,73,173]
- RHAE is squared: 2x human = 0.25, 3x = 0.11, 10x = 0.01
- Every action counts from first exposure (no replay opportunity)
- The agent must be efficient FROM THE START

Write a NEW choose_action function that improves on the best previous result.
Think about what strategies would reduce action count for level completion.
Focus on being MORE EFFICIENT (fewer actions to solve levels).

Return ONLY the Python function inside a code block. Available imports in the execution context: np, random, hashlib, defaultdict, deque.

```python
def choose_action(grid, available_actions, state):
    # Must return (action_value: int, data: dict or None)
    # data must have {{"x": int, "y": int}} for action 6
    ...
```"""

    # Write prompt to file, call claude CLI
    prompt_file = DATA_DIR / f"prompt_gen{generation}.txt"
    prompt_file.write_text(prompt)
    output_file = DATA_DIR / f"response_gen{generation}.txt"

    result = subprocess.run(
        f'claude -p "$(cat {prompt_file.as_posix()})" --output-format text < /dev/null > {output_file.as_posix()} 2>/dev/null',
        shell=True, timeout=180,
    )

    text = output_file.read_text() if output_file.exists() else ""
    if not text.strip():
        # Fallback: try direct invocation
        result = subprocess.run(
            ["claude", "-p", prompt[:3000], "--output-format", "text"],
            capture_output=True, text=True, timeout=180,
            stdin=subprocess.DEVNULL,
        )
        text = result.stdout or result.stderr or ""

    # Extract code block
    if "```python" in text:
        code = text.split("```python")[1].split("```")[0].strip()
    elif "```" in text:
        code = text.split("```")[1].split("```")[0].strip()
    else:
        code = text.strip()

    return code


# ─── Main Evolution Loop ─────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--time", type=int, default=120, help="Seconds per game")
    parser.add_argument("--actions", type=int, default=5000)
    parser.add_argument("--games", type=int, default=10, help="Number of games to evaluate")
    args = parser.parse_args()

    print("=" * 70)
    print(f"ARC-AGI-3 Agent Evolution: {args.generations} generations")
    print(f"  {args.games} games, {args.time}s/game, {args.actions} max actions")
    print("=" * 70)

    # Uses claude CLI (subscription, no API key needed)

    arcade = arc_agi.Arcade()
    envs = sorted(arcade.get_environments(), key=lambda e: sum(e.baseline_actions))[:args.games]

    # Seed population
    seeds = {
        "productive": '''
def choose_action(grid, available_actions, state):
    frame_hash = hashlib.md5(grid.tobytes()).hexdigest()
    tried = state["tried_actions"].get(frame_hash, set())
    untried = [a for a in available_actions if a not in tried]
    if untried:
        scored = [(a, state["globally_productive"].get(a, 0)) for a in untried]
        scored.sort(key=lambda x: -x[1])
        action = scored[0][0]
    else:
        productive = state["frame_change_actions"].get(frame_hash, set())
        prod_avail = [a for a in available_actions if a in productive]
        action = random.choice(prod_avail) if prod_avail else random.choice(available_actions)
    data = None
    if action == 6:
        nonzero = np.argwhere(grid != 0)
        if len(nonzero) > 0:
            colors = np.unique(grid[grid != 0])
            color = random.choice(colors)
            pixels = np.argwhere(grid == color)
            cy, cx = pixels.mean(axis=0).astype(int)
            data = {"x": int(cx), "y": int(cy)}
        else:
            data = {"x": random.randint(0, 63), "y": random.randint(0, 63)}
    return action, data
''',
    }

    archive = []
    best_rhae = 0.0
    best_code = None

    # Evaluate seeds
    print("\n--- Evaluating seeds ---")
    for name, code in seeds.items():
        ev = evaluate(code, envs, arcade, args.time, args.actions)
        archive.append({"generation": 0, "name": name, "code": code, "eval": ev})
        print(f"  {name}: RHAE={ev['mean_rhae']:.6f} levels={ev['total_levels']}")
        if ev["mean_rhae"] > best_rhae:
            best_rhae = ev["mean_rhae"]; best_code = code

    # Evolution loop
    for gen in range(1, args.generations + 1):
        print(f"\n--- Generation {gen}/{args.generations} ---")

        # Propose new agent
        try:
            new_code = propose_agent(archive, gen)
            print(f"  Proposed {len(new_code)} chars of code")
        except Exception as e:
            print(f"  Proposal failed: {e}")
            continue

        # Evaluate
        try:
            ev = evaluate(new_code, envs, arcade, args.time, args.actions)
        except Exception as e:
            print(f"  Evaluation failed: {e}")
            ev = {"mean_rhae": 0.0, "total_levels": 0, "per_game": []}

        archive.append({"generation": gen, "name": f"gen{gen}", "code": new_code, "eval": ev})

        improved = "NEW BEST!" if ev["mean_rhae"] > best_rhae else ""
        print(f"  RHAE={ev['mean_rhae']:.6f} levels={ev['total_levels']} {improved}")

        for g in ev.get("per_game", []):
            if g.get("levels", 0) > 0:
                print(f"    {g['title']}: L{g['levels']} RHAE={g['rhae']:.4f} acts={g['actions']}")

        if ev["mean_rhae"] > best_rhae:
            best_rhae = ev["mean_rhae"]
            best_code = new_code
            # Save best
            with open(DATA_DIR / "best_agent.py", "w") as f:
                f.write(new_code)
            print(f"  Saved best agent (RHAE={best_rhae:.6f})")

    # Final summary
    print(f"\n{'='*70}")
    print(f"EVOLUTION COMPLETE")
    print(f"  Best RHAE: {best_rhae:.6f}")
    print(f"  Target: 0.1")
    print(f"{'='*70}")

    # Save archive
    with open(DATA_DIR / "archive.json", "w") as f:
        json.dump(archive, f, indent=2, default=str)

    if best_code:
        with open(DATA_DIR / "best_agent.py", "w") as f:
            f.write(best_code)
        print(f"\nBest agent saved to {DATA_DIR}/best_agent.py")


if __name__ == "__main__":
    main()
