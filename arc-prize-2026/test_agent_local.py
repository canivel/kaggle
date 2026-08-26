"""Local agent simulation — runs FORGE v20 against real ARC games via the SDK.

This is the PRE-SUBMISSION GATE. Nothing gets pushed to Kaggle until this passes.

Usage:
    uv run python test_agent_local.py                     # quick: 3 games, 30s each
    uv run python test_agent_local.py --games 10 --time 60  # medium
    uv run python test_agent_local.py --full                # all 25, 120s each

Exit code 0 = all checks pass. Exit code 1 = agent crashes or scores 0.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

# ARC-AGI SDK
import arc_agi
from arcengine.enums import GameAction as GA, GameState as GS


def compute_rhae(level_actions, baseline_actions):
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


def run_agent_on_game(agent_module_path, arcade, env_info, time_budget=60, max_actions=50000):
    """Run agent on one game using the SDK. Returns metrics dict."""
    game_id = env_info.game_id
    baseline = env_info.baseline_actions

    # Load agent module fresh each game (isolate state)
    spec = importlib.util.spec_from_file_location("my_agent", agent_module_path)
    mod = importlib.util.module_from_spec(spec)

    # Mock the agents.agent.Agent base class for local testing
    import types
    agents_pkg = types.ModuleType("agents")
    agents_agent = types.ModuleType("agents.agent")

    class LocalAgent:
        MAX_ACTIONS = float("inf")
        _MAX_FRAMES = 10
        def __init__(self, *a, **kw):
            self.game_id = game_id
            self.frames = []
            self.guid = None
            self.is_playback = False
            self.action_counter = 0
            self.arc_env = arcade
        def append_frame(self, f):
            self.frames.append(f)
            if len(self.frames) > self._MAX_FRAMES:
                self.frames = self.frames[-self._MAX_FRAMES:]

    agents_agent.Agent = LocalAgent
    agents_agent.Playback = type("Playback", (), {})
    agents_pkg.agent = agents_agent
    sys.modules["agents"] = agents_pkg
    sys.modules["agents.agent"] = agents_agent

    spec.loader.exec_module(mod)
    MyAgent = mod.MyAgent

    # Create agent instance
    agent = MyAgent()
    agent.arc_env = arcade

    # Create game environment
    env = arcade.make(game_id)
    frame = env.reset()

    total_actions = 0
    levels_completed = 0
    level_actions = {}
    current_level_start = 0
    errors = []
    t0 = time.time()

    while time.time() - t0 < time_budget and total_actions < max_actions:
        if frame.state == GS.WIN:
            break

        try:
            agent.action_counter = total_actions
            action = agent.choose_action(agent.frames + [frame], frame)
        except Exception as e:
            errors.append(f"choose_action error at step {total_actions}: {e}")
            traceback.print_exc()
            action = GA.RESET

        # Check level change
        if frame.levels_completed > levels_completed:
            actions_this_level = total_actions - current_level_start
            level_actions[levels_completed] = actions_this_level
            levels_completed = frame.levels_completed
            current_level_start = total_actions

        # Execute action
        data = None
        if hasattr(action, "data") and action.data:
            data = action.data

        try:
            frame = env.step(action, data=data)
        except Exception as e:
            errors.append(f"env.step error at step {total_actions}: {e}")
            frame = env.step(GA.RESET)

        agent.append_frame(frame)
        total_actions += 1

    elapsed = time.time() - t0
    rhae = compute_rhae(level_actions, baseline)

    return {
        "game_id": game_id,
        "title": env_info.title or game_id.split("-")[0].upper(),
        "n_levels": len(baseline),
        "levels_completed": levels_completed,
        "rhae": round(rhae, 6),
        "total_actions": total_actions,
        "elapsed": round(elapsed, 1),
        "errors": errors,
        "human_baseline": sum(baseline),
    }


def main():
    parser = argparse.ArgumentParser(description="Local ARC agent simulation")
    parser.add_argument("--agent", default="notebooks/forge_agent/forge_v20_valuenet.py",
                        help="Path to agent .py file")
    parser.add_argument("--games", type=int, default=3, help="Number of games")
    parser.add_argument("--time", type=int, default=30, help="Seconds per game")
    parser.add_argument("--full", action="store_true", help="All 25 games, 120s each")
    args = parser.parse_args()

    if args.full:
        args.games = 25
        args.time = 120

    agent_path = args.agent
    if not os.path.exists(agent_path):
        print(f"Agent file not found: {agent_path}")
        sys.exit(1)

    # Step 1: Syntax check
    print("=" * 60)
    print(f"PRE-SUBMISSION GATE: {agent_path}")
    print("=" * 60)
    print("\n[1/4] Syntax check...", end=" ")
    import ast
    with open(agent_path, encoding="utf-8") as f:
        src = f.read()
    try:
        ast.parse(src)
        print("PASS")
    except SyntaxError as e:
        print(f"FAIL: {e}")
        sys.exit(1)

    # Step 2: Import + class instantiation check
    print("[2/4] Import + instantiation check...", end=" ")
    try:
        # Mock agents framework for import check
        import types
        if "agents" not in sys.modules:
            agents_pkg = types.ModuleType("agents")
            agents_agent = types.ModuleType("agents.agent")
            class _MockAgent:
                MAX_ACTIONS = float("inf"); _MAX_FRAMES = 10
                def __init__(self, *a, **kw):
                    self.game_id = "test"; self.frames = []; self.guid = None
                    self.is_playback = False; self.action_counter = 0; self.arc_env = None
                def append_frame(self, f): self.frames.append(f)
            agents_agent.Agent = _MockAgent
            agents_agent.Playback = type("Playback", (), {})
            agents_pkg.agent = agents_agent
            sys.modules["agents"] = agents_pkg
            sys.modules["agents.agent"] = agents_agent
        ns = {}
        exec(compile(src, agent_path, "exec"), ns)
        for cls_name in ["BFSSolver", "MyAgent"]:
            if cls_name not in ns:
                print(f"FAIL: {cls_name} not found")
                sys.exit(1)
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Step 3: ValueNet forward pass (if present)
    print("[3/4] Neural network check...", end=" ")
    try:
        import torch
        if "ValueNet" in ns:
            vn = ns["ValueNet"]()
            frame_oh = torch.zeros(1, 16, 64, 64)
            action_ids = torch.tensor([0])
            out = vn(frame_oh, action_ids)
            assert out.shape == (1,), f"Bad output shape: {out.shape}"
            print(f"PASS (ValueNet: {sum(p.numel() for p in vn.parameters()):,} params)")
            del vn
        elif "ForgeNet" in ns:
            fn = ns["ForgeNet"]()
            out = fn(torch.zeros(1, 26, 64, 64))
            print(f"PASS (ForgeNet: {out.shape})")
            del fn
        else:
            print("SKIP (no neural network found)")
    except Exception as e:
        print(f"FAIL: {e}")
        sys.exit(1)

    # Step 4: Live game simulation
    print(f"[4/4] Live game simulation ({args.games} games, {args.time}s each)...")
    arcade = arc_agi.Arcade()
    envs = sorted(arcade.get_environments(), key=lambda e: sum(e.baseline_actions))
    envs = envs[:args.games]

    results = []
    total_t0 = time.time()
    crashed = False

    for i, env_info in enumerate(envs):
        gid = env_info.game_id.split("-")[0]
        print(f"  [{i+1}/{len(envs)}] {(env_info.title or gid.upper()):5s} ...", end=" ", flush=True)

        try:
            r = run_agent_on_game(agent_path, arcade, env_info, args.time, 50000)
            results.append(r)
            if r["errors"]:
                print(f"ERRORS: {len(r['errors'])} (RHAE={r['rhae']:.4f})")
                for err in r["errors"][:3]:
                    print(f"    {err}")
                crashed = True
            elif r["levels_completed"] > 0:
                print(f"L{r['levels_completed']}/{r['n_levels']} RHAE={r['rhae']:.4f} acts={r['total_actions']} ({r['elapsed']:.0f}s)")
            else:
                print(f"--- no progress ({r['elapsed']:.0f}s)")
        except Exception as e:
            print(f"CRASH: {e}")
            traceback.print_exc()
            crashed = True
            results.append({"game_id": env_info.game_id, "rhae": 0.0, "errors": [str(e)]})

    total_elapsed = time.time() - total_t0
    mean_rhae = np.mean([r["rhae"] for r in results]) if results else 0

    print(f"\n{'=' * 60}")
    print(f"RESULTS: Mean RHAE = {mean_rhae:.6f}")
    print(f"Games: {len(results)} | Levels: {sum(r.get('levels_completed',0) for r in results)}")
    print(f"Time: {total_elapsed:.0f}s | Crashed: {crashed}")
    print(f"{'=' * 60}")

    if crashed:
        print("\nFAILED — DO NOT SUBMIT. Fix errors first.")
        sys.exit(1)
    elif mean_rhae == 0:
        print("\nWARNING — RHAE is 0.00. Agent may not be working correctly.")
        print("Run with --games 10 --time 60 for more thorough test.")
        sys.exit(1)
    else:
        print("\nPASSED — Safe to push and submit.")
        sys.exit(0)


if __name__ == "__main__":
    main()
