"""Local 25-game evaluation harness for forge_agent variants.

Runs an agent class against each public ARC-AGI-3 game by directly invoking
the game's perform_action() (bypassing the HTTP gateway). Bounded by an
action budget per game. Records per-level action counts and terminal state.

Use for A/B comparison between agent variants — public LB single-submission
noise is ~0.09 wide (v27=0.33 vs v28=0.24 on identical code), so we can't
detect real improvements via Kaggle submission. This harness gives a
reproducible per-game signal.

Usage:
    uv run python eval_harness.py --agent notebooks/forge_agent/v39_agent.py --budget 2000
    uv run python eval_harness.py --agent notebooks/forge_agent/v39_agent.py \
        --vs notebooks/forge_agent/v44_agent.py --budget 2000 --out runs/eval_v39_vs_v44.json
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import logging
import os
import sys
import time
import traceback
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent
ENV_DIR = ROOT / "kaggle-data" / "environment_files"
AGENTS_PKG_DIR = ROOT / "kaggle-data" / "ARC-AGI-3-Agents"

# Make agents.agent importable for forge_agent variants (they do `from agents.agent import Agent`)
sys.path.insert(0, str(AGENTS_PKG_DIR))
_pkg = types.ModuleType("agents"); _pkg.__path__ = [str(AGENTS_PKG_DIR / "agents")]
sys.modules.setdefault("agents", _pkg)
_spec = importlib.util.spec_from_file_location("agents.agent", str(AGENTS_PKG_DIR / "agents" / "agent.py"))
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
sys.modules["agents.agent"] = _mod; _pkg.agent = _mod

from arcengine import GameAction, GameState, ActionInput  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def discover_games() -> list[tuple[str, str, Path]]:
    """Return list of (gid, guid, game_py_path)."""
    out = []
    for gdir in sorted(p for p in ENV_DIR.iterdir() if p.is_dir()):
        for guid_dir in sorted(p for p in gdir.iterdir() if p.is_dir()):
            game_py = guid_dir / f"{gdir.name}.py"
            if game_py.exists():
                out.append((gdir.name, guid_dir.name, game_py))
                break  # one variant per game
    return out


def load_baseline_actions(game_py: Path) -> list[int]:
    """Load per-level human-action counts (`baseline_actions`) from the game's
    metadata.json. Returns [] if missing — caller should handle."""
    meta = game_py.parent / "metadata.json"
    if not meta.exists():
        return []
    try:
        d = json.loads(meta.read_text(encoding="utf-8"))
        return list(d.get("baseline_actions", []))
    except Exception:
        return []


def compute_rhae(level_actions: list[tuple[int, int]], baseline_actions: list[int]) -> tuple[float, list[dict]]:
    """Compute per-level RHAE = (human/agent)^2 capped at 1.0, and the per-game
    aggregate. `level_actions` is the per-level agent-action counts; index 0 =
    actions spent on level 0, etc. `baseline_actions[i]` = human actions on level i.

    Per Kaggle RHAE: only count SOLVED levels (where the agent moved past that
    level). Unsolved levels contribute 0. Aggregate = sum across solved levels.

    Returns: (total_rhae, per_level_details_list)
    """
    # level_actions is [(level_idx, n_actions_spent_on_that_level), ...]
    la = dict(level_actions)
    # Solved levels = levels i such that level i+1 was reached (i.e. agent
    # progressed past level i). Equivalently: max_level_reached - 1, OR all
    # levels in la EXCEPT the last (the one we died on without finishing).
    if not la:
        return 0.0, []
    max_reached = max(la.keys())
    details = []
    total = 0.0
    for i in range(max_reached + 1):
        agent_acts = la.get(i, 0)
        human_acts = baseline_actions[i] if i < len(baseline_actions) else None
        solved = i < max_reached  # only solved if we moved past this level
        if not solved or agent_acts <= 0 or human_acts is None:
            rhae_i = 0.0
        else:
            ratio = human_acts / agent_acts
            rhae_i = min(1.0, ratio * ratio)
        details.append({"level": i, "agent_actions": agent_acts,
                        "human_actions": human_acts, "solved": solved, "rhae": rhae_i})
        total += rhae_i
    return total, details


def load_agent_class(agent_py: Path, bfs_cap_s: int = 30):
    """Load the MyAgent class from a forge_agent variant .py file. Hard-cap any
    BFSSolver instance's bfs_timeout to bfs_cap_s so per-level BFS can't run
    longer than the harness budget regardless of the agent's adaptive logic."""
    spec = importlib.util.spec_from_file_location(f"agent_{agent_py.stem}", str(agent_py))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "BFSSolver"):
        BFS = mod.BFSSolver
        orig_init = BFS.__init__
        orig_setattr = BFS.__setattr__
        cap = bfs_cap_s
        def patched_init(self, *a, **kw):
            if "bfs_timeout" in kw:
                kw["bfs_timeout"] = min(kw["bfs_timeout"], cap)
            orig_init(self, *a, **kw)
            object.__setattr__(self, "bfs_timeout", min(self.bfs_timeout, cap))
        def patched_setattr(self, name, value):
            if name == "bfs_timeout" and isinstance(value, (int, float)):
                value = min(value, cap)
            orig_setattr(self, name, value)
        BFS.__init__ = patched_init
        BFS.__setattr__ = patched_setattr
    return getattr(mod, "MyAgent")


def load_game_class(game_py: Path):
    """Load the game class via importlib. The agent convention: class name is
    the gid capitalized (ft09 -> Ft09); otherwise pick the first ARCBaseGame-
    derived class."""
    import re
    src = game_py.read_text()
    spec = importlib.util.spec_from_file_location(f"game_mod_{game_py.stem}", str(game_py))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Conventional: ft09 -> Ft09
    conv = game_py.stem[0].upper() + game_py.stem[1:]
    if hasattr(mod, conv):
        return getattr(mod, conv)
    # Fallback: scan for any class extending ARCBaseGame
    for m in re.finditer(r"^class\s+(\w+)\s*\(ARCBaseGame", src, re.MULTILINE):
        return getattr(mod, m.group(1))
    raise RuntimeError(f"No ARCBaseGame class found in {game_py}")


class _FakeEnvInfo:
    def __init__(self, local_dir: str):
        self.local_dir = local_dir


class _FakeArcEnv:
    """Minimal stand-in for the Arcade env. Only environment_info.local_dir is read."""
    def __init__(self, local_dir: str):
        self.environment_info = _FakeEnvInfo(local_dir)


def run_one_game(AgentCls, gid: str, guid: str, game_py: Path, budget: int,
                 bfs_per_level_s: int = 30, wall_s_cap: float | None = None) -> dict:
    """Run a single agent instance against a single game, capped by action budget
    and (optionally) wall-clock seconds (whichever hits first)."""
    game_id = f"{gid}-{guid}"
    baseline = load_baseline_actions(game_py)
    rec: dict = {"game_id": game_id, "gid": gid, "guid": guid, "ok": False, "err": None,
                 "actions": 0, "levels_completed": 0, "terminal_state": None,
                 "level_actions": [], "wall_s": 0.0,
                 "baseline_actions": baseline, "rhae": 0.0, "rhae_details": []}
    t0 = time.time()
    try:
        GameCls = load_game_class(game_py)
        game = GameCls()
        # Two RESETs match gen_trajectories.py / agent framework pattern
        game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        first = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        if not first or not first.frame:
            rec["err"] = "no initial frame"
            return rec

        # Construct agent
        agent = AgentCls(card_id=None, game_id=game_id, agent_name="eval",
                         ROOT_URL="http://local", record=False,
                         arc_env=_FakeArcEnv(str(game_py.parent)))
        agent.frames = [first]
        agent.action_counter = 0
        agent.cl = -1  # force level-init path
        # BFS-per-level cap is enforced by load_agent_class via BFSSolver patch.

        cur_level = first.levels_completed or 0
        # `level_actions` accumulates ALL actions per level (across attempts/resets).
        # `first_clear_acts[L]` snapshots the actions-on-L count at the MOMENT we
        # first cleared L. RHAE is then computed against first_clear_acts (fairest
        # comparison to human's single-attempt count).
        level_actions = {cur_level: 0}
        first_clear_acts: dict[int, int] = {}
        terminal = None
        latest = first

        silent = not os.environ.get("EVAL_NO_SILENCE")
        for step in range(budget):
            if wall_s_cap is not None and (time.time() - t0) >= wall_s_cap:
                break
            # Build args matching base Agent: choose_action(frames, lf)
            # Silence the agent's noisy stderr (it does traceback.print_exc in
            # its outer except — useful in production, spammy in eval).
            try:
                if silent:
                    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                        action = agent.choose_action(agent.frames, latest)
                else:
                    action = agent.choose_action(agent.frames, latest)
            except Exception as e:
                rec["err"] = f"choose_action: {e!r}"
                traceback.print_exc()
                break
            data = action.action_data.model_dump() if hasattr(action, "action_data") else {}
            data = {k: v for k, v in data.items() if k != "reasoning"}
            # CRITICAL: record level we WERE on (where action was decided) BEFORE perform_action
            # advances the state. RHAE counts actions spent on each level, so the action that
            # CLEARS level 0 counts against level 0 (not the new level 1).
            level_before = latest.levels_completed or 0
            try:
                latest = game.perform_action(ActionInput(id=action, data=data), raw=True)
            except Exception as e:
                rec["err"] = f"perform_action: {e!r}"
                break
            if not latest or not latest.frame:
                rec["err"] = "empty frame after action"
                break
            agent.frames.append(latest)
            if len(agent.frames) > 12:
                agent.frames = agent.frames[-12:]
            agent.action_counter += 1
            rec["actions"] += 1
            level_actions[level_before] = level_actions.get(level_before, 0) + 1
            lvl = latest.levels_completed or 0
            if lvl > cur_level:
                # Snapshot the first-clear count for each newly-cleared level
                for L in range(cur_level, lvl):
                    first_clear_acts.setdefault(L, level_actions.get(L, 0))
                cur_level = lvl

            st = latest.state
            if st in (GameState.WIN, GameState.GAME_OVER):
                terminal = st.name if hasattr(st, "name") else str(st)
                if st == GameState.GAME_OVER:
                    # Match agent framework: RESET on GAME_OVER until budget out
                    try:
                        latest = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                        latest = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                        agent.frames.append(latest)
                    except Exception:
                        break
                else:  # WIN
                    break
        rec["levels_completed"] = cur_level
        rec["level_actions"] = sorted(level_actions.items())
        rec["first_clear_acts"] = sorted(first_clear_acts.items())
        rec["terminal_state"] = terminal or "BUDGET"
        rec["ok"] = rec["err"] is None
        # Compute RHAE against first-clear counts (fair single-attempt comparison
        # vs human's baseline_actions). Use the cur_level to mark which levels
        # count as solved (= for which we have first_clear_acts entries).
        rhae_input = [(L, n) for L, n in sorted(first_clear_acts.items())]
        # Add a sentinel for the unsolved "current" level so compute_rhae knows
        # the max_reached boundary.
        if cur_level not in first_clear_acts:
            rhae_input.append((cur_level, level_actions.get(cur_level, 0)))
        rec["rhae"], rec["rhae_details"] = compute_rhae(rhae_input, baseline)
    except Exception as e:
        rec["err"] = repr(e)[:300]
        traceback.print_exc()
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def run_agent_sweep(agent_py: Path, games: list[tuple[str, str, Path]], budget: int,
                    bfs_per_level_s: int = 30, wall_s_cap: float | None = None) -> dict:
    """Run an agent across all given games. Returns dict keyed by game_id."""
    AgentCls = load_agent_class(agent_py, bfs_cap_s=bfs_per_level_s)
    wallinfo = f"  wall_s_cap={wall_s_cap}" if wall_s_cap else ""
    print(f"\n=== Sweeping {agent_py.name} over {len(games)} games (budget={budget}, bfs/lvl={bfs_per_level_s}s{wallinfo}) ===")
    results = {}
    t0 = time.time()
    for i, (gid, guid, gpy) in enumerate(games):
        print(f"[{i+1}/{len(games)}] {gid} ... ", end="", flush=True)
        rec = run_one_game(AgentCls, gid, guid, gpy, budget,
                           bfs_per_level_s=bfs_per_level_s, wall_s_cap=wall_s_cap)
        results[rec["game_id"]] = rec
        if rec["ok"]:
            print(f"L={rec['levels_completed']} RHAE={rec['rhae']:.3f} acts={rec['actions']} {rec['terminal_state']} {rec['wall_s']}s")
        else:
            print(f"ERR: {rec['err']}")
    total = time.time() - t0
    print(f"\nTotal: {total:.0f}s")
    return results


def summarize(label: str, results: dict) -> None:
    total_levels = sum(r["levels_completed"] for r in results.values() if r["ok"])
    games_with_l = sum(1 for r in results.values() if r["ok"] and r["levels_completed"] > 0)
    wins = sum(1 for r in results.values() if r.get("terminal_state") == "WIN")
    errors = sum(1 for r in results.values() if not r["ok"])
    total_rhae = sum(r.get("rhae", 0.0) for r in results.values() if r["ok"])
    # Kaggle reports mean across the 25 public games (as a percentage)
    n_games = len(results) or 1
    mean_rhae_pct = (total_rhae / n_games) * 100
    print(f"  {label}: RHAE={total_rhae:.3f}  mean%={mean_rhae_pct:.2f}  total_levels={total_levels}  games_with_progress={games_with_l}  WIN={wins}  errors={errors}")


def compare(a_label: str, a_results: dict, b_label: str, b_results: dict) -> None:
    print(f"\n=== A/B comparison (RHAE primary) ===")
    print(f"{'game':14}  {a_label+' RHAE':>15}  {b_label+' RHAE':>15}  {'L_a':>4}  {'L_b':>4}  diff_rhae")
    a_total = b_total = 0
    wins_a = wins_b = 0
    for gid in sorted(set(a_results) | set(b_results)):
        ar = a_results.get(gid, {}).get("rhae", 0.0)
        br = b_results.get(gid, {}).get("rhae", 0.0)
        al = a_results.get(gid, {}).get("levels_completed", 0)
        bl = b_results.get(gid, {}).get("levels_completed", 0)
        a_total += ar; b_total += br
        if ar > br + 1e-6: wins_a += 1
        elif br > ar + 1e-6: wins_b += 1
        marker = "" if abs(ar - br) < 1e-6 else (" *" if ar < br else "")
        print(f"{gid:14}  {ar:>15.3f}  {br:>15.3f}  {al:>4}  {bl:>4}  {br - ar:+.3f}{marker}")
    n = max(len(a_results), len(b_results), 1)
    print(f"{'TOTAL':14}  {a_total:>15.3f}  {b_total:>15.3f}  diff: {b_total - a_total:+.3f}")
    print(f"Mean RHAE%:    {a_label}={100*a_total/n:.2f}  {b_label}={100*b_total/n:.2f}")
    print(f"Per-game wins: {a_label}={wins_a}  {b_label}={wins_b}  ties={n - wins_a - wins_b}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, help="agent .py to evaluate")
    ap.add_argument("--vs", default=None, help="optional second agent .py for A/B")
    ap.add_argument("--budget", type=int, default=2000, help="action budget per game")
    ap.add_argument("--bfs-s", type=int, default=30, help="BFS time budget per level (sec)")
    ap.add_argument("--wall-s", type=float, default=None, help="wall-clock cap per game (sec). If set, terminates a game when either action budget or wall-clock cap hits. Use for long-budget mode (e.g. --wall-s 1800 --budget 20000).")
    ap.add_argument("--games", default=None, help="comma-sep game gids to limit sweep")
    ap.add_argument("--out", default=None, help="path to save JSON results")
    ap.add_argument("--no-silence", action="store_true", help="show agent stderr (diag)")
    args = ap.parse_args()

    games = discover_games()
    if args.games:
        wanted = set(args.games.split(","))
        games = [g for g in games if g[0] in wanted]
    print(f"Discovered {len(games)} games")

    a_results = run_agent_sweep(Path(args.agent), games, args.budget, bfs_per_level_s=args.bfs_s, wall_s_cap=args.wall_s)
    summarize(Path(args.agent).name, a_results)

    payload = {"agent": args.agent, "budget": args.budget, "results_a": a_results}

    if args.vs:
        b_results = run_agent_sweep(Path(args.vs), games, args.budget, bfs_per_level_s=args.bfs_s, wall_s_cap=args.wall_s)
        summarize(Path(args.vs).name, b_results)
        compare(Path(args.agent).stem, a_results, Path(args.vs).stem, b_results)
        payload["vs"] = args.vs
        payload["results_b"] = b_results

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
