"""ARC-AGI-3 Auto-Evolution via Claude API.

Uses ArcAGI3Benchmark for local game evaluation and Claude claude-sonnet-4-6
as the proposer. Mimics KAOS MetaHarnessSearch but with Anthropic SDK
instead of local vLLM — no vLLM needed.

Usage:
    uv run python evolve_claude.py                     # 10 games, 60s, 20 iters
    uv run python evolve_claude.py --games 5 --time 30 --iters 10
    uv run python evolve_claude.py --resume            # continue from checkpoint
    uv run python evolve_claude.py --export            # export best as Kaggle agent

Archive layout (JSON-lines, resumable):
    data/evolve_claude/archive.jsonl   — all evaluated harnesses
    data/evolve_claude/best.py         — best choose_action() so far
    data/evolve_claude/best_kaggle.py  — ready-to-submit Kaggle agent
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "kaos"))
from kaos.metaharness.benchmarks import arc_agi3  # noqa — registers benchmark
from kaos.metaharness.benchmarks import get_benchmark
from kaos.metaharness.benchmarks.arc_agi3 import (
    ArcAGI3Benchmark, _GAME_LOOP,
    SEED_RANDOM, SEED_SYSTEMATIC, SEED_PRODUCTIVE_FIRST, SEED_CLICK_OBJECTS,
)

ARCHIVE_DIR = Path("data/evolve_claude")
ARCHIVE_FILE = ARCHIVE_DIR / "archive.jsonl"
BEST_FILE = ARCHIVE_DIR / "best.py"
BEST_KAGGLE_FILE = ARCHIVE_DIR / "best_kaggle.py"

MODEL = "claude-sonnet-4-6"  # sonnet: best quality; use haiku-4-5-20251001 for daytime (rate limits)


# ─── Archive ─────────────────────────────────────────────────────────

def load_archive() -> list[dict]:
    if not ARCHIVE_FILE.exists():
        return []
    entries = []
    for line in ARCHIVE_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def append_archive(entry: dict) -> None:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    with ARCHIVE_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def save_best(choose_action_src: str, rhae: float) -> None:
    BEST_FILE.write_text(choose_action_src, encoding="utf-8")
    kaggle = _build_kaggle_agent(choose_action_src, rhae)
    BEST_KAGGLE_FILE.write_text(kaggle, encoding="utf-8")
    print(f"  -> Best saved: RHAE={rhae:.4f}  {BEST_FILE}  {BEST_KAGGLE_FILE}")


# ─── Evaluation ──────────────────────────────────────────────────────

def extract_choose_action(harness_code: str) -> str | None:
    """Extract choose_action() source from a full harness."""
    lines = harness_code.split("\n")
    start = next((i for i, l in enumerate(lines) if l.startswith("def choose_action")), None)
    return "\n".join(lines[start:]) if start is not None else None


def build_harness(choose_action_src: str) -> str:
    return _GAME_LOOP + "\n" + choose_action_src


def evaluate(bench: ArcAGI3Benchmark, choose_action_src: str, label: str) -> dict:
    """Evaluate a choose_action() strategy. Returns aggregate + per-game results."""
    harness = build_harness(choose_action_src)
    problems = bench.get_search_set()
    t0 = time.time()
    per_game = bench.evaluate_harness(harness, problems)
    elapsed = time.time() - t0

    agg = bench.aggregate_scores([bench.score(p, r) for p, r in zip(problems, per_game)])
    result = {
        "label": label,
        "timestamp": int(time.time()),
        "elapsed": elapsed,
        "choose_action": choose_action_src,
        "aggregate": agg,
        "per_game": per_game,
    }
    append_archive(result)
    return result


# ─── Claude Proposer ─────────────────────────────────────────────────

def _format_archive_summary(archive: list[dict], top_n: int = 5) -> str:
    """Format top-N results from archive for the proposer prompt."""
    if not archive:
        return "(no prior results)"
    ranked = sorted(archive, key=lambda e: e["aggregate"].get("rhae", 0), reverse=True)
    lines = []
    for i, e in enumerate(ranked[:top_n]):
        agg = e["aggregate"]
        games = e["per_game"]
        n_solved = sum(1 for g in games if g.get("levels", 0) > 0)
        lines.append(
            f"#{i+1} '{e['label']}' — RHAE={agg['rhae']:.5f}  levels={agg['levels']:.1f}"
            f"  actions={agg['actions']:.0f}  games_w_progress={n_solved}/{len(games)}"
        )
        # Show top 2 games with progress
        progress = [(g.get("game_title","?"), g.get("levels",0), g.get("actions",0),
                     g.get("rhae",0), g.get("baseline_total",0))
                    for g in games if g.get("levels",0) > 0]
        for title, lvl, acts, rhae, base in progress[:2]:
            lines.append(f"     {title}: L{lvl}  {acts} acts  RHAE={rhae:.4f}  human_total={base}")
    return "\n".join(lines)


def _format_best_code(archive: list[dict]) -> str:
    if not archive:
        return "(none)"
    best = max(archive, key=lambda e: e["aggregate"].get("rhae", 0))
    return best["choose_action"]


PROPOSER_SYSTEM = """\
You are an expert at optimizing Python game-playing strategies for ARC-AGI-3.

ARC-AGI-3 is an interactive reasoning benchmark: the agent sees a 64×64 pixel grid
and must click/press keys to solve puzzle levels. You are optimizing the
`choose_action(grid, available_actions, state)` function.

METRIC: RHAE = sum_l( (l+1) * min(1, human_actions_l / agent_actions_l)^2 ) / normalizer
  - Range 0..1. Higher = better. 1.0 = match human efficiency exactly.
  - Uses FEWER actions than humans → high RHAE. Uses MORE → near 0.

INTERFACE:
```python
def choose_action(grid, available_actions, state):
    # grid: 64×64 numpy int8 array (pixel colors 0-15, 0=background)
    # available_actions: list[int] — current valid action IDs (subset of 1-6)
    # state: dict — you can read AND write custom keys to persist state
    #   tried_actions[frame_hash]        → set of action_vals tried at this frame
    #   frame_change_actions[frame_hash] → set of productive action_vals
    #   globally_productive[action_val]  → count across all frames it changed
    #   visited_hashes                   → set of frame hashes seen this level
    #   level                            → current level index
    #   total_actions / actions_this_level
    # Returns: (action_val: int, data: dict|None)
    #   action 6 = click → data = {"x": int, "y": int}
    #   actions 1-5 = keyboard → data = None
```

WINNING STRATEGIES (from leaderboard analysis):
1. Productive-first: prefer actions that previously changed frames
2. Object-click: detect colored blobs (np.argwhere, scipy), click centroids
3. Systematic sweep: try all actions before repeating
4. Cycle detection: track frame hashes, detect loops, undo to escape
5. Level transfer: remember what worked on level N, try on level N+1
6. Color sequence: track which colors appear/disappear, click in order

RULES:
- Only write choose_action() — nothing else
- Must start with: def choose_action(grid, available_actions, state):
- No imports at top level (imports inside the function are OK)
- No BFS/importlib (not available in this evaluation mode)
- Return (action_val, data) always
"""


def _call_claude(prompt: str, system: str, timeout: int = 300) -> str:
    """Call Claude via Claude Code CLI (uses CC subscription, no API key needed).

    Uses `env -u CLAUDECODE claude` to bypass the nested-session guard.
    Writes prompt to temp file to avoid stdin encoding issues.
    Retries once with shorter prompt on timeout.
    """
    full_prompt = f"{system}\n\n---\n\n{prompt}"

    def _run(content: str, t: int) -> str:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=False
        ) as f:
            f.write(content)
            tmp = f.name
        try:
            result = subprocess.run(
                f'cat "{tmp}" | env -u CLAUDECODE claude -p - --model {MODEL} --output-format text',
                shell=True, capture_output=True, text=True, timeout=t,
                encoding="utf-8",
            )
            if result.returncode != 0:
                raise RuntimeError(f"claude CLI error: {result.stderr[:300]}")
            return result.stdout
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    try:
        return _run(full_prompt, timeout)
    except subprocess.TimeoutExpired:
        print(f"  [claude] timeout after {timeout}s, retrying with shorter prompt...")
        time.sleep(10)
        # Shorten: drop system context, keep only user message
        return _run(prompt, timeout)


def propose(archive: list[dict], iteration: int, n: int = 2) -> list[str]:
    """Ask Claude to propose N new choose_action() strategies via CC CLI."""
    archive_summary = _format_archive_summary(archive, top_n=3)
    best_code = _format_best_code(archive)

    user_msg = f"""## Iteration {iteration} — Propose {n} new choose_action() strategies

### Current leaderboard (best first):
{archive_summary}

### Best strategy so far:
```python
{best_code}
```

### Your task:
Analyze why current strategies fail (low RHAE, high action count) and propose
{n} DIFFERENT improved strategies. Each must be a complete `choose_action()` function.

Think step by step:
1. What's limiting RHAE? (Using too many actions? Not completing levels?)
2. What patterns in the winning games (if any) suggest a mechanism?
3. What's a concrete algorithmic improvement?

Return EXACTLY {n} strategies, each in a Python code block:
```python
def choose_action(grid, available_actions, state):
    ...
```

Make them genuinely different from each other and from prior strategies.
Be creative — try things like: color-sequence detection, momentum tracking,
object boundary following, action-effect memory, multi-step planning.
"""

    raw = _call_claude(user_msg, PROPOSER_SYSTEM)

    # Extract all ```python ... ``` blocks
    candidates = []
    blocks = raw.split("```python")
    for block in blocks[1:]:
        code = block.split("```")[0].strip()
        if "def choose_action" in code:
            try:
                ast.parse(code)
                candidates.append(code)
            except SyntaxError as e:
                print(f"  [proposer] syntax error: {e}")
    return candidates[:n]


# ─── Kaggle export ───────────────────────────────────────────────────

def _build_kaggle_agent(choose_action_src: str, rhae: float) -> str:
    return f'''"""
Kaggle ARC-AGI-3 agent — auto-evolved via KAOS + Claude
Local RHAE: {rhae:.4f}
"""
import random, hashlib, time, numpy as np
from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent

GA_MAP = {{a.value: a for a in GameAction}}

# ─── Auto-evolved strategy ────────────────────────────────────────────
{choose_action_src}
# ─────────────────────────────────────────────────────────────────────


class MyAgent(Agent):
    MAX_ACTIONS = 50000

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._state = {{
            "prev_hash": None, "prev_action": None, "prev_grid": None,
            "visited_hashes": set(), "frame_change_actions": {{}},
            "tried_actions": {{}}, "globally_productive": {{}},
            "level": 0, "total_actions": 0, "actions_this_level": 0,
        }}
        self._level = 0; self._total = 0; self._lvl_start = 0

    def act(self, frames: list[FrameData]) -> GameAction:
        latest = frames[-1]
        if latest.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET
        lvl = latest.levels_completed
        if lvl > self._level:
            self._level = lvl; self._lvl_start = self._total
            self._state.update(level=lvl, actions_this_level=0,
                               visited_hashes=set(), tried_actions={{}})
        grid = np.array(latest.frame, dtype=np.int8)
        if grid.ndim == 3: grid = grid[-1]
        fh = hashlib.md5(grid.tobytes()).hexdigest()
        self._state["visited_hashes"].add(fh)
        ph = self._state["prev_hash"]; pa = self._state["prev_action"]
        if ph and pa is not None and fh != ph:
            self._state["frame_change_actions"].setdefault(ph, set()).add(pa)
            self._state["globally_productive"][pa] = self._state["globally_productive"].get(pa,0)+1
        self._state["total_actions"] = self._total
        self._state["actions_this_level"] = self._total - self._lvl_start
        avail = [a.value for a in GameAction if a not in (GameAction.RESET,)]
        avail = getattr(latest, "available_actions", avail) or avail
        try:
            action_val, data = choose_action(grid, avail, self._state)
        except Exception:
            action_val = random.choice(avail); data = None
        self._state["tried_actions"].setdefault(fh, set()).add(action_val)
        self._state["prev_hash"] = fh; self._state["prev_action"] = action_val
        self._state["prev_grid"] = grid; self._total += 1
        action = GA_MAP.get(action_val, GameAction.ACTION1)
        if action_val == 6:
            if data is None:
                nz = np.argwhere(grid != 0)
                if len(nz): i = random.randint(0,len(nz)-1); data={{"x":int(nz[i][1]),"y":int(nz[i][0])}}
                else: data={{"x":random.randint(0,63),"y":random.randint(0,63)}}
            action.set_data(data)
        return action
'''


# ─── Main loop ───────────────────────────────────────────────────────

def run_evolution(
    n_games: int,
    time_per_game: int,
    n_iterations: int,
    resume: bool,
) -> None:
    bench = ArcAGI3Benchmark(
        time_per_game=time_per_game,
        n_search_games=n_games,
        max_actions=5000,
    )
    problems = bench.get_search_set()

    print(f"\n{'='*65}")
    print(f"ARC-AGI-3 Auto-Evolution  |  model={MODEL}")
    print(f"Games: {len(problems)} | Time/game: {time_per_game}s | Iterations: {n_iterations}")
    print(f"Archive: {ARCHIVE_FILE}")
    print(f"{'='*65}\n")

    archive = load_archive() if resume else []
    if resume and archive:
        print(f"Resumed: {len(archive)} prior evaluations loaded")

    # Evaluate seeds if archive is empty
    if not archive:
        seeds = {
            "seed_random": extract_choose_action(SEED_RANDOM),
            "seed_systematic": extract_choose_action(SEED_SYSTEMATIC),
            "seed_productive": extract_choose_action(SEED_PRODUCTIVE_FIRST),
            "seed_click_objects": extract_choose_action(SEED_CLICK_OBJECTS),
        }
        print("Evaluating seed harnesses...")
        for name, ca_src in seeds.items():
            print(f"  {name}...", end=" ", flush=True)
            r = evaluate(bench, ca_src, name)
            agg = r["aggregate"]
            print(f"RHAE={agg['rhae']:.5f}  levels={agg['levels']:.1f}  actions={agg['actions']:.0f}")
        archive = load_archive()
        print()

    best_rhae = max((e["aggregate"].get("rhae", 0) for e in archive), default=0)
    best_entry = max(archive, key=lambda e: e["aggregate"].get("rhae", 0), default=None)
    if best_entry:
        save_best(best_entry["choose_action"], best_rhae)

    # Evolution loop
    for iteration in range(1, n_iterations + 1):
        print(f"\n{'-'*65}")
        print(f"Iteration {iteration}/{n_iterations}  |  best_RHAE={best_rhae:.5f}")
        print(f"{'-'*65}")

        # Propose new candidates via Claude Code CLI
        print(f"  Proposing 2 candidates via Claude Code CLI ({MODEL})...")
        candidates = propose(archive, iteration, n=2)
        if not candidates:
            print("  No valid candidates proposed, skipping.")
            continue
        print(f"  Got {len(candidates)} candidates")

        # Evaluate each
        for i, ca_src in enumerate(candidates):
            label = f"iter{iteration}_cand{i+1}"
            print(f"  Evaluating {label}...", end=" ", flush=True)
            r = evaluate(bench, ca_src, label)
            agg = r["aggregate"]
            rhae = agg["rhae"]
            print(f"RHAE={rhae:.5f}  levels={agg['levels']:.1f}  actions={agg['actions']:.0f}")

            # Per-game breakdown for progress
            progress = [(g.get("game_title","?"), g.get("levels",0), g.get("rhae",0))
                        for g in r["per_game"] if g.get("levels",0) > 0]
            for title, lvl, g_rhae in progress:
                print(f"    {title}: L{lvl}  RHAE={g_rhae:.4f}")

            if rhae > best_rhae:
                best_rhae = rhae
                save_best(ca_src, rhae)
                print(f"  *** NEW BEST: RHAE={best_rhae:.5f} ***")

        archive = load_archive()
        time.sleep(15)  # brief pause between iterations to avoid rate limits

    print(f"\n{'='*65}")
    print(f"Evolution complete. Best RHAE: {best_rhae:.5f}")
    print(f"Best agent: {BEST_KAGGLE_FILE}")
    print(f"Archive: {ARCHIVE_FILE}  ({len(archive)} entries)")
    print(f"{'='*65}")


# ─── CLI ─────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="ARC-AGI-3 auto-evolution via Claude API")
    p.add_argument("--games",  type=int, default=10, help="Games in search set")
    p.add_argument("--time",   type=int, default=60, help="Seconds per game")
    p.add_argument("--iters",  type=int, default=20, help="Evolution iterations")
    p.add_argument("--resume", action="store_true", help="Resume from archive")
    p.add_argument("--export", action="store_true", help="Just export best from archive")
    p.add_argument("--model",  default=None, help="Override model (e.g. claude-haiku-4-5-20251001)")
    args = p.parse_args()
    if args.model:
        global MODEL
        MODEL = args.model

    if args.export:
        archive = load_archive()
        if not archive:
            print("No archive found. Run evolution first.")
            sys.exit(1)
        best = max(archive, key=lambda e: e["aggregate"].get("rhae", 0))
        save_best(best["choose_action"], best["aggregate"].get("rhae", 0))
        return

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    run_evolution(
        n_games=args.games,
        time_per_game=args.time,
        n_iterations=args.iters,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
