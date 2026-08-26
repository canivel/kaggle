# ============================================================================
# duck_eval/scoring_oracle.py  --  ARC-AGI-3 offline leaderboard scoring oracle
# ============================================================================
"""Deterministic, offline, zero-cost oracle for the ARC-AGI-3 leaderboard score.

WHAT THIS IS
------------
A thin wrapper over the *real* shipped scorer
(`arc_agi.scorecard.EnvironmentScoreCalculator`) plus a 25-game baseline atlas
(`duck_eval/scoring_atlas.json`, read straight from the local
`environment_files/*/*/metadata.json` kernel lineage).  It lets you compute the
exact LB score of a candidate play offline, pre-build, to 1e-9 -- no API key,
no network, no GPU.  The formula is verified against both the notebook
`busyaprime/arc-agi-3-offline-atlas-and-scoring` (discussion 728299) and the
harness-reported scores in runs/kernel_pulls/sentinel_eval_v1 (see
runs/atlas_oracle/validation.md).

THE FORMULA (as implemented by the shipped EnvironmentScoreCalculator)
---------------------------------------------------------------------
Per level i (1-indexed), with human baseline b_i and actions taken a_i:
    level_score = min((b_i / a_i)**2 * 100, 100.0)   if the level was completed
    level_score = 0.0                                if not completed
A game's score is the LEVEL-NUMBER-WEIGHTED mean of its level scores:
    game_score = sum(level_score_i * i) / sum(i)      over all *attempted* levels
The final leaderboard number is the plain mean of game scores over all games.

This weighting is why finishing 4 of 6 levels (all perfect) scores 47.62, not
66.7: late levels carry more weight, and unfinished late levels (score 0) drag
the weighted mean down hard.  Build for finishing DEEP, not for shaving actions
on early levels you already clear.

NOTE ON THE 100 vs 115 CAP.  The notebook's hand-formula caps each level at
115 and adds an explicit completion-weight cap `min(tot/w, maxw/w*100)`.  The
*shipped* code caps each level at 100.0 and has no separate completion cap
(it falls out of scoring incomplete levels as 0).  On every realistic play the
two agree to 1e-9 (a completed level with a_i >= b_i can never exceed 100, and
scoring incomplete levels as 0 is exactly the completion cap).  This oracle
mirrors the SHIPPED code, which is the source of truth for the LB.

USAGE -- scoring an EWM (or any) candidate offline, pre-build
------------------------------------------------------------
    from duck_eval.scoring_oracle import score_game, score_run, ATLAS

    # a candidate: which levels it completes and the actions it spends on each
    #   completed[i] = did the agent finish level i+1?
    #   actions[i]   = actions the agent spent on level i+1
    s = score_game(
        "cd82-fb555c5d",
        actions=[41, 8, 30, 21, 19, 17],       # <- e.g. exactly human budget
        completed=[True, True, True, True, True, True],
    )                                          # -> 100.0 (perfect human parity)

    # baselines are auto-filled from the atlas; pass them explicitly to override.
    # actions/completed must be FULL length (one entry per level, unplayed = 0):
    s = score_game(
        "cd82-fb555c5d",
        actions=[123, 0, 0, 0, 0, 0],          # spent 123 on lvl1, quit
        completed=[False, False, False, False, False, False],
    )                                          # -> 0.0 (no level finished)

    # aggregate a whole 25-game run into the LB mean:
    lb = score_run({
        "cd82-fb555c5d": ([123, 0, 0, 0, 0, 0], [False] * 6),
        "r11l-495a7899": ([8, 88, 0, 0, 0, 0],
                          [True, False, False, False, False, False]),
        # ... one entry per game ...
    })
    print(lb["leaderboard_score"], lb["per_game"])

The oracle prefers the real `arc_agi.scorecard` when importable; if the package
is absent (e.g. a stripped dev box) it falls back to a pure-python
re-implementation that is byte-identical on the validated cases.

BASELINE DRIFT -- READ THIS BEFORE TRUSTING THE ATLAS
-----------------------------------------------------
Game bundles rotate by guid and their baseline_actions are version-specific.
Validation found that 20 of 25 games in the sentinel_eval_v1 run (2026-07-22)
were played on DIFFERENT guids than the local environment_files/ dirs, with
different baselines.  With the atlas baselines the oracle mismatched the harness
on 7 games; with each run's own benchmark.json `base_actions_per_level` it
matched all 25 to 0.00e+00.  So: to score a *specific* run, pass baselines from
`load_baselines_from_benchmark(<that run>/benchmark.json)`.  The atlas is the
best offline default only when no run-specific baseline is on hand (e.g. scoring
a hypothetical EWM candidate against the currently-bundled game version).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

# --- locate the atlas relative to this file (mount-path agnostic) -----------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ATLAS_PATH = os.path.join(_HERE, "scoring_atlas.json")


def _load_atlas() -> Dict[str, dict]:
    with open(_ATLAS_PATH, "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    return blob["games"]


ATLAS: Dict[str, dict] = _load_atlas()

# short-id -> full game_id index (e.g. "cd82" -> "cd82-fb555c5d")
_SHORT2FULL: Dict[str, str] = {v["short_id"]: gid for gid, v in ATLAS.items()}


# --- real-scorer detection --------------------------------------------------
try:
    from arc_agi.scorecard import EnvironmentScoreCalculator  # type: ignore

    _HAVE_REAL = True
except Exception:  # pragma: no cover - fallback path
    EnvironmentScoreCalculator = None  # type: ignore
    _HAVE_REAL = False


def using_real_scorer() -> bool:
    """True if the shipped arc_agi.scorecard is wrapped; False if the pure-python
    fallback is in use.  Both are validated to agree to 1e-9."""
    return _HAVE_REAL


# ---------------------------------------------------------------------------
def resolve_game_id(game: str) -> str:
    """Accept a full game_id ('cd82-fb555c5d') or a short id ('cd82')."""
    if game in ATLAS:
        return game
    if game in _SHORT2FULL:
        return _SHORT2FULL[game]
    # tolerate a versioned id whose short prefix we know
    short = game.split("-")[0]
    if short in _SHORT2FULL:
        return _SHORT2FULL[short]
    raise KeyError(f"unknown game {game!r}; not in atlas ({len(ATLAS)} games)")


def baseline_actions(game: str) -> List[int]:
    """Per-level human baseline actions for a game, from the atlas.

    WARNING: game bundles rotate by guid and baselines are version-specific.
    For scoring a *specific* run, prefer load_baselines_from_benchmark() on that
    run's benchmark.json (authoritative per-run).  The atlas value is the best
    offline default only when no run-specific baseline is available.
    """
    return list(ATLAS[resolve_game_id(game)]["baseline_actions"])


def load_baselines_from_benchmark(benchmark_json_path: str) -> Dict[str, List[int]]:
    """Authoritative per-run baselines: read `base_actions_per_level` for every
    game out of a harness benchmark.json.  These are the exact baselines the run
    was scored against and should override the atlas when re-scoring that run
    (the atlas can drift from a run's bundle version -- see scoring_atlas.json
    _meta.WARNING_baseline_drift)."""
    with open(benchmark_json_path, "r", encoding="utf-8") as fh:
        bench = json.load(fh)
    return {
        g["game_id"]: list(g["base_actions_per_level"])
        for g in bench.get("game_runs", [])
        if g.get("base_actions_per_level")
    }


# --- pure-python fallback (mirrors EnvironmentScoreCalculator exactly) -------
def _score_game_pure(
    baselines: Sequence[int],
    actions: Sequence[int],
    completed: Sequence[bool],
) -> float:
    total = 0.0
    weight = 0
    for i, (b, a, c) in enumerate(zip(baselines, actions, completed), start=1):
        if c and a > 0:
            s = min((b / a) ** 2 * 100.0, 100.0)
        else:
            s = 0.0
        total += s * i
        weight += i
    return total / weight if weight else 0.0


def _score_game_real(
    baselines: Sequence[int],
    actions: Sequence[int],
    completed: Sequence[bool],
) -> float:
    calc = EnvironmentScoreCalculator(id="oracle")  # type: ignore[operator]
    for i, (b, a, c) in enumerate(zip(baselines, actions, completed), start=1):
        calc.add_level(
            level_index=i,
            completed=bool(c),
            actions_taken=int(a),
            baseline_actions=int(b),
        )
    return calc.to_score().score


# ---------------------------------------------------------------------------
def score_game(
    game: str,
    actions: Sequence[int],
    completed: Sequence[bool],
    baselines: Optional[Sequence[int]] = None,
) -> float:
    """Exact LB score for a single game.

    Args:
        game: full game_id or short id (atlas lookup fills baselines).
        actions: actions spent on each level (1-indexed by position).
        completed: whether each level was completed.
        baselines: per-level human baselines; defaults to the atlas value.

    Returns:
        The game score in [0, 100] (LB scale), matching the shipped scorer.
    """
    if baselines is None:
        baselines = baseline_actions(game)
    if not (len(baselines) == len(actions) == len(completed)):
        raise ValueError(
            f"length mismatch for {game!r}: "
            f"baselines={len(baselines)} actions={len(actions)} "
            f"completed={len(completed)}"
        )
    if _HAVE_REAL:
        return _score_game_real(baselines, actions, completed)
    return _score_game_pure(baselines, actions, completed)


def score_run(
    games: Dict[str, Tuple[Sequence[int], Sequence[bool]]],
    n_total_games: Optional[int] = None,
) -> dict:
    """Aggregate per-game (actions, completed) into the leaderboard mean.

    Args:
        games: map game_id -> (actions, completed) for each played game.
        n_total_games: divisor for the LB mean.  Defaults to len(games).
            Pass the full roster size (25) if you only scored a subset and
            want the true LB mean (unscored games contribute 0).

    Returns:
        {"leaderboard_score": float, "per_game": {game_id: score}}
    """
    per_game: Dict[str, float] = {}
    for game, (actions, completed) in games.items():
        per_game[resolve_game_id(game)] = score_game(game, actions, completed)
    denom = n_total_games if n_total_games is not None else len(per_game)
    lb = (sum(per_game.values()) / denom) if denom else 0.0
    return {"leaderboard_score": lb, "per_game": per_game}


if __name__ == "__main__":
    # tiny self-check on two atlas games with known harness scores
    print("real scorer:", using_real_scorer())
    print(
        "r11l 1/6:",
        score_game(
            "r11l-495a7899",
            actions=[8, 88, 0, 0, 0, 0],
            completed=[True, False, False, False, False, False],
        ),
    )
    print(
        "cd82 human-parity:",
        score_game(
            "cd82-fb555c5d",
            actions=baseline_actions("cd82"),
            completed=[True] * 6,
        ),
    )
