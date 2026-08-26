"""A10 compressed-budget canary for the (a) budget sentinel -- LOCAL CPU only.

Pre-registered gate requirement (grinder_cracking_design.md §3, "compressed
budgets -- per-game action caps scaled to ~40% of the Qwen observed median
per-game use so that each window's trigger fires >=1/run on >=5 games. Canary
run verifies trigger counts BEFORE the gate seals"). This script REPLAYS the
recorded Qwen duck-harness action streams (runs/kernel_pulls/war_eval_v{1,2,3},
the same traces scripts/ewm_replay_dryrun.py reads) through a COMPRESSED
per-GAME action budget, drives the ACTUAL sentinel logic from
budget_sentinel_patch (_GameBudget -- not a reimplementation), and verifies:

  1. A10 canary (per run): the budget-threshold trigger fires >=1/run on
     >=5 games.

  2. Panel R15 O5 mechanism predicate (deterministic, code-checkable, replaces
     the underpowered "unseen budget deaths halved vs control seeds"):
     "sentinel fired before every budget death" -- for EVERY budget-attributable
     GAME_OVER in the trace (a recorded GAME_OVER at CUMULATIVE game actions
     >= the compressed budget B, i.e. the model overran the budget), there must
     exist a PRIOR sentinel threshold-firing earlier in the SAME game. Any
     death with no prior firing = a predicate violation (fails the mechanism
     prong).

  3. Secondary (binomial fallback): pooled firing counts across games x seeds,
     so a one-sided exact binomial on per-(game,seed) firing remains possible if
     the primary paired prong is underpowered.

  4. R16 Q2 condition 2 (defect-sensitive canary, rl-planning + prog-synthesis):
     counts CROSS-ATTEMPT-WASTE episodes -- (game,seed) units whose TOTAL game
     actions > B with no single level attempt > B/2. These are exactly the
     episodes the v1 per-attempt keying was structurally blind to. For each,
     verifies the v2 game-envelope sentinel (a) fired at all and (b) fired by
     0.9*B cumulative actions (warned inside the envelope). Also asserts >=1
     multi-attempt game is present in the canary set (attempt boundaries are
     tracked as metadata, so the unit bug cannot hide in a single-attempt set).

Compressed budget: per the design, ~40% of the Qwen observed median max-actions
per game (measured ~117-165 across the three seeds -> B=60 by default, override
with --budget). v2 unit (R16 repair): the budget is applied per GAME --
CUMULATIVE actions against B, each threshold fires at most once per game;
attempt boundaries only advance the event-metadata ordinal, exactly as the
live patch does.

Usage:
  uv run python duck_eval/sentinel/compressed_canary.py [--budget 60]
                                                        [--thresholds 0.5,0.75,0.9]
                                                        [--json out.json]
CPU-only, read-only w.r.t. traces, no network, no pushes.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import budget_sentinel_patch as sen  # noqa: E402  the REAL logic under test

SOURCES = {
    "war_eval_v1": REPO / "runs/kernel_pulls/war_eval_v1/artifacts",
    "war_eval_v2": REPO / "runs/kernel_pulls/war_eval_v2/artifacts",
    "war_eval_v3": REPO / "runs/kernel_pulls/war_eval_v3/artifacts",
}

# The 8 Delta-lc-positive / bankable games the (a) window targets
# (grinder_cracking_design.md §4); budget deaths were observed on
# {lp85, sb26, ft09, tu93} specifically (§2 (a)).
TARGET_GAMES = {"ft09", "ka59", "re86", "sc25", "tu93", "sb26", "lp85", "su15"}


def _replay_game(events, budget, thresholds):
    """Drive the REAL _GameBudget over one recorded action stream under a
    compressed per-attempt budget. Returns (firings, deaths, violations).

      firings   : list of (action_num, threshold, attempt)
      deaths    : list of budget-attributable GAME_OVERs
                  (action_num, attempt, consumed) with consumed >= budget
      violations: deaths with NO prior firing in the same attempt
    """
    gb = sen._GameBudget("g", thresholds)
    firings: list[tuple[int, float, int]] = []
    deaths: list[tuple[int, int, int]] = []
    # v2: game-level earliest firing action (for the "fired STRICTLY BEFORE the
    # death" check -- a firing on the same action as the death does not count
    # as a prior warning). Keyed 0 = the whole game.
    first_fire_action: dict[int, int] = {}
    last_level = 1
    attempt_base = 0
    attempt_starts = [0]     # cumulative action count at each attempt start
    total_actions = 0

    for e in events:
        if e.get("type") != "action":
            continue
        action_num = int(e.get("action_num") or 0)
        level = int(e.get("level") or last_level)
        game_over = bool(e.get("game_over"))
        level_up = level != last_level
        total_actions = max(total_actions, action_num)

        # v2: attempt boundaries only advance the metadata ordinal (no re-arm,
        # no clock restart) -- mirror the live patch's boundary detection.
        if level_up and action_num > attempt_base:
            gb.reset_attempt()
            last_level = level
            attempt_base = action_num - 1
            attempt_starts.append(attempt_base)

        # v2: consumed = CUMULATIVE game actions (matches the live patch's
        # game-envelope unit).
        consumed = max(0, action_num)
        cur_attempt = gb.attempt

        for th, _rem in gb.crossings(consumed, budget):
            firings.append((action_num, th, cur_attempt))
            first_fire_action.setdefault(0, action_num)

        # budget-attributable death: a recorded GAME_OVER at cumulative game
        # actions >= the compressed budget (the model overran B).
        if game_over and consumed >= budget:
            deaths.append((action_num, cur_attempt, consumed))

        # A GAME_OVER restart advances the metadata ordinal for subsequent
        # actions -- applied AFTER death attribution (no re-arm in v2).
        if game_over and action_num >= attempt_base:
            gb.reset_attempt()
            attempt_base = action_num
            attempt_starts.append(attempt_base)

    # R15 O5 violation: a budget death with NO threshold firing STRICTLY before
    # it (same game, earlier action_num). A firing on the death action itself
    # is not a prior warning.
    violations = []
    for death_act, attempt, _consumed in deaths:
        ff = first_fire_action.get(0)
        if ff is None or ff >= death_act:
            violations.append((death_act, attempt, _consumed))

    # attempt lengths from boundaries + final tail (mirrors
    # attempt_unit_analysis._analyze_game)
    bounds = attempt_starts + [total_actions]
    attempt_lens = [b2 - b1 for b1, b2 in zip(bounds, bounds[1:]) if b2 > b1]
    stats = {
        "total_actions": total_actions,
        "n_attempts": max(1, len(attempt_lens)),
        "max_attempt_len": max(attempt_lens) if attempt_lens else 0,
        "first_fire": first_fire_action.get(0),
    }
    return firings, deaths, violations, stats


def run(budget: int, thresholds: tuple[float, ...]):
    per_run = {}
    pool_firing_units = 0      # (game,seed) pairs that fired >=1
    pool_total_units = 0       # total (game,seed) pairs seen
    all_violations = 0
    all_deaths = 0

    waste_episodes = []        # R16 Q2 cond 2: total > B, no attempt > B/2
    multi_attempt_units = 0    # (game,seed) units with >1 level attempt

    for src, art in SOURCES.items():
        files = sorted(glob.glob(str(art / "*_events.jsonl")))
        game_rows = {}
        fired_games = []
        run_deaths = 0
        run_violations = 0
        for fp in files:
            gid = Path(fp).name.split("-")[0]
            with open(fp, encoding="utf-8") as f:
                events = [json.loads(ln) for ln in f if ln.strip()]
            firings, deaths, violations, stats = _replay_game(
                events, budget, thresholds)
            game_rows[gid] = {
                "firings": len(firings),
                "fired": len(firings) >= 1,
                "deaths": len(deaths),
                "violations": len(violations),
                "target": gid in TARGET_GAMES,
                "total_actions": stats["total_actions"],
                "n_attempts": stats["n_attempts"],
                "max_attempt_len": stats["max_attempt_len"],
                "first_fire": stats["first_fire"],
            }
            pool_total_units += 1
            if firings:
                pool_firing_units += 1
                fired_games.append(gid)
            if stats["n_attempts"] > 1:
                multi_attempt_units += 1
            if (stats["total_actions"] > budget
                    and stats["max_attempt_len"] <= budget / 2):
                ff = stats["first_fire"]
                waste_episodes.append({
                    "game": gid, "seed": src,
                    "total": stats["total_actions"],
                    "max_attempt": stats["max_attempt_len"],
                    "n_attempts": stats["n_attempts"],
                    "first_fire": ff,
                    "fired": ff is not None,
                    "warned_by_90pct": ff is not None and ff <= 0.9 * budget,
                    "target": gid in TARGET_GAMES,
                })
            run_deaths += len(deaths)
            run_violations += len(violations)
        all_deaths += run_deaths
        all_violations += run_violations
        per_run[src] = {
            "n_games": len(files),
            "n_fired_games": len(fired_games),
            "fired_games": sorted(fired_games),
            "canary_pass": len(fired_games) >= 5,
            "deaths": run_deaths,
            "violations": run_violations,
            "games": game_rows,
        }

    n_waste = len(waste_episodes)
    waste_fired = sum(1 for w in waste_episodes if w["fired"])
    waste_warned = sum(1 for w in waste_episodes if w["warned_by_90pct"])
    return {
        "budget": budget,
        "thresholds": list(thresholds),
        "per_run": per_run,
        "pool_firing_units": pool_firing_units,
        "pool_total_units": pool_total_units,
        "total_budget_deaths": all_deaths,
        "total_predicate_violations": all_violations,
        "canary_pass_all_runs": all(r["canary_pass"] for r in per_run.values()),
        "predicate_pass": all_violations == 0,
        # R16 Q2 condition 2: defect-sensitive counters
        "cross_attempt_waste_episodes": waste_episodes,
        "n_cross_attempt_waste": n_waste,
        "n_cross_attempt_waste_fired": waste_fired,
        "n_cross_attempt_waste_warned_by_90pct": waste_warned,
        "cross_attempt_waste_pass": (n_waste == 0
                                     or waste_warned == n_waste),
        "multi_attempt_units": multi_attempt_units,
        "multi_attempt_game_present": multi_attempt_units >= 1,
    }


def _print(agg):
    b = agg["budget"]
    th = "/".join(f"{int(round(t*100))}%" for t in agg["thresholds"])
    print(f"A10 compressed-budget canary | budget={b}/game | "
          f"thresholds={th}")
    print()
    for src, r in agg["per_run"].items():
        verdict = "PASS" if r["canary_pass"] else "FAIL"
        print(f"[{src}] canary={verdict} "
              f"fired_games={r['n_fired_games']}/25 (>=5 required); "
              f"budget_deaths={r['deaths']} predicate_violations={r['violations']}")
        # per-game firing table (target games first)
        rows = sorted(r["games"].items(),
                      key=lambda kv: (not kv[1]["target"], kv[0]))
        cells = []
        for gid, g in rows:
            if g["target"] or g["firings"]:
                mark = "*" if g["target"] else " "
                cells.append(f"{mark}{gid}:f{g['firings']}/d{g['deaths']}"
                             f"{'/V'+str(g['violations']) if g['violations'] else ''}")
        print("    " + "  ".join(cells))
    print()
    print(f"POOLED firing units (game,seed with >=1 firing): "
          f"{agg['pool_firing_units']}/{agg['pool_total_units']}  "
          f"(one-sided exact-binomial fallback basis)")
    print(f"R15 O5 predicate 'sentinel fired before every budget death': "
          f"total budget deaths={agg['total_budget_deaths']}, "
          f"violations={agg['total_predicate_violations']}  -> "
          f"{'PASS' if agg['predicate_pass'] else 'FAIL'}")
    print(f"A10 canary (>=5 games fire on EVERY run): "
          f"{'PASS' if agg['canary_pass_all_runs'] else 'FAIL'}")
    print(f"R16 Q2 cond-2 cross-attempt-waste (total>{b}, no attempt>{b//2}): "
          f"{agg['n_cross_attempt_waste']} episodes; fired="
          f"{agg['n_cross_attempt_waste_fired']}, warned_by_0.9B="
          f"{agg['n_cross_attempt_waste_warned_by_90pct']}  -> "
          f"{'PASS' if agg['cross_attempt_waste_pass'] else 'FAIL'}")
    for w in agg["cross_attempt_waste_episodes"]:
        mark = "*" if w["target"] else " "
        print(f"    {mark}{w['game']}/{w['seed']}: total={w['total']} "
              f"max_attempt={w['max_attempt']} n_attempts={w['n_attempts']} "
              f"first_fire={w['first_fire']}")
    print(f"multi-attempt game present in canary set: "
          f"{agg['multi_attempt_units']}/{agg['pool_total_units']} units  -> "
          f"{'PASS' if agg['multi_attempt_game_present'] else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=60,
                    help="compressed per-level-attempt action budget "
                         "(~40%% of Qwen median max-actions ~117-165)")
    ap.add_argument("--thresholds", default="0.5,0.75,0.9",
                    help="comma-separated budget-fraction thresholds")
    ap.add_argument("--json", help="write full aggregate to this JSON path")
    args = ap.parse_args()

    thresholds = tuple(sorted(
        float(t) for t in args.thresholds.split(",") if t.strip()))
    agg = run(args.budget, thresholds)
    _print(agg)
    if args.json:
        Path(args.json).write_text(json.dumps(agg, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    # exit non-zero if the canary, the predicate, or the R16 cond-2 checks fail
    ok = (agg["canary_pass_all_runs"] and agg["predicate_pass"]
          and agg["cross_attempt_waste_pass"]
          and agg["multi_attempt_game_present"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
