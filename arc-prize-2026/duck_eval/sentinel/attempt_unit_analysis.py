"""R16 Q2 discharge: per-attempt vs per-game-envelope denomination analysis.

Answers, from the SAME recorded certified-seed streams the B=150 canary used
(runs/kernel_pulls/war_eval_v{1,2,3}), the exact quantities the R16 reviewers
conditioned their SENTINEL_BUDGET=150 acceptance on:

  llm-agents N11 / Q1:
    (i)  fraction of budget-attributable GAME_OVERs (B=150, per-attempt unit)
         occurring in games with >1 level attempt;
    (ii) median per-game action count already consumed when the fatal attempt
         begins;
    (iii) per-game attempt-count distributions on the (a) target games.
  rl-planning (conditional accept):
    (iv) cross-attempt-waste episodes: games whose TOTAL actions > B with no
         single attempt > B/2 (waste the per-attempt keying structurally
         cannot warn about).
  prog-synthesis N5: the multi-attempt-game check = (i)+(ii) above.
  Decision metric (envelope-lateness): for every (game,seed) whose TOTAL
         actions >= B (i.e. the game would overrun a per-GAME envelope of B),
         did any per-attempt firing land at cumulative action <= 0.9*B? If per-
         attempt keying still warns before 90% of the per-game envelope in
         nearly all envelope-crossing games, the single-attempt approximation
         holds and B=150 may seal as denominated; otherwise re-key cumulative.

Usage: uv run python duck_eval/sentinel/attempt_unit_analysis.py
       [--budget 150] [--json runs/sentinel_attempt_unit_b150.json]
CPU-only, read-only w.r.t. traces, no network, no pushes.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
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
TARGET_GAMES = {"ft09", "ka59", "re86", "sc25", "tu93", "sb26", "lp85", "su15"}
THRESHOLDS = (0.5, 0.75, 0.9)


def _analyze_game(events, budget):
    """Replay one recorded stream with the REAL _GameBudget (per-attempt unit),
    tracking attempt boundaries + cumulative game actions.  Mirrors
    compressed_canary._replay_game's attempt bookkeeping exactly."""
    gb = sen._GameBudget("g", THRESHOLDS)
    last_level = 1
    attempt_base = 0
    attempt_starts = [0]        # cumulative action count at each attempt start
    firings = []                # (action_num, threshold, attempt)
    deaths = []                 # dicts, see below
    total_actions = 0

    for e in events:
        if e.get("type") != "action":
            continue
        action_num = int(e.get("action_num") or 0)
        level = int(e.get("level") or last_level)
        game_over = bool(e.get("game_over"))
        level_up = level != last_level
        total_actions = max(total_actions, action_num)

        if level_up and action_num > attempt_base:
            gb.reset_attempt()
            last_level = level
            attempt_base = action_num - 1
            attempt_starts.append(attempt_base)

        consumed = max(0, action_num - attempt_base)
        cur_attempt = gb.attempt
        for th, _rem in gb.crossings(consumed, budget):
            firings.append((action_num, th, cur_attempt))

        if game_over and consumed >= budget:
            deaths.append({
                "action_num": action_num,
                "attempt_index": cur_attempt,
                "consumed_in_attempt": consumed,
                "consumed_before_attempt": attempt_base,
                "n_attempts_at_death": len(attempt_starts),
            })

        if game_over and action_num >= attempt_base:
            gb.reset_attempt()
            attempt_base = action_num
            attempt_starts.append(attempt_base)

    # attempt lengths from boundaries + final tail
    bounds = attempt_starts + [total_actions]
    attempt_lens = [b2 - b1 for b1, b2 in zip(bounds, bounds[1:]) if b2 > b1]
    return {
        "total_actions": total_actions,
        "n_attempts": max(1, len(attempt_lens)),
        "attempt_lens": attempt_lens,
        "max_attempt_len": max(attempt_lens) if attempt_lens else 0,
        "firings": firings,
        "deaths": deaths,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=150)
    ap.add_argument("--json", default=str(REPO / "runs/sentinel_attempt_unit_b150.json"))
    args = ap.parse_args()
    B = args.budget

    rows = []          # one per (game, seed)
    all_deaths = []
    for src, art in SOURCES.items():
        for fp in sorted(glob.glob(str(art / "*_events.jsonl"))):
            gid = Path(fp).name.split("-")[0]
            with open(fp, encoding="utf-8") as f:
                events = [json.loads(ln) for ln in f if ln.strip()]
            g = _analyze_game(events, B)
            g.update({"game": gid, "seed": src, "target": gid in TARGET_GAMES})
            rows.append(g)
            for d in g["deaths"]:
                d.update({"game": gid, "seed": src, "target": gid in TARGET_GAMES})
                all_deaths.append(d)

    # (i) fraction of budget deaths in games with >1 attempt (at death time)
    multi = [d for d in all_deaths if d["n_attempts_at_death"] > 1]
    # (ii) median per-game actions consumed when the fatal attempt began
    consumed_before = [d["consumed_before_attempt"] for d in all_deaths]
    # (iii) attempt-count distribution, target games
    tdist = {}
    for r in rows:
        if r["target"]:
            tdist.setdefault(r["game"], []).append(r["n_attempts"])
    # (iv) cross-attempt-waste episodes (rl-planning definition)
    waste = [
        {"game": r["game"], "seed": r["seed"], "total": r["total_actions"],
         "max_attempt": r["max_attempt_len"], "n_attempts": r["n_attempts"]}
        for r in rows
        if r["total_actions"] > B and r["max_attempt_len"] <= B / 2
    ]
    # decision metric: envelope-crossing games where first firing came at
    # cumulative action <= 0.9*B (i.e. per-attempt keying still warned within
    # the per-game envelope)
    env_cross = [r for r in rows if r["total_actions"] >= B]
    early_warned = [
        r for r in env_cross
        if r["firings"] and min(a for a, _t, _at in r["firings"]) <= 0.9 * B
    ]
    late_or_never = [
        {"game": r["game"], "seed": r["seed"], "total": r["total_actions"],
         "first_fire": (min(a for a, _t, _at in r["firings"])
                        if r["firings"] else None),
         "n_attempts": r["n_attempts"]}
        for r in env_cross if r not in early_warned
    ]

    out = {
        "budget": B,
        "n_game_seed_units": len(rows),
        "n_budget_deaths": len(all_deaths),
        "deaths_in_multi_attempt_games": len(multi),
        "frac_deaths_multi_attempt": (len(multi) / len(all_deaths)) if all_deaths else None,
        "median_consumed_before_fatal_attempt": (
            statistics.median(consumed_before) if consumed_before else None),
        "consumed_before_fatal_attempt_all": sorted(consumed_before),
        "target_game_attempt_counts": {k: sorted(v) for k, v in sorted(tdist.items())},
        "cross_attempt_waste_episodes": waste,
        "n_cross_attempt_waste": len(waste),
        "envelope_crossing_units": len(env_cross),
        "envelope_crossing_warned_by_90pct": len(early_warned),
        "envelope_late_or_never": late_or_never,
        "deaths_detail": all_deaths,
    }
    Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"attempt-unit analysis | B={B} | {len(rows)} (game,seed) units")
    print(f"(i)   budget deaths: {len(all_deaths)}; in multi-attempt games: "
          f"{len(multi)} ({out['frac_deaths_multi_attempt']})")
    print(f"(ii)  median actions consumed before fatal attempt: "
          f"{out['median_consumed_before_fatal_attempt']} "
          f"(all: {out['consumed_before_fatal_attempt_all']})")
    print(f"(iii) target-game attempt counts (per seed): ")
    for k, v in out["target_game_attempt_counts"].items():
        print(f"        {k}: {v}")
    print(f"(iv)  cross-attempt-waste episodes (total>{B}, no attempt>{B//2}): "
          f"{len(waste)}")
    for w in waste:
        print(f"        {w}")
    print(f"(dec) envelope-crossing units: {len(env_cross)}; warned by "
          f"0.9*B cumulative: {len(early_warned)}; late/never: {len(late_or_never)}")
    for r in late_or_never:
        print(f"        {r}")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
