#!/usr/bin/env python3
"""Cadence instrument -- tokens-per-acting-turn and acting-turns-per-game.

The primary mechanism instruments for the CADENCE arm (see
``learnings/war_room/cadence_prereg_2026-08-22.md``).  They are computed
directly from a TAAF ``benchmark.json`` ``history`` stream, so they can be read
off any past pull as well as any future one -- no new logging required.

DEFINITIONS (validated against the 08-22 BP35 diagnostic, see --validate)
------------------------------------------------------------------------
A TAAF ``history`` entry is one ENVIRONMENT ACTION, not one model turn.  The
harness batches: one analyzer turn may emit several actions in a single
``action([...])`` call.  The harness attributes the whole turn of generated
tokens to the FIRST action of the batch and records ``generated_tokens == 0``
for the remaining actions of that batch.  Hence:

  acting turn      := a history entry with generated_tokens > 0
  turn actions     := that entry plus the following generated_tokens == 0 run
  tokens/turn      := sum(generated_tokens) / n_acting_turns
  actions/turn     := n_actions / n_acting_turns
  actions/game     := number of history entries (all environment actions)

Reconciliation check: the ``solver_note`` ``tokens=N`` total (the harness own
accounting) must be >= the sum of the entry ``generated_tokens``.  The residual
is real and is reported as ``tail_tokens_no_action``: generation that bought no
environment action at all (the give-up turn, plus any yielded turn still in
flight when the clock fired).  A run where the note total is SMALLER than the
attributed sum means the attribution model has broken and the cadence numbers
are not trustworthy -- that is what this check refuses.

Usage
-----
  uv run --no-project python duck_eval/cadence/cadence_instrument.py --validate
  uv run --no-project python duck_eval/cadence/cadence_instrument.py PATH [PATH...]
  ... --games bp35,r11l,sp80 --json out.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]

_TOKENS_RE = re.compile(r"tokens=(\d+)")


# --------------------------------------------------------------------------
# core
# --------------------------------------------------------------------------
def run_cadence(run: dict[str, Any]) -> dict[str, Any]:
    """Cadence metrics for one game-run (one entry of ``game_runs``)."""
    history = run.get("history") or []
    tokens = [int(e.get("generated_tokens") or 0) for e in history]
    n_actions = len(history)
    acting_idx = [i for i, t in enumerate(tokens) if t > 0]
    n_turns = len(acting_idx)
    total_tokens = sum(tokens)

    batch_sizes: list[int] = []
    for pos, i in enumerate(acting_idx):
        nxt = acting_idx[pos + 1] if pos + 1 < len(acting_idx) else n_actions
        batch_sizes.append(nxt - i)

    per_turn_tokens = [tokens[i] for i in acting_idx]

    note = str(run.get("solver_note") or "")
    m = _TOKENS_RE.search(note)
    note_tokens = int(m.group(1)) if m else None
    # The harness total must never be SMALLER than the sum attributed to
    # actions.  It is routinely LARGER: generation after the last recorded
    # action (the give-up turn, and any yielded turn that produced no action
    # before the clock fired) is counted by the solver and has no history
    # entry to land on.  That residual is itself a cadence metric -- tokens
    # that bought no action at all -- so it is exposed, not discarded.
    reconciled = None if note_tokens is None else (note_tokens >= total_tokens)
    tail_tokens = None if note_tokens is None else (note_tokens - total_tokens)

    wall = [float(e.get("wallclock_seconds") or 0.0) for e in history]
    turn_seconds: list[float] = []
    for pos, i in enumerate(acting_idx):
        prev_end = wall[acting_idx[pos - 1]] if pos > 0 else 0.0
        turn_seconds.append(max(0.0, wall[i] - prev_end))

    return {
        "game_id": run.get("game_id"),
        "state": run.get("state"),
        "levels_completed": run.get("levels_completed"),
        "final_score": run.get("final_score"),
        "actions": n_actions,
        "acting_turns": n_turns,
        "generated_tokens": total_tokens,
        "solver_note_tokens": note_tokens,
        "reconciled": reconciled,
        "tail_tokens_no_action": tail_tokens,
        "tokens_per_acting_turn": (total_tokens / n_turns) if n_turns else None,
        "median_tokens_per_acting_turn": (
            statistics.median(per_turn_tokens) if per_turn_tokens else None
        ),
        "max_tokens_in_one_turn": max(per_turn_tokens) if per_turn_tokens else None,
        "actions_per_acting_turn": (n_actions / n_turns) if n_turns else None,
        "median_batch_size": statistics.median(batch_sizes) if batch_sizes else None,
        "last_action_wallclock_s": wall[-1] if wall else None,
        "median_seconds_per_acting_turn": (
            statistics.median(turn_seconds) if turn_seconds else None
        ),
    }


def benchmark_cadence(path: Path) -> dict[str, Any]:
    """Cadence metrics for a whole ``benchmark.json``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    runs = [run_cadence(r) for r in data.get("game_runs", [])]
    ok = [r for r in runs if r["acting_turns"]]
    unrec = [r["game_id"] for r in runs if r["reconciled"] is False]

    def _agg(key: str) -> float | None:
        vals = [r[key] for r in ok if r[key] is not None]
        return statistics.median(vals) if vals else None

    tot_tokens = sum(r["generated_tokens"] for r in runs)
    tot_turns = sum(r["acting_turns"] for r in runs)
    tot_actions = sum(r["actions"] for r in runs)
    return {
        "path": str(path),
        "label": data.get("label"),
        "solver_label": data.get("solver_label"),
        "n_passes": data.get("n_passes"),
        "n_game_runs": len(runs),
        "pooled_tokens_per_acting_turn": (tot_tokens / tot_turns) if tot_turns else None,
        "pooled_actions_per_acting_turn": (tot_actions / tot_turns) if tot_turns else None,
        "median_tokens_per_acting_turn": _agg("tokens_per_acting_turn"),
        "median_acting_turns_per_game": _agg("acting_turns"),
        "median_actions_per_game": _agg("actions"),
        "total_generated_tokens": tot_tokens,
        "tail_tokens_no_action": sum(
            r["tail_tokens_no_action"] or 0 for r in runs
        ),
        "unreconciled_runs": unrec,
        "runs": runs,
    }


def by_game(bench: dict[str, Any], prefixes: list[str]) -> dict[str, dict[str, Any]]:
    """Median cadence per game-prefix (e.g. ``bp35``) across all passes."""
    out: dict[str, dict[str, Any]] = {}
    for p in prefixes:
        rs = [
            r for r in bench["runs"]
            if str(r["game_id"]).startswith(p) and r["acting_turns"]
        ]
        if not rs:
            continue
        out[p] = {
            "n": len(rs),
            "median_acting_turns": statistics.median([r["acting_turns"] for r in rs]),
            "median_actions": statistics.median([r["actions"] for r in rs]),
            "median_actions_per_turn": round(
                statistics.median([r["actions_per_acting_turn"] for r in rs]), 2
            ),
            "median_tokens_per_turn": round(
                statistics.median([r["tokens_per_acting_turn"] for r in rs])
            ),
        }
    return out


# --------------------------------------------------------------------------
# validation against artifacts already on disk
# --------------------------------------------------------------------------
# Numbers copied verbatim from duck_eval/p0/BP35_DIAGNOSTIC_2026-08-22.md sec 2,
# which were computed independently.  The instrument must reproduce them.
_EXPECTED = {
    "runs/tufa_example_run/benchmark.json": {
        "bp35": {"median_acting_turns": 45, "median_actions": 188.5,
                 "median_actions_per_turn": 4.1, "median_tokens_per_turn": 1009},
        "r11l": {"median_acting_turns": 49.5, "median_actions": 54.5,
                 "median_actions_per_turn": 1.0, "median_tokens_per_turn": 1265},
        "sp80": {"median_acting_turns": 46, "median_actions": 182.5,
                 "median_actions_per_turn": 3.6, "median_tokens_per_turn": 1177},
    },
    "runs/kernel_pulls/q38_field_v1/benchmark.json": {
        "bp35": {"median_acting_turns": 13, "median_actions": 55,
                 "median_actions_per_turn": 4.2, "median_tokens_per_turn": 7752},
        "r11l": {"median_acting_turns": 7, "median_actions": 7,
                 "median_actions_per_turn": 1.0, "median_tokens_per_turn": 9577},
        "sp80": {"median_acting_turns": 15, "median_actions": 62,
                 "median_actions_per_turn": 4.1, "median_tokens_per_turn": 6384},
    },
}

_REQUIRED_ARTIFACTS = [
    "runs/tufa_example_run/benchmark.json",
    "runs/kernel_pulls/q38_field_v1/benchmark.json",
    "runs/kernel_pulls/budget_t05_v1/benchmark.json",
    "runs/kernel_pulls/budget_t3_v1/benchmark.json",
    "runs/kernel_pulls/private_base_v1/benchmark.json",
    "runs/kernel_pulls/private_edge1_v2/benchmark.json",
]


def validate(verbose: bool = True) -> int:
    """Re-derive the BP35 diagnostic table from disk.  Returns failure count."""
    failures = 0
    checks = 0
    for rel, expected in _EXPECTED.items():
        path = REPO / rel
        if not path.exists():
            print(f"MISSING  {rel}")
            failures += 1
            continue
        bench = benchmark_cadence(path)
        got = by_game(bench, list(expected))
        for game, exp in expected.items():
            g = got.get(game)
            if g is None:
                print(f"FAIL     {rel} :: {game} absent")
                failures += 1
                continue
            for key, want in exp.items():
                checks += 1
                have = g[key]
                ok = abs(float(have) - float(want)) <= 0.051
                if not ok:
                    failures += 1
                    print(f"FAIL     {rel} :: {game}.{key}  want {want}  got {have}")
                elif verbose:
                    print(f"ok       {rel.split('/')[-2]:16s} {game}.{key:24s} = {have}")
        if bench["unreconciled_runs"]:
            failures += 1
            print(f"FAIL     {rel} :: solver_note reconciliation broke on "
                  f"{bench['unreconciled_runs']}")
        elif verbose:
            print(f"ok       {rel.split('/')[-2]:16s} solver_note reconciliation "
                  f"{bench['n_game_runs']}/{bench['n_game_runs']} runs "
                  f"(tail tokens with no action: {bench['tail_tokens_no_action']})")

    ref = benchmark_cadence(REPO / "runs/tufa_example_run/benchmark.json")
    ours = benchmark_cadence(REPO / "runs/kernel_pulls/q38_field_v1/benchmark.json")
    ratio = ours["pooled_tokens_per_acting_turn"] / ref["pooled_tokens_per_acting_turn"]
    checks += 1
    # POOLED over all 25 games.  The diagnostic's headline "6-8x" is PER GAME on
    # bp35/r11l/sp80 (exactly reproduced above); pooled over the whole board the
    # same artifacts give 3.5x, because the three diagnostic games are at the
    # heavy end.  Both numbers are real; the band below is the pooled one and
    # exists to catch a wholesale change of artifact or attribution rule.
    if not (3.0 <= ratio <= 4.5):
        failures += 1
        print(f"FAIL     pooled tokens/turn ratio {ratio:.2f} outside the 3.0-4.5x band")
    else:
        print(f"ok       pooled tokens/acting-turn  ref "
              f"{ref['pooled_tokens_per_acting_turn']:.0f}"
              f"  ours {ours['pooled_tokens_per_acting_turn']:.0f}"
              f"  ratio {ratio:.2f}x")

    for rel in _REQUIRED_ARTIFACTS:
        p = REPO / rel
        checks += 1
        if not p.exists():
            print(f"SKIP     {rel} (absent)")
            continue
        b = benchmark_cadence(p)
        if b["unreconciled_runs"]:
            failures += 1
            print(f"FAIL     {rel} reconciliation: {b['unreconciled_runs']}")

    print(f"\ncadence_instrument validate: {checks - failures}/{checks} checks passed")
    return failures


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("paths", nargs="*", help="benchmark.json paths (or run dirs)")
    ap.add_argument("--validate", action="store_true",
                    help="re-derive the 08-22 BP35 diagnostic table from disk")
    ap.add_argument("--games", default="", help="comma-separated game prefixes")
    ap.add_argument("--json", default="", help="write full results to this path")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.validate:
        return 1 if validate(verbose=not args.quiet) else 0

    if not args.paths:
        ap.error("give at least one benchmark.json (or --validate)")

    prefixes = [g for g in args.games.split(",") if g]
    results = []
    for raw in args.paths:
        p = Path(raw)
        if p.is_dir():
            p = p / "benchmark.json"
        b = benchmark_cadence(p)
        results.append(b)
        print(f"\n== {b['path']}")
        print(f"   label {b['label']}  solver {b['solver_label']}  "
              f"runs {b['n_game_runs']}")
        print(f"   tokens/acting-turn   pooled {b['pooled_tokens_per_acting_turn']:.0f}"
              f"   median-of-games {b['median_tokens_per_acting_turn']:.0f}")
        print(f"   acting-turns/game    median {b['median_acting_turns_per_game']:.1f}"
              f"   actions/game median {b['median_actions_per_game']:.1f}"
              f"   actions/turn pooled {b['pooled_actions_per_acting_turn']:.2f}")
        if b["unreconciled_runs"]:
            print(f"   !! UNRECONCILED: {b['unreconciled_runs']}")
        if prefixes:
            for g, v in by_game(b, prefixes).items():
                print(f"   {g}: turns {v['median_acting_turns']} "
                      f"actions {v['median_actions']} "
                      f"apt {v['median_actions_per_turn']} "
                      f"tpt {v['median_tokens_per_turn']}")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
