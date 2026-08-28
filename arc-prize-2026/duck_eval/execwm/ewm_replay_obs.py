#!/usr/bin/env python
"""
OBSERVATION-LAYER REPLAY for exec-WM v1.  CPU-only, no network, no GPU.

WHY THIS EXISTS.  exec-WM v1 fell back on 31 of 32 in-scope level-instances, and 26 of those
fell back at `no-verified-model`.  Two successive readings of that number were wrong:

  * "the probe budget is too small"  -- closed arithmetically by the probe histogram (the failing
    levels spent their FULL 16-20 budget).
  * "the observation layer is data-STARVED"  -- closed by this instrument's precursor: in the
    zero-candidate games the board changes on 85-100% of move actions, at a median changed-cell
    count (46) indistinguishable from the games that worked (54).  The frames are arriving and
    they are informative.  Nothing is starved.

So the loss is INSIDE the extractor, and this script measures exactly where, by replaying the
retained `*_events.jsonl` through the SHIPPED `exec_wm` code -- not a reimplementation, which
would only measure a second opinion about the bug.

It attributes every move-action transition to one of:
    noop         interior identical after masking
    move         a single translation explains EVERY interior diff cell
    UNEXPLAINED  -- with the reason the translation search rejected it:
                    residual    some diff cells are neither departures nor arrivals
                                (co-occurring change: animation, a second object, an enemy)
                    too-small   the best candidate moved < MIN_SPRITE_CELLS cells
                    no-candidate no (dr,dc) within +-MAX_DELTA explained anything

and then reports, per action, whether GATE A (`len(deltas) == 1` in mine()) would have admitted
a rule, and what a MAJORITY-vote gate would have admitted instead.  Those two columns are the
decision the arm's re-scope turns on.

Usage:
    python duck_eval/execwm/ewm_replay_obs.py                       # all retained games
    python duck_eval/execwm/ewm_replay_obs.py --game tr87 --verbose
    python duck_eval/execwm/ewm_replay_obs.py --json runs/execwm_obs_replay.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from exec_wm import (  # noqa: E402
    MAX_DELTA,
    MIN_SPRITE_CELLS,
    MOVE_ACTIONS,
    HudMask,
    Transition,
    color_counts,
    detect_translation,
    interior_diff,
)

REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_PULL = os.path.join(REPO, "runs", "kernel_pulls", "execwm_v1",
                            "artifacts", "artifacts")


def _why_unexplained(before, after, mask):
    """Re-run the translation search, but record WHICH guard rejected every candidate.

    Mirrors detect_translation exactly; it returns the reason instead of the verdict.  Kept
    beside it deliberately -- if detect_translation changes, this must change with it, and a
    drift shows up as `move` transitions that this function calls unexplained (asserted below).
    """
    diff = interior_diff(before, after, mask)
    if not diff:
        return "noop", 0
    diffset = set(diff)
    rows, cols = len(before), max(len(r) for r in before)
    best_dep, saw_residual, saw_small = 0, False, False
    for dr in range(-MAX_DELTA, MAX_DELTA + 1):
        for dc in range(-MAX_DELTA, MAX_DELTA + 1):
            if dr == 0 and dc == 0:
                continue
            departures, arrivals = set(), set()
            for (r, c) in diff:
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < rows and 0 <= c2 < cols and (r2, c2) in diffset \
                        and after[r2][c2] == before[r][c]:
                    departures.add((r, c))
                    arrivals.add((r2, c2))
            if not departures:
                continue
            best_dep = max(best_dep, len(departures))
            if len(departures) < MIN_SPRITE_CELLS:
                saw_small = True
                continue
            if diffset - departures - arrivals:
                saw_residual = True
                continue
            return "move", len(diffset)
    if saw_residual:
        return "residual", len(diffset)
    if saw_small:
        return "too-small", len(diffset)
    return "no-candidate", len(diffset)


def transitions_from_events(path):
    """Rebuild (action, before, after, level) transitions from a retained events.jsonl.

    The viewer log interleaves board-carrying action events with commentary events; only
    consecutive board frames straddling one action are a transition.
    """
    out, prev, prev_level = [], None, None
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        board = d.get("board")
        if board is None:
            continue
        level = int(d.get("level", 1) or 1)
        act = d.get("action_name")
        if act and prev is not None:
            out.append(Transition(str(act).strip().upper(), prev, board,
                                  prev_level, level))
        prev, prev_level = board, level
    return out


def replay_game(path, verbose=False):
    trs = transitions_from_events(path)
    mask = HudMask()
    for t in trs:                       # mask converges on ALL pairs, as in the live arm
        mask.observe(t.before, t.after)

    per_action = collections.defaultdict(list)
    reasons = collections.Counter()
    for t in trs:
        if t.level_before != t.level_after or t.action == "RESET":
            continue
        if t.action not in MOVE_ACTIONS:
            continue
        kind, info = detect_translation(t.before, t.after, mask)
        why, ncells = _why_unexplained(t.before, t.after, mask)
        # drift guard: the shipped classifier and the reason-tracer must agree on `move`
        assert (kind == "move") == (why == "move"), \
            "reason-tracer drifted from detect_translation on %s" % os.path.basename(path)
        reasons[why] += 1
        delta = tuple(info[:2]) if kind == "move" else None
        per_action[t.action].append(delta)

    rows = []
    for action in sorted(per_action):
        deltas = [d for d in per_action[action] if d is not None]
        distinct = set(deltas)
        gate_a = len(distinct) == 1                    # what the arm shipped
        maj = None
        if deltas:
            maj, cnt = collections.Counter(deltas).most_common(1)[0]
            gate_maj = cnt / len(deltas) >= 0.60 and cnt >= 3
        else:
            gate_maj = False
        # The proposed repair is the UNION, not a replacement: majority alone would DROP a
        # unanimous rule seen only once or twice (re86 loses one that way).  Nothing the
        # shipped gate admits may be lost by a repair whose whole claim is "admit more".
        gate_union = gate_a or gate_maj
        rows.append(dict(action=action, n=len(per_action[action]), moves=len(deltas),
                         distinct=len(distinct), gate_a=gate_a, gate_majority=gate_maj,
                         gate_union=gate_union,
                         majority_delta=maj,
                         top_share=(collections.Counter(deltas).most_common(1)[0][1]
                                    / len(deltas)) if deltas else 0.0))
    return reasons, rows, len(trs)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pull", default=DEFAULT_PULL)
    p.add_argument("--game", default=None, help="substring filter, e.g. tr87")
    p.add_argument("--json", default=None)
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)

    files = sorted(f for f in os.listdir(a.pull) if f.endswith("_events.jsonl"))
    if a.game:
        files = [f for f in files if a.game in f]
    if not files:
        print("no events.jsonl under %s" % a.pull)
        return 2

    pooled = collections.Counter()
    recovered = []
    out = {}
    print("%-8s %6s %6s %6s %9s %9s %11s   %s"
          % ("game", "trans", "noop", "move", "residual", "too-small", "no-cand", "rules A -> union"))
    for f in files:
        gid = f.split("-")[0]
        reasons, rows, ntr = replay_game(os.path.join(a.pull, f), a.verbose)
        pooled.update(reasons)
        na = sum(1 for r in rows if r["gate_a"])
        nm = sum(1 for r in rows if r["gate_union"])
        if nm > na:
            recovered.append((gid, na, nm))
        out[gid] = {"reasons": dict(reasons), "actions": rows, "transitions": ntr}
        print("%-8s %6d %6d %6d %9d %9d %11d   %d -> %d"
              % (gid, ntr, reasons["noop"], reasons["move"], reasons["residual"],
                 reasons["too-small"], reasons["no-candidate"], na, nm))
        if a.verbose:
            for r in rows:
                print("      %-8s n=%-4d moves=%-4d distinct=%-3d top_share=%.2f  A=%s maj=%s %s"
                      % (r["action"], r["n"], r["moves"], r["distinct"], r["top_share"],
                         r["gate_a"], r["gate_majority"], r["majority_delta"]))

    tot = sum(pooled.values())
    print("\nPOOLED move-action transitions: %d" % tot)
    for k in ("noop", "move", "residual", "too-small", "no-candidate"):
        if tot:
            print("   %-13s %6d  %5.1f%%" % (k, pooled[k], 100 * pooled[k] / tot))
    print("\nGATE A (shipped, len(deltas)==1) vs UNION(A or majority>=60%%,n>=3):")
    if recovered:
        for gid, na, nm in recovered:
            print("   %-8s %d -> %d rules recovered" % (gid, na, nm))
    else:
        print("   no game gains a rule from the majority gate.")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump({"pooled": dict(pooled), "games": out}, fh, indent=1)
        print("\nwrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
