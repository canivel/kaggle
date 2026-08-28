"""Sub-classify the exec-WM GATE B `residual` bucket.

The 08-27 replay verdict (exp 61) measured that GATE B -- detect_translation's
`if diffset - departures - arrivals: continue`, i.e. "a move must explain EVERY
interior diff cell" -- discards 641 of 2394 move-action transitions (26.8%), the
single largest loss channel. That verdict named the next measurement and called
it free: "the `residual` bucket is counted but not sub-classified (animation vs.
second object vs. enemy)".

This is that measurement. It decides GATE B's REPAIR, which is not one design:

  CO-MOVER      leftover is itself explained by ONE other translation
                -> the repair is multi-delta support (two objects, two deltas)
  BYSTANDER     leftover is small and spatially disjoint from the sprite
                -> the repair is a bounded-leftover tolerance (HUD tick, counter,
                   a second object that changed without translating)
  OVERLAP       leftover touches the sprite's own footprint
                -> occlusion / partial redraw; a tolerance would mis-mine here
  DIFFUSE       leftover is large or many-component
                -> animation or global repaint; NOT recoverable by either repair

Reuses the SHIPPED primitives (interior_diff, detect_translation, MAX_DELTA,
MIN_SPRITE_CELLS). A reimplementation would only measure a second opinion about
the bug -- the 08-27 lesson, kept.
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
    HudMask,
    detect_translation,
    interior_diff,
)
from ewm_replay_obs import (  # noqa: E402
    DEFAULT_PULL,
    MOVE_ACTIONS,
    _why_unexplained,
    transitions_from_events,
)

# A leftover this small is a tick/counter rather than an object, in the sense
# that a bounded-leftover tolerance would admit it without admitting a scene.
BYSTANDER_MAX_CELLS = 6


def _components(cells):
    """4-connected components of a cell set."""
    todo, comps = set(cells), []
    while todo:
        seed = todo.pop()
        comp, stack = {seed}, [seed]
        while stack:
            r, c = stack.pop()
            for nr, nc in ((r+1, c), (r-1, c), (r, c+1), (r, c-1)):
                if (nr, nc) in todo:
                    todo.discard((nr, nc))
                    comp.add((nr, nc))
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _best_residual_candidate(before, after, mask):
    """The candidate GATE B rejected: most departures, >= MIN_SPRITE_CELLS, leftover non-empty.

    Returns (departures, arrivals, leftover, diffset) or None when this transition
    is not a GATE B rejection at all.
    """
    diff = interior_diff(before, after, mask)
    if not diff:
        return None
    diffset = set(diff)
    rows, cols = len(before), max(len(r) for r in before)
    best = None
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
            if len(departures) < MIN_SPRITE_CELLS:
                continue
            leftover = diffset - departures - arrivals
            if not leftover:
                return None  # a clean move exists; not a GATE B rejection
            if best is None or len(departures) > len(best[0]):
                best = (departures, arrivals, leftover, diffset)
    return best


def _explained_by_one_translation(leftover, before, after, diffset):
    """Is the leftover itself a single translation? (the CO-MOVER test)"""
    rows, cols = len(before), max(len(r) for r in before)
    for dr in range(-MAX_DELTA, MAX_DELTA + 1):
        for dc in range(-MAX_DELTA, MAX_DELTA + 1):
            if dr == 0 and dc == 0:
                continue
            dep, arr = set(), set()
            for (r, c) in leftover:
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < rows and 0 <= c2 < cols and (r2, c2) in diffset \
                        and after[r2][c2] == before[r][c]:
                    dep.add((r, c))
                    arr.add((r2, c2))
            if dep and not (leftover - dep - arr):
                return (dr, dc), len(dep)
    return None


def classify(before, after, mask):
    got = _best_residual_candidate(before, after, mask)
    if got is None:
        return None
    departures, arrivals, leftover, diffset = got
    sprite = departures | arrivals

    co = _explained_by_one_translation(leftover, before, after, diffset)
    if co is not None:
        return "co-mover", len(leftover), co[0]

    comps = _components(leftover)
    touches = any(
        any((r+dr, c+dc) in sprite
            for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)))
        for comp in comps for (r, c) in comp
    )
    if len(leftover) <= BYSTANDER_MAX_CELLS and len(comps) <= 2 and not touches:
        return "bystander", len(leftover), None
    if touches:
        return "overlap", len(leftover), None
    return "diffuse", len(leftover), None


def main(argv=None):
    global BYSTANDER_MAX_CELLS
    p = argparse.ArgumentParser()
    p.add_argument("--pull", default=DEFAULT_PULL)
    p.add_argument("--json", default=None)
    p.add_argument("--bystander-max", type=int, default=BYSTANDER_MAX_CELLS,
                   help="sensitivity knob for the bystander/diffuse split")
    a = p.parse_args(argv)
    BYSTANDER_MAX_CELLS = a.bystander_max

    files = sorted(f for f in os.listdir(a.pull) if f.endswith("_events.jsonl"))
    pooled = collections.Counter()
    per_game = {}
    cells = collections.defaultdict(list)
    n_residual = 0

    print("%-8s %9s %9s %10s %9s %9s" % ("game", "residual", "co-mover",
                                         "bystander", "overlap", "diffuse"))
    for f in files:
        gid = f.split("-")[0]
        trs = transitions_from_events(os.path.join(a.pull, f))
        # Converge the mask over ALL pairs first, exactly as replay_game and the
        # live arm do. A fresh mask is a different instrument and silently
        # reclassifies every transition.
        mask = HudMask()
        for t in trs:
            mask.observe(t.before, t.after)
        local = collections.Counter()
        for t in trs:
            if t.level_before != t.level_after or t.action == "RESET":
                continue
            if t.action not in MOVE_ACTIONS:
                continue
            # `residual` is the TRACER's label; detect_translation only ever
            # returns noop/move/unexplained.
            why, _ncells = _why_unexplained(t.before, t.after, mask)
            if why != "residual":
                continue
            n_residual += 1
            got = classify(t.before, t.after, mask)
            if got is None:
                local["unclassified"] += 1
                continue
            label, nleft, _delta = got
            local[label] += 1
            cells[label].append(nleft)
        pooled.update(local)
        per_game[gid] = dict(local)
        if sum(local.values()):
            print("%-8s %9d %9d %10d %9d %9d"
                  % (gid, sum(local.values()), local["co-mover"],
                     local["bystander"], local["overlap"], local["diffuse"]))

    tot = sum(pooled.values())
    print("\nPOOLED GATE B rejections sub-classified: %d "
          "(the tracer called %d of them residual)" % (tot, n_residual))
    for k in ("co-mover", "bystander", "overlap", "diffuse", "unclassified"):
        if tot and pooled[k]:
            med = sorted(cells[k])[len(cells[k]) // 2] if cells[k] else 0
            print("   %-13s %6d  %5.1f%%   median leftover %d cells"
                  % (k, pooled[k], 100 * pooled[k] / tot, med))

    rec = pooled["co-mover"] + pooled["bystander"]
    print("\nRECOVERABLE by a GATE B repair: %d of %d (%.1f%%)"
          % (rec, tot, 100 * rec / tot if tot else 0))
    print("   co-mover  -> needs MULTI-DELTA support (two objects, two deltas)")
    print("   bystander -> needs a BOUNDED-LEFTOVER tolerance (<= %d cells, disjoint)"
          % BYSTANDER_MAX_CELLS)
    print("   overlap/diffuse are NOT recoverable by either and must stay rejected.")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump({"pooled": dict(pooled), "games": per_game,
                       "n_residual": n_residual,
                       "bystander_max_cells": BYSTANDER_MAX_CELLS}, fh, indent=1)
        print("\nwrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
