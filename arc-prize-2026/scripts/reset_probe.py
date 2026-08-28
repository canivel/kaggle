"""OFFLINE RESET PROBE — is RESET actually a perfect undo, on OUR instrument?

WHY
    `thtennant/arc3-duck-v27` reports, from its own archive:
        states with no way back to the level opening   2210/2454 = 90.1%
        RESET restored the level opening exactly        306/306  = 100.0%
        board-changing moves ever walked back            31/1310 = 2.4%
    and separately that RESET appears NOWHERE in the prompt text -- the model
    meets it as a bare token in `valid_actions` and chooses it deliberately
    3 times in ~1980 actions.

    Our own certified pull agrees on the use rate: `affordance_audit.py` on
    runs/kernel_pulls/private_edge2_v3 gives RESET 7/1555 = 0.45%.

    A 100%-reliable escape hatch that is never advertised, in a 90% one-way
    world, is the strongest candidate arm on the board. Before it earns a slot
    it needs the two RATES replicated on OUR instrument -- another team's
    numbers are evidence, not our measurement.

WHAT THIS MEASURES
    (1) RESET FIDELITY. From a level opening, take k random valid actions, then
        RESET, then compare the board to the opening byte-for-byte. This is the
        load-bearing claim: if RESET is not a perfect undo, advertising it is
        worthless.
    (2) ONE-WAYNESS PROXY. Over the same random walks, how often does the board
        return to the opening on its own? The complement bounds how much a
        perfect undo is worth.

    NO LLM, NO GPU, NO KAGGLE SLOT. Pure engine driving, seconds per game.

USAGE
    scripts/reset_probe.py                    # all 25 official games
    scripts/reset_probe.py --games 5 --walks 8 --depth 6
    scripts/reset_probe.py --json out.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

_SELF = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != _SELF]

ROOT = Path(__file__).resolve().parents[1]


def board_of(state) -> str:
    """A hashable, exact rendering of the visible board.

    frame may hold numpy arrays, so hash the bytes rather than JSON-encode.
    """
    raw = state.raw
    frame = getattr(raw, "frame", None)
    if frame is None:
        return ""
    h = hashlib.sha256()
    def feed(o):
        if hasattr(o, "tobytes"):
            h.update(o.tobytes()); h.update(str(getattr(o, "shape", "")).encode())
        elif isinstance(o, (list, tuple)):
            for x in o: feed(x)
        elif isinstance(o, dict):
            for k in sorted(o): h.update(str(k).encode()); feed(o[k])
        else:
            h.update(repr(o).encode())
    feed(frame)
    return h.hexdigest()


def probe(n_games: int, walks: int, depth: int, seed: int) -> dict:
    import arcengine
    import re_arc
    import taaf.game_api

    # arcengine.GameAction(3) RAISES "3 is not a valid GameAction" even though
    # member ACTION3 has value 3 -- the enum's value lookup is broken (custom
    # metaclass). Build the map from the members themselves and never call the
    # constructor.
    BY_VALUE = {m.value: m for m in arcengine.GameAction}

    rng = random.Random(seed)
    official = sorted(re_arc.list_game_ids(datasets=["train", "eval"],
                                           include_tags="official"))[:n_games]

    reset_ok = reset_tried = 0
    returned_naturally = walk_count = 0
    changed_moves = 0
    per_game = []

    for gid in official:
        api = taaf.game_api.GameAPI(env_name=gid)
        session = taaf.game.RunSession()
        state = api.start_game(session=session)
        opening = board_of(state)
        g_ok = g_tried = g_ret = 0

        for _ in range(walks):
            # walk away from the opening
            moved = False
            for _ in range(depth):
                # available_actions are GameAction ints; 0 is RESET, and
                # ACTION6 needs x/y data (feedback_arc_set_data_bug).
                valid = [int(a) for a in (state.available_actions or []) if int(a) != 0]
                if not valid:
                    break
                act = rng.choice(valid)
                before = board_of(state)
                ga = BY_VALUE[int(act)]
                data = ({"x": rng.randrange(64), "y": rng.randrange(64)}
                        if ga.name == "ACTION6" else {})
                try:
                    state = api._execute_action(arcengine.ActionInput(id=ga, data=data))
                except Exception:  # noqa: BLE001
                    break
                if board_of(state) != before:
                    changed_moves += 1
                    moved = True
            walk_count += 1
            if board_of(state) == opening:
                returned_naturally += 1
                g_ret += 1
            if not moved:
                continue

            # now the load-bearing test: does RESET restore the opening exactly?
            try:
                state = api._execute_action(
                    arcengine.ActionInput(id=BY_VALUE[0], data={}))
            except Exception:  # noqa: BLE001
                continue
            reset_tried += 1
            g_tried += 1
            if board_of(state) == opening:
                reset_ok += 1
                g_ok += 1

        per_game.append({"game": gid, "reset_ok": g_ok, "reset_tried": g_tried,
                         "returned_naturally": g_ret, "walks": walks})
        try:
            api._finish_game()
        except Exception:  # noqa: BLE001
            pass

    return {"games": len(official), "walks_per_game": walks, "depth": depth,
            "seed": seed, "reset_ok": reset_ok, "reset_tried": reset_tried,
            "reset_fidelity": reset_ok / reset_tried if reset_tried else None,
            "walks": walk_count, "returned_naturally": returned_naturally,
            "return_rate": returned_naturally / walk_count if walk_count else None,
            "board_changing_moves": changed_moves, "per_game": per_game}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=25)
    ap.add_argument("--walks", type=int, default=6)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    r = probe(a.games, a.walks, a.depth, a.seed)
    bar = "=" * 78
    print(bar)
    print("  OFFLINE RESET PROBE   no LLM, no GPU, no slot")
    print(f"  {r['games']} games x {r['walks_per_game']} walks x depth {r['depth']}"
          f"   seed {r['seed']}")
    print(bar)
    print(f"  board-changing moves observed : {r['board_changing_moves']}")
    if r["reset_tried"]:
        print(f"  RESET restored the opening    : {r['reset_ok']}/{r['reset_tried']}"
              f" = {100*r['reset_fidelity']:.1f}%      [their claim: 100.0%, 306/306]")
    else:
        print("  RESET restored the opening    : no trials (no board-changing walks)")
    print(f"  walks that returned unaided   : {r['returned_naturally']}/{r['walks']}"
          f" = {100*r['return_rate']:.1f}%       [their one-wayness: 90.1% cannot]")
    imperfect = [g for g in r["per_game"] if g["reset_tried"] and g["reset_ok"] < g["reset_tried"]]
    if imperfect:
        print(f"\n  GAMES WHERE RESET WAS NOT EXACT ({len(imperfect)}):")
        for g in imperfect:
            print(f"    {g['game']:<22}{g['reset_ok']}/{g['reset_tried']}")
    else:
        print("\n  RESET was byte-exact in every game with a trial.")
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=2))
        print(f"  wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
