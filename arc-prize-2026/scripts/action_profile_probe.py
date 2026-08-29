"""OFFLINE ACTION-PROFILE PROBE -- is the never-pressed control actually LIVE?

WHY
    `scripts/untried_probe.py` established, on our own 662-pass archive:

        ACTION7 is DECLARED in 137 passes and PRESSED 0 times in 33,820 turns.

    That is the single starkest affordance miss the campaign has recorded -- but
    it is only half a finding. A control that is never pressed matters only if
    pressing it would have DONE something. Because we never pressed it, our
    archive cannot say. This probe answers the other half on the real engine.

    It is the same instrument that replicated RESET at 192/192 on 25/25 official
    games (scripts/reset_probe.py, 2026-08-28), so RESET-between-presses is a
    verified perfect undo and each action is tested from a byte-identical opening.

WHAT IT MEASURES  (no LLM, no GPU, no Kaggle slot -- seconds per game)
    For every official game, at the LEVEL-1 OPENING, press each declared action
    once and record whether the board changed.

    SCOPE, STATED UP FRONT: this is the level-1 opening only. thtennant's sweep
    covered 6 levels x 25 games = 600 action-level pairs by driving to each level;
    `taaf.game_api.GameAPI` exposes `number_of_levels` but no setter, so deeper
    levels need real play and are NOT covered here. Do not quote this as a
    replication of his 600-pair number -- it is the level-1 slice of it.

    ACTION6 (MOUSE) is parameterised by (x, y). One press is not a fair test of a
    64x64 coordinate space, so it is sampled over --mouse-samples random
    coordinates and reported SEPARATELY. Its rate is "did ANY sampled coordinate
    move the board", which is an UPPER bound on one-press liveness.

USAGE
    scripts/action_profile_probe.py
    scripts/action_profile_probe.py --games 5 --json out.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

_SELF = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != _SELF]

ROOT = Path(__file__).resolve().parents[1]
# taaf ships as bundle source, not as an installed package (the box that had it
# installed was the Mac, and the Mac is gone).
sys.path.insert(0, str(ROOT / "duck_eval" / "taaf_bundle" / "src" /
                       "tufa-arc-agi-framework" / "src"))


def board_hash(state) -> str:
    h = hashlib.sha256()

    def feed(o):
        if hasattr(o, "tobytes"):
            h.update(o.tobytes())
            h.update(str(getattr(o, "shape", "")).encode())
        elif isinstance(o, (list, tuple)):
            for x in o:
                feed(x)
        elif isinstance(o, dict):
            for k in sorted(o):
                h.update(str(k).encode())
                feed(o[k])
        else:
            h.update(repr(o).encode())
    feed(getattr(state, "frame", state))
    return h.hexdigest()


def probe(n_games: int, mouse_samples: int, seed: int) -> dict:
    import arcengine
    import re_arc
    import taaf.game_api
    import taaf.game

    # arcengine.GameAction(3) RAISES even though member ACTION3 has value 3 --
    # the enum's value lookup is broken (custom metaclass). Build the map from
    # the members and never call the constructor. (Inherited from reset_probe.)
    BY_VALUE = {m.value: m for m in arcengine.GameAction}

    rng = random.Random(seed)
    games = sorted(re_arc.list_game_ids(datasets=["train", "eval"],
                                        include_tags="official"))[:n_games]

    per_game, live, tried = [], collections.Counter(), collections.Counter()
    declared_count = collections.Counter()
    errors = []

    for gid in games:
        try:
            api = taaf.game_api.GameAPI(env_name=gid)
            session = taaf.game.RunSession()
            state = api.start_game(session=session)
        except Exception as e:                                    # noqa: BLE001
            errors.append({"game": gid, "stage": "start", "err": repr(e)[:200]})
            continue

        opening = board_hash(state)
        declared = [int(a) for a in (state.available_actions or [])]
        row = {"game": gid, "declared": [BY_VALUE[a].name for a in declared],
               "n_levels": None, "results": {}}
        try:
            row["n_levels"] = api.number_of_levels
        except Exception:                                          # noqa: BLE001
            pass

        for a in declared:
            ga = BY_VALUE[a]
            if ga.value == 0:                       # RESET is the undo, not a probe
                continue
            declared_count[ga.name] += 1
            samples = mouse_samples if ga.name == "ACTION6" else 1
            changed = False
            for _ in range(samples):
                data = ({"x": rng.randrange(64), "y": rng.randrange(64)}
                        if ga.name == "ACTION6" else {})
                try:
                    st = api._execute_action(arcengine.ActionInput(id=ga, data=data))
                except Exception as e:                             # noqa: BLE001
                    errors.append({"game": gid, "action": ga.name, "err": repr(e)[:200]})
                    break
                if board_hash(st) != opening:
                    changed = True
                # RESET back to the opening so every press starts identically
                try:
                    st = api._execute_action(arcengine.ActionInput(id=BY_VALUE[0], data={}))
                except Exception:                                  # noqa: BLE001
                    break
                if board_hash(st) != opening:
                    errors.append({"game": gid, "action": ga.name,
                                   "err": "RESET did not restore the opening"})
                    break
                if changed:
                    break
            tried[ga.name] += 1
            live[ga.name] += changed
            row["results"][ga.name] = {"moved_board": changed,
                                       "samples": samples}
        per_game.append(row)
        try:
            api._finish_game()
        except Exception:                                          # noqa: BLE001
            pass

    return {"games": len(per_game), "seed": seed, "mouse_samples": mouse_samples,
            "scope": "level-1 opening only",
            "live": dict(live), "tried": dict(tried),
            "declared_count": dict(declared_count),
            "per_game": per_game, "errors": errors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=25)
    ap.add_argument("--mouse-samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    r = probe(a.games, a.mouse_samples, a.seed)
    print("OFFLINE ACTION-PROFILE PROBE -- %d official games, LEVEL-1 OPENING ONLY\n"
          % r["games"])
    print("   %-10s %8s %8s %10s" % ("action", "declared", "moved", "rate"))
    for act in sorted(r["tried"], key=lambda k: -r["tried"][k]):
        t, l = r["tried"][act], r["live"].get(act, 0)
        note = "  (ANY of %d coords -- upper bound)" % r["mouse_samples"] if act == "ACTION6" else ""
        print("   %-10s %8d %8d %9.1f%%%s" % (act, t, l, 100.0 * l / t, note))

    a7 = [g["game"] for g in r["per_game"]
          if g["results"].get("ACTION7", {}).get("moved_board")]
    a7d = [g["game"] for g in r["per_game"] if "ACTION7" in g["results"]]
    print("\nACTION7 -- the control our archive says was declared 137x and pressed 0x")
    print("   declared at level-1 opening in : %d/%d games" % (len(a7d), r["games"]))
    print("   moved the board on ONE press   : %d/%d of those" % (len(a7), len(a7d) or 1))
    if a7:
        print("   games where it is LIVE         : %s" % ", ".join(a7))
    if r["errors"]:
        print("\nERRORS (%d) -- reported, never silently dropped:" % len(r["errors"]))
        for e in r["errors"][:10]:
            print("   %s" % e)

    if a.json:
        out = Path(a.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(r, indent=1), encoding="utf-8")
        print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
