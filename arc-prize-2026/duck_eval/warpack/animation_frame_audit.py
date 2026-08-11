"""Animation-frame audit across ALL 25 official games (ADOPT #1, sweep 2026-08-11).

The sweep found that ``taaf.game.GameState.frame`` returns only ``raw.frame[-1]``
while ``arcengine`` returns a *list* of frames per action
(``GameState.all_frames`` / ``animation_frames``), and that
``duck_eval/taaf_bundle/src/ARC3-Inference/`` has ZERO consumers of either.
The sweep supplied mechanism, not efficacy, and quoted a competitor's numbers
(13/24 games multi-frame; ft09/sb26 "type 1"). This audit reproduces the
measurement **on our own engines, with our own recorded agent behaviour**, so
the evidence base is ours and not a quote.

Two probes, both LM-free:

  probe A ("recorded")  -- the real recorded action history of a full 25-game
      kernel run (default ``runs/kernel_pulls/a22_v2_1/benchmark.json``) replayed
      verbatim through ``env.step``. This is the *realistic* action distribution:
      whatever the agent actually did is what determines how much signal it lost.
  probe B ("seeded")    -- a fixed seeded script over the game's initial
      non-RESET available actions (ACTION6 gets seeded x/y in 0..63), to cover
      action/coordinate space the recorded run never touched.

Per action we record, from the RAW arcengine response:
  n_frames          len(resp.frame)
  first_eq_last     frames[0] == frames[-1]              (within-response)
  mid_differs       any frames[i] (i<last) != frames[-1]  (within-response)
  settled_changed   frames[-1] != previous settled frame  (what the agent sees)

and derive the two headline quantities:

  MULTI       n_frames > 1                     -- an animation happened
  INVISIBLE   settled_changed == False AND mid_differs == True
              -- the agent was shown a board byte-identical to the one before
                 its action, while the engine had in fact rendered something.
                 This is the previously-invisible signal, and it is the exact
                 false-no-op / state-aliasing failure class in project memory.

``TYPE1`` (per the sweep's taxonomy) is reported as a per-game verdict:
  type1  -- has >=1 INVISIBLE action  (signal lives only between frames)
  type2  -- multi-frame but 0 INVISIBLE (pure motion interpolation)
  single -- no multi-frame response observed

Run from the repo root:
    .venv/Scripts/python.exe duck_eval/warpack/animation_frame_audit.py
Optional:
    --bench runs/kernel_pulls/<pull>/benchmark.json
    --games ft09,sb26,r11l,sk48,ls20     (default: all 25)
    --max-actions 400                    (per game, probe A)
    --seeded-steps 64                    (probe B)
Output: runs/animation/frame_audit.json (+ .md table)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUNDLE = REPO / "duck_eval" / "taaf_bundle" / "src"
ENV_FILES = REPO / "kaggle-data" / "environment_files"

sys.path.insert(0, str(BUNDLE / "tufa-arc-agi-framework" / "src"))

DEFAULT_BENCH = REPO / "runs" / "kernel_pulls" / "a22_v2_1" / "benchmark.json"
OUT_JSON = REPO / "runs" / "animation" / "frame_audit.json"
OUT_MD = REPO / "runs" / "animation" / "frame_audit.md"

GRID = 64
ALL_GAMES = [
    "ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52",
    "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26", "sc25", "sk48",
    "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30",
]


# --------------------------------------------------------------------------
# frame helpers (raw arcengine frames are list[list[list[int]]] or similar)
# --------------------------------------------------------------------------

def _norm(frame) -> tuple:
    """Hashable, comparable normalisation of one raw frame."""
    try:
        return tuple(tuple(int(c) for c in row) for row in frame)
    except Exception:  # noqa: BLE001
        return (repr(frame),)


def _cells_differing(a: tuple, b: tuple) -> int:
    """Count of differing cells between two normalised frames (-1 on shape mismatch)."""
    if len(a) != len(b):
        return -1
    n = 0
    for ra, rb in zip(a, b):
        if len(ra) != len(rb):
            return -1
        for ca, cb in zip(ra, rb):
            if ca != cb:
                n += 1
    return n


def analyse_response(resp, prev_settled: tuple | None) -> dict:
    """All per-action animation quantities, from the RAW response."""
    frames = list(getattr(resp, "frame", None) or [])
    norm = [_norm(f) for f in frames]
    settled = norm[-1] if norm else None
    n = len(norm)
    first_eq_prev = bool(n > 0 and prev_settled is not None and norm[0] == prev_settled)
    first_eq_last = bool(n > 1 and norm[0] == norm[-1])
    mid_differs = bool(n > 1 and any(f != norm[-1] for f in norm[:-1]))
    settled_changed = bool(prev_settled is not None and settled is not None
                           and settled != prev_settled)
    if prev_settled is None:
        settled_changed = True  # initial frame: not a comparison
    transient = 0
    if mid_differs and settled is not None:
        transient = max(
            (c for c in (_cells_differing(f, settled) for f in norm[:-1]) if c >= 0),
            default=0,
        )
    return {
        "n_frames": n,
        "unique_frames": len(set(norm)),
        "first_eq_prev": first_eq_prev,
        "first_eq_last": first_eq_last,
        "mid_differs": mid_differs,
        "settled_changed": settled_changed,
        "invisible": bool((not settled_changed) and mid_differs),
        "max_transient_cells": transient,
        "state": getattr(getattr(resp, "state", None), "name", "?"),
        "lc": int(getattr(resp, "levels_completed", 0) or 0),
        "settled": settled,
    }


# --------------------------------------------------------------------------
# scripts
# --------------------------------------------------------------------------

def load_recorded_script(bench_path: Path, prefix: str, max_actions: int) -> list[dict]:
    """Recorded history for ``prefix`` from a kernel-pull benchmark.json."""
    if not bench_path.exists():
        return []
    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    best: list | None = None
    for r in bench.get("game_runs", []):
        if not str(r.get("game_id", "")).startswith(prefix):
            continue
        hist = r.get("history") or []
        if best is None or len(hist) > len(best):
            best = hist
    if not best:
        return []
    return [
        {"name": a["action"]["id"], "data": dict(a["action"].get("data") or {})}
        for a in best[:max_actions]
    ]


def make_seeded_script(prefix: str, available: list[int], n_steps: int) -> list[dict]:
    import arcengine

    rng = random.Random(f"animation_frame_audit:{prefix}")
    ids = [a for a in available if a != 0]
    if not ids:
        return []
    by_value = {a.value: a for a in arcengine.GameAction}
    out: list[dict] = []
    for _ in range(n_steps):
        act = by_value[rng.choice(ids)]
        data = ({"x": rng.randrange(GRID), "y": rng.randrange(GRID)}
                if act.name == "ACTION6" else {})
        out.append({"name": act.name, "data": data})
    return out


# --------------------------------------------------------------------------
# probe driver
# --------------------------------------------------------------------------

def run_script(env, script: list[dict], initial_settled: tuple | None) -> dict:
    """Feed the script verbatim; aggregate animation stats. Never raises."""
    import arcengine

    prev = initial_settled
    per_action: list[dict] = []
    errors = 0
    recoveries = 0
    for step in script:
        try:
            resp = env.step(arcengine.GameAction.from_name(step["name"]),
                            data=dict(step["data"]))
        except Exception:  # noqa: BLE001
            errors += 1
            continue
        if resp is None or not getattr(resp, "frame", None):
            # Engine refused to advance (typically a non-RESET action after
            # GAME_OVER). The real solver issues RESET here; mirror it so a
            # dead-end does not silently truncate the probe.
            recoveries += 1
            try:
                resp = env.step(arcengine.GameAction.RESET, data={})
            except Exception:  # noqa: BLE001
                resp = None
            if resp is None or not getattr(resp, "frame", None):
                errors += 1
                continue
        rec = analyse_response(resp, prev)
        prev = rec.pop("settled")
        rec["action"] = step["name"]
        per_action.append(rec)
    return {"per_action": per_action, "errors": errors,
            "recoveries": recoveries, "final_settled": prev}


def summarise(per_action: list[dict]) -> dict:
    n = len(per_action)
    multi = [r for r in per_action if r["n_frames"] > 1]
    return {
        "actions": n,
        "multi_frame": len(multi),
        "multi_frame_pct": round(100.0 * len(multi) / n, 1) if n else 0.0,
        "max_frames": max((r["n_frames"] for r in per_action), default=0),
        "mean_frames": round(sum(r["n_frames"] for r in per_action) / n, 2) if n else 0.0,
        "first_eq_prev": sum(1 for r in per_action if r["first_eq_prev"]),
        "first_eq_last": sum(1 for r in multi if r["first_eq_last"]),
        "first_eq_last_mid_differs": sum(
            1 for r in multi if r["first_eq_last"] and r["mid_differs"]),
        "settled_unchanged": sum(1 for r in per_action if not r["settled_changed"]),
        "settled_unchanged_multi": sum(
            1 for r in per_action if not r["settled_changed"] and r["n_frames"] > 1),
        "invisible": sum(1 for r in per_action if r["invisible"]),
        "invisible_pct_of_actions": (
            round(100.0 * sum(1 for r in per_action if r["invisible"]) / n, 1) if n else 0.0),
        "invisible_pct_of_noops": (
            round(100.0 * sum(1 for r in per_action if r["invisible"])
                  / max(1, sum(1 for r in per_action if not r["settled_changed"])), 1)),
        "max_transient_cells": max((r["max_transient_cells"] for r in per_action), default=0),
    }


def _fresh_play(arcade, prefix: str):
    """A brand-new play of ``prefix``. Each probe gets its own: running probe B
    on the env probe A left behind measures whatever end-state probe A reached,
    not the game (observed 2026-08-11: ft09 seeded INVISIBLE 281/300 on a fresh
    play vs 0/300 when chained after the recorded probe)."""
    env = arcade.make(prefix)
    if env is None:
        return None, None, []
    # ONLY_RESET_LEVELS mirrors GameAPI._start_game (set AFTER make()).
    os.environ["ONLY_RESET_LEVELS"] = "true"
    initial = env.observation_space
    frames = list(getattr(initial, "frame", None) or []) if initial is not None else []
    settled = _norm(frames[-1]) if frames else None
    available = list(getattr(initial, "available_actions", []) or []) if initial is not None else []
    return env, settled, available


def audit_game(arcade, prefix: str, bench: Path, max_actions: int, seeded_steps: int) -> dict:
    t0 = time.time()
    out: dict = {"game": prefix}
    try:
        env_a, initial_settled, available = _fresh_play(arcade, prefix)
    except Exception as exc:  # noqa: BLE001
        return {"game": prefix, "error": f"make_failed:{type(exc).__name__}:{exc}"}
    if env_a is None:
        return {"game": prefix, "error": "make_returned_none"}
    out["initial_available_actions"] = [int(a) for a in available]

    # probe A: recorded history, on its own fresh play
    script_a = load_recorded_script(bench, prefix, max_actions)
    res_a = run_script(env_a, script_a, initial_settled) if script_a else {
        "per_action": [], "errors": 0, "recoveries": 0, "final_settled": initial_settled}
    out["recorded"] = summarise(res_a["per_action"])
    out["recorded"]["errors"] = res_a["errors"]
    out["recorded"]["recoveries"] = res_a["recoveries"]
    out["recorded"]["script_len"] = len(script_a)

    # probe B: seeded random, on its own fresh play
    env_b, settled_b, _ = _fresh_play(arcade, prefix)
    script_b = make_seeded_script(prefix, [int(a) for a in available], seeded_steps)
    res_b = run_script(env_b, script_b, settled_b) if (script_b and env_b is not None) else {
        "per_action": [], "errors": 0, "recoveries": 0, "final_settled": settled_b}
    out["seeded"] = summarise(res_b["per_action"])
    out["seeded"]["errors"] = res_b["errors"]
    out["seeded"]["recoveries"] = res_b["recoveries"]

    merged = res_a["per_action"] + res_b["per_action"]
    out["combined"] = summarise(merged)
    if out["combined"]["multi_frame"] == 0:
        out["type"] = "single"
    elif out["combined"]["invisible"] > 0:
        out["type"] = "type1"
    else:
        out["type"] = "type2"
    out["wallclock_s"] = round(time.time() - t0, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=str(DEFAULT_BENCH))
    ap.add_argument("--games", default="")
    ap.add_argument("--max-actions", type=int, default=400)
    ap.add_argument("--seeded-steps", type=int, default=300)
    ap.add_argument("--out-json", default=str(OUT_JSON))
    ap.add_argument("--out-md", default=str(OUT_MD))
    args = ap.parse_args()

    import arc_agi

    games = [g.strip() for g in args.games.split(",") if g.strip()] or ALL_GAMES
    bench = Path(args.bench)
    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.OFFLINE,
        environments_dir=str(ENV_FILES),
    )

    results = []
    for g in games:
        r = audit_game(arcade, g, bench, args.max_actions, args.seeded_steps)
        results.append(r)
        if "error" in r:
            print(f"[{g}] ERROR {r['error']}", flush=True)
        else:
            c = r["combined"]
            print(f"[{g}] type={r['type']:6s} actions={c['actions']:4d} "
                  f"multi={c['multi_frame']:4d} ({c['multi_frame_pct']:5.1f}%) "
                  f"maxframes={c['max_frames']:3d} invisible={c['invisible']:4d} "
                  f"({c['invisible_pct_of_actions']:4.1f}% of actions, "
                  f"{c['invisible_pct_of_noops']:5.1f}% of no-ops) "
                  f"{r['wallclock_s']}s", flush=True)

    ok = [r for r in results if "error" not in r]
    totals = {
        "games_audited": len(ok),
        "games_multi_frame": sum(1 for r in ok if r["combined"]["multi_frame"] > 0),
        "games_type1": sum(1 for r in ok if r["type"] == "type1"),
        "games_type2": sum(1 for r in ok if r["type"] == "type2"),
        "games_single": sum(1 for r in ok if r["type"] == "single"),
        "actions": sum(r["combined"]["actions"] for r in ok),
        "multi_frame": sum(r["combined"]["multi_frame"] for r in ok),
        "settled_unchanged": sum(r["combined"]["settled_unchanged"] for r in ok),
        "invisible": sum(r["combined"]["invisible"] for r in ok),
    }
    totals["multi_frame_pct"] = (
        round(100.0 * totals["multi_frame"] / max(1, totals["actions"]), 1))
    totals["invisible_pct_of_actions"] = (
        round(100.0 * totals["invisible"] / max(1, totals["actions"]), 1))
    totals["invisible_pct_of_noops"] = (
        round(100.0 * totals["invisible"] / max(1, totals["settled_unchanged"]), 1))

    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "bench": str(bench),
        "env_files": str(ENV_FILES),
        "max_actions": args.max_actions,
        "seeded_steps": args.seeded_steps,
        "totals": totals,
        "games": results,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Animation-frame audit -- 25 official games (our engines, our recorded behaviour)",
        "",
        f"generated {payload['generated']} | bench `{bench.name}` | "
        f"probe A = recorded history (<= {args.max_actions} actions), "
        f"probe B = {args.seeded_steps} seeded actions | LM-free, offline",
        "",
        "`MULTI` = engine returned >1 frame for one action. "
        "`INVISIBLE` = settled board identical to the previous settled board "
        "AND at least one intermediate frame differed -- the agent saw \"nothing happened\" "
        "while the engine had rendered something.",
        "",
        "| game | type | actions | MULTI | MULTI% | max frames | first==last | "
        "settled-unchanged | INVISIBLE | INV% actions | INV% of no-ops | max transient cells | "
        "INV (probe A recorded) | INV (probe B seeded) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['game']} | ERROR | | | | | | | | | | {r['error']} | | |")
            continue
        c = r["combined"]
        lines.append(
            f"| {r['game']} | {r['type']} | {c['actions']} | {c['multi_frame']} | "
            f"{c['multi_frame_pct']}% | {c['max_frames']} | {c['first_eq_last']} | "
            f"{c['settled_unchanged']} | **{c['invisible']}** | "
            f"{c['invisible_pct_of_actions']}% | {c['invisible_pct_of_noops']}% | "
            f"{c['max_transient_cells']} | "
            f"{r['recorded']['invisible']}/{r['recorded']['actions']} | "
            f"{r['seeded']['invisible']}/{r['seeded']['actions']} |"
        )
    lines += [
        "",
        f"**Totals:** {totals['games_multi_frame']}/{totals['games_audited']} games return "
        f"multi-frame responses ({totals['games_type1']} type-1, {totals['games_type2']} type-2, "
        f"{totals['games_single']} single-frame). "
        f"{totals['multi_frame']}/{totals['actions']} actions "
        f"({totals['multi_frame_pct']}%) were animated. "
        f"**{totals['invisible']} actions ({totals['invisible_pct_of_actions']}% of all, "
        f"{totals['invisible_pct_of_noops']}% of apparent no-ops) carried signal the agent "
        f"could not see.**",
        "",
    ]
    out_md = Path(args.out_md)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines[-4:]))
    print(f"\nwrote {out_json}\nwrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
