"""EFFNOTE offline replay / CONTROL-SPREAD precompute -- CPU only, read-only.

WHY THIS FILE EXISTS, AND WHY IT RUNS FIRST.
Mechanism C (P1's non-truncatable memory block) looked like a 4.4x behavioural
win inside its own arm (33.9% -> 7.8%, z=10.9) and was REGRESSION TO THE MEAN:
the identical statistic on three block-free CONTROL runs spans 5.3-23.1%, the
arm's 7.8% sits INSIDE it, and on one control the within-run direction
REVERSES. The lesson, now a standing rule: **compute the control-side statistic
for every behavioural metric BEFORE reading the arm.**

So this tool reconstructs, on three block-free control runs, exactly what the
EFFNOTE note WOULD have said at every turn -- driving the SHIPPED pure
functions imported from ``_kaggle_dataset/effnote_patch.py``, not a
re-implementation -- and reports the spread of every behavioural metric the
prereg will screen the arm on. Its output is sealed into
``learnings/war_room/effnote_prereg_2026-08-13.md`` before any push.

Controls (all block-free, all the same duck harness, none carrying EFFNOTE):
  runs/kernel_pulls/animation_v1   (25 games, 17 cleared levels, 2026-08-11)
  runs/a22_v2_seed1                (14 cleared levels)
  runs/a22_compaction_v1           (17 cleared levels)

Definitions (identical for arm and control -- that symmetry is the point):
  TURN            one analysis step carrying >=1 action. The note is built from
                  the frame history as of the LAST action of the PREVIOUS step,
                  which is exactly when ``_build_user_prompt`` runs.
  D1 note_rate    turns whose note is non-empty / turns
  D2 stall_rate   turns where >=1 detector fired / turns   (>40% = nagging)
  D3 over_rate    turns where used > the clamped proxy target / turns
  D4 chars        mean / max rendered note length (the CHARACTER cost bound)
  B1 post-stall revisit rate  of the actions issued on a STALL turn, the
                  fraction landing on a board state already visited on that
                  level. "Did the agent break the loop it was just told about?"
                  Reported with its within-run non-stall counterpart, because
                  a within-run first->second-half fall is exactly the artefact
                  that fooled Mechanism C.
  B2 post-stall no-op rate    same denominator, actions with board_changed=False
  B3 over-target burn         actions spent on a level AFTER it first crossed
                  the target (per cleared level; run median + total)
  B4 stall-turn size          mean actions per stall turn vs per non-stall turn
  M0 median actions per cleared level (per game, then the run median)

Usage:
  .venv/Scripts/python.exe duck_eval/warpack/effnote_replay.py
  .venv/Scripts/python.exe duck_eval/warpack/effnote_replay.py --json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE / "_kaggle_dataset"))

import effnote_patch as EN  # noqa: E402

# The three block-free CONTROL runs. Their spread was computed and SEALED into
# learnings/war_room/effnote_prereg_2026-08-13.md BEFORE the arm existed.
RUNS = [
    "runs/kernel_pulls/animation_v1",
    "runs/a22_v2_seed1",
    "runs/a22_compaction_v1",
]

# The ARM. Added 2026-08-13 AFTER the seal, scored by the SAME reconstructor on
# the SAME definitions -- that symmetry is the whole point. Pass --arm to
# include it; the control spread is always computed from RUNS alone so it can
# never be contaminated by the arm.
ARM = "runs/kernel_pulls/effnote_v1"
OUT_DIR = REPO / "runs" / "effnote_replay"


# --------------------------------------------------------------------------- #
# a Frame stand-in: the shipped detectors only ever test grid EQUALITY and
# level equality, so a board digest is an exact substitute for the grid.
# --------------------------------------------------------------------------- #
class F:
    __slots__ = ("grid", "level", "shape")

    def __init__(self, grid: str, level: int, shape: tuple[int, int]) -> None:
        self.grid = grid
        self.level = level
        self.shape = shape


class E:
    """HistoryEntry stand-in."""
    __slots__ = ("frame",)

    def __init__(self, frame: F) -> None:
        self.frame = frame


def _digest(board) -> str:
    return hashlib.blake2b(
        json.dumps(board, separators=(",", ":")).encode(), digest_size=16
    ).hexdigest()


# --------------------------------------------------------------------------- #
# trace loading
# --------------------------------------------------------------------------- #
def load_game(path: Path) -> dict | None:
    """Recorded jsonl -> per-action records + the initial frame."""
    acts: list[dict] = []
    init = None
    shape = (0, 0)
    actions_seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            t = ev.get("type")
            if t == "initial" and init is None:
                board = ev.get("board") or []
                shape = (len(board), len(board[0]) if board else 0)
                init = {"hash": _digest(board),
                        "level": int(ev.get("level") or 1)}
                continue
            if t != "action" or "board" not in ev:
                continue
            name = str(ev.get("action_name") or "?")
            if name != "RESET":
                actions_seen.add(name)
            acts.append({
                "hash": _digest(ev["board"]),
                "level": int(ev.get("level") or 1),
                "bc": bool(ev.get("board_changed")),
                "lc": bool(ev.get("level_completed")),
                "step": ev.get("analysis_step"),
                "anum": ev.get("action_num"),
                "name": name,
            })
    if init is None or not acts:
        return None
    return {"init": init, "acts": acts, "shape": shape,
            "valid_actions": len(actions_seen)}


# --------------------------------------------------------------------------- #
# the replay -- drives the SHIPPED pure functions
# --------------------------------------------------------------------------- #
def replay_game(g: dict) -> dict:
    """Reconstruct the per-turn note over one recorded game and score the
    behavioural metrics. No policy is applied -- the recorded actions are held
    fixed; we only ask what the note WOULD have said and what the agent then
    did."""
    shape = g["shape"]
    board_cells = shape[0] * shape[1]
    nvalid = g["valid_actions"]
    target = EN.heuristic_action_target(nvalid, board_cells)

    frames: list[F] = [F(g["init"]["hash"], g["init"]["level"], shape)]
    hist: list[E] = [E(frames[0])]

    # group actions into turns by analysis_step, preserving order
    turns: list[list[dict]] = []
    cur_step = object()
    for a in g["acts"]:
        if a["step"] != cur_step:
            turns.append([])
            cur_step = a["step"]
        turns[-1].append(a)

    # per-level bookkeeping
    visited: dict[int, set[str]] = {}
    level_actions: dict[int, int] = {}
    level_over_at: dict[int, int] = {}   # level -> action index when it crossed
    level_cleared: set[int] = set()

    cur_level = frames[0].level
    visited.setdefault(cur_level, set()).add(frames[0].grid)

    stats = {
        "turns": 0, "noted": 0, "chars_sum": 0, "chars_max": 0,
        "over": 0, "stall": 0, "nz": 0, "stag": 0, "rev": 0,
        "stall_actions": 0, "stall_revisits": 0, "stall_noops": 0,
        "nonstall_actions": 0, "nonstall_revisits": 0, "nonstall_noops": 0,
        "actions": 0, "target": target, "valid_actions": nvalid,
    }

    for turn in turns:
        # --- the note as of the PREVIOUS action (this is when the prompt is
        #     built) -------------------------------------------------------
        cur = frames[-1]
        used = level_actions.get(cur.level, 0)
        nz = EN.detect_net_zero_cycle(cur, frames)
        stag = EN.detect_stagnation(cur, frames)
        rev = EN.count_recent_revisits(cur, frames)
        note = EN.build_efficiency_note(
            level_number=cur.level, actions_this_level=used, target=target,
            net_zero_actions=nz, stagnation_actions=stag, revisit_count=rev)
        fired = bool(nz) or bool(stag) or (rev >= EN.CFG.revisit_min)

        stats["turns"] += 1
        if note:
            stats["noted"] += 1
            stats["chars_sum"] += len(note)
            stats["chars_max"] = max(stats["chars_max"], len(note))
        if used > target:
            stats["over"] += 1
            level_over_at.setdefault(cur.level, used)
        if nz:
            stats["nz"] += 1
        if stag:
            stats["stag"] += 1
        if rev >= EN.CFG.revisit_min:
            stats["rev"] += 1
        if fired:
            stats["stall"] += 1

        # --- what the agent then did ------------------------------------
        bucket = "stall" if fired else "nonstall"
        for a in turn:
            seen = visited.setdefault(a["level"], set())
            prior = visited.get(cur.level, set())
            stats["actions"] += 1
            stats[f"{bucket}_actions"] += 1
            if a["hash"] in prior:
                stats[f"{bucket}_revisits"] += 1
            if not a["bc"]:
                stats[f"{bucket}_noops"] += 1

            level_actions[cur.level] = level_actions.get(cur.level, 0) + 1
            seen.add(a["hash"])
            frames.append(F(a["hash"], a["level"], shape))
            hist.append(E(frames[-1]))
            if a["lc"]:
                level_cleared.add(cur.level)
            cur = frames[-1]

    # B3: actions burned on a level AFTER it first crossed the target
    burn = {}
    for lvl, at in level_over_at.items():
        total = level_actions.get(lvl, 0)
        burn[lvl] = max(0, total - at)
    cleared_costs = [level_actions.get(l, 0) for l in sorted(level_cleared)]

    stats["burn_total"] = sum(burn.values())
    stats["burn_cleared"] = sum(v for k, v in burn.items() if k in level_cleared)
    stats["levels_over"] = len(burn)
    stats["levels_cleared"] = len(level_cleared)
    stats["cleared_costs"] = cleared_costs
    stats["median_actions_per_cleared_level"] = (
        statistics.median(cleared_costs) if cleared_costs else None)
    return stats


def _rate(num: int, den: int) -> float:
    return (num / den) if den else 0.0


def replay_run(rel: str) -> dict:
    root = REPO / rel
    art = root / "artifacts"
    bench = json.loads((root / "benchmark.json").read_text(encoding="utf-8"))
    runs = bench if isinstance(bench, list) else bench["game_runs"]
    per_game = {}
    for r in runs:
        p = art / f"{r['game_id']}_p0_events.jsonl"
        if not p.is_file():
            continue
        g = load_game(p)
        if g is None:
            continue
        gid = str(r["game_id"]).split("-")[0]
        per_game[gid] = replay_game(g)

    agg = {k: sum(v[k] for v in per_game.values()) for k in (
        "turns", "noted", "chars_sum", "over", "stall", "nz", "stag", "rev",
        "stall_actions", "stall_revisits", "stall_noops",
        "nonstall_actions", "nonstall_revisits", "nonstall_noops",
        "actions", "burn_total", "burn_cleared", "levels_over",
        "levels_cleared")}
    chars_max = max((v["chars_max"] for v in per_game.values()), default=0)
    all_cleared = [c for v in per_game.values() for c in v["cleared_costs"]]
    med_per_game = [v["median_actions_per_cleared_level"]
                    for v in per_game.values()
                    if v["median_actions_per_cleared_level"] is not None]

    out = {
        "run": rel,
        "games": len(per_game),
        "targets": sorted({v["target"] for v in per_game.values()}),
        "valid_action_counts": sorted({v["valid_actions"] for v in per_game.values()}),
        "D1_note_rate": _rate(agg["noted"], agg["turns"]),
        "D2_stall_rate": _rate(agg["stall"], agg["turns"]),
        "D3_over_rate": _rate(agg["over"], agg["turns"]),
        "D4_chars_mean": _rate(agg["chars_sum"], agg["noted"]),
        "D4_chars_max": chars_max,
        "detector_games": {
            "net_zero": sum(1 for v in per_game.values() if v["nz"]),
            "stagnation": sum(1 for v in per_game.values() if v["stag"]),
            "revisit": sum(1 for v in per_game.values() if v["rev"]),
        },
        "detector_turn_rate": {
            "net_zero": _rate(agg["nz"], agg["turns"]),
            "stagnation": _rate(agg["stag"], agg["turns"]),
            "revisit": _rate(agg["rev"], agg["turns"]),
        },
        "B1_post_stall_revisit_rate": _rate(agg["stall_revisits"], agg["stall_actions"]),
        "B1c_nonstall_revisit_rate": _rate(agg["nonstall_revisits"], agg["nonstall_actions"]),
        "B2_post_stall_noop_rate": _rate(agg["stall_noops"], agg["stall_actions"]),
        "B2c_nonstall_noop_rate": _rate(agg["nonstall_noops"], agg["nonstall_actions"]),
        "B3_over_target_burn_total": agg["burn_total"],
        "B3_over_target_burn_cleared": agg["burn_cleared"],
        "B3_levels_over_target": agg["levels_over"],
        "B4_stall_turn_size": _rate(agg["stall_actions"], agg["stall"]),
        "B4_nonstall_turn_size": _rate(agg["nonstall_actions"], agg["turns"] - agg["stall"]),
        "M0_median_actions_per_cleared_level": (
            statistics.median(all_cleared) if all_cleared else None),
        "M0_median_of_per_game_medians": (
            statistics.median(med_per_game) if med_per_game else None),
        "levels_cleared": agg["levels_cleared"],
        "turns": agg["turns"],
        "actions": agg["actions"],
        "stall_actions": agg["stall_actions"],
        "per_game": per_game,
    }
    return out


SPREAD_KEYS = [
    "D1_note_rate", "D2_stall_rate", "D3_over_rate", "D4_chars_mean",
    "D4_chars_max",
    "B1_post_stall_revisit_rate", "B1c_nonstall_revisit_rate",
    "B2_post_stall_noop_rate", "B2c_nonstall_noop_rate",
    "B3_over_target_burn_total", "B3_over_target_burn_cleared",
    "B4_stall_turn_size", "B4_nonstall_turn_size",
    "M0_median_actions_per_cleared_level", "M0_median_of_per_game_medians",
    "levels_cleared",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--arm", action="store_true",
                    help="also score the ARM run and read it against the "
                         "SEALED control spread (0.3986 on B1)")
    args = ap.parse_args()

    results = [replay_run(r) for r in RUNS]
    spread = {}
    for k in SPREAD_KEYS:
        vals = [r[k] for r in results if r.get(k) is not None]
        if not vals:
            continue
        spread[k] = {"min": min(vals), "max": max(vals),
                     "values": vals,
                     "mean": sum(vals) / len(vals)}

    arm = replay_run(ARM) if args.arm else None
    payload = {"controls": results, "control_spread": spread,
               "effnote_version": EN.VERSION,
               "max_chars_bound": EN.CFG.max_chars}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if arm is None:
        (OUT_DIR / "control_spread.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
    else:
        payload["arm"] = arm
        # The SEALED gate. Verbatim from effnote_prereg_2026-08-13.md sec3/sec5:
        # PASS requires the arm strictly BELOW the control-spread MINIMUM.
        gate_min = spread["B1_post_stall_revisit_rate"]["min"]
        payload["gate"] = {
            "metric": "B1_post_stall_revisit_rate",
            "sealed_threshold": 0.3986,
            "control_spread_min": gate_min,
            "arm": arm["B1_post_stall_revisit_rate"],
            "verdict": ("PASS" if arm["B1_post_stall_revisit_rate"] < 0.3986
                        else "FAIL"),
            "note": ("The arm's own first-half/second-half contrast is NOT an "
                     "endpoint and may not be cited (P1 mechanism C)."),
        }
        (OUT_DIR / "arm_vs_control.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print("EFFNOTE CONTROL SPREAD (block-free runs; note reconstructed, never applied)")
    print(f"shipped module: effnote_patch {EN.VERSION}   char bound: {EN.CFG.max_chars}\n")
    cols = results + ([arm] if arm else [])
    hdr = f"{'metric':<40}" + "".join(
        f"{('ARM ' + Path(r['run']).name) if arm and r is arm else Path(r['run']).name:>22}"
        for r in cols) + f"{'CONTROL SPREAD':>26}"
    print(hdr)
    print("-" * len(hdr))
    for k in SPREAD_KEYS:
        cells = ""
        for r in cols:
            v = r.get(k)
            cells += f"{'-':>22}" if v is None else (
                f"{v:>22.4f}" if isinstance(v, float) else f"{v:>22}")
        sp = spread.get(k)
        tail = "" if not sp else (
            f"{sp['min']:.4f} - {sp['max']:.4f}" if isinstance(sp["min"], float)
            else f"{sp['min']} - {sp['max']}")
        print(f"{k:<40}{cells}{tail:>26}")
    print()
    if arm:
        g = payload["gate"]
        print("=" * 96)
        print(f"SEALED GATE  {g['metric']}  arm={g['arm']:.4f}  "
              f"threshold=<{g['sealed_threshold']} "
              f"(control-spread min {g['control_spread_min']:.4f})  "
              f"-> {g['verdict']}")
        print("=" * 96)
        print()
    for r in results:
        print(f"{Path(r['run']).name}: games={r['games']} turns={r['turns']} "
              f"actions={r['actions']} targets={r['targets']} "
              f"valid_action_counts={r['valid_action_counts']} "
              f"detector_games={r['detector_games']} "
              f"detector_turn_rate={ {k: round(v, 4) for k, v in r['detector_turn_rate'].items()} }")
    print(f"\nwrote {OUT_DIR / 'control_spread.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
