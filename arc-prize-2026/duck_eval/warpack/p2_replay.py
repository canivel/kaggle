"""P2 offline replay / SAFETY PROOF / CONTROL-SPREAD precompute.

CPU only, read-only, no network, no spend.

WHY THIS FILE RUNS BEFORE ANYTHING IS PUSHED.

1. **Safety.** The level-completing batches in our recorded traces OPEN WITH
   RE-TRAVERSAL (sp80 L1 = RIGHT(rev) x3 then SPACE; ar25 L1 = LEFT(rev) x5,
   DOWN x9, then the completing DOWN). A naive position cap DELETES those
   actions - the same wall that forced ``P1_ABORT_REVISIT=0`` on 08-12. So the
   cap is replayed against every recorded run and the number of
   LEVEL-COMPLETING and BOARD-CHANGING actions it cuts is measured BEFORE a
   kernel is built. Zero level-completing cuts is a hard precondition to push.

2. **Regression to the mean.** P1 mechanism C looked like a 4.4x behavioural win
   inside its own arm and was regression to the mean; EFFNOTE's B1 sat inside a
   control spread that had been sealed first. So the P2 behavioural endpoint
   (G1, actions REQUESTED per stall-gated batch) gets its control spread
   computed here, on block-free runs, BEFORE the arm exists.

Everything is driven by the SHIPPED module ``_kaggle_dataset/p2_batchgate_patch.py``
(its detectors, its ``should_abort_remainder`` predicate, its fingerprint) and
scored with the EXACT scorer ``scripts/phase1_gate.py:rhae_score``. Nothing here
is a re-implementation.

Controls (all block-free, all the same duck harness, none carrying P2):
  runs/kernel_pulls/animation_v1   (25 games, 17 cleared levels, 2026-08-11)
  runs/a22_v2_seed1                (14 cleared levels)
  runs/a22_compaction_v1           (17 cleared levels)
  runs/kernel_pulls/effnote_v1     (16 cleared levels, 2026-08-13) -- the run
                                   whose B4 = 11.11 actions/stall-turn is the
                                   reason P2 exists. Included as a FOURTH
                                   recorded run for the SAFETY proof; the
                                   control spread for G1 is computed from the
                                   three block-free runs only (see --spread3).

Definitions (identical for arm and control):
  BATCH        one ``step_env`` call = the actions of one tool call. Recovered
               exactly from the recorded ``batch_index``/``batch_size`` fields
               (a batch starts at ``batch_index == 1``).
  GATED BATCH  a batch issued while >=1 of the three shipped stall detectors
               fires on the frame history as of the last executed action.
  G1           mean actions REQUESTED per gated batch. The cap truncates
               EXECUTION, it never shrinks the REQUEST, so G1 stays a free
               measure of what the agent chose - it cannot be moved
               mechanically by the cap.
  G1c          the same quantity on ungated batches (within-run control).
  SAVED        actions the cap would not have charged, holding the recorded
               request stream fixed.
  LC-CUT       level-completing actions inside a cut suffix. MUST BE ZERO.
  BC-CUT       board-changing actions inside a cut suffix. Reported for every
               candidate; a rule with BC-CUT = 0 is state-preserving, i.e. its
               replay is EXACT rather than counterfactual.

Usage:
  .venv/Scripts/python.exe duck_eval/warpack/p2_replay.py            # sweep
  .venv/Scripts/python.exe duck_eval/warpack/p2_replay.py --json
  .venv/Scripts/python.exe duck_eval/warpack/p2_replay.py --arm      # score arm
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE / "_kaggle_dataset"))


def _load_scorer():
    """scripts/ is NOT put on sys.path -- it contains queue.py, which shadows
    the stdlib ``queue`` for everything imported afterwards."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_p2_phase1_gate", REPO / "scripts" / "phase1_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.rhae_score


rhae_score = _load_scorer()

import p2_batchgate_patch as P2  # noqa: E402

# Block-free CONTROL runs -- the G1 spread comes from these three ONLY.
CONTROL_RUNS = [
    "runs/kernel_pulls/animation_v1",
    "runs/a22_v2_seed1",
    "runs/a22_compaction_v1",
]
# A fourth recorded run, used for the SAFETY proof (more level-completing
# batches to try to break the cap on) and reported separately.
SAFETY_EXTRA = ["runs/kernel_pulls/effnote_v1"]
ARM = "runs/kernel_pulls/p2_v1"
OUT_DIR = REPO / "runs" / "p2_replay"

# Games with a run-dependent latent state (efficiency_diagnosis sec5 P1,
# measured on animation_v1). Used ONLY to certify that lc survives on them;
# the shipped patch never reads a game id.
LATENT_STATE_GAMES = {"m0r0", "re86", "sk48", "ka59", "cd82", "g50t", "dc22",
                      "wa30", "cn04", "sc25"}


# --------------------------------------------------------------------------- #
# a Frame stand-in: the shipped detectors only ever test grid EQUALITY and
# level equality, so a board digest is an exact substitute for the grid.
# --------------------------------------------------------------------------- #
class F:
    __slots__ = ("grid", "level")

    def __init__(self, grid: str, level: int) -> None:
        self.grid = grid
        self.level = level


def _fp(board) -> str:
    return P2.board_fingerprint(board)


# --------------------------------------------------------------------------- #
# trace loading
# --------------------------------------------------------------------------- #
def load_game(path: Path) -> dict | None:
    """Recorded jsonl -> per-action records + the initial frame.

    ``level`` is the ENGINE level (used by the detectors, which stop scanning
    at a level boundary); ``lvidx`` is the 0-based cleared-level index (used to
    index ``actions_per_level`` for the scorer)."""
    acts: list[dict] = []
    init = None
    lvidx = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            t = ev.get("type")
            if t == "initial" and init is None:
                init = {"fp": _fp(ev.get("board") or []),
                        "level": int(ev.get("level") or 1)}
                continue
            if t != "action" or "board" not in ev:
                continue
            acts.append({
                "fp": _fp(ev["board"]),
                "level": int(ev.get("level") or 1),
                "lvidx": lvidx,
                "bc": bool(ev.get("board_changed")),
                "lc": bool(ev.get("level_completed")),
                "bi": int(ev.get("batch_index") or 1),
                "bs": int(ev.get("batch_size") or 1),
                "step": ev.get("analysis_step"),
                "name": str(ev.get("action_display")
                            or ev.get("action_name") or "?"),
            })
            if ev.get("level_completed"):
                lvidx += 1
    if init is None or not acts:
        return None
    return {"init": init, "acts": acts}


def load_run(rel: str) -> list[tuple[dict, dict | None]]:
    bench = json.loads((REPO / rel / "benchmark.json").read_text(encoding="utf-8"))
    runs = bench if isinstance(bench, list) else bench["game_runs"]
    art = REPO / rel / "artifacts"
    out = []
    for r in runs:
        meta = {k: r.get(k) for k in ("game_id", "number_of_levels",
                                      "base_actions_per_level",
                                      "actions_per_level", "levels_completed")}
        p = art / f"{r['game_id']}_p0_events.jsonl"
        out.append((meta, load_game(p) if p.is_file() else None))
    return out


# --------------------------------------------------------------------------- #
# the replay -- drives the SHIPPED detectors and the SHIPPED predicate
# --------------------------------------------------------------------------- #
def replay_game(g: dict, rule: dict) -> dict:
    """Replay one recorded game under one candidate rule.

    The model's requests are held FIXED and we count the actions the runner
    would not have charged. Two honesty counters make the counterfactual
    visible: ``bc_cut`` (board-changing actions inside a cut suffix -- if this
    is 0 the replay is exact, because every cut action left the board
    byte-identical) and ``lc_cut`` (level-completing actions inside a cut
    suffix -- MUST BE ZERO)."""
    frames = [F(g["init"]["fp"], g["init"]["level"])]

    # split into batches: a batch starts wherever batch_index == 1
    batches: list[list[dict]] = []
    for a in g["acts"]:
        if a["bi"] == 1 or not batches:
            batches.append([])
        batches[-1].append(a)

    persist = bool(rule.get("persist", False))
    visited: set[str] = set()
    cur_lvidx = None
    carry_noop = 0
    carry_stale = 0
    st = {
        "batches": 0, "gated": 0, "gated_multi": 0,
        "requested": 0, "gated_requested": 0, "ungated_requested": 0,
        "executed": 0, "saved": 0, "aborts": 0,
        "nz": 0, "stag": 0, "rev": 0,
        "bc_cut": 0, "lc_cut": 0,
        "gated_executed": 0, "ungated_executed": 0,
        "saved_per_level": {},
        "lc_cut_levels": [],
    }

    for batch in batches:
        size = len(batch)
        st["batches"] += 1
        st["requested"] += size

        current = frames[-1]
        fired = P2.detectors_for(current, frames)
        if fired:
            st["gated"] += 1
            st["gated_requested"] += size
            st["nz"] += 1 if "nz" in fired else 0
            st["stag"] += 1 if "stag" in fired else 0
            st["rev"] += 1 if "rev" in fired else 0
        else:
            st["ungated_requested"] += size
        gated = bool(fired) and size >= rule.get("min_batch", 2)
        if gated:
            st["gated_multi"] += 1

        # seed the level's visited set with the pre-batch state
        if batch[0]["lvidx"] != cur_lvidx:
            cur_lvidx = batch[0]["lvidx"]
            visited = set()
            carry_noop = carry_stale = 0
        visited.add(current.grid)

        executed = 0
        # ``persist`` carries the run across batches within a level -- the
        # shipped ``P2_PERSIST`` behaviour. The ``executed >= 1`` guard inside
        # ``should_abort_remainder`` still holds, so the first action of a
        # batch ALWAYS executes and the stock aggregation path is always used.
        noop_run = carry_noop if persist else 0
        stale_run = carry_stale if persist else 0
        aborted_at = None
        for i, a in enumerate(batch):
            if gated:
                reason = P2.should_abort_remainder(
                    executed=executed,
                    consecutive_noops=noop_run,
                    consecutive_stale=stale_run,
                    cap=rule.get("cap", 0),
                    noop_run=rule.get("noop_run", 0),
                    stale_run=rule.get("stale_run", 0),
                )
                if reason:
                    aborted_at = i
                    break
            # execute
            executed += 1
            st["executed"] += 1
            if gated:
                st["gated_executed"] += 1
            else:
                st["ungated_executed"] += 1
            noop_run = 0 if a["bc"] else noop_run + 1
            if a["fp"] not in visited:
                visited.add(a["fp"])
                stale_run = 0
            else:
                stale_run += 1
            frames.append(F(a["fp"], a["level"]))
            if a["lc"]:
                # a completed level resets the state space
                cur_lvidx = a["lvidx"] + 1
                visited = set()
                noop_run = stale_run = 0
        carry_noop, carry_stale = noop_run, stale_run

        if aborted_at is not None:
            cut = batch[aborted_at:]
            st["aborts"] += 1
            st["saved"] += len(cut)
            lv = batch[aborted_at]["lvidx"]
            st["saved_per_level"][lv] = st["saved_per_level"].get(lv, 0) + len(cut)
            for a in cut:
                if a["bc"]:
                    st["bc_cut"] += 1
                if a["lc"]:
                    st["lc_cut"] += 1
                    st["lc_cut_levels"].append(a["lvidx"])
            # The cut suffix did not run. Every recorded frame after it is a
            # counterfactual, so we DO NOT append them to ``frames`` -- the
            # detector history follows the actions that actually executed.
    return st


def replay_run(rel: str, rule: dict, games: list | None = None) -> dict:
    games = games if games is not None else load_run(rel)
    agg = {k: 0 for k in ("batches", "gated", "gated_multi", "requested",
                          "gated_requested", "ungated_requested", "executed",
                          "saved", "aborts", "nz", "stag", "rev", "bc_cut",
                          "lc_cut", "gated_executed", "ungated_executed")}
    per_game: dict[str, dict] = {}
    as_run = new = honest = 0.0
    saved_cleared = 0
    lc_cut_games: list[str] = []
    detector_games = {"nz": 0, "stag": 0, "rev": 0}
    for meta, g in games:
        gid = str(meta["game_id"]).split("-")[0]
        nlev = int(meta["number_of_levels"] or 0)
        apl = list(meta["actions_per_level"] or [])
        lc = int(meta["levels_completed"] or 0)
        s0 = rhae_score(meta["base_actions_per_level"], apl, lc, nlev)
        as_run += s0
        if g is None:
            new += s0
            honest += s0
            continue
        st = replay_game(g, rule)
        for k in agg:
            agg[k] += st[k]
        for d in detector_games:
            detector_games[d] += 1 if st[d] else 0
        napl = list(apl)
        for lv, saved in st["saved_per_level"].items():
            if 0 <= lv < len(napl):
                napl[lv] = max(0, napl[lv] - saved)
                if lv < lc:
                    saved_cleared += saved
        s1 = rhae_score(meta["base_actions_per_level"], napl, lc, nlev)
        new += s1
        # LC-HONEST score. ``s1`` credits the cap with removing actions from a
        # level it also just prevented from completing -- an over-estimate that
        # every published multiplier for an lc-cutting rule silently carries.
        # Here the level (and every deeper one) is scored as NOT completed.
        if st["lc_cut"]:
            lc_cut_games.append(gid)
            lost_at = min(st["lc_cut_levels"])
            honest += rhae_score(meta["base_actions_per_level"], napl,
                                 min(lc, lost_at), nlev)
        else:
            honest += s1
        per_game[gid] = {
            "as_run": s0, "p2": s1,
            "saved": st["saved"], "aborts": st["aborts"],
            "bc_cut": st["bc_cut"], "lc_cut": st["lc_cut"],
            "gated": st["gated"], "batches": st["batches"],
            "G1": (st["gated_requested"] / st["gated"]) if st["gated"] else None,
            "latent_state": gid in LATENT_STATE_GAMES,
        }
    n = len(games) or 1
    ungated_batches = max(0, agg["batches"] - agg["gated"])
    return {
        "run": rel,
        "rule": rule,
        "games": len(games),
        "batches": agg["batches"],
        "gated_batches": agg["gated"],
        "gate_rate": agg["gated"] / (agg["batches"] or 1),
        "gated_multi": agg["gated_multi"],
        "requested": agg["requested"],
        "executed": agg["executed"],
        "saved": agg["saved"],
        "saved_rate": agg["saved"] / (agg["requested"] or 1),
        "saved_cleared_levels": saved_cleared,
        "aborts": agg["aborts"],
        "BC_CUT": agg["bc_cut"],
        "LC_CUT": agg["lc_cut"],
        "lc_cut_games": sorted(set(lc_cut_games)),
        "G1_gated_batch_size": (agg["gated_requested"] / agg["gated"]
                                if agg["gated"] else 0.0),
        "G1c_ungated_batch_size": (agg["ungated_requested"] / ungated_batches
                                   if ungated_batches else 0.0),
        "detector_games": detector_games,
        "as_run_score": as_run / n,
        "p2_score": new / n,
        "p2_score_lc_honest": honest / n,
        "multiplier": (new / as_run) if as_run else 1.0,
        "multiplier_lc_honest": (honest / as_run) if as_run else 1.0,
        "per_game": per_game,
    }


# --------------------------------------------------------------------------- #
# candidate rules
# --------------------------------------------------------------------------- #
def candidate_rules() -> list[dict]:
    rules: list[dict] = []
    # (a) the naive POSITION cap the brief proposed -- measured so the refusal
    #     is on the record with numbers, not with an argument.
    for cap in (1, 2, 3, 4, 6):
        rules.append({"name": f"cap{cap}", "cap": cap, "noop_run": 0,
                      "stale_run": 0, "min_batch": 2, "persist": False})
    # (b) the pure no-op run rule (P1 mechanism B's rule, restricted to gated
    #     batches -- a strict SUBSET of what already shipped live on 08-12)
    for k in (1, 2, 3):
        rules.append({"name": f"noop{k}", "cap": 0, "noop_run": k,
                      "stale_run": 0, "min_batch": 2, "persist": False})
    # (c) the k-tolerant stale-state run, per batch
    for k in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12):
        rules.append({"name": f"stale{k}", "cap": 0, "noop_run": 0,
                      "stale_run": k, "min_batch": 2, "persist": False})
    # (d) the same rules with the run PERSISTED across batches within a level.
    #     Needed because the agent issues 2-4 SHORT batches per analysis turn,
    #     so a per-batch counter is reset by its own tool-call cadence.
    for k in (1, 2, 3, 4):
        rules.append({"name": f"P-noop{k}", "cap": 0, "noop_run": k,
                      "stale_run": 0, "min_batch": 2, "persist": True})
    for k in (4, 6, 8, 10, 12, 16, 20):
        rules.append({"name": f"P-stale{k}", "cap": 0, "noop_run": 0,
                      "stale_run": k, "min_batch": 2, "persist": True})
    return rules


def sweep(runs: list[str]) -> dict:
    loaded = {r: load_run(r) for r in runs}
    out: dict[str, list[dict]] = {}
    for rule in candidate_rules():
        rows = [replay_run(r, rule, loaded[r]) for r in runs]
        out[rule["name"]] = rows
    return out


def summarise(sweep_out: dict) -> list[dict]:
    rows = []
    for name, res in sweep_out.items():
        rows.append({
            "rule": name,
            "params": res[0]["rule"],
            "LC_CUT": sum(r["LC_CUT"] for r in res),
            "BC_CUT": sum(r["BC_CUT"] for r in res),
            "saved": sum(r["saved"] for r in res),
            "requested": sum(r["requested"] for r in res),
            "saved_rate": (sum(r["saved"] for r in res)
                           / max(1, sum(r["requested"] for r in res))),
            "saved_cleared": sum(r["saved_cleared_levels"] for r in res),
            "aborts": sum(r["aborts"] for r in res),
            "multipliers": [round(r["multiplier"], 4) for r in res],
            "mult_mean": statistics.mean([r["multiplier"] for r in res]),
            "mult_mean_lc_honest": statistics.mean(
                [r["multiplier_lc_honest"] for r in res]),
            "lc_cut_games": sorted({g for r in res for g in r["lc_cut_games"]}),
            "SAFE": sum(r["LC_CUT"] for r in res) == 0,
            "EXACT": sum(r["BC_CUT"] for r in res) == 0,
        })
    rows.sort(key=lambda r: (-r["SAFE"], -r["saved"]))
    return rows


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--arm", action="store_true",
                    help="score the ARM against the sealed control spread")
    ap.add_argument("--rule", default=None,
                    help="replay a single named rule (e.g. stale6)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_runs = CONTROL_RUNS + SAFETY_EXTRA

    if args.arm:
        rule = json.loads((OUT_DIR / "sealed_rule.json").read_text())
        spread = json.loads((OUT_DIR / "control_spread.json").read_text())
        arm = replay_run(ARM, rule)
        gate_min = spread["G1_gated_batch_size"]["min"]
        payload = {"rule": rule, "arm": arm, "control_spread": spread,
                   "gate": {"metric": "G1_gated_batch_size",
                            "control_spread_min": gate_min,
                            "arm": arm["G1_gated_batch_size"],
                            "verdict": "PASS" if arm["G1_gated_batch_size"] < gate_min
                                       else "FAIL"}}
        (OUT_DIR / "arm_vs_control.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload["gate"], indent=2))
        return 0

    sweep_out = sweep(all_runs)
    rows = summarise(sweep_out)

    # the G1 control spread comes from the THREE block-free runs only, and is
    # a property of the trace (not of any rule), so read it off a no-op rule.
    zero = {"name": "none", "cap": 0, "noop_run": 0, "stale_run": 0,
            "min_batch": 2}
    ctrl = [replay_run(r, zero) for r in CONTROL_RUNS]
    spread = {}
    for k in ("G1_gated_batch_size", "G1c_ungated_batch_size", "gate_rate",
              "gated_batches", "batches", "requested"):
        vals = [c[k] for c in ctrl]
        spread[k] = {"min": min(vals), "max": max(vals),
                     "values": vals, "mean": statistics.mean(vals)}
    payload = {"sweep": rows, "control_spread": spread,
               "controls": ctrl, "runs": all_runs,
               "p2_version": P2.VERSION}
    (OUT_DIR / "sweep.json").write_text(json.dumps(payload, indent=2),
                                        encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print("P2 CAP SWEEP -- 4 recorded runs, shipped detectors + shipped predicate")
    print(f"module: p2_batchgate_patch {P2.VERSION}\n")
    hdr = (f"{'rule':<10}{'LC_CUT':>8}{'BC_CUT':>8}{'saved':>8}"
           f"{'saved%':>8}{'clr':>6}{'aborts':>8}{'mult':>9}{'mult-HONEST':>13}  safe")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['rule']:<10}{r['LC_CUT']:>8}{r['BC_CUT']:>8}{r['saved']:>8}"
              f"{100 * r['saved_rate']:>7.2f}%{r['saved_cleared']:>6}"
              f"{r['aborts']:>8}{r['mult_mean']:>9.4f}"
              f"{r['mult_mean_lc_honest']:>13.4f}   "
              f"{'Y' if r['SAFE'] else 'N'}"
              + (f"   lc-cut: {','.join(r['lc_cut_games'])}"
                 if r["lc_cut_games"] else ""))
    print("\nG1 CONTROL SPREAD (3 block-free runs; actions REQUESTED per gated batch)")
    for k, v in spread.items():
        print(f"  {k:<28} min={v['min']:.4f}  max={v['max']:.4f}  "
              f"values={[round(x, 4) for x in v['values']]}")
    print(f"\nwrote {OUT_DIR / 'sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
