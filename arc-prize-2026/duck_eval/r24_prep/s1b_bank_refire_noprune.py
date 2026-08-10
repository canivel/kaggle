"""S1b — offline banking re-fire with `prune_trace` DISABLED, all 25 traces.

R24 proposal `learnings/war_room/r24_successor_lane_proposal_2026-08-08.md` §4 row
S1b / §3(c) "Minimal first experiment":

    "Re-run bank_fire_validation offline with pruning disabled (full replay from
     RESET) across recorded traces for all 25 games; report per game fired /
     aborted / abort-step histogram / score-invariance."
    Gate: "step-0 `frame_divergence` must clear on the 11-game
     prefix-splice-safe set."

THIN WRAPPER. It re-uses, unmodified:
  * `duck_eval/warpack/warpack_patch.py` — `TraceStep`, `prune_trace`, and the
    EXACT divergence semantics of `_bank` (per-step `grid_hash` + `lc_after`
    comparison, new play opened with <=2 RESETs under ONLY_RESET_LEVELS=false
    until `full_reset`).
  * `runs/kernel_pulls/war_eval_v1/benchmark.json` — the 25 recorded traces
    (3,638 actions, the ledger-OFF seed-1 control run).
  * `kaggle-data/environment_files` + the local `arcengine`/`arc_agi` engines,
    i.e. the same engines `bank_fire_validation.py` and
    `determinism_audit_25.py` ran on.

`prune_trace` is a module-level function with NO env kill-switch
(`WARPACK_BANK_STRICT` gates frame CHECKING, not pruning), so "pruning disabled"
is implemented here as an arm selector over the trace, not as an edit to
warpack. Three arms are replayed per game on three successive fresh plays so
the comparison is paired:

  pruned         prune_trace(trace)  — the 2026-07-15 baseline that aborted at
                                       step 0 on sc25 / m0r0
  trailing_only  drop everything after the last level completion; keep every
                 other recorded step INCLUDING no-ops and RESETs (the
                 "trailing-only-pruned" variant R16 flagged as also viable)
  unpruned_full  every recorded step up to the target lc, verbatim

Usage
-----
    .venv/Scripts/python.exe duck_eval/r24_prep/s1b_bank_refire_noprune.py --dry-run
    .venv/Scripts/python.exe duck_eval/r24_prep/s1b_bank_refire_noprune.py \
        --out runs/r24_prep/s1b_bank_refire.json

CPU-only, local offline engines, no network, no Kaggle push, $0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WARPACK = ROOT / "duck_eval" / "warpack"
ENV_FILES = ROOT / "kaggle-data" / "environment_files"
BENCH = ROOT / "runs" / "kernel_pulls" / "war_eval_v1" / "benchmark.json"

sys.path.insert(0, str(WARPACK))
sys.path.insert(0, str(ROOT / "duck_eval" / "taaf_bundle" / "src"
                       / "tufa-arc-agi-framework" / "src"))

# R17 SEALED §7 carrier set (learnings/war_room/grinder_design_R17_sealing.md
# L319-L324): 10 CONFIRMED CLEAN + tn36 admitted WITH ITS FLAG = the proposal's
# 11-game "prefix-splice-safe set". The sealed BANKING rule (same doc, L67-71)
# restricts prefix-splice to the 10 and excludes tn36. Both are reported.
CONFIRMED_CLEAN_10 = ["ar25", "bp35", "ft09", "lf52", "lp85", "ls20", "r11l",
                      "sp80", "su15", "tu93"]
SPLICE_SAFE_11 = sorted(CONFIRMED_CLEAN_10 + ["tn36"])

ARMS = ("pruned", "trailing_only", "unpruned_full")


# ------------------------------------------------------------------ utilities
def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_info() -> dict:
    def _run(*a):
        try:
            return subprocess.run(a, cwd=str(ROOT), capture_output=True,
                                  text=True, timeout=20).stdout.strip()
        except Exception:  # noqa: BLE001
            return "unavailable"
    return {"commit": _run("git", "rev-parse", "HEAD"),
            "dirty": bool(_run("git", "status", "--porcelain"))}


def ghash(frame_rows):
    """Verbatim warpack_patch._grid_hash_from_frame semantics."""
    try:
        return hash(tuple(tuple(int(c) for c in row) for row in frame_rows))
    except Exception:  # noqa: BLE001
        return None


def trailing_only(trace):
    """Drop everything after the last level completion; prune nothing else."""
    last = max((i for i, s in enumerate(trace) if s.level_completed), default=-1)
    return list(trace[:last + 1]) if last >= 0 else []


def open_new_play(env) -> bool:
    """_bank's new-play opener: <=2 RESETs under ONLY_RESET_LEVELS=false."""
    import arcengine

    prev = os.environ.get("ONLY_RESET_LEVELS")
    try:
        os.environ["ONLY_RESET_LEVELS"] = "false"
        for _ in range(2):
            resp = env.step(arcengine.GameAction.RESET, data={})
            if resp is None:
                return False
            if getattr(resp, "full_reset", False):
                return True
    finally:
        if prev is None:
            os.environ.pop("ONLY_RESET_LEVELS", None)
        else:
            os.environ["ONLY_RESET_LEVELS"] = prev
    return False


def replay_arm(env, steps, max_actions: int) -> dict:
    """Replay `steps` with _bank's exact divergence semantics."""
    import arcengine

    out = {"n_replay_actions": len(steps)}
    if not steps:
        out.update(outcome="bank_skip_trace", abort_step=None, abort_kind=None,
                   step0_frame_divergence=False, final_lc=None)
        return out
    if len(steps) > max_actions:
        out.update(outcome="bank_skip_trace",
                   abort_step=None, abort_kind="over_max_replay_actions",
                   step0_frame_divergence=False, final_lc=None)
        return out

    prev = os.environ.get("ONLY_RESET_LEVELS")
    os.environ["ONLY_RESET_LEVELS"] = "true"   # replay under recorded semantics
    outcome, abort_step, abort_kind, final_lc = "survived", None, None, 0
    try:
        for i, st in enumerate(steps):
            try:
                resp = env.step(arcengine.GameAction.from_name(st.name),
                                data=dict(st.data))
            except Exception as exc:  # noqa: BLE001
                outcome, abort_step, abort_kind = "aborted", i, f"step_error:{exc}"
                break
            if resp is None or not getattr(resp, "frame", None):
                outcome, abort_step, abort_kind = "aborted", i, "empty_frame"
                break
            final_lc = int(getattr(resp, "levels_completed", 0) or 0)
            if final_lc != st.lc_after:
                outcome, abort_step, abort_kind = "aborted", i, "lc_divergence"
                break
            rh = ghash(resp.frame[-1])
            if st.grid_hash is not None and rh is not None and rh != st.grid_hash:
                outcome, abort_step, abort_kind = "aborted", i, "frame_divergence"
                break
    finally:
        if prev is None:
            os.environ.pop("ONLY_RESET_LEVELS", None)
        else:
            os.environ["ONLY_RESET_LEVELS"] = prev

    out.update(outcome=outcome, abort_step=abort_step, abort_kind=abort_kind,
               final_lc=final_lc,
               step0_frame_divergence=(abort_step == 0
                                       and abort_kind == "frame_divergence"))
    return out


def scorecard(game) -> dict:
    try:
        eng_id = game.env.environment_info.game_id
        sc = game._arcade.scorecard_manager.scorecards.get(game._scorecard_id)
        card = sc.cards.get(eng_id) if sc else None
        if card is None:
            return {}
        return {"total_plays": card.total_plays,
                "levels_per_play": list(card.levels_completed),
                "actions_per_play": list(card.actions)}
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------- per-game run
def run_game(prefix: str, rec: dict, max_actions: int) -> dict:
    import arc_agi
    import arcengine
    from taaf.game_api import ArcadeSpec, GameAPI
    from warpack_patch import TraceStep, prune_trace

    t0 = time.time()
    target_lc = int(rec["levels_completed"])
    history = [a["action"] for a in rec["history"]]
    out = {"game": prefix, "benchmark_game_id": rec["game_id"],
           "recorded_lc": target_lc, "recorded_actions": len(history),
           "in_splice_safe_11": prefix in SPLICE_SAFE_11,
           "in_confirmed_clean_10": prefix in CONFIRMED_CLEAN_10}

    if target_lc < 1:
        out.update(status="NO_BANK_TRACE",
                   reason="recorded levels_completed = 0; _bank never fires",
                   arms={}, wallclock_s=round(time.time() - t0, 1))
        return out

    os.environ.pop("ONLY_RESET_LEVELS", None)
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE,
                            environments_dir=str(ENV_FILES))
    local_id = next(e.game_id for e in arcade.available_environments
                    if e.game_id.startswith(prefix))
    out["local_game_id"] = local_id
    out["version_mismatch_vs_kaggle"] = local_id != rec["game_id"]

    game = None
    try:
        game = GameAPI(env_name=local_id,
                       arcade_spec=ArcadeSpec(
                           operation_mode=arc_agi.OperationMode.OFFLINE,
                           environments_dir=str(ENV_FILES)))
        game.start_game()
        game._finish_game = lambda: None      # keep the engine card open
        env = game.env
        seed_frame = game.current_state.raw.frame[-1]
        out["driver"] = "gameapi"
    except Exception as exc:  # noqa: BLE001
        os.environ.pop("ONLY_RESET_LEVELS", None)
        env = arcade.make(local_id, scorecard_id=arcade.create_scorecard())
        if env is None or env.observation_space is None:
            out.update(status="UNTESTABLE", reason=f"make failed: {exc}",
                       arms={}, wallclock_s=round(time.time() - t0, 1))
            return out
        os.environ["ONLY_RESET_LEVELS"] = "true"
        seed_frame = env.observation_space.frame[-1]
        out["driver"] = f"raw_env (gameapi failed: {type(exc).__name__})"

    # ---- pass 1: rebuild the TraceStep record on the local engine ----------
    prev_hash, prev_lc = ghash(seed_frame), 0
    trace: list = []
    for act in history:
        resp = env.step(arcengine.GameAction[act["id"]],
                        data=dict(act["data"] or {}))
        h = ghash(resp.frame[-1]) if getattr(resp, "frame", None) else None
        lc = int(getattr(resp, "levels_completed", 0) or 0)
        trace.append(TraceStep(name=act["id"], data=dict(act["data"] or {}),
                               board_changed=(h != prev_hash),
                               level_completed=(lc > prev_lc), lc_after=lc,
                               grid_hash=h, state_name=resp.state.name))
        prev_hash, prev_lc = h, lc
        if lc >= target_lc:
            break
    out["actions_fed"] = len(trace)
    out["lc_reached_on_local_engine"] = prev_lc
    out["local_engine_reproduced_recorded_lc"] = prev_lc >= target_lc

    variants = {
        "pruned": prune_trace(trace),
        "trailing_only": trailing_only(trace),
        "unpruned_full": list(trace),
    }
    out["arm_lengths"] = {k: len(v) for k, v in variants.items()}
    out["n_dropped_by_prune"] = len(trace) - len(variants["pruned"])

    # index in the recorded trace of pruned[0] -> what prune dropped before it
    if variants["pruned"]:
        p0 = variants["pruned"][0]
        idx0 = next((i for i, s in enumerate(trace)
                     if (s.name, s.data, s.grid_hash) == (p0.name, p0.data,
                                                          p0.grid_hash)), None)
        dropped = trace[:idx0] if idx0 is not None else []
        out["prune_dropped_before_pruned0"] = {
            "recorded_index_of_pruned0": idx0,
            "noops": sum(1 for s in dropped
                         if not s.board_changed and s.name != "RESET"),
            "board_changing": sum(1 for s in dropped
                                  if s.board_changed and s.name != "RESET"),
            "resets": sum(1 for s in dropped if s.name == "RESET"),
        }

    # ---- pass 2: one fresh play per arm -----------------------------------
    arms = {}
    for arm in ARMS:
        if not open_new_play(env):
            arms[arm] = {"outcome": "no_new_play", "abort_step": None,
                         "abort_kind": "no_new_play",
                         "step0_frame_divergence": False,
                         "n_replay_actions": len(variants[arm]),
                         "final_lc": None}
            continue
        arms[arm] = replay_arm(env, variants[arm], max_actions)
        arms[arm]["reached_recorded_lc"] = (
            arms[arm].get("final_lc") is not None
            and arms[arm]["final_lc"] >= target_lc)
    out["arms"] = arms
    if game is not None:
        out["scorecard"] = scorecard(game)
    out["status"] = "TESTED"
    out["wallclock_s"] = round(time.time() - t0, 1)
    return out


# ----------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(
        description="S1b offline bank re-fire, prune_trace DISABLED (offline, $0).")
    ap.add_argument("--games", default="", help="comma-separated game filter")
    ap.add_argument("--out", default="runs/r24_prep/s1b_bank_refire.json")
    ap.add_argument("--max-replay-actions", type=int, default=1500,
                    help="warpack cfg.bank_max_replay_actions (default 1500)")
    ap.add_argument("--authorized-by", default="UNSEALED-R24-PENDING")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify every asset + import, print the plan, "
                         "TOUCH NO ENGINE")
    args = ap.parse_args()

    missing = [str(p) for p in (BENCH, ENV_FILES, WARPACK / "warpack_patch.py")
               if not p.exists()]
    if missing:
        print("MISSING ASSETS:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 2

    bench = json.loads(BENCH.read_text(encoding="utf-8"))
    runs = {r["game_id"][:4]: r for r in bench["game_runs"]}
    only = {g.strip() for g in args.games.split(",") if g.strip()}
    games = sorted(g for g in runs if not only or g in only)

    if args.dry_run:
        print(f"S1b dry-run — repo {ROOT}")
        print(f"benchmark : {BENCH.relative_to(ROOT)}  ({len(runs)} game_runs, "
              f"{sum(len(r['history']) for r in runs.values())} recorded actions)")
        n_env = len([p for p in ENV_FILES.iterdir() if p.is_dir()])
        print(f"env files : {ENV_FILES.relative_to(ROOT)}  ({n_env} engines)")
        try:
            import arc_agi  # noqa: F401
            import arcengine  # noqa: F401
            from taaf.game_api import GameAPI  # noqa: F401
            from warpack_patch import TraceStep, prune_trace  # noqa: F401
            print("imports   : arcengine, arc_agi, taaf.game_api, "
                  "warpack_patch.{TraceStep,prune_trace}  OK")
        except Exception as exc:  # noqa: BLE001
            print(f"imports   : FAIL {type(exc).__name__}: {exc}")
            return 1
        bankable = [g for g in games if runs[g]["levels_completed"] >= 1]
        print(f"arms      : {', '.join(ARMS)}")
        print(f"planned   : {len(games)} games, {len(bankable)} with lc>=1 "
              f"(bankable); {len(games) - len(bankable)} NO_BANK_TRACE")
        print(f"gate set  : splice-safe 11 = {', '.join(SPLICE_SAFE_11)}")
        print(f"            confirmed-clean 10 = {', '.join(CONFIRMED_CLEAN_10)}")
        gate_bankable = [g for g in SPLICE_SAFE_11
                         if runs.get(g, {}).get("levels_completed", 0) >= 1]
        print(f"            of the 11, lc>=1 in war_eval_v1: "
              f"{len(gate_bankable)} ({', '.join(gate_bankable)})")
        print("no engine touched (--dry-run)")
        return 0

    t0 = time.time()
    rows = []
    for g in games:
        try:
            row = run_game(g, runs[g], args.max_replay_actions)
        except Exception as exc:  # noqa: BLE001
            row = {"game": g, "status": "ERROR",
                   "reason": f"{type(exc).__name__}: {exc}", "arms": {}}
        rows.append(row)
        a = row.get("arms", {})
        print(f"{g}: {row.get('status')} "
              + "  ".join(
                  f"{k}={a[k]['outcome']}@{a[k]['abort_step']}" for k in ARMS
                  if k in a), flush=True)

    def step0(arm):
        return sorted(r["game"] for r in rows
                      if r.get("arms", {}).get(arm, {}).get(
                          "step0_frame_divergence"))

    def survived(arm):
        return sorted(r["game"] for r in rows
                      if r.get("arms", {}).get(arm, {}).get("outcome")
                      == "survived")

    tested_11 = [r["game"] for r in rows
                 if r["game"] in SPLICE_SAFE_11 and r.get("status") == "TESTED"]
    s0_unpruned_11 = [g for g in step0("unpruned_full") if g in SPLICE_SAFE_11]
    s0_unpruned_10 = [g for g in step0("unpruned_full") if g in CONFIRMED_CLEAN_10]

    out = {
        "provenance": {
            "script": "duck_eval/r24_prep/s1b_bank_refire_noprune.py",
            "script_sha256": sha256_file(Path(__file__)),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "git": git_info(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "proposal": "learnings/war_room/r24_successor_lane_proposal_2026-08-08.md §4 S1b",
            "authorized_by": args.authorized_by,
            "cost": {"usd": 0, "kaggle_pushes": 0, "gpu": "none (CPU only)"},
            "rng": "none — replay is fully deterministic; no seeds are drawn",
            "trace_source": "runs/kernel_pulls/war_eval_v1/benchmark.json",
            "trace_source_sha256": sha256_file(BENCH),
            "warpack_patch_sha256": sha256_file(WARPACK / "warpack_patch.py"),
            "engines": "kaggle-data/environment_files (local offline arcengine)",
            "wallclock_s": round(time.time() - t0, 1),
        },
        "config": {
            "arms": list(ARMS),
            "divergence_semantics": "warpack _bank verbatim: per-step grid_hash "
                                    "+ levels_completed comparison; new play via "
                                    "<=2 RESETs at ONLY_RESET_LEVELS=false",
            "bank_max_replay_actions": args.max_replay_actions,
            "prune_disable_method": "arm selector over the trace; warpack_patch "
                                    "is imported unmodified",
            "splice_safe_11": SPLICE_SAFE_11,
            "confirmed_clean_10": CONFIRMED_CLEAN_10,
            "gate": "step-0 frame_divergence must clear on the 11-game "
                    "prefix-splice-safe set with pruning disabled",
        },
        "games": rows,
        "summary": {
            "n_games": len(rows),
            "n_tested": sum(1 for r in rows if r.get("status") == "TESTED"),
            "n_no_bank_trace": sum(1 for r in rows
                                   if r.get("status") == "NO_BANK_TRACE"),
            "step0_frame_divergence": {a: step0(a) for a in ARMS},
            "survived": {a: survived(a) for a in ARMS},
            "gate_11_tested": sorted(tested_11),
            "gate_11_step0_divergent_unpruned": s0_unpruned_11,
            "gate_11_verdict": "CLEAR" if not s0_unpruned_11 else "FAIL",
            "gate_10_strict_step0_divergent_unpruned": s0_unpruned_10,
            "gate_10_verdict": "CLEAR" if not s0_unpruned_10 else "FAIL",
            "note": "gate_11 follows the R24 proposal text; gate_10 follows the "
                    "R17 SEALED banking rule, which excludes tn36 from splice",
        },
    }
    op = ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nstep-0 frame_divergence (unpruned_full): "
          f"{step0('unpruned_full') or 'NONE'}")
    print(f"gate_11 = {out['summary']['gate_11_verdict']}  "
          f"gate_10 = {out['summary']['gate_10_verdict']}")
    print(f"written: {op}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
