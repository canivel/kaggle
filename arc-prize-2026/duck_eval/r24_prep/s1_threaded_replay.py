"""S1 / L0 — state-threaded, abstention-aware re-verification of the exec_wm sims.

R24 proposal `learnings/war_room/r24_successor_lane_proposal_2026-08-08.md` §4 row S1:

    "offline re-verification of the 24 existing sims under Tycho's protocol:
     replay from level frame 0 WITH STATE THREADING, report accepted transition
     match AND coverage per game, on-trajectory"

This is a THIN WRAPPER: it imports nothing from the sims but their public
`simulate(state, action_id, x, y) -> (next_state, reward_class, done)` contract
(the same contract `exec_wm/validate_sim.py` documents) and re-uses the trace
parsing conventions of `scripts/ewm_replay_dryrun.py`. It does not modify any
existing campaign file.

WHAT IS DIFFERENT FROM THE 2026-07-18 DRY-RUN (`scripts/ewm_replay_dryrun.py`)
-----------------------------------------------------------------------------
1. **State threading.** The old harness was TEACHER-FORCED: every prediction
   started from the recorded pre-action frame, so mismatches never cascaded and
   the reported number was a per-frame independent accuracy. Here the sim's own
   output is fed back in (`pred_{t+1} = simulate(pred_t, a_t)`) for the whole
   level-segment, which is what an executor would actually experience. The
   teacher-forced number is still computed in the same pass so the two
   protocols are reported side by side (arm A/B, zero extra runs).
2. **Level-segment episodes.** Replay restarts at "level frame 0": the recorded
   `initial` frame, every post-RESET frame, and every post-level-completion
   frame. The old harness never restarted and never reset sim module state.
3. **Module-state reset.** g50t/re86/tr87 carry module-level hidden counters
   (`reset_state` / `reset_phase` / `reset_step_parity`). The old harness loaded
   the module once per game and never reset it, so its counters were desynced
   from the first segment boundary onwards. Here the reset hook (when the sim
   exposes one) is called at every segment start.
4. **Coverage / abstention.** The sims have NO abstention channel (no
   `UNKNOWN=-1`; the return is a 3-tuple with a concrete grid). Coverage is
   therefore reported under an explicit operational proxy, three ways, so the
   panel can pick the one it wants to seal:
       coverage_strict     = (steps - errors) / steps
       coverage_committed  = (steps - errors - identity_abstentions) / steps
       identity abstention = the sim returned its input frame unchanged WHILE
                             the recorded transition did change the board
                             (several sims document this as their explicit
                             "we can't tell from one frame" fallback, e.g.
                             g50t action 5).
   `accepted_match` is the match rate over COMMITTED steps only, per Tycho.
   `match_all_steps` (matches / all steps) is also emitted so the result is
   directly comparable to the 2026-07-18 numbers.

Usage
-----
    .venv/Scripts/python.exe duck_eval/r24_prep/s1_threaded_replay.py --dry-run
    .venv/Scripts/python.exe duck_eval/r24_prep/s1_threaded_replay.py \
        --source war_eval_v1 --out runs/r24_prep/s1_threaded_replay.json

CPU-only, read-only w.r.t. sims and traces, no network, no Kaggle push, $0.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SIM_DIR = ROOT / "exec_wm" / "sims"

SOURCES = {
    "war_eval_v1": ROOT / "runs/kernel_pulls/war_eval_v1/artifacts",
    "war_eval_v2": ROOT / "runs/kernel_pulls/war_eval_v2/artifacts",
    "war_eval_v3": ROOT / "runs/kernel_pulls/war_eval_v3/artifacts",
    "gpt56_full": ROOT / "runs/gpt56_probe/experiment_full/artifacts",
}
PRIMARY_SOURCE = "war_eval_v1"

MOUSE_RE = re.compile(r"MOUSE\(row=(\d+), col=(\d+)\)")
RESET_HOOKS = ("reset_state", "reset_phase", "reset_step_parity")

# Held-out state_exact% from exec_wm/scale_summary.md (24 games; bp35 was never
# scale-validated and is carried as None).
HELD_OUT = {
    "ar25": 80.0, "cd82": 60.5, "cn04": 77.5, "dc22": 50.5, "ft09": 100.0,
    "g50t": 73.0, "ka59": 60.5, "lf52": 100.0, "lp85": 100.0, "ls20": 100.0,
    "m0r0": 57.5, "r11l": 23.0, "re86": 90.5, "s5i5": 99.5, "sb26": 100.0,
    "sc25": 72.5, "sk48": 38.0, "sp80": 100.0, "su15": 99.5, "tn36": 100.0,
    "tr87": 100.0, "tu93": 100.0, "vc33": 99.5, "wa30": 65.0, "bp35": None,
}

# Prior carrier set that S1's gate must beat, from
# learnings/stuck_review_v2_2026-07-23.md L13:
#   "EWM clean carrier set shrinks to {tn36, tu93, ls20, ft09-L1}"
PRIOR_CARRIER_SET = ["ft09", "ls20", "tn36", "tu93"]

# Engine-determinism verdicts, learnings/daily_brief_2026-07-20.md L37.
ENGINE_VERDICT = {
    **{g: "CLEAN" for g in ("ar25", "bp35", "ft09", "lf52", "lp85", "ls20",
                            "r11l", "sp80", "su15", "tn36", "tu93")},
    **{g: "ALIASED-RESOLVABLE" for g in ("cd82", "cn04", "dc22", "ka59", "re86",
                                         "s5i5", "sb26", "sc25", "tr87", "vc33",
                                         "wa30")},
    **{g: "ALIASED-UNRESOLVED" for g in ("g50t", "sk48", "m0r0")},
}

# PROPOSED, NOT SEALED. The R24 minute must fix these before the run counts as
# evidence, otherwise the gate is post-hoc. Derived from the criterion actually
# used on 2026-07-20 ("step_acc 0.92-0.97 AND clean/resolvable").
DEFAULT_CARRIER_MATCH_MIN = 0.92
DEFAULT_CARRIER_COVERAGE_MIN = 0.50
DEFAULT_CARRIER_REQUIRE_RESOLVABLE = True


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


def load_sim(game_id: str, tag: str):
    """Fresh, independently-namespaced instance of exec_wm/sims/<gid>_sim.py."""
    p = SIM_DIR / f"{game_id}_sim.py"
    spec = importlib.util.spec_from_file_location(f"{game_id}_sim_{tag}", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "simulate"):
        raise AttributeError(f"{p}: no simulate()")
    return mod


def sim_reset(mod) -> str | None:
    for h in RESET_HOOKS:
        fn = getattr(mod, h, None)
        if callable(fn):
            fn()
            return h
    return None


def parse_action(ev):
    """action event -> (aid, x, y); None for RESET/unknown (control events)."""
    name = ev.get("action_name") or ""
    if name == "RESET":
        return None
    m = re.fullmatch(r"ACTION(\d)", name)
    if not m:
        return None
    aid = int(m.group(1))
    x = y = 0
    if aid == 6:
        mm = MOUSE_RE.search(ev.get("action_display") or "")
        if mm:
            y, x = int(mm.group(1)), int(mm.group(2))  # engine data{x=col,y=row}
    return aid, x, y


def segments(events):
    """Split a recorded event stream into level-episodes ('level frame 0').

    A segment is (seed_board, seed_level, [action_events]). A new segment
    starts at the `initial` frame, at every post-RESET frame, and at every
    post-level-completion frame.
    """
    segs, seed, seed_lvl, cur = [], None, 0, []
    for ev in events:
        t = ev.get("type")
        if t == "initial":
            seed, seed_lvl, cur = ev["board"], ev.get("level", 0), []
            continue
        if t != "action":
            continue
        if (ev.get("action_name") or "") == "RESET":
            if seed is not None and cur:
                segs.append((seed, seed_lvl, cur))
            seed, seed_lvl, cur = ev.get("board"), ev.get("level", 0), []
            continue
        if seed is None:
            continue
        cur.append(ev)
        if ev.get("level_completed"):
            segs.append((seed, seed_lvl, cur))
            seed, seed_lvl, cur = ev.get("board"), ev.get("level", 0), []
    if seed is not None and cur:
        segs.append((seed, seed_lvl, cur))
    return segs


# --------------------------------------------------------------------- replay
def replay_game(gid: str, fp: Path) -> dict:
    """State-threaded (+ teacher-forced A/B) replay of one recorded trace."""
    with open(fp, encoding="utf-8") as f:
        events = [json.loads(ln) for ln in f if ln.strip()]

    mod_a = load_sim(gid, "a")
    mod_b = load_sim(gid, "b")
    sim_a, sim_b = mod_a.simulate, mod_b.simulate
    reset_hook = None

    r = {
        "n_segments": 0, "n_steps": 0,
        "n_error": 0, "n_selfdiff": 0,
        "n_identity_abstain": 0, "n_obs_changed": 0,
        "n_match_threaded": 0, "n_match_threaded_committed": 0,
        "n_committed": 0,
        "n_match_teacher_forced": 0,
        "n_done_agree": 0, "n_done_total": 0,
        "n_reward_total": 0,
        "survival_steps": [],      # threaded steps before first mismatch, per segment
        "segment_lengths": [],
        "per_action": {},          # aid -> {n, match_threaded, match_tf, abstain, error}
    }

    for seed_board, _seed_lvl, acts in segments(events):
        steps = [e for e in acts if parse_action(e) is not None]
        if not steps or seed_board is None:
            continue
        r["n_segments"] += 1
        r["segment_lengths"].append(len(steps))
        reset_hook = sim_reset(mod_a) or reset_hook
        sim_reset(mod_b)

        pred = [list(row) for row in seed_board]
        obs_prev = seed_board
        survived = 0
        broken = False

        for ev in steps:
            aid, x, y = parse_action(ev)
            obs_next = ev["board"]
            pa = r["per_action"].setdefault(
                str(aid), {"n": 0, "match_threaded": 0, "match_tf": 0,
                           "abstain": 0, "error": 0})
            pa["n"] += 1
            r["n_steps"] += 1
            obs_changed = obs_next != obs_prev
            if obs_changed:
                r["n_obs_changed"] += 1

            # ---- threaded prediction (the S1 protocol) -------------------
            err = False
            try:
                p1 = sim_a([list(row) for row in pred], aid, x, y)
                p2 = sim_b([list(row) for row in pred], aid, x, y)
                a1 = np.asarray(p1[0], dtype=np.uint8)
                a2 = np.asarray(p2[0], dtype=np.uint8)
                if a1.shape != (64, 64):
                    raise ValueError(f"shape {a1.shape}")
                if not np.array_equal(a1, a2):
                    r["n_selfdiff"] += 1
                pred_next = a1.tolist()
            except Exception:  # noqa: BLE001
                err = True
                r["n_error"] += 1
                pa["error"] += 1
                pred_next = pred  # hold state; the step is uncommitted

            if not err:
                identity = pred_next == pred
                abstain = identity and obs_changed
                if abstain:
                    r["n_identity_abstain"] += 1
                    pa["abstain"] += 1
                else:
                    r["n_committed"] += 1
                match = pred_next == obs_next
                if match:
                    r["n_match_threaded"] += 1
                    pa["match_threaded"] += 1
                    if not abstain:
                        r["n_match_threaded_committed"] += 1
                    if not broken:
                        survived += 1
                else:
                    broken = True
                r["n_done_total"] += 1
                if bool(p1[2]) == bool(ev.get("level_completed", False)):
                    r["n_done_agree"] += 1
                r["n_reward_total"] += 1
            else:
                broken = True

            # ---- teacher-forced prediction (legacy protocol, same pass) ---
            try:
                q = sim_a([list(row) for row in obs_prev], aid, x, y)
                qa = np.asarray(q[0], dtype=np.uint8)
                if qa.shape == (64, 64) and qa.tolist() == obs_next:
                    r["n_match_teacher_forced"] += 1
                    pa["match_tf"] += 1
            except Exception:  # noqa: BLE001
                pass

            pred = pred_next
            obs_prev = obs_next

        r["survival_steps"].append(survived)

    r["reset_hook"] = reset_hook
    return r


def score_game(gid: str, source: str, trace_id: str, fp: Path, r: dict,
               cfg: dict) -> dict:
    steps = r["n_steps"]
    committed = r["n_committed"]
    out = {
        "game": gid,
        "source": source,
        "trace_id": trace_id,
        "trace_file": str(fp.relative_to(ROOT)).replace("\\", "/"),
        "sim_file": f"exec_wm/sims/{gid}_sim.py",
        "sim_sha256": sha256_file(SIM_DIR / f"{gid}_sim.py"),
        "sim_reset_hook": r["reset_hook"],
        "sim_has_module_state": r["reset_hook"] is not None,
        "engine_determinism_verdict": ENGINE_VERDICT.get(gid, "UNKNOWN"),
        "held_out_state_exact_pct": HELD_OUT.get(gid),
        "n_segments": r["n_segments"],
        "n_steps": steps,
        "segment_lengths": r["segment_lengths"],
        "n_error": r["n_error"],
        "n_selfdiff": r["n_selfdiff"],
        "n_obs_changed": r["n_obs_changed"],
        "n_identity_abstain": r["n_identity_abstain"],
        "n_committed": committed,
        # ---- the two S1 metrics ------------------------------------------
        "accepted_match": (r["n_match_threaded_committed"] / committed
                           if committed else None),
        "coverage": (committed / steps) if steps else None,
        "coverage_strict": ((steps - r["n_error"]) / steps) if steps else None,
        # ---- comparability channels --------------------------------------
        "match_all_steps": (r["n_match_threaded"] / steps) if steps else None,
        "teacher_forced_match_all_steps": (r["n_match_teacher_forced"] / steps
                                           if steps else None),
        "mean_survival_steps": (sum(r["survival_steps"]) / len(r["survival_steps"])
                                if r["survival_steps"] else None),
        "max_survival_steps": max(r["survival_steps"], default=None),
        "done_flag_agree": (f"{r['n_done_agree']}/{r['n_done_total']}"
                            if r["n_done_total"] else None),
        "per_action": r["per_action"],
    }
    am, cv = out["accepted_match"], out["coverage"]
    reasons = []
    if am is None or am < cfg["carrier_match_min"]:
        reasons.append(f"accepted_match {am} < {cfg['carrier_match_min']}")
    if cv is None or cv < cfg["carrier_coverage_min"]:
        reasons.append(f"coverage {cv} < {cfg['carrier_coverage_min']}")
    if (cfg["carrier_require_resolvable"]
            and out["engine_determinism_verdict"] == "ALIASED-UNRESOLVED"):
        reasons.append("engine ALIASED-UNRESOLVED")
    out["carrier"] = not reasons
    out["carrier_fail_reasons"] = reasons
    return out


# ----------------------------------------------------------------------- main
def discover(source: str) -> list[tuple[str, str, Path]]:
    art = SOURCES[source]
    found = []
    for fp in sorted(glob.glob(str(art / "*_events.jsonl"))):
        name = Path(fp).name             # e.g. sp80-589a99af_p0_events.jsonl
        trace_id = name.split("_")[0]
        gid = trace_id.split("-")[0]
        if not (SIM_DIR / f"{gid}_sim.py").exists():
            continue
        found.append((gid, trace_id, Path(fp)))
    return found


def main() -> int:
    ap = argparse.ArgumentParser(
        description="S1/L0 state-threaded exec_wm re-verification (offline, $0).")
    ap.add_argument("--source", default=PRIMARY_SOURCE,
                    choices=[*SOURCES, "all"],
                    help="recorded trace source (default: war_eval_v1)")
    ap.add_argument("--games", default="", help="comma-separated game filter")
    ap.add_argument("--out", default="runs/r24_prep/s1_threaded_replay.json")
    ap.add_argument("--carrier-match-min", type=float,
                    default=DEFAULT_CARRIER_MATCH_MIN)
    ap.add_argument("--carrier-coverage-min", type=float,
                    default=DEFAULT_CARRIER_COVERAGE_MIN)
    ap.add_argument("--allow-unresolvable-carriers", action="store_true")
    ap.add_argument("--authorized-by", default="UNSEALED-R24-PENDING",
                    help="R24 minute reference; recorded in the provenance header")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve + import every asset, print the plan, run NOTHING")
    args = ap.parse_args()

    srcs = list(SOURCES) if args.source == "all" else [args.source]
    only = {g.strip() for g in args.games.split(",") if g.strip()}
    cfg = {
        "mode": "state_threaded",
        "segment_rule": "initial | post-RESET frame | post-level-completion frame",
        "reset_module_state_at_segment_start": True,
        "teacher_forced_ab_in_same_pass": True,
        "coverage_definition": {
            "commit": "sim returned a valid 64x64 grid AND did not return its "
                      "input unchanged while the recorded transition changed "
                      "the board",
            "abstention_channel_in_sims": False,
            "note": "the sims expose no UNKNOWN/-1 channel; identity-on-change "
                    "is an OPERATIONAL PROXY and must be sealed by R24",
        },
        "carrier_match_min": args.carrier_match_min,
        "carrier_coverage_min": args.carrier_coverage_min,
        "carrier_require_resolvable": not args.allow_unresolvable_carriers,
        "carrier_thresholds_status": "PROPOSED — NOT SEALED BY R24",
        "prior_carrier_set": PRIOR_CARRIER_SET,
        "gate": "carrier set must EXPAND beyond the prior 4",
    }

    plan = []
    for s in srcs:
        if not SOURCES[s].is_dir():
            print(f"MISSING source dir: {SOURCES[s]}", file=sys.stderr)
            return 2
        for gid, tid, fp in discover(s):
            if only and gid not in only:
                continue
            plan.append((s, gid, tid, fp))

    if args.dry_run:
        print(f"S1 dry-run — repo {ROOT}")
        print(f"sim dir: {SIM_DIR}  ({len(list(SIM_DIR.glob('*_sim.py')))} *_sim.py "
              f"files, active sims exclude *_v1/_v2 variants)")
        bad = 0
        seen = set()
        for s, gid, tid, fp in plan:
            if gid in seen:
                continue
            seen.add(gid)
            try:
                m = load_sim(gid, "dry")
                hook = next((h for h in RESET_HOOKS if callable(getattr(m, h, None))),
                            None)
                print(f"  OK  {gid:5s} sim import + simulate(); reset_hook={hook}; "
                      f"trace={fp.name}")
            except Exception as exc:  # noqa: BLE001
                bad += 1
                print(f"  FAIL {gid:5s} {type(exc).__name__}: {exc}")
        print(f"\nplanned: {len(plan)} (source, game) replays over "
              f"{len(seen)} distinct sims; {bad} sim import failures")
        print(f"carrier thresholds (PROPOSED, NOT SEALED): "
              f"accepted_match >= {cfg['carrier_match_min']}, "
              f"coverage >= {cfg['carrier_coverage_min']}, "
              f"require_resolvable={cfg['carrier_require_resolvable']}")
        print("no experiment executed (--dry-run)")
        return 0 if bad == 0 else 1

    t0 = time.time()
    rows = []
    for s, gid, tid, fp in plan:
        r = replay_game(gid, fp)
        row = score_game(gid, s, tid, fp, r, cfg)
        rows.append(row)
        print(f"[{s}] {gid}: accepted_match={row['accepted_match']} "
              f"coverage={row['coverage']} "
              f"(all-steps {row['match_all_steps']}, "
              f"TF {row['teacher_forced_match_all_steps']}) "
              f"carrier={row['carrier']}", flush=True)

    primary = [r for r in rows if r["source"] == PRIMARY_SOURCE] or rows
    carriers = sorted({r["game"] for r in primary if r["carrier"]})
    out = {
        "provenance": {
            "script": "duck_eval/r24_prep/s1_threaded_replay.py",
            "script_sha256": sha256_file(Path(__file__)),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "git": git_info(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "proposal": "learnings/war_room/r24_successor_lane_proposal_2026-08-08.md §4 S1",
            "authorized_by": args.authorized_by,
            "cost": {"usd": 0, "kaggle_pushes": 0, "gpu": "none (CPU only)"},
            "rng": "none — replay is fully deterministic; no seeds are drawn",
            "sources": {s: str(SOURCES[s].relative_to(ROOT)).replace("\\", "/")
                        for s in srcs},
            "wallclock_s": round(time.time() - t0, 1),
        },
        "config": cfg,
        "games": rows,
        "summary": {
            "n_rows": len(rows),
            "primary_source": PRIMARY_SOURCE,
            "n_games_primary": len({r["game"] for r in primary}),
            "carrier_set": carriers,
            "n_carriers": len(carriers),
            "prior_carrier_set": PRIOR_CARRIER_SET,
            "n_prior_carriers": len(PRIOR_CARRIER_SET),
            "gained": sorted(set(carriers) - set(PRIOR_CARRIER_SET)),
            "lost": sorted(set(PRIOR_CARRIER_SET) - set(carriers)),
            "gate": "carrier set must EXPAND beyond ~4 games",
            "verdict": ("EXPANDED" if len(carriers) > len(PRIOR_CARRIER_SET)
                        else "NOT_EXPANDED"),
            "verdict_status": "ADVISORY until R24 seals the carrier thresholds",
        },
    }
    op = ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\ncarrier set ({len(carriers)}): {', '.join(carriers) or '-'}  "
          f"prior {PRIOR_CARRIER_SET} -> {out['summary']['verdict']}")
    print(f"written: {op}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
