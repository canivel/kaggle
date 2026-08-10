"""S1 / L0 — state-threaded re-verification of the exec_wm sims.

SEALED SPEC: `duck_eval/r24_prep/s1_sealed_spec_2026-08-10.md` §9, committed at
`fd57f31bda84260d6f45a5de13b73b101316902f` BEFORE this runner was amended and
before any result existed. The prereg commit SHA is recorded in every artifact
this script writes.

WHAT CHANGED ON 2026-08-10 (§9.3 of the sealed spec)
----------------------------------------------------
The original version of this file (preserved at the prereg commit) implemented
the R24 proposal's S1 as written. R24 held that S1 because its endpoint was
uninterpretable, and the seal retired it:

  * REMOVED — `coverage`, `coverage_strict`, `accepted_match`,
    `n_identity_abstain`, `n_committed`, `carrier*`, `verdict: EXPANDED/
    NOT_EXPANDED`, `gained`, `lost`.  `coverage_strict` is measured at exactly
    1.0 (0 sim errors, 0 selfdiffs over 4,996 banked steps) and the
    identity-abstention proxy was computed from the observed label, so it was
    circular AND mechanically inflated `accepted_match`.  Emitting any of these
    fields VOIDS the run.
  * NEW PRIMARY (E1) — the threaded survival horizon.  Per (game, source,
    segment): the number of consecutive threaded steps from segment start that
    exactly match the recorded settled frame, before the first mismatch.  Per
    (game, source): the MEDIAN over that source's segments.  Per game:
    `H_g = MIN over the three war_eval sources` of that median.  Threshold
    `H_g >= 10` (R16 §9.2's registered executor plan depth).
    E1 is computed ONLY over the 13 games that have never been replayed
    on-trajectory; the 12 SAT12 games are replayed as an anchor and are
    explicitly barred from contributing.
  * NEW E3 — `module_reset_delta_steps`: the whole replay is run a second time
    with the module reset hooks suppressed, and the two prediction streams are
    compared step by step.  Pre-registered to be exactly 0 for tr87 on all
    three sources and for g50t on v1/v2; a non-zero value there is a runner
    defect and VOIDS the run.
  * NEW E4 — `engine_matches_kaggle` on every row; every summary reported both
    pooled and split on it.
  * SOURCES — `gpt56_full` removed (2 of its 5 streams carry different engine
    hashes from the war-eval build).  `--source` now takes a comma-separated
    list and defaults to all three war_eval pulls.

`match_all_steps` and `teacher_forced_match_all_steps` are retained as
comparability channels to `runs/ewm_dryrun/report.md` and are explicitly
NON-GATING.

CPU-only, read-only w.r.t. sims and traces, no network, no Kaggle push, $0.
Deterministic replay of recorded data: no RNG is drawn, so one run is the
complete result and there is no seed to vary.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import platform
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SIM_DIR = ROOT / "exec_wm" / "sims"
SPEC = "duck_eval/r24_prep/s1_sealed_spec_2026-08-10.md"
PREREG_COMMIT = "fd57f31bda84260d6f45a5de13b73b101316902f"

# gpt56_full REMOVED — sealed spec §5.2 (su15 4c352 vs 1944f, vc33 9851e vs 54305).
SOURCES = {
    "war_eval_v1": ROOT / "runs/kernel_pulls/war_eval_v1/artifacts",
    "war_eval_v2": ROOT / "runs/kernel_pulls/war_eval_v2/artifacts",
    "war_eval_v3": ROOT / "runs/kernel_pulls/war_eval_v3/artifacts",
}
ALL_SOURCES = list(SOURCES)

MOUSE_RE = re.compile(r"MOUSE\(row=(\d+), col=(\d+)\)")
RESET_HOOKS = ("reset_state", "reset_phase", "reset_step_parity")

# ---- the sealed E1 partition (spec §3, §9.1). Frozen; may not be edited. -----
NEVER_MEASURED_13 = ["ar25", "bp35", "cd82", "cn04", "dc22", "g50t", "ka59",
                     "m0r0", "r11l", "re86", "sc25", "sk48", "wa30"]
ANCHOR_SAT12 = ["ft09", "lf52", "lp85", "ls20", "s5i5", "sb26", "sp80", "su15",
                "tn36", "tr87", "tu93", "vc33"]

# ---- the sealed threshold and decision rule (spec §9.1, §9.2). Frozen. -------
H_THRESHOLD = 10
PREREGISTERED_E1_EXPECTATION = 0
# E3 void conditions: (game, source) pairs whose module_reset_delta MUST be 0.
E3_PREREGISTERED_ZEROS = [("tr87", "war_eval_v1"), ("tr87", "war_eval_v2"),
                          ("tr87", "war_eval_v3"), ("g50t", "war_eval_v1"),
                          ("g50t", "war_eval_v2")]

# In-sample state_exact% from exec_wm/scale_summary.md:22-45 (NOT held out —
# --split all; see the CORRECTION block in that file). bp35 never validated.
IN_SAMPLE_STATE_EXACT = {
    "ar25": 80.0, "cd82": 60.5, "cn04": 77.5, "dc22": 50.5, "ft09": 100.0,
    "g50t": 73.0, "ka59": 60.5, "lf52": 100.0, "lp85": 100.0, "ls20": 100.0,
    "m0r0": 57.5, "r11l": 23.0, "re86": 90.5, "s5i5": 99.5, "sb26": 100.0,
    "sc25": 72.5, "sk48": 38.0, "sp80": 100.0, "su15": 99.5, "tn36": 100.0,
    "tr87": 100.0, "tu93": 100.0, "vc33": 99.5, "wa30": 65.0, "bp35": None,
}


# ------------------------------------------------------------------ utilities
def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def hash_grid(g) -> str:
    return hashlib.blake2b(json.dumps(g, separators=(",", ":")).encode(),
                           digest_size=8).hexdigest()


def git_info() -> dict:
    def _run(*a):
        try:
            return subprocess.run(a, cwd=str(ROOT), capture_output=True,
                                  text=True, timeout=20).stdout.strip()
        except Exception:  # noqa: BLE001
            return "unavailable"
    return {"commit": _run("git", "rev-parse", "HEAD"),
            "dirty": bool(_run("git", "status", "--porcelain"))}


def engine_match_table() -> dict:
    """engine_matches_kaggle per game, from the 25-game determinism audit."""
    p = ROOT / "runs" / "war_eval_v1" / "determinism_audit_25.json"
    out = {}
    if p.exists():
        for g in json.loads(p.read_text(encoding="utf-8")).get("games", []):
            gid = (g.get("game") or "").split("-")[0]
            if gid:
                out[gid] = not g.get("version_mismatch_vs_kaggle", False)
    return out


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
def replay_game(gid: str, fp: Path, do_reset: bool = True) -> dict:
    """State-threaded replay of one recorded trace.

    `do_reset=True`  -> module reset hook fired at every segment start (S1).
    `do_reset=False` -> hooks never fired, i.e. the 2026-07-18 harness's
                        behaviour; used only to measure `module_reset_delta_steps`.
    The teacher-forced channel is computed in the same pass (arm A/B, no extra
    run) and is NON-GATING.
    """
    with open(fp, encoding="utf-8") as f:
        events = [json.loads(ln) for ln in f if ln.strip()]

    mod_a = load_sim(gid, "a")
    mod_b = load_sim(gid, "b")
    sim_a, sim_b = mod_a.simulate, mod_b.simulate
    reset_hook = next((h for h in RESET_HOOKS
                       if callable(getattr(mod_a, h, None))), None)

    r = {
        "n_segments": 0, "n_steps": 0,
        "n_error": 0, "n_selfdiff": 0,
        "n_match_threaded": 0, "n_match_teacher_forced": 0,
        "n_done_agree": 0, "n_done_total": 0,
        "segment_survivals": [],   # threaded steps survived, per segment
        "segment_lengths": [],
        "pred_hashes": [],         # per-step predicted-state hash (for E3)
        "reset_hook": reset_hook,
    }

    for seed_board, _seed_lvl, acts in segments(events):
        steps = [e for e in acts if parse_action(e) is not None]
        if not steps or seed_board is None:
            continue
        r["n_segments"] += 1
        r["segment_lengths"].append(len(steps))
        if do_reset:
            sim_reset(mod_a)
            sim_reset(mod_b)

        pred = [list(row) for row in seed_board]
        obs_prev = seed_board
        survived = 0
        broken = False

        for ev in steps:
            aid, x, y = parse_action(ev)
            obs_next = ev["board"]
            r["n_steps"] += 1

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
                pred_next = pred  # hold state

            r["pred_hashes"].append(hash_grid(pred_next))

            if not err:
                if pred_next == obs_next:
                    r["n_match_threaded"] += 1
                    if not broken:
                        survived += 1
                else:
                    broken = True
                r["n_done_total"] += 1
                if bool(p1[2]) == bool(ev.get("level_completed", False)):
                    r["n_done_agree"] += 1
            else:
                broken = True

            # ---- teacher-forced channel (legacy protocol, NON-GATING) -----
            try:
                q = sim_a([list(row) for row in obs_prev], aid, x, y)
                qa = np.asarray(q[0], dtype=np.uint8)
                if qa.shape == (64, 64) and qa.tolist() == obs_next:
                    r["n_match_teacher_forced"] += 1
            except Exception:  # noqa: BLE001
                pass

            pred = pred_next
            obs_prev = obs_next

        r["segment_survivals"].append(survived)

    return r


def score_row(gid: str, source: str, trace_id: str, fp: Path, r: dict,
              delta_steps: int, engine_match) -> dict:
    steps = r["n_steps"]
    surv = r["segment_survivals"]
    return {
        "game": gid,
        "source": source,
        "trace_id": trace_id,
        "trace_file": str(fp.relative_to(ROOT)).replace("\\", "/"),
        "sim_file": f"exec_wm/sims/{gid}_sim.py",
        "sim_sha256": sha256_file(SIM_DIR / f"{gid}_sim.py"),
        "sim_reset_hook": r["reset_hook"],
        "sim_has_module_state": r["reset_hook"] is not None,
        "engine_matches_kaggle": engine_match,
        "in_sample_state_exact_pct": IN_SAMPLE_STATE_EXACT.get(gid),
        "n_segments": r["n_segments"],
        "n_steps": steps,
        "segment_lengths": r["segment_lengths"],
        # ---- E1 primary --------------------------------------------------
        "segment_survivals": surv,
        "median_survival": (statistics.median(surv) if surv else None),
        "mean_survival": (round(sum(surv) / len(surv), 3) if surv else None),
        "max_survival": (max(surv) if surv else None),
        # ---- E3 ----------------------------------------------------------
        "module_reset_delta_steps": delta_steps,
        # ---- comparability channels, NON-GATING --------------------------
        "match_all_steps": (round(r["n_match_threaded"] / steps, 4)
                            if steps else None),
        "teacher_forced_match_all_steps": (
            round(r["n_match_teacher_forced"] / steps, 4) if steps else None),
        "n_error": r["n_error"],
        "n_selfdiff": r["n_selfdiff"],
        "done_flag_agree": (f"{r['n_done_agree']}/{r['n_done_total']}"
                            if r["n_done_total"] else None),
    }


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


def horizon_block(games: list[str], per_game: dict) -> dict:
    """H_g = min over sources of median_survival, + the sealed >=10 test."""
    rows, passes = {}, []
    for g in games:
        med = per_game.get(g, {})
        vals = [med[s] for s in ALL_SOURCES if med.get(s) is not None]
        H = min(vals) if len(vals) == len(ALL_SOURCES) else None
        rows[g] = {
            "median_survival_per_source": {s: med.get(s) for s in ALL_SOURCES},
            "H_g": H,
            "n_sources": len(vals),
            "passes_H_ge_10": (H is not None and H >= H_THRESHOLD),
        }
        if rows[g]["passes_H_ge_10"]:
            passes.append(g)
    return {"n_games": len(games), "games": rows,
            "n_pass_H_ge_10": len(passes), "pass_list": sorted(passes)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="S1/L0 state-threaded exec_wm horizon measurement "
                    "(offline, $0). Sealed spec: " + SPEC)
    ap.add_argument("--source", default=",".join(ALL_SOURCES),
                    help="comma-separated list (default: all three war_eval pulls)")
    ap.add_argument("--games", default="", help="comma-separated game filter")
    ap.add_argument("--out", default="runs/r24_prep/s1_threaded_replay.json")
    ap.add_argument("--authorized-by", default="UNSEALED",
                    help="sealed-spec reference; recorded in the provenance header")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve + import every asset, print the plan, run NOTHING")
    args = ap.parse_args()

    srcs = [s.strip() for s in args.source.split(",") if s.strip()]
    for s in srcs:
        if s not in SOURCES:
            print(f"unknown source {s!r}; known: {ALL_SOURCES}", file=sys.stderr)
            return 2
    only = {g.strip() for g in args.games.split(",") if g.strip()}
    emt = engine_match_table()

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
        print(f"S1 dry-run — repo {ROOT}; spec {SPEC} @ {PREREG_COMMIT[:12]}")
        bad, seen = 0, set()
        for s, gid, tid, fp in plan:
            if gid in seen:
                continue
            seen.add(gid)
            try:
                m = load_sim(gid, "dry")
                hook = next((h for h in RESET_HOOKS
                             if callable(getattr(m, h, None))), None)
                print(f"  OK  {gid:5s} reset_hook={hook}; trace={fp.name}")
            except Exception as exc:  # noqa: BLE001
                bad += 1
                print(f"  FAIL {gid:5s} {type(exc).__name__}: {exc}")
        print(f"\nplanned: {len(plan)} (source, game) replays over {len(seen)} "
              f"sims; {bad} import failures")
        print(f"E1 population (13): {NEVER_MEASURED_13}")
        print(f"anchor, non-contributing (12): {ANCHOR_SAT12}")
        print(f"threshold H_g >= {H_THRESHOLD}; pre-registered E1 = "
              f"{PREREGISTERED_E1_EXPECTATION}")
        print("no experiment executed (--dry-run)")
        return 0 if bad == 0 else 1

    t0 = time.time()
    rows = []
    med_by_game: dict[str, dict] = {}
    for s, gid, tid, fp in plan:
        r = replay_game(gid, fp, do_reset=True)
        # E3: same trace, same order, reset hooks suppressed.
        if r["reset_hook"] is None:
            delta = 0
        else:
            r0 = replay_game(gid, fp, do_reset=False)
            delta = sum(1 for a, b in zip(r["pred_hashes"], r0["pred_hashes"])
                        if a != b) + abs(len(r["pred_hashes"])
                                         - len(r0["pred_hashes"]))
        row = score_row(gid, s, tid, fp, r, delta, emt.get(gid))
        rows.append(row)
        med_by_game.setdefault(gid, {})[s] = row["median_survival"]
        print(f"[{s}] {gid}: segs={row['n_segments']} steps={row['n_steps']} "
              f"median_surv={row['median_survival']} max={row['max_survival']} "
              f"(all-steps {row['match_all_steps']}, TF "
              f"{row['teacher_forced_match_all_steps']}) "
              f"reset_delta={delta}", flush=True)

    e1 = horizon_block(NEVER_MEASURED_13, med_by_game)
    anchor = horizon_block(ANCHOR_SAT12, med_by_game)

    # ---- E3 void check (sealed spec §4.4 R5) -----------------------------
    violations = [{"game": g, "source": s,
                   "module_reset_delta_steps": next(
                       (r["module_reset_delta_steps"] for r in rows
                        if r["game"] == g and r["source"] == s), None)}
                  for g, s in E3_PREREGISTERED_ZEROS]
    violations = [v for v in violations
                  if v["module_reset_delta_steps"] not in (0, None)]
    void = bool(violations)

    # ---- the sealed decision rule (spec §9.2) ----------------------------
    n = e1["n_pass_H_ge_10"]
    if void:
        verdict, action = "VOID", "runner defect: E3 pre-registered zero violated"
    elif n == 0:
        verdict, action = "L1 NO-GO", (
            "S5 does not open; bank the clean second negative on THIS endpoint; "
            "exec-wm closes as an execution substrate, C1/C2/C3 retained as "
            "schema only; lane (a) proceeds on P1/P3")
    elif n <= 2:
        verdict, action = "L1 HELD", (
            "author the abstention interface for the identified games only; "
            "re-read at R26")
    else:
        verdict, action = "L1 GO", (
            "authorise L1 for the identified set, conditional on the already-"
            "ratified workstation-authoring rule")

    # ---- E4 engine stratification ----------------------------------------
    def _strat(pred):
        sel = [r for r in rows if pred(r) and r["median_survival"] is not None]
        return {"n_rows": len(sel),
                "mean_median_survival": (
                    round(sum(r["median_survival"] for r in sel) / len(sel), 3)
                    if sel else None),
                "mean_match_all_steps": (
                    round(sum(r["match_all_steps"] for r in sel) / len(sel), 4)
                    if sel else None)}

    out = {
        "provenance": {
            "script": "duck_eval/r24_prep/s1_threaded_replay.py",
            "script_sha256": sha256_file(Path(__file__)),
            "sealed_spec": SPEC,
            "sealed_spec_sha256": sha256_file(ROOT / SPEC),
            "prereg_commit": PREREG_COMMIT,
            "prereg_commit_note": "the sealed spec and its expectation were "
                                  "committed at this SHA BEFORE this run; the "
                                  "prereg is provably prior to the outcome",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "git": git_info(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "authorized_by": args.authorized_by,
            "cost": {"usd": 0, "kaggle_pushes": 0, "gpu": "none (CPU only)"},
            "rng": "none — replay is fully deterministic; no seeds are drawn",
            "sources": {s: str(SOURCES[s].relative_to(ROOT)).replace("\\", "/")
                        for s in srcs},
            "wallclock_s": round(time.time() - t0, 1),
        },
        "config": {
            "mode": "state_threaded",
            "segment_rule": "initial | post-RESET frame | post-level-completion frame",
            "reset_module_state_at_segment_start": True,
            "teacher_forced_ab_in_same_pass": True,
            "primary_statistic": "survival = consecutive threaded steps from "
                                 "segment start matching the recorded settled "
                                 "frame exactly, before the first mismatch",
            "per_source_aggregate": "median over that source's segments",
            "per_game_aggregate": "H_g = min over the 3 war_eval sources",
            "threshold": f"H_g >= {H_THRESHOLD}",
            "threshold_provenance": "R16 §9.2 registered executor plan depth <=10",
            "E1_population_13": NEVER_MEASURED_13,
            "anchor_population_12_non_contributing": ANCHOR_SAT12,
            "prereg_expectation_E1": PREREGISTERED_E1_EXPECTATION,
            "decision_rule": "0 -> L1 NO-GO; 1-2 -> L1 HELD; >=3 -> L1 GO",
            "removed_by_seal": ["coverage", "coverage_strict", "accepted_match",
                                "n_identity_abstain", "n_committed", "carrier",
                                "carrier_set", "gained", "lost",
                                "EXPANDED/NOT_EXPANDED"],
            "non_gating_channels": ["match_all_steps",
                                    "teacher_forced_match_all_steps",
                                    "n_error", "n_selfdiff", "done_flag_agree"],
        },
        "rows": rows,
        "E1_never_measured_13": e1,
        "anchor_sat12_12": anchor,
        "E3_module_reset": {
            "preregistered_zeros": [list(x) for x in E3_PREREGISTERED_ZEROS],
            "violations": violations,
            "void": void,
            "nonzero_rows": [{"game": r["game"], "source": r["source"],
                              "delta": r["module_reset_delta_steps"]}
                             for r in rows if r["module_reset_delta_steps"]],
        },
        "E4_engine_stratification": {
            "engine_matched": _strat(lambda r: r["engine_matches_kaggle"] is True),
            "engine_mismatched": _strat(lambda r: r["engine_matches_kaggle"] is False),
            "pooled": _strat(lambda r: True),
        },
        "result": {
            "E1_n_pass_H_ge_10": n,
            "E1_pass_list": e1["pass_list"],
            "prereg_expectation": PREREGISTERED_E1_EXPECTATION,
            "matched_expectation": n == PREREGISTERED_E1_EXPECTATION,
            "verdict": verdict,
            "action": action,
            "anchor_n_pass_non_contributing": anchor["n_pass_H_ge_10"],
            "anchor_pass_list_non_contributing": anchor["pass_list"],
        },
    }
    op = ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\n{'='*78}")
    print(f"E1 (13 never-measured games): {n} reach H_g >= {H_THRESHOLD}  "
          f"{e1['pass_list'] or '-'}")
    print(f"pre-registered expectation: {PREREGISTERED_E1_EXPECTATION}  -> "
          f"{'MATCHED' if n == PREREGISTERED_E1_EXPECTATION else 'NOT MATCHED'}")
    print(f"anchor SAT12 (non-contributing): {anchor['n_pass_H_ge_10']} "
          f"{anchor['pass_list'] or '-'}")
    print(f"E3 void: {void}  {violations if violations else ''}")
    print(f"VERDICT: {verdict} — {action}")
    print(f"written: {op}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
