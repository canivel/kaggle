"""S1 seal audit — evidence for the six R24 §5.2 re-scope items.

READ-ONLY. Runs no experiment, writes one JSON report, touches no campaign file.
Every number the sealed spec (`duck_eval/r24_prep/s1_sealed_spec_2026-08-10.md`)
asserts about the sims / traces / engines is produced here so it can be
re-derived by a third party.

Checks
------
A. ABSTENTION (§5.2 i)   — AST + token scan of every active sim for any channel
                           by which `simulate()` could decline to predict:
                           an `UNKNOWN`/`-1`/`None` state return, a 4-tuple, a
                           documented abstain flag. Also counts how many sims
                           can return their input UNCHANGED (the identity
                           proxy's structural precondition).
B. MODULE STATE (§5.2 i) — AST scan for module-level mutable globals that any
                           function rebinds via `global`, plus reset hooks.
C. STREAMS (§5.2 iii)    — enumerate games with a recorded per-action event
                           stream per source, and how many of those streams
                           carry `board` frames on every action event.
D. DRYRUN COVERAGE (iv)  — which games the 2026-07-18 `ewm_replay_dryrun`
                           artifact actually covers, intersected with the
                           module-state set.
E. ENGINE VERSIONS (v)   — trace-id hash per game per source, local engine dir
                           hash, and the recorded mismatch flags.
F. BANKED TF NUMBERS     — the already-banked teacher-forced on-trajectory
                           accuracies (runs/ewm_dryrun/raw.json), their
                           across-source range, and a stratification by
                           local-vs-Kaggle engine match. Teacher forcing is an
                           upper bound on state-threaded matching (the two are
                           bit-identical up to and including the first
                           mismatch), so this bounds what S1 can report.
G. SEGMENT BOUNDARIES    — per-source count and position of the segment
                           restarts ('level frame 0') for the three stateful
                           sims; this sizes the module-state bug's blast radius
                           exactly.

Usage:  .venv/Scripts/python.exe duck_eval/r24_prep/s1_seal_audit.py
        (add --out PATH to redirect; default duck_eval/r24_prep/s1_seal_audit.json)
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIM_DIR = ROOT / "exec_wm" / "sims"

SOURCES = {
    "war_eval_v1": ROOT / "runs/kernel_pulls/war_eval_v1/artifacts",
    "war_eval_v2": ROOT / "runs/kernel_pulls/war_eval_v2/artifacts",
    "war_eval_v3": ROOT / "runs/kernel_pulls/war_eval_v3/artifacts",
    "gpt56_full": ROOT / "runs/gpt56_probe/experiment_full/artifacts",
}

RESET_HOOKS = ("reset_state", "reset_phase", "reset_step_parity")
ABSTAIN_TOKENS = ("UNKNOWN", "ABSTAIN", "abstain", "NO_PREDICTION", "unsupported")


def active_sims() -> list[str]:
    """<gid>_sim.py only — excludes *_sim_v2.py / *_sim_v1*.py / backups."""
    out = []
    for p in sorted(SIM_DIR.glob("*_sim.py")):
        gid = p.name[: -len("_sim.py")]
        if gid.startswith("_") or re.search(r"_v\d$", gid):
            continue
        out.append(gid)
    return out


# ------------------------------------------------------------------ A. abstain
def audit_abstention(gid: str) -> dict:
    p = SIM_DIR / f"{gid}_sim.py"
    src = p.read_text(encoding="utf-8")
    tree = ast.parse(src)

    fn = next((n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name == "simulate"), None)
    rec: dict = {
        "game": gid,
        "has_simulate": fn is not None,
        "n_args": len(fn.args.args) if fn else None,
        "return_arities": [],
        "returns_none_state": False,
        "returns_negative_state": False,
        "abstain_tokens": sorted({t for t in ABSTAIN_TOKENS if t in src}),
        "can_return_input_unchanged": False,
    }
    if fn is None:
        return rec

    for n in ast.walk(fn):
        if not isinstance(n, ast.Return) or n.value is None:
            continue
        v = n.value
        if isinstance(v, ast.Tuple):
            rec["return_arities"].append(len(v.elts))
            first = v.elts[0]
            if isinstance(first, ast.Constant) and first.value is None:
                rec["returns_none_state"] = True
            if (isinstance(first, ast.UnaryOp) and isinstance(first.op, ast.USub)):
                rec["returns_negative_state"] = True
        else:
            rec["return_arities"].append(-1)  # non-tuple return

    # identity-return capability: does any return path hand back the state var
    # (or a shallow copy of it) as element 0?
    state_name = fn.args.args[0].arg if fn.args.args else "state"
    for n in ast.walk(fn):
        if not isinstance(n, ast.Return) or not isinstance(n.value, ast.Tuple):
            continue
        first = n.value.elts[0]
        names = {x.id for x in ast.walk(first) if isinstance(x, ast.Name)}
        if state_name in names or "grid" in names or "g" in names or "s" in names:
            rec["can_return_input_unchanged"] = True
            break

    rec["return_arities"] = sorted(set(rec["return_arities"]))
    rec["has_abstention_channel"] = bool(
        rec["returns_none_state"] or rec["returns_negative_state"]
        or any(a not in (3,) for a in rec["return_arities"])
        or rec["abstain_tokens"])
    return rec


# -------------------------------------------------------------- B. module state
def audit_module_state(gid: str) -> dict:
    p = SIM_DIR / f"{gid}_sim.py"
    src = p.read_text(encoding="utf-8")
    tree = ast.parse(src)

    module_names: set[str] = set()
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    module_names.add(t.id)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            module_names.add(n.target.id)

    mutated: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Global):
            mutated.update(x for x in n.names if x in module_names)

    hooks = [h for h in RESET_HOOKS
             if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and n.name == h for n in tree.body)]
    # any zero-arg module-level function whose body only rebinds globals
    other_resetters = [
        n.name for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name not in hooks
        and not n.args.args
        and any(isinstance(c, ast.Global) for c in ast.walk(n))
    ]
    return {
        "game": gid,
        "module_level_names": sorted(module_names),
        "globals_rebound_in_functions": sorted(mutated),
        "has_module_state": bool(mutated),
        "reset_hooks": hooks,
        "other_zero_arg_global_rebinders": other_resetters,
        "sim_lines": len(src.splitlines()),
    }


# ------------------------------------------------------------------ C. streams
def audit_streams() -> dict:
    out = {}
    for src, art in SOURCES.items():
        if not art.is_dir():
            out[src] = {"exists": False}
            continue
        games = {}
        for fp in sorted(glob.glob(str(art / "*_events.jsonl"))):
            tid = Path(fp).name.split("_")[0]
            gid = tid.split("-")[0]
            n_action = n_board = n_initial = 0
            n_reset = 0
            n_levelcomp = 0
            with open(fp, encoding="utf-8") as f:
                for ln in f:
                    if not ln.strip():
                        continue
                    ev = json.loads(ln)
                    t = ev.get("type")
                    if t == "initial":
                        n_initial += 1
                    elif t == "action":
                        n_action += 1
                        if ev.get("board") is not None:
                            n_board += 1
                        if (ev.get("action_name") or "") == "RESET":
                            n_reset += 1
                        if ev.get("level_completed"):
                            n_levelcomp += 1
            games[gid] = {
                "trace_id": tid,
                "engine_hash": tid.split("-")[1] if "-" in tid else None,
                "n_initial_frames": n_initial,
                "n_action_events": n_action,
                "n_action_events_with_board": n_board,
                "n_reset_actions": n_reset,
                "n_level_completions": n_levelcomp,
                "full_board_coverage": n_action == n_board and n_action > 0,
                "has_sim": (SIM_DIR / f"{gid}_sim.py").exists(),
            }
        out[src] = {
            "exists": True,
            "n_streams": len(games),
            "games": games,
        }
    return out


# ------------------------------------------------------- D. old dryrun coverage
def audit_dryrun() -> dict:
    raw_p = ROOT / "runs" / "ewm_dryrun" / "raw.json"
    script_p = ROOT / "scripts" / "ewm_replay_dryrun.py"
    src = script_p.read_text(encoding="utf-8")
    rec = {
        "script": "scripts/ewm_replay_dryrun.py",
        "loads_sim_once_per_game": "load_sim(gid, \"a\")" in src,
        "calls_any_reset_hook": any(h in src for h in RESET_HOOKS),
        "teacher_forced": "boards[id(ev)] = prev_board" in src,
        "sat12_filter_present": "if gid not in SAT12" in src,
    }
    if raw_p.exists():
        raw = json.loads(raw_p.read_text(encoding="utf-8"))
        rec["sat12"] = sorted(raw.get("sat12", {}))
        rec["per_source_games"] = {
            s: sorted(v.get("shadow", {})) for s, v in raw.get("sources", {}).items()}
    return rec


# ----------------------------------------------------------- E. engine versions
def audit_engines() -> dict:
    envdir = ROOT / "kaggle-data" / "environment_files"
    local = {}
    if envdir.is_dir():
        for d in sorted(envdir.iterdir()):
            if not d.is_dir():
                continue
            name = d.name
            gid = name.split("-")[0]
            local[gid] = {"dir": name,
                          "engine_hash": name.split("-")[1] if "-" in name else None}
    det = ROOT / "runs" / "war_eval_v1" / "determinism_audit_25.json"
    det_rec = {}
    if det.exists():
        d = json.loads(det.read_text(encoding="utf-8"))
        games = d.get("games", d if isinstance(d, list) else [])
        if isinstance(games, list):
            for g in games:
                gid = (g.get("game") or "").split("-")[0]
                if gid:
                    det_rec[gid] = {
                        k: g.get(k) for k in
                        ("version_mismatch_vs_kaggle", "local_game_id",
                         "benchmark_game_id", "verdict", "determinism")
                        if k in g}
    return {"local_engine_dirs": local,
            "determinism_audit_25_present": det.exists(),
            "determinism_audit_25_per_game": det_rec}


# ---------------------------------------------- F. banked teacher-forced rates
def audit_banked_tf(det_per_game: dict) -> dict:
    raw_p = ROOT / "runs" / "ewm_dryrun" / "raw.json"
    if not raw_p.exists():
        return {"present": False}
    raw = json.loads(raw_p.read_text(encoding="utf-8"))
    war = [s for s in ("war_eval_v1", "war_eval_v2", "war_eval_v3")
           if s in raw["sources"]]
    games = sorted(raw["sources"][war[0]]["shadow"])
    rows, tot_err, tot_sd, tot_steps = {}, 0, 0, 0
    for g in games:
        acc = {}
        for s in war:
            sh = raw["sources"][s]["shadow"][g]
            acc[s] = sh["exact"] / sh["steps"] if sh["steps"] else None
        vals = [v for v in acc.values() if v is not None]
        rows[g] = {
            "tf_acc": {k: round(v, 4) for k, v in acc.items()},
            "across_source_range": round(max(vals) - min(vals), 4),
            "first_divergence_step": {
                s: raw["sources"][s]["shadow"][g]["first_div"] for s in war},
            "engine_matches_kaggle": not det_per_game.get(g, {}).get(
                "version_mismatch_vs_kaggle", None),
        }
    for s, v in raw["sources"].items():
        for sh in v["shadow"].values():
            tot_err += sh["sim_error"]
            tot_sd += sh["selfdiff"]
            tot_steps += sh["steps"]
    ranges = sorted(r["across_source_range"] for r in rows.values())
    med = round(statistics.median(ranges), 4)
    matched = [g for g, r in rows.items() if r["engine_matches_kaggle"]]
    return {
        "present": True,
        "n_games": len(games),
        "games": rows,
        "total_steps_all_sources": tot_steps,
        "total_sim_error": tot_err,
        "total_selfdiff": tot_sd,
        "coverage_strict_is_1_0_everywhere": tot_err == 0,
        "median_across_source_range": med,
        "n_ge_0_92_per_source": {
            s: sorted(g for g in games
                      if (rows[g]["tf_acc"][s] or 0) >= 0.92) for s in war},
        "engine_matched_games": sorted(matched),
        "engine_mismatched_games": sorted(set(games) - set(matched)),
        "mean_v1_tf_acc_matched": round(
            sum(rows[g]["tf_acc"]["war_eval_v1"] for g in matched)
            / max(1, len(matched)), 4),
        "mean_v1_tf_acc_mismatched": round(
            sum(rows[g]["tf_acc"]["war_eval_v1"]
                for g in set(games) - set(matched))
            / max(1, len(games) - len(matched)), 4),
    }


# ------------------------------------------------------- G. segment boundaries
def audit_segments(stateful: list[str]) -> dict:
    out = {}
    for src in ("war_eval_v1", "war_eval_v2", "war_eval_v3"):
        art = SOURCES[src]
        if not art.is_dir():
            continue
        for g in stateful:
            hits = glob.glob(str(art / f"{g}-*_events.jsonl"))
            if not hits:
                continue
            n_calls, bounds = 0, []
            with open(hits[0], encoding="utf-8") as f:
                for ln in f:
                    if not ln.strip():
                        continue
                    ev = json.loads(ln)
                    if ev.get("type") != "action":
                        continue
                    if (ev.get("action_name") or "") == "RESET":
                        bounds.append(["RESET", n_calls])
                        continue
                    n_calls += 1
                    if ev.get("level_completed"):
                        bounds.append(["LEVEL_COMPLETED", n_calls])
            out.setdefault(g, {})[src] = {
                "n_sim_calls": n_calls,
                "segment_boundaries": bounds,
                "n_extra_segments": len(bounds),
                # tr87's counter is mod-2; a boundary at an EVEN call index is
                # a no-op for it because reset_step_parity(0) == n_calls % 2.
                "boundary_call_indices_parity": [b[1] % 2 for b in bounds],
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="duck_eval/r24_prep/s1_seal_audit.json")
    args = ap.parse_args()

    gids = active_sims()
    abst = [audit_abstention(g) for g in gids]
    mods = [audit_module_state(g) for g in gids]
    streams = audit_streams()
    dry = audit_dryrun()
    eng = audit_engines()

    stateful = sorted(m["game"] for m in mods if m["has_module_state"])
    with_hook = sorted(m["game"] for m in mods if m["reset_hooks"])
    abstainers = sorted(a["game"] for a in abst if a["has_abstention_channel"])
    identity_capable = sorted(a["game"] for a in abst
                              if a["can_return_input_unchanged"])

    rep = {
        "n_active_sims": len(gids),
        "active_sims": gids,
        "A_abstention": {
            "n_with_abstention_channel": len(abstainers),
            "games_with_abstention_channel": abstainers,
            "n_identity_return_capable": len(identity_capable),
            "identity_return_capable": identity_capable,
            "per_sim": abst,
        },
        "B_module_state": {
            "n_stateful": len(stateful),
            "stateful": stateful,
            "with_named_reset_hook": with_hook,
            "stateful_without_reset_hook": sorted(set(stateful) - set(with_hook)),
            "per_sim": mods,
        },
        "C_streams": streams,
        "D_old_dryrun": dry,
        "E_engines": eng,
        "F_banked_tf": audit_banked_tf(eng["determinism_audit_25_per_game"]),
        "G_segment_boundaries": audit_segments(stateful),
    }
    op = ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print(f"active sims: {len(gids)}")
    print(f"A. sims with an abstention channel: {len(abstainers)} {abstainers}")
    print(f"   sims able to return input unchanged: {len(identity_capable)}")
    print(f"B. sims with module state: {len(stateful)} {stateful}")
    print(f"   named reset hooks: {with_hook}")
    print(f"   stateful WITHOUT reset hook: {sorted(set(stateful)-set(with_hook))}")
    for s, v in streams.items():
        if v.get("exists"):
            full = sum(1 for g in v["games"].values() if g["full_board_coverage"])
            print(f"C. {s}: {v['n_streams']} streams, {full} with board on every action")
    print(f"D. old dryrun games: "
          f"{ {s: len(g) for s, g in dry.get('per_source_games', {}).items()} }")
    print(f"   calls a reset hook: {dry['calls_any_reset_hook']}; "
          f"teacher_forced: {dry['teacher_forced']}")
    f = rep["F_banked_tf"]
    if f.get("present"):
        print(f"F. banked TF: {f['total_steps_all_sources']} steps, "
              f"sim_error={f['total_sim_error']}, selfdiff={f['total_selfdiff']} "
              f"=> coverage_strict==1.0 everywhere: "
              f"{f['coverage_strict_is_1_0_everywhere']}")
        print(f"   median across-source range = {f['median_across_source_range']}")
        print(f"   TF acc >= 0.92: {f['n_ge_0_92_per_source']}")
        print(f"   mean v1 TF acc  engine-matched={f['mean_v1_tf_acc_matched']} "
              f"mismatched={f['mean_v1_tf_acc_mismatched']}")
    for g, per in rep["G_segment_boundaries"].items():
        for s, v in per.items():
            print(f"G. {g} {s}: {v['n_sim_calls']} sim calls, "
                  f"boundaries {v['segment_boundaries']}")
    print(f"written: {op}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
