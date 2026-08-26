"""P1 offline replay validation -- CPU only, read-only, no network, no spend.

Re-scores three recorded runs with the P1 suppressor applied, using the EXACT
scorer (``scripts/phase1_gate.py:rhae_score``) and the SHIPPED policy objects
imported from ``_kaggle_dataset/p1_suppressor_patch.py`` -- the replay drives
``P1State``/``Config`` themselves, so it validates the code that ships, not a
re-implementation of it.

Runs:
  runs/kernel_pulls/animation_v1   (25 games, 17 cleared levels, 2026-08-11)
  runs/a22_v2_seed1                (14 cleared levels)
  runs/a22_compaction_v1           (17 cleared levels)

Reports, per run and per arm:
  as-run score / P1 score / multiplier
  declined + aborted actions on cleared levels        (the saving)
  BOARD-CHANGING declines                             (path-refusal risk)
  LEVEL-COMPLETING actions declined or aborted        (the fatal canary)
  duplicate re-execution rate  and  blind-batch-tail rate, before and after
  the online latent-state detector's per-game verdict

Usage:
  .venv/Scripts/python.exe duck_eval/warpack/p1_replay_validate.py
  .venv/Scripts/python.exe duck_eval/warpack/p1_replay_validate.py --json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE / "_kaggle_dataset"))

# NOTE: scripts/ is NOT put on sys.path -- it contains queue.py, which would
# shadow the stdlib ``queue`` module for everything imported afterwards
# (urllib3 -> requests -> arc_agi -> taaf all break). Load the scorer by path.
def _load_scorer():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_p1_phase1_gate", REPO / "scripts" / "phase1_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.rhae_score


rhae_score = _load_scorer()
import p1_suppressor_patch as P1  # noqa: E402

RUNS = [
    "runs/kernel_pulls/animation_v1",
    "runs/a22_v2_seed1",
    "runs/a22_compaction_v1",
]
OUT_DIR = REPO / "runs" / "p1_replay"
CACHE = OUT_DIR / "traces_cache.pkl"

# Published latent-state set (efficiency_diagnosis_2026-08-12.md sec5 P1),
# measured on animation_v1. Used ONLY to certify the online detector; the
# shipped patch never reads a game id.
PUBLISHED_AMBIGUOUS = {"m0r0": 55, "re86": 19, "sk48": 11, "ka59": 10,
                       "cd82": 8, "g50t": 4, "dc22": 3, "wa30": 2}

# Arms. "shipped" == the module defaults. "published_M1M3" == the mechanism
# exactly as written in the diagnosis (memo on every repeat, batch abort on
# no-op OR revisit) -- the arm whose +0.184 / x1.10 we are checking.
ARMS = {
    "shipped": {
        "P1_MEMO": "1", "P1_MEMO_MODE": "noop", "P1_CONFIRM": "2",
        "P1_MAX_DECLINES": "1", "P1_ABORT": "1", "P1_ABORT_NOOP_STREAK": "1",
        "P1_ABORT_CYCLE": "1", "P1_ABORT_REVISIT": "0",
    },
    "shipped_memo_only": {
        "P1_MEMO": "1", "P1_MEMO_MODE": "noop", "P1_CONFIRM": "2",
        "P1_MAX_DECLINES": "1", "P1_ABORT": "0", "P1_ABORT_CYCLE": "0",
        "P1_ABORT_REVISIT": "0",
    },
    "shipped_abort_only": {
        "P1_MEMO": "0", "P1_ABORT": "1", "P1_ABORT_NOOP_STREAK": "1",
        "P1_ABORT_CYCLE": "1", "P1_ABORT_REVISIT": "0",
    },
    "memo_all": {
        "P1_MEMO": "1", "P1_MEMO_MODE": "all", "P1_CONFIRM": "2",
        "P1_MAX_DECLINES": "1", "P1_ABORT": "1", "P1_ABORT_NOOP_STREAK": "1",
        "P1_ABORT_CYCLE": "1", "P1_ABORT_REVISIT": "0",
    },
    "published_M1M3": {
        "P1_MEMO": "1", "P1_MEMO_MODE": "all", "P1_CONFIRM": "2",
        "P1_MAX_DECLINES": "1000000", "P1_ABORT": "1",
        "P1_ABORT_NOOP_STREAK": "1", "P1_ABORT_CYCLE": "1",
        "P1_ABORT_REVISIT": "1",
    },
}


# --------------------------------------------------------------------------- #
# trace loading
# --------------------------------------------------------------------------- #
def _bh(board) -> str:
    return P1.board_fingerprint(board)


def _replay_events(path: Path) -> list[dict]:
    """Recorded jsonl -> compact per-action records (boards dropped)."""
    acts: list[dict] = []
    prev = None
    level = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            t = ev.get("type")
            if t == "initial":
                prev = _bh(ev["board"])
                continue
            if t != "action" or "board" not in ev or prev is None:
                continue
            out = _bh(ev["board"])
            acts.append({
                "prev": prev, "out": out,
                "act": ev.get("action_display") or ev.get("action_name") or "?",
                "bc": bool(ev.get("board_changed", prev != out)),
                "lc": bool(ev.get("level_completed")),
                "level": level,
                "step": ev.get("analysis_step"),
                "bi": ev.get("batch_index") or 1,
                "bs": ev.get("batch_size") or 1,
            })
            prev = out
            if ev.get("level_completed"):
                level += 1
    return acts


def load_traces(rebuild: bool = False) -> dict:
    if CACHE.is_file() and not rebuild:
        try:
            return pickle.loads(CACHE.read_bytes())
        except Exception:  # noqa: BLE001
            pass
    out: dict = {}
    for rd in RUNS:
        bench = json.loads((REPO / rd / "benchmark.json").read_text())
        runs = bench if isinstance(bench, list) else bench["game_runs"]
        art = REPO / rd / "artifacts"
        games = []
        for r in runs:
            meta = {k: r[k] for k in ("game_id", "number_of_levels",
                                      "base_actions_per_level",
                                      "actions_per_level", "levels_completed")}
            p = art / f"{r['game_id']}_p0_events.jsonl"
            games.append((meta, _replay_events(p) if p.is_file() else None))
        out[rd] = games
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps(out))
    return out


# --------------------------------------------------------------------------- #
# the replay -- drives the SHIPPED P1State / Config
# --------------------------------------------------------------------------- #
def simulate(acts: list[dict], cleared_levels: int) -> dict:
    """Replay one recorded game through the shipped P1 policy.

    Counting convention (identical to the diagnosis): the model's requests are
    held fixed and we count the actions the runner would NOT have charged.
    Two honesty counters make the counterfactual visible:
      ``diverged``  declines whose true recorded outcome was NOT a no-op (in
                    ``noop`` mode this must be 0: the board really is unchanged,
                    so the replay is exact and not a counterfactual);
      ``blind_tail`` aborted actions -- these genuinely did not run, so the
                    board after the batch differs from the recording.
    """
    st = P1.P1State("replay")
    per_level: dict[int, dict[str, int]] = {}
    dead = False
    cur_batch = None

    def slot(lv: int) -> dict[str, int]:
        return per_level.setdefault(lv, {"declined": 0, "aborted": 0,
                                         "diverged": 0, "lc_lost": 0,
                                         "charged": 0, "dup_exec": 0,
                                         "blind_seen": 0})

    for a in acts:
        lv = a["level"]
        s = slot(lv)
        st.sync_level(lv)
        if st.last_hash is None:
            st.last_hash = a["prev"]
            st.visited.add(a["prev"])
            st.batch_states.add(a["prev"])
        bkey = (a["step"], a["bs"])
        if bkey != cur_batch:
            cur_batch = bkey
            dead = False
            st.batch_dead = False
            st.batch_states = {st.last_hash} if st.last_hash else set()
            st._noop_streak = 0
        # -- B: already dead
        if dead and a["bs"] > 1 and P1.CFG.abort:
            s["aborted"] += 1
            if a["lc"]:
                s["lc_lost"] += 1
            continue
        key = (a["prev"], a["act"])
        # -- A: decline?
        ent = None
        if key in st.memo:
            st.dup_requests += 1
            ent = st.should_decline(key)
        if ent is not None:
            ent.declines += 1
            s["declined"] += 1
            if a["bc"]:
                s["diverged"] += 1
            if a["lc"]:
                s["lc_lost"] += 1
            continue
        # -- executed
        s["charged"] += 1
        if key in st.memo:
            s["dup_exec"] += 1
        cycle = a["out"] in st.batch_states
        revisit = a["out"] in st.visited
        noop = not a["bc"]
        st.record(key, a["out"], noop)
        if P1.CFG.abort and a["bs"] > 1 and not dead:
            kill = False
            if noop and P1.CFG.abort_noop_streak <= 1:
                kill = True
            elif noop:
                st._noop_streak = getattr(st, "_noop_streak", 0) + 1
                kill = st._noop_streak >= P1.CFG.abort_noop_streak
            else:
                st._noop_streak = 0
            if not kill and P1.CFG.abort_cycle and cycle and not noop:
                kill = True
            if not kill and P1.CFG.abort_revisit and revisit:
                kill = True
            if kill and not a["lc"]:
                dead = True
    return {
        "per_level": per_level,
        "ambiguous": st.ambiguous,
        "ambiguity_pairs": len(st.ambiguity_pairs),
    }


def baseline_stats(acts: list[dict], cleared: int) -> dict:
    """As-run duplicate + blind-tail rates on CLEARED levels (the canaries)."""
    seen: dict[int, set] = {}
    visited: dict[int, set] = {}
    dup = blind = total = 0
    dead = False
    cur = None
    for a in acts:
        lv = a["level"]
        if lv >= cleared:
            continue
        sp = seen.setdefault(lv, set())
        vs = visited.setdefault(lv, set())
        if not vs:
            vs.add(a["prev"])
        if (a["step"], a["bs"]) != cur:
            cur = (a["step"], a["bs"])
            dead = False
        total += 1
        key = (a["prev"], a["act"])
        if key in sp:
            dup += 1
        if dead:
            blind += 1
        sp.add(key)
        rev = a["out"] in vs
        vs.add(a["out"])
        if a["bs"] > 1 and not dead and ((not a["bc"]) or rev):
            dead = True
    return {"actions": total, "dup": dup, "blind": blind}


def score_arm(traces: dict, run_dir: str, arm: str) -> dict:
    for k, v in ARMS[arm].items():
        os.environ[k] = v
    games = traces[run_dir]
    as_run = new = 0.0
    agg = {"declined": 0, "aborted": 0, "diverged": 0, "lc_lost": 0,
           "dup_exec": 0, "charged": 0}
    base = {"actions": 0, "dup": 0, "blind": 0}
    ambiguous_games: dict[str, int] = {}
    per_game = {}
    for meta, acts in games:
        nlev = meta["number_of_levels"]
        apl = list(meta["actions_per_level"])
        lc = meta["levels_completed"]
        s0 = rhae_score(meta["base_actions_per_level"], apl, lc, nlev)
        as_run += s0
        if acts is None:
            new += s0
            continue
        res = simulate(acts, lc)
        if res["ambiguous"]:
            ambiguous_games[meta["game_id"][:4]] = res["ambiguity_pairs"]
        b = baseline_stats(acts, lc)
        for k in base:
            base[k] += b[k]
        napl = list(apl)
        for lv in range(nlev):
            sl = res["per_level"].get(lv)
            if not sl:
                continue
            napl[lv] = max(0, apl[lv] - sl["declined"] - sl["aborted"])
            if lv < lc:
                for k in agg:
                    agg[k] += sl[k]
        s1 = rhae_score(meta["base_actions_per_level"], napl, lc, nlev)
        new += s1
        per_game[meta["game_id"][:4]] = {"as_run": s0, "p1": s1,
                                         "actions": apl[:nlev],
                                         "p1_actions": napl[:nlev]}
    n = len(games)
    return {
        "run": run_dir, "arm": arm,
        "as_run": as_run / n, "p1": new / n,
        "multiplier": (new / as_run) if as_run else 1.0,
        "delta": (new - as_run) / n,
        **agg,
        "baseline": base,
        "dup_rate_before": base["dup"] / max(1, base["actions"]),
        "blind_rate_before": base["blind"] / max(1, base["actions"]),
        "dup_rate_after": agg["dup_exec"] / max(1, agg["charged"]),
        "ambiguous_games": ambiguous_games,
        "per_game": per_game,
    }


# --------------------------------------------------------------------------- #
# the diagnosis's own arithmetic, reproduced independently of the shipped
# Config (which clamps P1_CONFIRM >= 2 so the detector can never be blinded).
# Buckets exactly as efficiency_diagnosis_2026-08-12.md sec1 defines them:
#   (b) dup   -- (board, action) already executed on this level
#   (c) blind -- fired after the batch went dead (earlier action no-opped OR
#                landed on an already-visited board), dup taking precedence
# Drop (b)+(c) from actions_per_level and re-score. This reproduces the
# published act / dup / retraversal / RESET / analysis-step columns EXACTLY
# for all 17 cleared levels of animation_v1.
# --------------------------------------------------------------------------- #
def published_arithmetic(traces: dict) -> list[dict]:
    rows = []
    for rd in RUNS:
        as_run = new = 0.0
        dup_t = blind_t = lc_lost = 0
        games = traces[rd]
        for meta, acts in games:
            nlev = meta["number_of_levels"]
            apl = list(meta["actions_per_level"])
            lc = meta["levels_completed"]
            s0 = rhae_score(meta["base_actions_per_level"], apl, lc, nlev)
            as_run += s0
            if acts is None:
                new += s0
                continue
            seen: dict[int, set] = {}
            visited: dict[int, set] = {}
            rm = [0] * nlev
            dead = False
            cur = None
            for a in acts:
                lv = a["level"]
                sp = seen.setdefault(lv, set())
                vs = visited.setdefault(lv, set())
                if not vs:
                    vs.add(a["prev"])
                if (a["step"], a["bs"]) != cur:
                    cur = (a["step"], a["bs"])
                    dead = False
                key = (a["prev"], a["act"])
                is_dup = key in sp
                bucket = "dup" if is_dup else ("blind" if dead else "nec")
                if bucket != "nec" and lv < nlev:
                    rm[lv] += 1
                    if lv < lc:
                        dup_t += bucket == "dup"
                        blind_t += bucket == "blind"
                        lc_lost += bool(a["lc"])
                sp.add(key)
                rev = a["out"] in vs
                vs.add(a["out"])
                if a["bs"] > 1 and not dead and ((not a["bc"]) or rev):
                    dead = True
            napl = [max(0, apl[i] - rm[i]) for i in range(nlev)]
            new += rhae_score(meta["base_actions_per_level"], napl, lc, nlev)
        n = len(games)
        rows.append({"run": rd, "as_run": as_run / n, "m1m3": new / n,
                     "multiplier": new / as_run, "dup": dup_t,
                     "blind": blind_t, "lc_lost": lc_lost})
    return rows


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args()

    traces = load_traces(rebuild=args.rebuild_cache)
    out: dict = {"arms": {}, "detector": {}}

    # ---- detector certification (must reproduce the published 8-game set) --
    for k, v in ARMS["shipped"].items():
        os.environ[k] = v
    det = {}
    for meta, acts in traces[RUNS[0]]:
        if acts is None:
            continue
        st = P1.P1State(meta["game_id"][:4])
        # game-scoped detector run (memo not level-cleared) -- this is the
        # published definition; the shipped patch clears per level, which is
        # strictly MORE conservative (fewer declines), never less safe.
        seen: dict[tuple[str, str], str] = {}
        n = 0
        for a in acts:
            key = (a["prev"], a["act"])
            if key in seen and seen[key] != a["out"]:
                n += 1
            seen.setdefault(key, a["out"])
        if n:
            det[meta["game_id"][:4]] = n
    out["detector"] = {"observed": det, "published": PUBLISHED_AMBIGUOUS,
                       "exact_match": det == PUBLISHED_AMBIGUOUS}
    print("=" * 78)
    print("ONLINE LATENT-STATE DETECTOR vs the published set (animation_v1)")
    print(f"  observed : {det}")
    print(f"  published: {PUBLISHED_AMBIGUOUS}")
    print(f"  EXACT MATCH: {out['detector']['exact_match']}")

    # ---- the diagnosis's own arithmetic, reproduced -----------------------
    pub = published_arithmetic(traces)
    out["published_arithmetic"] = pub
    print("=" * 78)
    print("PUBLISHED M1+M3 ARITHMETIC, reproduced independently")
    print("  (diagnosis sec2: 1.6352->1.8188 x1.11 / 1.4075->1.5627 x1.11 /"
          " 1.4509->1.5794 x1.09)")
    for r in pub:
        print(f"  {r['run'].split('/')[-1]:20s} {r['as_run']:.4f} ->"
              f" {r['m1m3']:.4f}  x{r['multiplier']:.4f}"
              f"   dup={r['dup']} blind={r['blind']}"
              f"   LEVEL-COMPLETING actions deleted={r['lc_lost']}")

    for arm in ARMS:
        out["arms"][arm] = []
        print("=" * 78)
        print(f"ARM {arm}")
        for rd in RUNS:
            r = score_arm(traces, rd, arm)
            out["arms"][arm].append(r)
            print(f"  {rd.split('/')[-1]:20s} {r['as_run']:.4f} -> {r['p1']:.4f}"
                  f"  x{r['multiplier']:.4f}  delta={r['delta']:+.4f}")
            print(f"     saved: declined={r['declined']:4d} aborted={r['aborted']:4d}"
                  f"   RISK: board-changing declines={r['diverged']:3d}"
                  f"   LEVEL-COMPLETING actions lost={r['lc_lost']}")
            print(f"     dup rate {r['dup_rate_before']*100:5.2f}% ->"
                  f" {r['dup_rate_after']*100:5.2f}%   blind-tail before"
                  f" {r['blind_rate_before']*100:5.2f}%"
                  f"   ambiguous games={len(r['ambiguous_games'])}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("=" * 78)
    print(f"wrote {OUT_DIR / 'report.json'}")
    if args.json:
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
