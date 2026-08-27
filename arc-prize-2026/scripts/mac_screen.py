"""LOCAL SCREENING RUNNER  [MAC-SCREEN]  (lane: local-rail)

Runs the campaign's REAL vehicle -- the frozen fork's HarnessSolver + ToolAgent
-- against the REAL 25 official games, driven by the local Qwen3.8-27B served
by mlx_lm.server. Emits the campaign's canonical `benchmark.json` so every
existing scorer and `local_gate` read a local run unchanged.

WHAT THIS IS FOR -- and the timing forces the answer
    MEASURED on this box with the 4-bit build and the prompt cache on: ~73 s
    per LLM call (was 182 s on 8-bit). The harness issues ~2.4 calls per
    action, so a 1-game/60-action probe is ~2.9 h and a full 25-game sweep is
    still ~231 HOURS. Local therefore CANNOT rank arms by score -- and the
    field floor's own draw noise (1.14 / 1.16 / 1.92 on IDENTICAL code) would
    swamp any local ordering anyway. `--estimate` prints the projection.

    What it CAN do cheaply -- and this is the high-value use -- is measure
    MECHANICAL and BEHAVIOURAL failure in a handful of turns:
      * does the arm act at all, or does tool-call parsing silently fail?
      * does the model USE an affordance it was given, or ignore it?
      * does the observation layer actually yield usable data?
    Those are exactly the failures that killed the last three arms: P2
    CERTIFIED but dead on delivery (10.73% use against a 25% bar), and exec-WM
    starved of transitions (9/18 games yielded zero). Both are visible in a
    1-game / 8-action probe costing ~25 MINUTES on 4-bit -- no Kaggle slot, no
    GPU spend. This rail has already earned that: reading ONE reasoning trace
    found the `animation()` phantom-tool defect, which was costing 29 actions
    across 13 of 25 games on the certified Kaggle vehicle.
    The campaign's own standing rule says it: PRE-MEASURE THE USE, NOT JUST
    THE FIRE. That is what this box is for.

WHAT THIS IS NOT
    A verdict, and not a score ranker. Env-mismatch is confirmed 5x and the Mac
    widens it (4-bit MLX on Metal vs FP8 on CUDA). Every
    number this writes is stamped [MAC-SCREEN]. No sealed verdict, no queue-head
    promotion, no band read comes from it. See MIGRATION_MACBOOK.md and the
    local_gate footer.

USAGE
    # terminal 1
    ./scripts/serve_local_model.sh
    # terminal 2
    .venv/bin/python scripts/mac_screen.py --label baseline --games 3 --max-actions 40
    .venv/bin/python scripts/mac_screen.py --label baseline --games 25 --estimate
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

# scripts/ contains queue.py, which shadows the stdlib `queue` and breaks every
# downstream harness import (urllib3/threading/concurrent.futures pull it in).
# Running `python scripts/mac_screen.py` puts scripts/ on sys.path[0], so strip
# it before anything else imports. Same guard as local_gate.py.
_SELF_DIR = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != _SELF_DIR]

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "duck_eval" / "private" / "bundle_20260815" / "src" / "ARC3-Inference"
OUT_ROOT = ROOT / "runs" / "mac_screen"
INDEX = OUT_ROOT / "INDEX.jsonl"

DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"

# MEASURED END-TO-END ON THIS BOX, 2026-08-27, against the real harness.
# Do NOT estimate from tok/s: generation is 18.1 tok/s but PREFILL dominates,
# because the harness resends a long game-state+history prefix every turn.
# Measured seconds per /chat/completions call, one game, identical workload:
#   no prompt cache : 326 s/call  (21 calls / 108.5 min)
#   prompt cache on : 182 s/call  (1.8x faster; still ~3 min/call)
# Re-measured 2026-08-27 on the 4-bit build (now the default): generation runs
# 32.9 tok/s vs 13.1 on 8-bit, and generation is ~75% of wall-clock, so call
# latency falls roughly 2.5x. 8-bit figures kept for comparison.
SEC_PER_CALL_CACHED = 39.0        # 4-bit MEASURED under the real harness
                                  # (73.0 was extrapolated from an isolated
                                  # benchmark and too pessimistic: the harness
                                  # issues many SHORT calls where 4-bit's faster
                                  # prefill pays too, not just generation)
SEC_PER_CALL_UNCACHED = 130.0     # 4-bit; was 326.0 on 8-bit
SEC_PER_CALL_8BIT = 182.0
# The harness issues more LLM calls than actions (analyzer + agent, retries).
# Observed ~2.4 calls per action on the ft09 smoke.
CALLS_PER_ACTION = 2.4
# Observed in real artifacts (runs/kernel_pulls/q38_v2, phase1_v2_screen, null10).
ACTIONS_PER_GAME = 190


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
def check_server(base_url: str) -> tuple[bool, str]:
    """Is a local OpenAI-shaped server up, and which model is it serving?"""
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=10) as r:
            data = json.load(r)
        ids = [m.get("id") for m in data.get("data", [])]
        if not ids:
            return False, "server up but serving no model"
        return True, ids[0]
    except urllib.error.URLError as exc:
        return False, f"cannot reach {base_url} ({exc.reason})"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def estimate(n_games: int, max_actions: int | None, cached: bool = True) -> dict:
    """Project wall-clock from MEASURED call latency, not from tok/s.

    Thinking on/off barely moves this: prefill of the resent prefix dominates,
    not generation length.
    """
    per_game = min(max_actions, ACTIONS_PER_GAME) if max_actions else ACTIONS_PER_GAME
    actions = n_games * per_game
    calls = actions * CALLS_PER_ACTION
    sec = SEC_PER_CALL_CACHED if cached else SEC_PER_CALL_UNCACHED
    hours = calls * sec / 3600
    return {
        "games": n_games,
        "actions_per_game": per_game,
        "total_actions": actions,
        "projected_llm_calls": round(calls),
        "sec_per_call_measured": sec,
        "prompt_cache": cached,
        "projected_hours": round(hours, 2),
    }


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
def build_games(n_games: int, game_ids: list[str] | None):
    import re_arc
    import taaf.game_api

    official = sorted(re_arc.list_game_ids(datasets=["train", "eval"], include_tags="official"))
    if game_ids:
        wanted = []
        for want in game_ids:
            hit = [g for g in official if g == want or g.split("-")[0] == want]
            if not hit:
                raise SystemExit(f"unknown game id {want!r}; official prefixes: "
                                 f"{sorted({g.split('-')[0] for g in official})}")
            wanted.extend(hit)
        chosen = wanted
    else:
        chosen = official[:n_games]
    return chosen, [taaf.game_api.GameAPI(env_name=g) for g in chosen]


def run_benchmark(args, model_id: str, chosen: list[str], games) -> Path:
    import taaf.benchmark
    sys.path.insert(0, str(BUNDLE))
    from inference.framework.solver import HarnessSolver

    solver = HarnessSolver(
        label=f"mac-screen-{args.label}",
        model=model_id,
        concurrency=args.concurrency,
        max_actions_per_game=args.max_actions,
        kaggle_enable_vllm=False,   # the model is already served by mlx_lm.server
        start_local_server=False,   # ... so the harness must not start its own
        analyzer_timeout=args.analyzer_timeout,
        animation_awareness=True,
        animation_retrieval=False,
        hard_noop_guard=True,
        save_request_logs=True,
    )
    job = Path(tempfile.mkdtemp(prefix=f"macscreen-{args.label}-"))
    bm = taaf.benchmark.Benchmark(
        label=f"[MAC-SCREEN] {args.label}",
        games=games,
        solver=solver,
        n_passes=1,
        job_dir=job,
        periodic_save_interval_s=60.0,
    )
    asyncio.run(bm.run())
    return job


# ---------------------------------------------------------------------------
# diagnosis -- "what is going wrong and what to fix next"
# ---------------------------------------------------------------------------
def diagnose(bench: dict) -> dict:
    runs = bench.get("game_runs", [])
    per_game, cleared, stuck, zero_action = [], [], [], []
    scores, lc_total = [], 0

    for r in runs:
        gid = r.get("game_id", "?")
        lc = r.get("levels_completed", 0) or 0
        nl = r.get("number_of_levels", 0) or 0
        acts = sum(r.get("actions_per_level") or [])
        score = r.get("final_score", 0.0) or 0.0
        note = (r.get("solver_note") or "").strip()
        state = r.get("state", "")
        lc_total += lc
        scores.append(score)
        row = {"game_id": gid, "levels_completed": lc, "number_of_levels": nl,
               "actions": acts, "final_score": score, "state": state,
               "solver_note": note[:400]}
        per_game.append(row)
        if lc > 0:
            cleared.append(gid)
        if acts == 0:
            zero_action.append(gid)
        elif lc == 0:
            stuck.append(gid)

    findings = []
    if zero_action:
        findings.append({
            "severity": "critical",
            "what": f"{len(zero_action)} game(s) took ZERO actions: {zero_action[:6]}",
            "means": "the solver never issued an action -- harness wiring, model "
                     "endpoint, or tool-call parsing failed, not a reasoning failure",
            "fix_next": "check the request logs; confirm the model returns tool_calls "
                        "in a shape the ToolAgent parses (mlx_lm.server has no "
                        "vLLM tool-call parser)",
        })
    if stuck:
        findings.append({
            "severity": "high",
            "what": f"{len(stuck)} game(s) acted but cleared no level: {stuck[:6]}",
            "means": "the agent is acting but not solving -- a genuine reasoning or "
                     "planning failure, which IS what screening should surface",
            "fix_next": "read solver_note per game; compare against a Kaggle run of "
                        "the same arm to separate local quantisation drift from a "
                        "real arm regression",
        })
    if not runs:
        findings.append({
            "severity": "critical",
            "what": "no game_runs in the artifact",
            "means": "the benchmark produced nothing",
            "fix_next": "check the harness raised before the first game started",
        })

    return {
        "n_games": len(runs),
        "levels_completed_total": lc_total,
        "games_cleared": cleared,
        "games_stuck": stuck,
        "games_zero_action": zero_action,
        "mean_score": round(statistics.fmean(scores), 6) if scores else 0.0,
        "total_actions": sum(p["actions"] for p in per_game),
        "per_game": per_game,
        "findings": findings,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Local [MAC-SCREEN] runner for the ARC campaign")
    p.add_argument("--label", required=True, help="short slug for this iteration")
    p.add_argument("--games", type=int, default=3, help="how many official games (default 3)")
    p.add_argument("--game-ids", default=None, help="comma-separated ids/prefixes, e.g. ft09,ar25")
    p.add_argument("--max-actions", type=int, default=40, help="action cap per game (default 40)")
    p.add_argument("--concurrency", type=int, default=1, help="parallel games (default 1; each "
                                                              "holds the single local server)")
    p.add_argument("--analyzer-timeout", type=float, default=600.0)
    p.add_argument("--base-url", default=os.environ.get("LOCAL_LLM_BASE_URL", DEFAULT_BASE_URL))
    p.add_argument("--no-think", action="store_true",
                   help="assert the SERVER is running with thinking disabled. The harness "
                        "sends no chat_template_kwargs, so thinking is a SERVER-side setting: "
                        "start it with LOCAL_MODEL_THINKING=0 ./scripts/serve_local_model.sh")
    p.add_argument("--note", default="", help="free text recorded with the iteration")
    p.add_argument("--estimate", action="store_true", help="print the time projection and exit")
    args = p.parse_args()

    n_games = len(args.game_ids.split(",")) if args.game_ids else args.games
    est = estimate(n_games, args.max_actions, cached=True)

    if args.estimate:
        print(json.dumps(est, indent=2))
        print(f"\n  ~{est['projected_hours']:.1f} h projected "
              f"({est['projected_llm_calls']} calls x {est['sec_per_call_measured']:.0f}s measured)")
        return 0

    if args.no_think:
        print("[mac_screen] --no-think: verify the server was started with "
              "LOCAL_MODEL_THINKING=0; this flag cannot change a running server.")

    ok, model_id = check_server(args.base_url)
    if not ok:
        print(f"[mac_screen] LOCAL SERVER NOT READY: {model_id}", file=sys.stderr)
        print("  start it with:  ./scripts/serve_local_model.sh", file=sys.stderr)
        return 2
    print(f"[mac_screen] server OK at {args.base_url}, serving {model_id}")
    print(f"[mac_screen] projection: ~{est['projected_hours']:.1f} h "
          f"({est['total_actions']} actions)")

    os.environ["OPENAI_BASE_URL"] = args.base_url
    os.environ.setdefault("OPENAI_API_KEY", "local")

    chosen, games = build_games(args.games, args.game_ids.split(",") if args.game_ids else None)
    print(f"[mac_screen] games: {chosen}")

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = OUT_ROOT / f"{stamp}_{args.label}"
    out.mkdir(parents=True, exist_ok=True)

    started = time.time()
    started_iso = _now()
    error = None
    try:
        job = run_benchmark(args, model_id, chosen, games)
    except KeyboardInterrupt:
        print("\n[mac_screen] interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        job = None
        print(f"[mac_screen] RUN FAILED: {error}", file=sys.stderr)
    elapsed = time.time() - started

    bench = {}
    if job:
        # Keep the WHOLE job dir. The harness writes its evidence to the job
        # ROOT, not to tidy subdirectories: `<game>_requests.jsonl` (every
        # request AND response snapshot, so the model's raw content -- which on
        # mlx_lm.server carries the <think> block inline, there being no
        # reasoning parser), `prompts/<game>.log`, intermediate states, movies.
        # Cherry-picking named subdirs silently dropped all of it. Copy
        # everything; disk is cheap and these traces ARE the diagnosis.
        for item in job.iterdir():
            dest = out / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
        if (out / "benchmark.json").is_file():
            bench = json.loads((out / "benchmark.json").read_text(encoding="utf-8"))

    diag = diagnose(bench) if bench else {"findings": [
        {"severity": "critical", "what": "no benchmark.json produced",
         "means": error or "the harness exited before writing an artifact",
         "fix_next": "read the traceback above"}]}

    record = {
        "label": args.label,
        "stamp": stamp,
        "started": started_iso,
        "finished": _now(),
        "elapsed_s": round(elapsed, 1),
        "lane": "MAC-SCREEN",
        "certifies": False,
        "model": model_id,
        "base_url": args.base_url,
        "thinking": not args.no_think,
        "games": chosen,
        "max_actions_per_game": args.max_actions,
        "concurrency": args.concurrency,
        "note": args.note,
        "estimate": est,
        "error": error,
        "results": diag,
        "results_path": str(out.relative_to(ROOT)),
    }
    (out / "iteration.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with INDEX.open("a", encoding="utf-8") as fh:
        slim = {k: record[k] for k in ("label", "stamp", "elapsed_s", "model", "thinking",
                                       "games", "max_actions_per_game", "results_path", "error")}
        slim["levels_completed_total"] = diag.get("levels_completed_total")
        slim["mean_score"] = diag.get("mean_score")
        fh.write(json.dumps(slim) + "\n")

    # ---- report -----------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"  [MAC-SCREEN] {args.label}   {elapsed/60:.1f} min")
    print("=" * 78)
    if bench:
        print(f"  games {diag['n_games']}   levels_completed {diag['levels_completed_total']}"
              f"   mean_score {diag['mean_score']}   actions {diag['total_actions']}")
        for row in diag["per_game"]:
            mark = "OK " if row["levels_completed"] else ("!! " if row["actions"] == 0 else " . ")
            print(f"   {mark}{row['game_id']:<20} lc={row['levels_completed']}/{row['number_of_levels']:<3}"
                  f" actions={row['actions']:<5} score={row['final_score']}")
    if diag["findings"]:
        print("\n  WHAT IS GOING WRONG / WHAT TO FIX NEXT")
        for f in diag["findings"]:
            print(f"   [{f['severity'].upper()}] {f['what']}")
            print(f"        means    : {f['means']}")
            print(f"        fix next : {f['fix_next']}")
    print(f"\n  artifact : {out.relative_to(ROOT)}/benchmark.json")
    print(f"  iteration: {out.relative_to(ROOT)}/iteration.json")
    print("  SCREEN ONLY -- this licenses a Kaggle build, never a verdict.")
    print("=" * 78)
    return 0 if not error else 1


if __name__ == "__main__":
    raise SystemExit(main())
