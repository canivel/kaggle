#!/usr/bin/env python
"""Quantify the free-rail (Kaggle interactive BUILD) vs scored-rail (competition
RERUN) throughput-regime gap for the duck harness.

R25 systems FATAL: "free-rail and scored rail are different throughput regimes ->
'build-rail instrument is our most important asset' is unestablished for the
scored twin."

$0, offline, reads only artifacts already on disk. No Kaggle API, no pushes, no
network. Emits a text report to stdout and JSON to
runs/rail_regime_gap_2026-08-10.json.

Sources (all repo-local, cited in learnings/sweeps/rail_regime_gap_2026-08-10.md):
  runs/kernel_pulls/*/benchmark.json          per-game wall clock, actions, score
  runs/kernel_pulls/*/*.log                   kernel-level wall clock (setup+teardown)
  runs/kernel_pulls/*/vllm-openai-server.log  engine Running/Waiting/throughput/KV
  runs/null10/seed*/benchmark.json            10 same-config build-rail nulls
  duck_eval/taaf_bundle/preamble.txt          deployed solver config (concurrency=28)
  duck_eval/taaf_bundle/deploy_target.pkl     KaggleTarget.max_runtime_s
  duck_eval/taaf_bundle/src/.../run.py:583    wave arithmetic
  runs/kernel_pulls/a22_v2_1/m1m2m3_screen.json  LLM turn accounting
  runs/lb_ground_truth.md                     scored-rail draw ledger
"""

from __future__ import annotations

import io
import json
import math
import pickletools
import re
import statistics as st
from datetime import datetime
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "runs" / "rail_regime_gap_2026-08-10.json"

# --- constants read from repo files -------------------------------------------
DEPLOYED_CONCURRENCY = 28          # duck_eval/taaf_bundle/preamble.txt:2
DEPLOYED_GUILLOTINE_S = 7920.0     # ditto (max_runtime_s_per_game)
DEPLOYED_N_PASSES = 1
BUILD_GAME_COUNT = 25              # bundled offline environment_files
OFFICIAL_GAME_COUNT = 110          # learnings/gap_forensics_2026-07-09.md:29
                                   # runs/lb_process_model/lb_process_model.py:49
GATEWAY_WAIT_CAP_S = 600.0         # notebook cell 14 _wait_for_gateway(timeout_s=600)

# runs/kernel_pulls/a22_v2_1/m1m2m3_screen.json -> M2 (war-eval seed-1, ledger-OFF)
WAR_TURNS = 1686
WAR_ACTIONS = 3638
WAR_GEN_TOKENS = 1569582

# Tycho budgets, learnings/war_room/tycho_portability_2026-08-08.md §2.7 / §6.2
TYCHO_LM_CALLS_PER_GAME = 3500
TYCHO_BUILDER_CALLS_TOTAL = 147
TYCHO_BUILDER_GAMES = 25
TYCHO_TOOL_STEPS_PER_TURN = 40
TYCHO_ANSWER_TOKENS_PER_CALL = 24000

# runs/lb_ground_truth.md -- frozen-fork scored draw ledger, oldest..newest (n=27)
FROZEN_DRAWS = [
    0.82, 0.89, 0.93, 1.02, 0.95, 1.33, 0.92, 0.93, 1.14, 0.82,
    1.05, 0.84, 1.02, 0.90, 1.03, 0.85, 1.10, 0.65, 0.68, 0.99,
    0.97, 1.21, 0.77, 0.78, 0.87, 0.89, 1.05,
]


def wave_count(games: int, n_passes: int, concurrency: int) -> int:
    """Mirror of inference/framework/run.py:_wave_count (:583)."""
    return math.ceil(games * n_passes / concurrency)


def deploy_target_max_runtime_s(path: Path) -> float | None:
    """Read KaggleTarget.max_runtime_s out of the pickle WITHOUT unpickling."""
    buf = io.StringIO()
    try:
        pickletools.dis(open(path, "rb"), out=buf)
    except Exception:
        return None
    lines = buf.getvalue().splitlines()
    for i, ln in enumerate(lines):
        if "'max_runtime_s'" in ln:
            for nxt in lines[i + 1 : i + 5]:
                m = re.search(r"BINFLOAT\s+([0-9.eE+-]+)", nxt)
                if m:
                    return float(m.group(1))
    return None


# ------------------------------------------------------------------ build rail
def read_benchmark(path: Path) -> dict | None:
    try:
        d = json.load(open(path))
    except Exception:
        return None
    gr = d.get("game_runs") or []
    if not gr:
        return None
    walls = [g["final_wallclock_seconds"] for g in gr]
    acts = [
        sum(g["actions_per_level"]) if isinstance(g["actions_per_level"], list) else 0
        for g in gr
    ]
    starts = sorted(g["started_at"] for g in gr)
    t0 = datetime.fromisoformat(d["start_time"])
    t1 = datetime.fromisoformat(d["end_time"])
    overall = (t1 - t0).total_seconds()
    return {
        "run": path.parent.name,
        "label": d.get("label", ""),
        "n_games": len(gr),
        "bench_wall_s": overall,
        "sum_game_wall_s": sum(walls),
        "effective_parallelism": sum(walls) / overall,
        "launch_spread_s": (
            datetime.fromisoformat(starts[-1]) - datetime.fromisoformat(starts[0])
        ).total_seconds(),
        "wall_min": min(walls),
        "wall_max": max(walls),
        "n_at_guillotine": sum(1 for w in walls if w >= DEPLOYED_GUILLOTINE_S),
        "total_actions": sum(acts),
        "s_per_action": sum(walls) / max(sum(acts), 1),
        "levels_completed": sum(g["levels_completed"] for g in gr),
        "mean_score": st.mean(g["final_score"] for g in gr),
    }


def kernel_wall_s(run_dir: Path) -> float | None:
    """Last stream timestamp in the Kaggle kernel log = total notebook wall."""
    logs = [p for p in run_dir.glob("*.log") if "vllm" not in p.name]
    if not logs:
        return None
    tail = logs[0].read_bytes()[-8000:].decode("utf8", "ignore")
    ts = re.findall(r'"time":([0-9.]+)', tail)
    return float(ts[-1]) if ts else None


VLLM_RE = re.compile(
    r"Avg prompt throughput: ([0-9.]+) tokens/s, "
    r"Avg generation throughput: ([0-9.]+) tokens/s, "
    r"Running: (\d+) reqs, Waiting: (\d+) reqs, "
    r"GPU KV cache usage: ([0-9.]+)%"
)


def read_vllm(path: Path) -> list[tuple[float, float, int, int, float]]:
    rows = []
    try:
        with open(path, "r", errors="ignore") as fh:
            for line in fh:
                m = VLLM_RE.search(line)
                if m:
                    rows.append(
                        (float(m.group(1)), float(m.group(2)), int(m.group(3)),
                         int(m.group(4)), float(m.group(5)))
                    )
    except OSError:
        pass
    return rows


def main() -> None:
    rep: dict = {"generated": "2026-08-10"}

    # ---------------- build-rail runs
    pulls, nulls = [], []
    for p in sorted(glob(str(ROOT / "runs" / "kernel_pulls" / "*" / "benchmark.json"))):
        r = read_benchmark(Path(p))
        if r:
            r["kernel_wall_s"] = kernel_wall_s(Path(p).parent)
            if r["kernel_wall_s"]:
                r["overhead_s"] = r["kernel_wall_s"] - r["bench_wall_s"]
            pulls.append(r)
    for p in sorted(glob(str(ROOT / "runs" / "null10" / "seed*" / "benchmark.json"))):
        r = read_benchmark(Path(p))
        if r:
            r["run"] = "null10/" + Path(p).parent.name
            nulls.append(r)
    rep["build_rail_pulls"] = pulls
    rep["build_rail_nulls"] = nulls

    full = [r for r in pulls + nulls if r["n_games"] == 25]
    # overhead only from FULL 25-game kernel pulls (a17 4-game pulls are a
    # different shape and would bias the setup+teardown estimate)
    ovh = [r["overhead_s"] for r in pulls
           if r.get("overhead_s") and r["n_games"] == 25]
    rep["build_rail_summary"] = {
        "n_runs_25game": len(full),
        "n_kernel_pull_25game": len([r for r in pulls if r["n_games"] == 25]),
        "bench_wall_s": {"min": min(r["bench_wall_s"] for r in full),
                         "max": max(r["bench_wall_s"] for r in full),
                         "mean": st.mean(r["bench_wall_s"] for r in full)},
        "kernel_wall_s_observed": {"min": min(ovh and [r["kernel_wall_s"] for r in pulls
                                                       if r.get("kernel_wall_s")] or [0]),
                                   "max": max([r["kernel_wall_s"] for r in pulls
                                               if r.get("kernel_wall_s")] or [0])},
        "overhead_s": {"min": min(ovh), "max": max(ovh), "mean": st.mean(ovh)} if ovh else None,
        "effective_parallelism": {"min": min(r["effective_parallelism"] for r in full),
                                  "max": max(r["effective_parallelism"] for r in full)},
        "launch_spread_s_max": max(r["launch_spread_s"] for r in full),
        "frac_games_at_guillotine": sum(r["n_at_guillotine"] for r in full)
        / sum(r["n_games"] for r in full),
        "s_per_action": {"min": min(r["s_per_action"] for r in full),
                         "max": max(r["s_per_action"] for r in full),
                         "mean": st.mean(r["s_per_action"] for r in full)},
        "actions_per_game": {
            "min": min(r["total_actions"] / r["n_games"] for r in full),
            "max": max(r["total_actions"] / r["n_games"] for r in full),
            "mean": st.mean(r["total_actions"] / r["n_games"] for r in full)},
    }

    # ---------------- wave arithmetic + ceiling
    tgt = deploy_target_max_runtime_s(ROOT / "duck_eval" / "taaf_bundle" / "deploy_target.pkl")
    wb = wave_count(BUILD_GAME_COUNT, DEPLOYED_N_PASSES, DEPLOYED_CONCURRENCY)
    ws = wave_count(OFFICIAL_GAME_COUNT, DEPLOYED_N_PASSES, DEPLOYED_CONCURRENCY)
    proj_scored = ws * DEPLOYED_GUILLOTINE_S
    mean_ovh = st.mean(ovh) if ovh else 0.0
    rep["wave_arithmetic"] = {
        "concurrency": DEPLOYED_CONCURRENCY,
        "guillotine_s": DEPLOYED_GUILLOTINE_S,
        "deploy_target_max_runtime_s": tgt,
        "build": {"games": BUILD_GAME_COUNT, "waves": wb,
                  "projected_bench_s": wb * DEPLOYED_GUILLOTINE_S,
                  "steady_children": min(BUILD_GAME_COUNT, DEPLOYED_CONCURRENCY)},
        "scored": {"games": OFFICIAL_GAME_COUNT, "waves": ws,
                   "projected_bench_s": proj_scored,
                   "projected_bench_h": proj_scored / 3600,
                   "steady_children": DEPLOYED_CONCURRENCY,
                   "last_wave_children": OFFICIAL_GAME_COUNT
                   - DEPLOYED_CONCURRENCY * (ws - 1)},
        "scored_total_with_observed_overhead_s": proj_scored + mean_ovh,
        "scored_margin_vs_target_s": (tgt - (proj_scored + mean_ovh)) if tgt else None,
        "scored_margin_if_gateway_wait_maxed_s": (
            tgt - (proj_scored + mean_ovh + GATEWAY_WAIT_CAP_S)) if tgt else None,
        "wall_ratio_scored_over_build": ws / wb,
        "concurrency_ratio_scored_over_build":
            DEPLOYED_CONCURRENCY / min(BUILD_GAME_COUNT, DEPLOYED_CONCURRENCY),
    }

    # ---------------- vLLM saturation (25-game pulls only)
    runs25 = ["war_eval_v1", "war_eval_v2", "war_eval_v3", "a22_v2_1", "phase1_v5",
              "gate_eval_v1", "gate_eval_v2", "sched_v1", "w0_eval_s1", "w0_cont_eval",
              "sentinel_eval_v1", "sentinel_eval_v2", "war_v2_eval_s1"]
    pooled: dict[int, list[float]] = {}
    waiting, kvs = {}, {}
    for run in runs25:
        p = ROOT / "runs" / "kernel_pulls" / run / "vllm-openai-server.log"
        if not p.exists():
            continue
        rws = read_vllm(p)
        for _pt, gen, running, _w, _kv in rws:
            if running > 0:
                pooled.setdefault(running, []).append(gen)
        hi = [(w, kv) for _pt, _g, r, w, kv in rws if r >= 20]
        if hi:
            waiting[run] = {"n": len(hi),
                            "frac_waiting_gt0": sum(1 for w, _ in hi if w > 0) / len(hi),
                            "max_waiting": max(w for w, _ in hi)}
            kvs[run] = {"median_kv_pct": st.median([kv for _, kv in hi]),
                        "max_kv_pct": max(kv for _, kv in hi)}
    sat = {r: {"n": len(v), "median_total_tok_s": st.median(v),
               "median_per_req_tok_s": st.median(v) / r}
           for r, v in sorted(pooled.items()) if len(v) >= 40}
    rep["vllm_saturation_25game"] = sat
    rep["vllm_queueing"] = waiting
    rep["vllm_kv"] = kvs

    # log-log slope of per-request throughput vs Running over the dense region
    dense = {r: v for r, v in sat.items() if r >= 20}
    if len(dense) >= 3:
        xs = [math.log(r) for r in dense]
        ys = [math.log(v["median_per_req_tok_s"]) for v in dense.values()]
        mx, my = st.mean(xs), st.mean(ys)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        at25 = dense.get(25, {}).get("median_per_req_tok_s")
        rep["saturation_fit"] = {
            "region": sorted(dense),
            "loglog_slope_per_req_vs_running": slope,
            "interpretation": "slope <= -1 => engine saturated; extra children cost "
                              "per-child throughput ~1:1 or worse",
            "per_req_tok_s_at_25": at25,
            "extrapolated_per_req_tok_s_at_28_fitfit":
                at25 * (28 / 25) ** slope if at25 else None,
            "extrapolated_per_req_tok_s_at_28_pure_1_over_R":
                at25 * (25 / 28) if at25 else None,
            "derating_28_over_25_fit": (28 / 25) ** slope,
            "derating_28_over_25_pure": 25 / 28,
        }

    # ---------------- LLM turn / action economics (used by the port-scope memo)
    per_game_turns = WAR_TURNS / BUILD_GAME_COUNT
    s_per_turn = DEPLOYED_GUILLOTINE_S / per_game_turns
    tycho_builder_cadence = TYCHO_BUILDER_CALLS_TOTAL / TYCHO_BUILDER_GAMES
    rep["turn_economics"] = {
        "war_turns_total": WAR_TURNS, "war_actions_total": WAR_ACTIONS,
        "turns_per_game": per_game_turns,
        "actions_per_game": WAR_ACTIONS / BUILD_GAME_COUNT,
        "s_per_turn": s_per_turn,
        "s_per_action": DEPLOYED_GUILLOTINE_S / (WAR_ACTIONS / BUILD_GAME_COUNT),
        "gen_tokens_per_turn": WAR_GEN_TOKENS / WAR_TURNS,
        "tycho_lm_calls_per_game": TYCHO_LM_CALLS_PER_GAME,
        "lm_call_gap_x": TYCHO_LM_CALLS_PER_GAME / per_game_turns,
        "tycho_answer_tokens_per_call": TYCHO_ANSWER_TOKENS_PER_CALL,
        "answer_token_gap_x": TYCHO_ANSWER_TOKENS_PER_CALL / (WAR_GEN_TOKENS / WAR_TURNS),
        "tycho_builder_calls_per_game": tycho_builder_cadence,
        "builder_call_cost_s_at_40_steps": TYCHO_TOOL_STEPS_PER_TURN * s_per_turn,
        "builder_cost_frac_of_game_budget_at_tycho_cadence":
            tycho_builder_cadence * TYCHO_TOOL_STEPS_PER_TURN * s_per_turn / DEPLOYED_GUILLOTINE_S,
        "builder_turn_frac_if_5_turns_per_call":
            tycho_builder_cadence * 5 / per_game_turns,
        "builder_turn_frac_if_1_turn_per_call":
            tycho_builder_cadence * 1 / per_game_turns,
    }

    # ---------------- scored rail: ledger + resolving power
    n = len(FROZEN_DRAWS)
    mu, sd = st.mean(FROZEN_DRAWS), st.stdev(FROZEN_DRAWS)
    z_a, z_b = 1.959964, 0.841621  # alpha=.05 two-sided, power .80
    def n_per_arm(d: float) -> float:
        return 2 * sd * sd * (z_a + z_b) ** 2 / (d * d)
    rep["scored_rail"] = {
        "n_draws": n, "mean": mu, "sd": sd, "cv": sd / mu,
        "min": min(FROZEN_DRAWS), "max": max(FROZEN_DRAWS),
        "draws_needed_per_arm_alpha05_power80": {
            f"{d:.2f}": n_per_arm(d) for d in (0.05, 0.10, 0.15, 0.25, 0.40)},
        "note": "one draw = one calendar day (daily submit cap); two arms => 2x days",
    }

    # ---------------- information-rate asymmetry
    builds_per_week_lo, builds_per_week_hi = 12, 13   # r24 minutes §5.4
    rep["information_rate"] = {
        "build_rail_game_obs_per_week": [builds_per_week_lo * 25, builds_per_week_hi * 25],
        "build_rail_scalars_per_run": 25,
        "scored_rail_scalars_per_week": 7,
        "ratio_obs_per_week": builds_per_week_lo * 25 / 7,
    }

    # ---------------- cross-rail score ratio (confounded, stated as such)
    pull25 = [r for r in pulls if r["n_games"] == 25]
    bmean = st.mean(r["mean_score"] for r in pull25)
    rep["cross_rail_score"] = {
        "build_rail_mean_per_game_score_kernel_pulls": bmean,
        "build_rail_13seed_grand_mean_lb_process_model": 1.593972115811736,
        "scored_rail_mean_draw": mu,
        "ratio_scored_over_build_13seed": mu / 1.593972115811736,
        "note": "CONFOUNDED: 25 public offline games vs 110 official gateway games. "
                "This is a game-set + rail composite, not a pure rail effect. "
                "lb_process_model.py already uses c=0.58 as the difficulty calibration.",
    }

    # ---------------- wave-4 truncation sensitivity
    # If the kernel is cut at the ceiling mid-wave-4, the last wave's games score 0.
    # How much of the apparent "official set is harder" discount (c=0.58) could that
    # explain, with no difficulty difference at all?
    last_wave = OFFICIAL_GAME_COUNT - DEPLOYED_CONCURRENCY * (ws - 1)
    surviving = 1 - last_wave / OFFICIAL_GAME_COUNT
    grand13 = 1.593972115811736
    rep["truncation_sensitivity"] = {
        "last_wave_games": last_wave,
        "fraction_of_official_set": last_wave / OFFICIAL_GAME_COUNT,
        "observed_scored_mean": mu,
        "implied_untruncated_mean_if_last_wave_zeroed": mu / surviving,
        "c_difficulty_no_truncation": mu / grand13,
        "c_difficulty_with_full_last_wave_truncation": (mu / surviving) / grand13,
        "note": "NOT a measurement — a bound on how much of lb_process_model's c=0.58 "
                "could be rail truncation rather than official-set difficulty. "
                "Nothing on disk distinguishes the two.",
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(OUT_JSON, "w"), indent=2)

    # ----------------------------------------------------------------- report
    P = print
    P("=" * 78)
    P("RAIL REGIME GAP — build (Kaggle interactive) vs scored (competition rerun)")
    P("=" * 78)
    s = rep["build_rail_summary"]
    P(f"\n[BUILD RAIL] {s['n_runs_25game']} full 25-game runs "
      f"({s['n_kernel_pull_25game']} kernel pulls + 10 null10 seeds)")
    P(f"  benchmark wall   {s['bench_wall_s']['min']:.0f}–{s['bench_wall_s']['max']:.0f} s "
      f"(mean {s['bench_wall_s']['mean']:.0f} s = {s['bench_wall_s']['mean']/3600:.2f} h)")
    P(f"  kernel wall      {s['kernel_wall_s_observed']['min']:.0f}–"
      f"{s['kernel_wall_s_observed']['max']:.0f} s")
    if s["overhead_s"]:
        P(f"  setup+teardown   {s['overhead_s']['min']:.0f}–{s['overhead_s']['max']:.0f} s "
          f"(mean {s['overhead_s']['mean']:.0f} s)")
    P(f"  effective parallelism {s['effective_parallelism']['min']:.2f}–"
      f"{s['effective_parallelism']['max']:.2f} ; max launch spread "
      f"{s['launch_spread_s_max']:.2f} s -> ONE wave")
    P(f"  games ending AT the 7920 s guillotine: {s['frac_games_at_guillotine']*100:.1f}%")
    P(f"  s / scored action {s['s_per_action']['min']:.1f}–{s['s_per_action']['max']:.1f} "
      f"(mean {s['s_per_action']['mean']:.1f})")
    P(f"  actions / game    {s['actions_per_game']['min']:.0f}–"
      f"{s['actions_per_game']['max']:.0f} (mean {s['actions_per_game']['mean']:.0f})")

    w = rep["wave_arithmetic"]
    P(f"\n[WAVE ARITHMETIC] concurrency={w['concurrency']}, guillotine={w['guillotine_s']:.0f}s, "
      f"KaggleTarget.max_runtime_s={w['deploy_target_max_runtime_s']}")
    P(f"  build : {w['build']['games']:>3} games -> {w['build']['waves']} wave  "
      f"{w['build']['projected_bench_s']/3600:.2f} h bench, "
      f"{w['build']['steady_children']} steady children")
    P(f"  scored: {w['scored']['games']:>3} games -> {w['scored']['waves']} waves "
      f"{w['scored']['projected_bench_h']:.2f} h bench, "
      f"{w['scored']['steady_children']} steady children "
      f"(last wave {w['scored']['last_wave_children']})")
    P(f"  scored total w/ observed overhead: {w['scored_total_with_observed_overhead_s']:.0f} s "
      f"= {w['scored_total_with_observed_overhead_s']/3600:.2f} h")
    P(f"  margin vs target ceiling: {w['scored_margin_vs_target_s']:.0f} s  "
      f"(if gateway wait maxes at 600 s: {w['scored_margin_if_gateway_wait_maxed_s']:.0f} s)")
    P(f"  wall ratio scored/build {w['wall_ratio_scored_over_build']:.1f}x ; "
      f"concurrency ratio {w['concurrency_ratio_scored_over_build']:.2f}x")

    P("\n[vLLM SATURATION] pooled over 25-game pulls")
    P(f"  {'Running':>8} {'n':>6} {'tot tok/s':>11} {'per-req tok/s':>14}")
    for r, v in rep["vllm_saturation_25game"].items():
        P(f"  {r:>8} {v['n']:>6} {v['median_total_tok_s']:>11.1f} "
          f"{v['median_per_req_tok_s']:>14.2f}")
    if "saturation_fit" in rep:
        f = rep["saturation_fit"]
        P(f"  log-log slope (per-req tok/s vs Running, R>=20): {f['loglog_slope_per_req_vs_running']:.3f}")
        P(f"  de-rating 25 -> 28 children: fit {f['derating_28_over_25_fit']:.3f}x, "
          f"pure 1/R {f['derating_28_over_25_pure']:.3f}x")

    P("\n[vLLM QUEUEING / KV at Running>=20]")
    for run in rep["vllm_queueing"]:
        q, k = rep["vllm_queueing"][run], rep["vllm_kv"][run]
        P(f"  {run:<18} frac(Waiting>0)={q['frac_waiting_gt0']*100:5.1f}%  "
          f"maxWait={q['max_waiting']}  KV median {k['median_kv_pct']:.1f}% max {k['max_kv_pct']:.1f}%")

    t = rep["turn_economics"]
    P(f"\n[TURN ECONOMICS] {t['turns_per_game']:.1f} LLM turns/game, "
      f"{t['actions_per_game']:.1f} actions/game, {t['s_per_turn']:.1f} s/turn, "
      f"{t['s_per_action']:.1f} s/action, {t['gen_tokens_per_turn']:.0f} gen tok/turn")
    P(f"  Tycho gap: LM calls {t['lm_call_gap_x']:.1f}x, answer tokens {t['answer_token_gap_x']:.1f}x")
    P(f"  one Tycho builder call at 40 tool steps = {t['builder_call_cost_s_at_40_steps']:.0f} s "
      f"= {t['builder_call_cost_s_at_40_steps']/DEPLOYED_GUILLOTINE_S*100:.0f}% of a game's wall budget")
    P(f"  Tycho cadence {t['tycho_builder_calls_per_game']:.1f} calls/game -> "
      f"{t['builder_cost_frac_of_game_budget_at_tycho_cadence']*100:.0f}% of the game budget")
    P(f"  truncated to 5 turns/call -> {t['builder_turn_frac_if_5_turns_per_call']*100:.0f}% "
      f"of the actor's turn budget; 1 turn/call -> "
      f"{t['builder_turn_frac_if_1_turn_per_call']*100:.0f}%")

    sr = rep["scored_rail"]
    P(f"\n[SCORED RAIL] n={sr['n_draws']} draws  mean {sr['mean']:.4f}  sd {sr['sd']:.4f}  "
      f"CV {sr['cv']*100:.1f}%")
    P("  draws needed PER ARM (alpha .05, power .80) to resolve an LB delta:")
    for d, v in sr["draws_needed_per_arm_alpha05_power80"].items():
        P(f"    delta {d}: {v:6.1f} draws/arm  ({2*v:6.1f} calendar days for a 2-arm test)")
    ir = rep["information_rate"]
    P(f"  information rate: build rail {ir['build_rail_game_obs_per_week'][0]}–"
      f"{ir['build_rail_game_obs_per_week'][1]} per-game observations/week vs "
      f"scored rail {ir['scored_rail_scalars_per_week']} scalars/week "
      f"({ir['ratio_obs_per_week']:.0f}x)")

    cs = rep["cross_rail_score"]
    P(f"\n[CROSS-RAIL SCORE] build 13-seed grand mean "
      f"{cs['build_rail_13seed_grand_mean_lb_process_model']:.3f} (25 public) vs scored "
      f"{cs['scored_rail_mean_draw']:.4f} (110 official) -> "
      f"{cs['ratio_scored_over_build_13seed']:.3f}")
    P(f"  {cs['note']}")
    ts = rep["truncation_sensitivity"]
    P(f"\n[TRUNCATION SENSITIVITY] wave 4 = {ts['last_wave_games']} games "
      f"({ts['fraction_of_official_set']*100:.1f}% of the official set)")
    P(f"  c if pure difficulty: {ts['c_difficulty_no_truncation']:.3f}; "
      f"c if wave 4 is fully lost to the ceiling: "
      f"{ts['c_difficulty_with_full_last_wave_truncation']:.3f}")
    P(f"  {ts['note']}")
    P(f"\nJSON -> {OUT_JSON}")


if __name__ == "__main__":
    main()
