"""ITEM 1 (R24 §5.3 / §3.1) — latency instrumentation + matched-action-prefix endpoint.

WHY
---
`runs/kernel_pulls/war_eval_v1/arc3-duck-war-eval.log:657` prints
`max_actions_per_game=None, max_runtime_s_per_game=7920.0, concurrency=28`.
Every game in every pull terminates on the wall-clock guillotine, so the number
of scored actions a run gets is an OUTPUT of throughput, not a fixed budget.
Consequence (minutes §3.1): a Δ levels-completed endpoint measured at equal WALL
CLOCK cannot separate "the mechanism is harmful" from "the mechanism is slower".

THIS MODULE
-----------
1. `latency_table(pull)`   -- per-game wall-clock instrumentation from
   benchmark.json (cumulative `history[i].wallclock_seconds`) plus tool-call
   counts from the transcripts.
2. `matched_prefix(arm, base)` -- for each game, truncate BOTH arms to
   K = min(actions_arm, actions_base) scored actions and recompute levels
   completed at that prefix. `GameRun.lc_at_action_prefix` is exact: the pull
   records `actions_per_level`, and `sum(actions_per_level) == len(history)`
   holds for 25/25 games in war_eval_v1/v2/v3 (checked by `pull_io.load_pull`).
3. `verdict(...)` -- the pre-registerable decision rule that keeps a slowdown
   from being recorded as a mechanism negative:

     full Δlc PASS                                   -> PASS
     full Δlc FAIL, action deficit > 10%,
       matched-prefix Δlc PASS                       -> INCONCLUSIVE-ON-LATENCY
     full Δlc FAIL, matched-prefix Δlc also FAIL     -> FAIL (harm, K3' eligible)
     full Δlc FAIL, action deficit <= 10%            -> FAIL (harm, K3' eligible)

   INCONCLUSIVE-ON-LATENCY is explicitly NOT a K3' strike.

The gate line is K3' (`learnings/sweeps/gate_recalibration_2026-08-09.md:198-215`),
never the vacated −0.128.

Usage:
  uv run python duck_eval/p1_instr/latency_prefix.py                 # latency only
  uv run python duck_eval/p1_instr/latency_prefix.py ARM BASE [BASE2 BASE3 ...]
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pull_io import Pull, load_calls, load_pull  # noqa: E402

# t(0.95, df) one-sided, df = m-1
_T95 = {1: 6.3138, 2: 2.9200, 3: 2.3534, 4: 2.1318, 5: 2.0150, 6: 1.9432,
        7: 1.8946, 8: 1.8595, 9: 1.8331, 10: 1.8125}
K3P_FALLBACK_M1 = -0.200
K3P_FALLBACK_M3 = -0.190
LATENCY_DEFICIT_TRIGGER = 0.10   # >10% action deficit, systems review §MAJOR


# --------------------------------------------------------------------------
# 1. latency instrumentation
# --------------------------------------------------------------------------
@dataclass
class GameLatency:
    game_id: str
    n_actions: int
    total_wallclock_s: float
    hit_runtime_cap: bool
    s_per_action_mean: float
    s_per_action_median: float
    s_per_action_p90: float
    generated_tokens: int
    tokens_per_s: float
    n_tool_calls: int
    n_analysis_steps: int
    tool_calls_per_action: float
    levels_completed: int


def _p(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def latency_table(pull: Pull, *, with_transcripts: bool = True) -> list[GameLatency]:
    cap = float(pull.banner.get("max_runtime_s_per_game") or 0.0)
    calls = load_calls(pull) if with_transcripts else {}
    rows: list[GameLatency] = []
    for gid in sorted(pull.games):
        g = pull.games[gid]
        gaps = [b - a for a, b in zip([0.0] + g.cum_wallclock[:-1], g.cum_wallclock)]
        gaps = [x for x in gaps if x >= 0.0]
        toks = sum(g.generated_tokens)
        gcalls = calls.get(gid, [])
        rows.append(GameLatency(
            game_id=gid,
            n_actions=g.n_actions,
            total_wallclock_s=round(g.final_wallclock_seconds, 3),
            hit_runtime_cap=bool(cap and g.final_wallclock_seconds >= cap),
            s_per_action_mean=round(st.mean(gaps), 4) if gaps else 0.0,
            s_per_action_median=round(st.median(gaps), 4) if gaps else 0.0,
            s_per_action_p90=round(_p(gaps, 0.90), 4),
            generated_tokens=toks,
            tokens_per_s=round(toks / g.final_wallclock_seconds, 3) if g.final_wallclock_seconds else 0.0,
            n_tool_calls=len(gcalls),
            n_analysis_steps=len({c.analysis_step for c in gcalls}),
            tool_calls_per_action=round(len(gcalls) / g.n_actions, 4) if g.n_actions else 0.0,
            levels_completed=g.levels_completed,
        ))
    return rows


# --------------------------------------------------------------------------
# 2. matched-action-prefix endpoint
# --------------------------------------------------------------------------
@dataclass
class PrefixRow:
    game_id: str
    actions_arm: int
    actions_base: float
    k_matched: int
    lc_arm_full: int
    lc_base_full: float
    dlc_full: float
    lc_arm_at_k: float
    lc_base_at_k: float
    dlc_matched: float
    action_deficit: float          # (base - arm)/base ; >0 means arm did fewer
    s_per_action_arm: float
    s_per_action_base: float
    latency_ratio: float           # arm / base seconds-per-action


def matched_prefix(arm: Pull, bases: list[Pull]) -> list[PrefixRow]:
    """Per-game matched-action-prefix comparison of `arm` against m baselines.

    The baseline leg is the per-game MEAN over the m same-config baseline runs
    (K3' requires m >= 3). lc-at-prefix is averaged across the baseline runs at
    the arm-vs-that-run matched K, then re-averaged; K reported is the mean K.
    """
    rows: list[PrefixRow] = []
    common = sorted(set(arm.games) & set.intersection(*[set(b.games) for b in bases]))
    for gid in common:
        a = arm.games[gid]
        ks, lc_b_at_k, lc_a_at_k, base_actions, base_spa = [], [], [], [], []
        for b in bases:
            g = b.games[gid]
            k = min(a.n_actions, g.n_actions)
            ks.append(k)
            lc_b_at_k.append(g.lc_at_action_prefix(k))
            lc_a_at_k.append(a.lc_at_action_prefix(k))
            base_actions.append(g.n_actions)
            base_spa.append(g.final_wallclock_seconds / g.n_actions if g.n_actions else 0.0)
        mean_base_actions = st.mean(base_actions)
        arm_spa = a.final_wallclock_seconds / a.n_actions if a.n_actions else 0.0
        mean_base_spa = st.mean(base_spa)
        rows.append(PrefixRow(
            game_id=gid,
            actions_arm=a.n_actions,
            actions_base=round(mean_base_actions, 2),
            k_matched=int(round(st.mean(ks))),
            lc_arm_full=a.levels_completed,
            lc_base_full=round(st.mean(b.games[gid].levels_completed for b in bases), 4),
            dlc_full=round(a.levels_completed - st.mean(b.games[gid].levels_completed for b in bases), 4),
            lc_arm_at_k=round(st.mean(lc_a_at_k), 4),
            lc_base_at_k=round(st.mean(lc_b_at_k), 4),
            dlc_matched=round(st.mean(lc_a_at_k) - st.mean(lc_b_at_k), 4),
            action_deficit=round((mean_base_actions - a.n_actions) / mean_base_actions, 4)
            if mean_base_actions else 0.0,
            s_per_action_arm=round(arm_spa, 3),
            s_per_action_base=round(mean_base_spa, 3),
            latency_ratio=round(arm_spa / mean_base_spa, 4) if mean_base_spa else 0.0,
        ))
    return rows


# --------------------------------------------------------------------------
# 3. K3' line + verdict
# --------------------------------------------------------------------------
def k3prime_line(bases: list[Pull], *, n_games: int = 25) -> dict[str, float | int | str]:
    """K3' threshold: -t(0.95, df=m-1) * s_base * sqrt(1 + 1/m).

    s_base = sd over the m baseline runs of (run lc total / n_games).
    Spec: learnings/sweeps/gate_recalibration_2026-08-09.md:198-215.
    """
    m = len(bases)
    per_run = [b.total_lc / n_games for b in bases]
    if m == 1:
        return {"m": 1, "s_base": 0.0, "line": K3P_FALLBACK_M1, "rule": "fallback_m1"}
    s_base = st.stdev(per_run)
    t = _T95.get(m - 1, 1.8125)
    line = -t * s_base * math.sqrt(1.0 + 1.0 / m)
    out: dict[str, float | int | str] = {
        "m": m,
        "s_base": round(s_base, 6),
        "t95_df": t,
        "line": round(line, 6),
        "rule": "K3prime",
        "fallback_m3": K3P_FALLBACK_M3,
    }
    # R25 methodology N1: K3' at small m can be LOOSER than its own fallback.
    if line < K3P_FALLBACK_M3:
        out["r25_n1_flag"] = (
            "K3' line is looser than the m>=3 fallback (-0.190); R25 methodology N1 "
            "filed this as a type-II miscalibration. Report BOTH lines."
        )
    return out


def verdict(rows: list[PrefixRow], line: float) -> dict[str, object]:
    mean_full = st.mean(r.dlc_full for r in rows)
    mean_matched = st.mean(r.dlc_matched for r in rows)
    # Pooled deficit is the decision statistic: the per-game mean is dominated
    # by a handful of games where the arm ran 3x the actions (wa30 in every
    # pull), which is exactly the wall-clock-throughput artifact we are trying
    # to price. Pooled == the systems review's "3,638 vs 3,994 actions" framing.
    tot_arm = sum(r.actions_arm for r in rows)
    tot_base = sum(r.actions_base for r in rows)
    pooled_deficit = (tot_base - tot_arm) / tot_base if tot_base else 0.0
    mean_deficit = st.mean(r.action_deficit for r in rows)
    median_deficit = st.median(r.action_deficit for r in rows)
    pooled_latency_ratio = (
        (sum(r.s_per_action_arm * r.actions_arm for r in rows) / tot_arm)
        / (sum(r.s_per_action_base * r.actions_base for r in rows) / tot_base)
    ) if tot_arm and tot_base else 0.0
    full_pass = mean_full >= line
    matched_pass = mean_matched >= line
    if full_pass:
        v, k3 = "PASS", False
    elif pooled_deficit > LATENCY_DEFICIT_TRIGGER and matched_pass:
        v, k3 = "INCONCLUSIVE-ON-LATENCY", False
    else:
        v, k3 = "FAIL", True
    return {
        "line": round(line, 6),
        "mean_dlc_full": round(mean_full, 6),
        "mean_dlc_matched_prefix": round(mean_matched, 6),
        "pooled_action_deficit": round(pooled_deficit, 6),
        "mean_action_deficit": round(mean_deficit, 6),
        "median_action_deficit": round(median_deficit, 6),
        "pooled_latency_ratio": round(pooled_latency_ratio, 6),
        "total_actions_arm": tot_arm,
        "total_actions_base": round(tot_base, 1),
        "full_pass": full_pass,
        "matched_pass": matched_pass,
        "verdict": v,
        "counts_as_K3prime_strike": k3,
        "n_games": len(rows),
    }


# --------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    out_path = None
    if len(argv) > 2 and argv[1] == "--json":
        out_path = Path(argv[2])
        argv = [argv[0], *argv[3:]]
    if out_path is not None:
        arm = load_pull(argv[1])
        bases = [load_pull(n) for n in argv[2:]]
        rows = matched_prefix(arm, bases)
        gate = k3prime_line(bases)
        payload = {
            "arm": arm.name,
            "baselines": [b.name for b in bases],
            "banner": arm.banner,
            "k3prime": gate,
            "verdict": verdict(rows, float(gate["line"])),
            "latency_table_arm": [asdict(r) for r in latency_table(arm)],
            "matched_prefix_rows": [asdict(r) for r in rows],
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload["verdict"], indent=2))
        print(f"wrote {out_path}")
        return 0
    if len(argv) <= 1:
        pull = load_pull("war_eval_v1")
        rows = latency_table(pull)
        print(f"# latency instrumentation :: {pull.name} ({pull.label})")
        print(f"banner: {pull.banner}")
        capped = sum(1 for r in rows if r.hit_runtime_cap)
        print(f"games={len(rows)} hit_runtime_cap={capped}/{len(rows)}")
        print(f"{'game':16} {'act':>5} {'wall_s':>9} {'s/act':>7} {'p90':>8} "
              f"{'calls':>6} {'steps':>6} {'c/a':>6} {'lc':>3}")
        for r in rows:
            print(f"{r.game_id:16} {r.n_actions:5d} {r.total_wallclock_s:9.1f} "
                  f"{r.s_per_action_mean:7.2f} {r.s_per_action_p90:8.2f} "
                  f"{r.n_tool_calls:6d} {r.n_analysis_steps:6d} "
                  f"{r.tool_calls_per_action:6.2f} {r.levels_completed:3d}")
        tot_a = sum(r.n_actions for r in rows)
        tot_w = sum(r.total_wallclock_s for r in rows)
        print(f"TOTAL actions={tot_a} wallclock={tot_w:.1f}s "
              f"=> {tot_w / tot_a:.2f} s per scored action")
        return 0

    arm = load_pull(argv[1])
    bases = [load_pull(n) for n in argv[2:]] or [load_pull("war_eval_v1")]
    rows = matched_prefix(arm, bases)
    gate = k3prime_line(bases)
    res = verdict(rows, float(gate["line"]))
    print(f"# matched-action-prefix :: arm={arm.name} vs baselines={[b.name for b in bases]}")
    print(f"K3' gate: {json.dumps(gate)}")
    print(f"{'game':16} {'aA':>5} {'aB':>7} {'K':>5} {'dlc_full':>9} {'dlc_K':>8} "
          f"{'deficit':>8} {'lat_x':>7}")
    for r in rows:
        print(f"{r.game_id:16} {r.actions_arm:5d} {r.actions_base:7.1f} {r.k_matched:5d} "
              f"{r.dlc_full:9.3f} {r.dlc_matched:8.3f} {r.action_deficit:8.3f} "
              f"{r.latency_ratio:7.3f}")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
