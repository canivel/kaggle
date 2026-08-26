"""boristown readiness-gate — non-harm SCREEN vs null10 (entry-gate #2).

Mirrors scripts/war_eval_screen.py (the sentinel used it verbatim:
runs/sentinel_eval_v1/screen_report.md). Same validated RHAE scorer
(phase1_gate.py, 0e+00 vs Tufa's 500 runs), same PRIMARY statistic (paired
delta levels_completed, exact sign-flip) + secondary delta log1p(RHAE) vs the
null10 per-game means.

ENTRY-GATE #2 CRITERION (intent boristown_ab_intent_2026-07-28.md §"Entry gates"
#2; prereg boristown_ab_prereg_2026-07-29_DRAFT.md BLOCKER 3): the gate passes
the non-harm screen iff BOTH:
  (a) the MECHANISM FIRES — the pulled eval-kernel log carries the gate's
      observed-firing telemetry ("A17-GATE observed-firing
      vllm_ready_latency_s=... : GATE fired", latency <= 180 s) AND boris's own
      "vLLM server ready" line; AND
  (b) Δ levels-completed NOT MATERIALLY NEGATIVE — same criterion the sentinel
      screen used: the gate is left-tail insurance run BEFORE bm.run, so a
      materially negative Δlc would mean the gate itself harmed the run. The
      sentinel's own screen (1 seed, NOT a gate look) reported Δlc = -0.128 and
      was still admitted to the draw because it was within noise (the gate here
      touches nothing the solver sees, so the honest prior is Δlc ~= 0). The
      "materially negative" bar is the sentinel precedent's: a paired-Δlc mean
      that is not significantly < 0 (one-sided exact sign-flip) AND no
      catastrophic per-game collapse. Threshold below is the DECLARED bar.

*** DO NOT RUN AGAINST LIVE DATA YET. *** The gate-eval kernels are unpushed
(the orchestrator holds the push slots). This script no-ops with a clear message
if the pull directory / benchmark.json is absent. Once the orchestrator has
pulled the eval-kernel outputs into runs/kernel_pulls/<pull_name>/ (benchmark.json
+ the kernel .log), run:

    uv run python scripts/gate_eval_screen.py gate_eval_v1
    uv run python scripts/gate_eval_screen.py gate_eval_v2

Writes runs/<pull_name>/screen_report.md (+ screen_raw.json) exactly like the
war/sentinel screen, plus a NON-HARM VERDICT block.
"""
from __future__ import annotations

import io
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from phase1_gate import load_null, load_our_seed, signflip_p_exact  # noqa: E402

NULL10 = ROOT / "runs" / "null10"
PULL_NAME = sys.argv[1] if len(sys.argv) > 1 else "gate_eval_v1"
PULL = ROOT / "runs" / "kernel_pulls" / PULL_NAME
OUT_DIR = ROOT / "runs" / PULL_NAME

# --- DECLARED non-harm bar (stated pre-run; sentinel-precedent criterion) -----
# (a) mechanism-fired: these greppable markers must appear in the pulled kernel log.
GATE_FIRED_RE = re.compile(
    r"A17-GATE observed-firing vllm_ready_latency_s=([0-9.]+)\s*:?\s*GATE fired")
GATE_ARMED_MARK = "A17-GATE"  # + "GATE armed"
BORIS_READY_MARK = "vLLM server ready"
LATENCY_CAP_S = 180.0
# (b) Δlc not materially negative: not significantly < 0 (one-sided exact
#     sign-flip p on the NEGATIVE tail) AND no game collapses by > 1 level below
#     null relative to the field. Sentinel precedent admitted Δlc = -0.128
#     (p=0.9495 on the positive tail, i.e. FAR from significant harm).
DLC_HARM_ALPHA = 0.05           # one-sided; harm asserted only if p_neg < alpha
DLC_CATASTROPHIC_PER_GAME = -1.0  # any single game losing > 1 lvl vs null flags review


def find_log(pull_dir: Path) -> Path | None:
    cands = sorted(pull_dir.glob("*.log"))
    # prefer the eval kernel's own log if present
    for c in cands:
        if "gate" in c.name.lower():
            return c
    return cands[0] if cands else None


def mechanism_check(pull_dir: Path) -> dict:
    log = find_log(pull_dir)
    if log is None:
        return {"log": None, "armed": False, "fired": False, "boris_ready": False,
                "latency_s": None, "latency_ok": False,
                "note": "no *.log in pull dir — cannot verify mechanism firing"}
    text = log.read_text(encoding="utf-8", errors="replace")
    armed = (GATE_ARMED_MARK in text) and ("GATE armed" in text)
    m = GATE_FIRED_RE.search(text)
    latency = float(m.group(1)) if m else None
    return {
        "log": log.name,
        "armed": armed,
        "fired": m is not None,
        "boris_ready": BORIS_READY_MARK in text,
        "latency_s": latency,
        "latency_ok": latency is not None and latency <= LATENCY_CAP_S,
    }


def main() -> int:
    if not (PULL / "benchmark.json").is_file():
        print(f"[STAGED-ONLY] no {PULL/'benchmark.json'} yet.")
        print("The gate-eval kernels are unpushed; this screen is prepared but NOT run.")
        print("Once the orchestrator pulls the eval outputs, re-run:")
        print(f"    uv run python scripts/gate_eval_screen.py {PULL_NAME}")
        return 0

    # --- (b) Δlc vs null10 (identical machinery to war/sentinel screen) --------
    null_games, max_err, n_checked, overall, _ = load_null(
        ROOT / "runs" / "tufa_example_run" / "benchmark.json",
        ROOT / "runs" / "tufa_example_run" / "score.json")
    assert max_err < 1e-9, f"scorer validation failed: {max_err}"

    seed_files = sorted(NULL10.glob("vanilla_seed*.json"))
    assert len(seed_files) == 10
    seeds = {sf.stem.replace("vanilla_", ""): load_our_seed(sf, null_games)
             for sf in seed_files}
    gate = load_our_seed(PULL / "benchmark.json", null_games)

    prefixes = sorted(gate)
    rows, d_lc, d_lg = [], [], []
    for p in prefixes:
        n_lc = st.mean(seeds[s][p]["lc"] for s in seeds if p in seeds[s])
        n_sc = st.mean(seeds[s][p]["score"] for s in seeds if p in seeds[s])
        dlc = gate[p]["lc"] - n_lc
        dlg = math.log1p(gate[p]["score"]) - math.log1p(n_sc)
        d_lc.append(dlc)
        d_lg.append(dlg)
        rows.append((p, gate[p]["lc"], n_lc, dlc, gate[p]["score"], n_sc, dlg,
                     ",".join(gate[p]["flags"]) or "-"))

    n = len(d_lc)
    # p on the POSITIVE tail (>= observed) as war/sentinel report it, plus the
    # NEGATIVE-tail p (harm test): exact sign-flip of the negated deltas.
    p_lc_pos, _ = signflip_p_exact(d_lc, sum(d_lc))
    p_lc_neg, _ = signflip_p_exact([-d for d in d_lc], -sum(d_lc))
    p_lg_pos, _ = signflip_p_exact(d_lg, sum(d_lg))

    mech = mechanism_check(PULL)
    worst_game = min(rows, key=lambda r: r[3]) if rows else None
    catastrophic = worst_game is not None and worst_game[3] < DLC_CATASTROPHIC_PER_GAME
    dlc_harmful = p_lc_neg < DLC_HARM_ALPHA
    mechanism_ok = mech["armed"] and mech["fired"] and mech["boris_ready"] and mech["latency_ok"]
    nonharm_ok = mechanism_ok and (not dlc_harmful) and (not catastrophic)

    res = {
        "arm": f"boristown readiness-gate (kernel pull {PULL_NAME}, offline eval build)",
        "seeds": 1,
        "n_games": n,
        "scorer_validation_max_err": max_err,
        "mechanism": mech,
        "primary_dlc": {"mean": st.mean(d_lc), "sd_games": st.stdev(d_lc),
                        "signflip_p_pos": p_lc_pos, "signflip_p_neg": p_lc_neg,
                        "wins": sum(d > 0 for d in d_lc), "losses": sum(d < 0 for d in d_lc)},
        "secondary_dlog1p": {"mean": st.mean(d_lg), "signflip_p_pos": p_lg_pos},
        "worst_game": {"game": worst_game[0], "dlc": worst_game[3]} if worst_game else None,
        "nonharm_verdict": {
            "mechanism_fired": mechanism_ok,
            "dlc_not_materially_negative": (not dlc_harmful) and (not catastrophic),
            "PASS": nonharm_ok,
            "criterion": ("mechanism fires (armed+fired+boris-ready, latency<=180s) "
                          "AND Δlc not significantly<0 (one-sided sign-flip p_neg>=0.05) "
                          "AND no game collapses >1 lvl vs null"),
        },
        "per_game": [dict(zip(("game", "gate_lc", "null_lc", "dlc",
                               "gate_rhae", "null_rhae", "dlog1p", "flags"), r))
                     for r in rows],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "screen_raw.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    lines = [
        f"# boristown readiness-gate — non-harm SCREEN vs null10 ({PULL_NAME})",
        "",
        f"Scorer validated: max err {max_err:.1e} over {n_checked} checks.",
        "",
        "## Entry-gate #2 non-harm verdict",
        f"- **NON-HARM: {'PASS' if nonharm_ok else 'FAIL'}**",
        f"- mechanism fired: {mechanism_ok} "
        f"(armed={mech['armed']}, fired={mech['fired']}, boris_ready={mech['boris_ready']}, "
        f"latency_s={mech['latency_s']} ok={mech['latency_ok']}, log={mech['log']})",
        f"- Δlc not materially negative: {(not dlc_harmful) and (not catastrophic)} "
        f"(harm-tail p_neg={p_lc_neg:.4f} vs α={DLC_HARM_ALPHA}; "
        f"worst game {res['worst_game']} vs cap {DLC_CATASTROPHIC_PER_GAME})",
        "",
        f"- PRIMARY paired Δlc: mean {res['primary_dlc']['mean']:+.3f} "
        f"(sd {res['primary_dlc']['sd_games']:.3f}, "
        f"{res['primary_dlc']['wins']}W/{res['primary_dlc']['losses']}L, "
        f"pos-tail p={p_lc_pos:.4f}, harm-tail p={p_lc_neg:.4f})",
        f"- Secondary Δlog1p(RHAE): mean {res['secondary_dlog1p']['mean']:+.3f} (p={p_lg_pos:.4f})",
        "",
        "| game | gate lc | null lc | Δlc | gate RHAE | null RHAE | Δlog1p | flags |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]:.2f} | {r[3]:+.2f} | "
                     f"{r[4]:.2f} | {r[5]:.2f} | {r[6]:+.2f} | {r[7]} |")
    (OUT_DIR / "screen_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:14]))
    print(f"\nwritten: {OUT_DIR / 'screen_report.md'}")
    return 0


if __name__ == "__main__":
    main()
