"""War-v1 build-rail screen vs null10 (prereg 2026-07-14 gate statistics).

Scores runs/kernel_pulls/war_eval_v1/benchmark.json with the validated RHAE
scorer (phase1_gate.py, 0e+00 vs Tufa's 500 runs) and contrasts against the
null10 per-game means under the NEW primary statistic (paired delta
levels_completed, exact sign-flip) + secondary delta log1p(RHAE).

1-seed SCREEN, not a gate look (prereg section 2: 3-seed minimum for gates).

Usage: uv run python scripts/war_eval_screen.py [pull_name]
  pull_name defaults to war_eval_v1; e.g. war_eval_v2 scores seed 2 and
  writes to runs/war_eval_v2/.
Output: runs/<pull_name>/screen_report.md (+ raw JSON)
"""
from __future__ import annotations

import io
import json
import math
import statistics as st
import sys

# Windows console defaults to cp1252, which can't encode the report's Δ
# characters; the files are written utf-8 regardless, this guards the print.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from phase1_gate import load_null, load_our_seed, signflip_p_exact  # noqa: E402

NULL10 = ROOT / "runs" / "null10"
PULL_NAME = sys.argv[1] if len(sys.argv) > 1 else "war_eval_v1"
WAR = ROOT / "runs" / "kernel_pulls" / PULL_NAME
OUT_DIR = ROOT / "runs" / PULL_NAME


def main():
    null_games, max_err, n_checked, overall, _ = load_null(
        ROOT / "runs" / "tufa_example_run" / "benchmark.json",
        ROOT / "runs" / "tufa_example_run" / "score.json")
    assert max_err < 1e-9, f"scorer validation failed: {max_err}"

    seed_files = sorted(NULL10.glob("vanilla_seed*.json"))
    assert len(seed_files) == 10
    seeds = {sf.stem.replace("vanilla_", ""): load_our_seed(sf, null_games)
             for sf in seed_files}
    war = load_our_seed(WAR / "benchmark.json", null_games)

    prefixes = sorted(war)
    rows, d_lc, d_lg = [], [], []
    for p in prefixes:
        n_lc = st.mean(seeds[s][p]["lc"] for s in seeds if p in seeds[s])
        n_sc = st.mean(seeds[s][p]["score"] for s in seeds if p in seeds[s])
        dlc = war[p]["lc"] - n_lc
        dlg = math.log1p(war[p]["score"]) - math.log1p(n_sc)
        d_lc.append(dlc)
        d_lg.append(dlg)
        rows.append((p, war[p]["lc"], n_lc, dlc, war[p]["score"], n_sc, dlg,
                     ",".join(war[p]["flags"]) or "-"))

    n = len(d_lc)
    p_lc, _ = signflip_p_exact(d_lc, sum(d_lc))
    p_lg, _ = signflip_p_exact(d_lg, sum(d_lg))
    res = {
        "arm": f"war-v1 (kernel pull {PULL_NAME}, WARPACK_FORCE_OFFLINE_BENCH=1)",
        "seeds": 1,
        "n_games": n,
        "scorer_validation_max_err": max_err,
        "primary_dlc": {"mean": st.mean(d_lc), "sd_games": st.stdev(d_lc),
                        "signflip_p_exact": p_lc,
                        "wins": sum(d > 0 for d in d_lc),
                        "losses": sum(d < 0 for d in d_lc)},
        "secondary_dlog1p": {"mean": st.mean(d_lg), "sd_games": st.stdev(d_lg),
                             "signflip_p_exact": p_lg},
        "war_lc_total": sum(war[p]["lc"] for p in prefixes),
        "null_lc_total_mean": sum(
            st.mean(seeds[s][p]["lc"] for s in seeds if p in seeds[s])
            for p in prefixes),
        "war_rhae_mean": st.mean(war[p]["score"] for p in prefixes),
        "null_rhae_mean": st.mean(
            st.mean(seeds[s][p]["score"] for s in seeds if p in seeds[s])
            for p in prefixes),
        "per_game": [dict(zip(("game", "war_lc", "null_lc", "dlc",
                               "war_rhae", "null_rhae", "dlog1p", "flags"), r))
                     for r in rows],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "screen_raw.json").write_text(encoding="utf-8", data=json.dumps(res, indent=2))

    lines = [
        "# war-v1 build-rail SCREEN vs null10 — 1 seed (NOT a gate look)",
        "",
        f"Scorer validated: max err {max_err:.1e} over {n_checked} checks.",
        "",
        f"- **PRIMARY paired Δlc: mean {res['primary_dlc']['mean']:+.3f}**"
        f" (sd {res['primary_dlc']['sd_games']:.3f}, "
        f"{res['primary_dlc']['wins']}W/{res['primary_dlc']['losses']}L, "
        f"exact sign-flip p = {p_lc:.4f})",
        f"- Secondary Δlog1p(RHAE): mean {res['secondary_dlog1p']['mean']:+.3f}"
        f" (p = {p_lg:.4f})",
        f"- lc totals: war {res['war_lc_total']} vs null {res['null_lc_total_mean']:.1f}",
        f"- RHAE run-mean: war {res['war_rhae_mean']:.3f} vs null {res['null_rhae_mean']:.3f}",
        "",
        "| game | war lc | null lc | Δlc | war RHAE | null RHAE | Δlog1p | flags |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]:.2f} | {r[3]:+.2f} | "
                     f"{r[4]:.2f} | {r[5]:.2f} | {r[6]:+.2f} | {r[7]} |")
    (OUT_DIR / "screen_report.md").write_text(encoding="utf-8", data="\n".join(lines) + "\n")
    print("\n".join(lines[:12]))
    print(f"\nwritten: {OUT_DIR / 'screen_report.md'}")


if __name__ == "__main__":
    main()
