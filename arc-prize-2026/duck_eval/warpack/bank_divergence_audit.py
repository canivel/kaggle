"""Banking replay divergence audit across ALL winning eval runs (panel R12 N5).

R11's engineered validation (bank_fire_validation.py) tested 4 named games and
found frame_divergence aborts on sc25/m0r0 — including the flagship warpack win
(sc25, +1.8 lc both seeds). Panel R12 llm-agents [N5]: before any R2 window is
licensed, measure the divergence fraction over the whole panel, because if most
games randomize per play, bank_strict replay degenerates to a no-op and the R2
reach-table row is unfounded.

Coverage: every (game, seed) pair with levels_completed >= 1 in the war-eval
seed-1 or seed-2 benchmarks (banking is structurally unreachable at lc=0 — a
game with no win has no trace to replay; those games are reported as N/A, not
counted in the divergence denominator).

Run from repo root:
    .venv/Scripts/python.exe duck_eval/warpack/bank_divergence_audit.py
Output: runs/war_eval_v1/bank_divergence_audit.json (+ .md summary table)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from bank_fire_validation import run_case  # noqa: E402  (pulls smoke_test env)

BENCHES = {
    "seed1": REPO / "runs" / "kernel_pulls" / "war_eval_v1" / "benchmark.json",
    "seed2": REPO / "runs" / "kernel_pulls" / "war_eval_v2" / "benchmark.json",
}
OUT_JSON = REPO / "runs" / "war_eval_v1" / "bank_divergence_audit.json"
OUT_MD = REPO / "runs" / "war_eval_v1" / "bank_divergence_audit.md"


def classify(case: dict) -> str:
    if case["replay_succeeded"]:
        return "replayed"
    aborts = [e for e in case["events"] if str(e[0]) == "bank_abort"]
    if aborts:
        return str(aborts[-1][1])  # last abort reason, e.g. frame_divergence
    skips = [e for e in case["events"] if str(e[0]) == "bank_skip"]
    if skips:
        return f"skip:{skips[-1][1]}"
    return "no_bank_event"


def main() -> int:
    import warpack_patch

    cfg = warpack_patch.apply()
    assert cfg.enable and cfg.enable_banking

    cases = []
    for seed, path in BENCHES.items():
        bench = json.loads(path.read_text(encoding="utf-8"))
        for r in bench["game_runs"]:
            prefix = r["game_id"][:4]
            lc = int(r["levels_completed"])
            if lc < 1:
                continue
            print(f"== {seed} {prefix}: {len(r['history'])} actions, lc {lc} ==",
                  flush=True)
            case = run_case(prefix, r["history"], lc)
            case["seed"] = seed
            case["outcome"] = classify(case)
            print(f"   outcome={case['outcome']} events={case['events']}", flush=True)
            cases.append(case)

    n = len(cases)
    replayed = [c for c in cases if c["outcome"] == "replayed"]
    diverged = [c for c in cases if "divergence" in c["outcome"]]
    other = [c for c in cases if c not in replayed and c not in diverged]
    games_replayed = sorted({c["game"] for c in replayed})
    games_diverged = sorted({c["game"] for c in diverged})

    summary = {
        "n_winning_cases": n,
        "replayed": len(replayed),
        "diverged": len(diverged),
        "other": [{ "game": c["game"], "seed": c["seed"], "outcome": c["outcome"]}
                  for c in other],
        "divergence_fraction": round(len(diverged) / n, 3) if n else None,
        "games_replayable": games_replayed,
        "games_divergent": games_diverged,
        "all_replays_verbatim_invariant": all(
            c.get("replay_verbatim") and c.get("score_invariant") for c in replayed),
    }
    OUT_JSON.write_text(json.dumps({"summary": summary, "cases": cases}, indent=2),
                        encoding="utf-8")

    lines = [
        "# Banking replay divergence audit — all winning (game, seed) cases (R12 N5)",
        "",
        f"- winning cases tested: {n} (union of war-eval seeds 1-2, lc >= 1)",
        f"- replayed verbatim+invariant: {len(replayed)} "
        f"({', '.join(games_replayed) or '-'})",
        f"- diverged (strict-frame/lc abort): {len(diverged)} "
        f"({', '.join(games_divergent := games_diverged) or '-'})",
        f"- **divergence fraction: {summary['divergence_fraction']}**",
        "",
        "| seed | game | lc | outcome |",
        "|---|---|---|---|",
    ]
    for c in cases:
        lines.append(f"| {c['seed']} | {c['game']} | {c['target_lc']} | {c['outcome']} |")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"written: {OUT_JSON}\n         {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
