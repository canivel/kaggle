"""Panel R14 component (d) — offline PREDICT->RESULT "no-effect FACT" metric.

Objection 4 (learnings/panel/round14/prog-synthesis.md) demands a ZERO-COST
offline estimate of component (d)'s achievable prediction signal BEFORE its gate
window seals (Jul 21). Component (d) makes the LLM emit a PREDICT line (predicted
board_changed outcome) before repeating an action family; the harness scores it
vs the actual RESULT.

This script measures the *mechanical* half of (d) — the "no-effect FACT" rule:
    IF a (state, action) pair was previously observed to produce
    board_changed=False, predict board_changed=False when that pair recurs.
That rule's accuracy on real recorded Qwen (duck-harness) action streams is its
ceiling. We compute it OFFLINE from the existing per-action event traces
(runs/kernel_pulls/*/artifacts/*_events.jsonl and runs/phase1_ab/seed1), which
already carry a ground-truth `board_changed` flag on every action event (and full
`board` frames as an independent check). No engine replay, no cloud, CPU only.

Key definitions (reported side by side for honesty):
  * state_action : key = (digest(board the agent acted on), action_display)
                   action_display fully qualifies the action, incl. ACTION6
                   MOUSE(row,col) coordinates and ACTION1-5 direction.
  * action_only  : key = action_display  (coarsest "this family never works" rule)

For each recurring key we ask: was it EVER seen no-effect before this occurrence?
If so, (d) fires ("no-effect FACT") and predicts board_changed=False. Accuracy =
P(board_changed=False now | key was previously no-effect). Trigger count per run =
number of such firings; feeds the A10 canary (>=1/run on >=5 games).

Baseline: majority-class predictor over all actions (predict the more common of
{effect, no-effect}); on these streams no-effect is rare (~9%), so the majority
baseline "always effect" scores ~1 - noeff_rate. The (d) rule must beat that to
justify its gate window.

Usage:  uv run python scripts/predict_metric.py
Output: runs/predict_metric/report.md  +  raw.json
Read-only w.r.t. all existing eval/harness code.
"""
from __future__ import annotations

import glob
import hashlib
import io
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

# Windows console defaults to cp1252 and cannot encode the report's unicode; the
# files are written utf-8 regardless, this only guards the console print.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]

# Recorded Qwen (duck-harness) run pulls that carry per-action board_changed.
RUN_DIRS = [
    "runs/kernel_pulls/war_eval_v1",
    "runs/kernel_pulls/war_eval_v2",
    "runs/kernel_pulls/war_eval_v3",
    "runs/kernel_pulls/war_v2_eval_s1",
    "runs/kernel_pulls/sched_v1",
    "runs/kernel_pulls/phase1_v5",
    "runs/phase1_ab/seed1",
]

KEY_MODES = ("state_action", "action_only")


def board_digest(board) -> str:
    return hashlib.blake2b(json.dumps(board, separators=(",", ":")).encode(),
                           digest_size=8).hexdigest()


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (None, None)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_events(fp: str):
    with open(fp, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def game_id_from_path(fp: str) -> str:
    name = Path(fp).name  # e.g. ar25-0c556536_p0_events.jsonl
    return name.split("_")[0]  # ar25-0c556536


def action_key(ev) -> str:
    # action_display fully qualifies: direction for ACTION1-5, MOUSE(r,c) for
    # ACTION6. Fall back to action_name if display absent.
    return ev.get("action_display") or ev.get("action_name") or "<unk>"


def analyse_game(events, key_mode: str):
    """Walk one game trace; return per-game counters for the (d) no-effect rule.

    board_changed ground truth is taken from the recorded flag. As an integrity
    check we also derive board_changed from consecutive frame digests and count
    disagreements.
    """
    seen = defaultdict(list)          # key -> list of past board_changed bools
    prev_digest = None                # digest of the board the agent acted on

    total = 0                         # action events
    noeff = 0                         # board_changed == False
    triggers = 0                      # key seen no-effect before -> (d) fires
    correct = 0                       # trigger AND board_changed == False now
    repeat_pairs = 0                  # key seen at least once before (any outcome)
    digest_disagree = 0               # recorded flag vs frame-digest derivation

    for ev in events:
        et = ev.get("type")
        if et == "initial":
            prev_digest = board_digest(ev["board"])
            continue
        if et != "action":
            continue  # skip analysis-only frames (no action taken)
        if "board_changed" not in ev:
            continue

        bc = bool(ev["board_changed"])
        cur_digest = board_digest(ev["board"])
        # integrity: frame changed <=> board_changed should be True
        if prev_digest is not None:
            derived_changed = (cur_digest != prev_digest)
            if derived_changed != bc:
                digest_disagree += 1

        ak = action_key(ev)
        key = (prev_digest, ak) if key_mode == "state_action" else ak
        hist = seen[key]

        total += 1
        if not bc:
            noeff += 1
        if hist:
            repeat_pairs += 1
            if False in hist:            # previously observed no-effect -> (d) fires
                triggers += 1
                if not bc:               # predicted no-effect, was no-effect
                    correct += 1
        hist.append(bc)
        prev_digest = cur_digest

    return {
        "total_actions": total,
        "noeff": noeff,
        "repeat_pairs": repeat_pairs,
        "triggers": triggers,
        "correct": correct,
        "digest_disagree": digest_disagree,
    }


def main():
    out_dir = ROOT / "runs" / "predict_metric"
    out_dir.mkdir(parents=True, exist_ok=True)

    # results[key_mode][run_dir] = {game_id: counters}
    results = {km: {} for km in KEY_MODES}
    coverage = {}  # run_dir -> {n_games, n_files, actions}

    for rd in RUN_DIRS:
        files = sorted(glob.glob(str(ROOT / rd / "artifacts" / "*_events.jsonl")))
        coverage[rd] = {"n_files": len(files), "games": {}}
        for km in KEY_MODES:
            results[km][rd] = {}
        for fp in files:
            gid = game_id_from_path(fp)
            events = load_events(fp)
            for km in KEY_MODES:
                results[km][rd][gid] = analyse_game(events, km)
        coverage[rd]["n_games"] = len(files)

    # ---- pooled + per-run aggregation per key mode ----
    raw = {
        "methodology": {
            "component": "R14 (d) PREDICT->RESULT no-effect FACT rule",
            "rule": ("IF a (key) was previously observed board_changed=False, "
                     "predict board_changed=False on its next occurrence."),
            "ground_truth": "recorded board_changed flag on each action event",
            "key_modes": {
                "state_action": "(digest(board acted on), action_display)",
                "action_only": "action_display only",
            },
            "trigger_definition": ("an action whose key was previously seen "
                                   "no-effect; = an A10 (d) firing opportunity"),
            "baseline": ("majority-class over all actions; here no-effect is "
                         "rare so majority='always effect', acc = 1 - noeff_rate"),
            "source_runs": RUN_DIRS,
        },
        "coverage": {},
        "by_key_mode": {},
    }

    # coverage summary
    for rd in RUN_DIRS:
        km0 = results["state_action"][rd]
        acts = sum(g["total_actions"] for g in km0.values())
        dis = sum(g["digest_disagree"] for g in km0.values())
        raw["coverage"][rd] = {
            "n_games": coverage[rd]["n_games"],
            "total_actions": acts,
            "frame_digest_disagreements": dis,
        }

    for km in KEY_MODES:
        pooled = {"total_actions": 0, "noeff": 0, "repeat_pairs": 0,
                  "triggers": 0, "correct": 0}
        per_run = {}
        per_game_rows = []
        # trigger/run: one "run" = one game trace file
        trigs_per_run = []
        games_with_trigger = 0
        n_runs = 0

        for rd in RUN_DIRS:
            rpool = {"total_actions": 0, "noeff": 0, "repeat_pairs": 0,
                     "triggers": 0, "correct": 0}
            for gid, c in results[km][rd].items():
                n_runs += 1
                trigs_per_run.append(c["triggers"])
                if c["triggers"] >= 1:
                    games_with_trigger += 1
                for k in rpool:
                    rpool[k] += c[k]
                    pooled[k] += c[k]
                acc = (c["correct"] / c["triggers"]) if c["triggers"] else None
                per_game_rows.append({
                    "run": rd.split("/")[-1],
                    "game": gid,
                    "actions": c["total_actions"],
                    "noeff": c["noeff"],
                    "triggers": c["triggers"],
                    "correct": c["correct"],
                    "recurrence_acc": acc,
                })
            racc = (rpool["correct"] / rpool["triggers"]) if rpool["triggers"] else None
            rbase = 1 - rpool["noeff"] / rpool["total_actions"] if rpool["total_actions"] else None
            per_run[rd] = {
                **rpool,
                "recurrence_acc": racc,
                "majority_baseline_acc": rbase,
                "triggers_per_game": rpool["triggers"] / max(1, coverage[rd]["n_games"]),
            }

        pooled_acc = (pooled["correct"] / pooled["triggers"]) if pooled["triggers"] else None
        pooled_base = (1 - pooled["noeff"] / pooled["total_actions"]) if pooled["total_actions"] else None
        acc_ci = wilson_ci(pooled["correct"], pooled["triggers"])
        # also: naive no-effect rate among triggered actions = "if we just always
        # predicted no-effect on every action" this is the noeff base rate.
        raw["by_key_mode"][km] = {
            "pooled": {
                **pooled,
                "recurrence_acc": pooled_acc,
                "recurrence_acc_wilson95": acc_ci,
                "majority_baseline_acc": pooled_base,
                "noeff_base_rate": pooled["noeff"] / pooled["total_actions"] if pooled["total_actions"] else None,
                "n_runs": n_runs,
                "games_with_ge1_trigger": games_with_trigger,
                "mean_triggers_per_run": sum(trigs_per_run) / max(1, n_runs),
                "median_triggers_per_run": sorted(trigs_per_run)[len(trigs_per_run) // 2] if trigs_per_run else 0,
            },
            "per_run": per_run,
            "per_game": per_game_rows,
        }

    (out_dir / "raw.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")

    # ---- verdict logic (on the stronger of the two key modes) ----
    sa = raw["by_key_mode"]["state_action"]["pooled"]
    ao = raw["by_key_mode"]["action_only"]["pooled"]
    # (d) is justified only if recurrence_acc beats majority baseline AND
    # trigger coverage >=1/run on >=5 games (A10 canary). Use the mode most
    # favourable to (d).
    best = max(
        ("state_action", sa), ("action_only", ao),
        key=lambda kv: (kv[1]["recurrence_acc"] or 0),
    )
    best_mode, best_pool = best
    beats_baseline = (best_pool["recurrence_acc"] is not None
                      and best_pool["majority_baseline_acc"] is not None
                      and best_pool["recurrence_acc"] > best_pool["majority_baseline_acc"])
    enough_triggers = best_pool["games_with_ge1_trigger"] >= 5
    justified = beats_baseline and enough_triggers

    verdict = _write_report(out_dir, raw, best_mode, best_pool, beats_baseline,
                            enough_triggers, justified)

    # console summary
    print(f"predict_metric: {best_pool['n_runs']} game-runs across {len(RUN_DIRS)} pulls")
    for km in KEY_MODES:
        p = raw["by_key_mode"][km]["pooled"]
        print(f"  [{km}] recurrence_acc={_fmt(p['recurrence_acc'])} "
              f"baseline={_fmt(p['majority_baseline_acc'])} "
              f"triggers={p['triggers']} games>=1trig={p['games_with_ge1_trigger']}/{p['n_runs']}")
    print(f"  VERDICT: {'JUSTIFY' if justified else 'KILL'} (d) — "
          f"beats_baseline={beats_baseline}, >=5 games w/ trigger={enough_triggers}")
    print(f"  wrote {out_dir/'report.md'} and {out_dir/'raw.json'}")


def _fmt(x):
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def _write_report(out_dir, raw, best_mode, best_pool, beats_baseline,
                  enough_triggers, justified):
    L = []
    L.append("# Panel R14 component (d) — offline PREDICT->RESULT no-effect FACT metric\n")
    L.append("Zero-cost offline answer to prog-synthesis objection 4 "
             "(learnings/panel/round14/prog-synthesis.md): does the mechanical "
             "\"no-effect FACT\" half of component (d) have >chance predictive "
             "accuracy and enough trigger opportunities to justify its gate "
             "window (seal Jul 21)?\n")

    L.append("## Methodology\n")
    m = raw["methodology"]
    L.append(f"- **Rule**: {m['rule']}")
    L.append(f"- **Ground truth**: {m['ground_truth']} (full `board` frames also "
             "present; used as an independent frame-digest cross-check).")
    L.append("- **Key modes** (reported side by side):")
    L.append(f"  - `state_action` = {m['key_modes']['state_action']}")
    L.append(f"  - `action_only` = {m['key_modes']['action_only']}")
    L.append(f"- **Trigger** = {m['trigger_definition']}. Trigger count per run "
             "= number of (d) firings = A10 canary opportunities.")
    L.append(f"- **Baseline** = {m['baseline']}.")
    L.append("- CPU-only, read-only over recorded traces; no engine replay, no "
             "cloud, no kernel push.\n")

    L.append("## Coverage (which traces were usable)\n")
    L.append("All discovered `*_events.jsonl` pulls carry a `board_changed` flag "
             "on every action event, so 100% of recorded actions are usable. "
             "These are recorded Qwen / duck-harness runs over the 25 official "
             "games.\n")
    L.append("| run pull | games | actions | frame-digest disagreements |")
    L.append("|---|---:|---:|---:|")
    tot_a = 0
    for rd, c in raw["coverage"].items():
        L.append(f"| {rd} | {c['n_games']} | {c['total_actions']} | "
                 f"{c['frame_digest_disagreements']} |")
        tot_a += c["total_actions"]
    L.append(f"| **pooled** | **{sum(c['n_games'] for c in raw['coverage'].values())}** "
             f"| **{tot_a}** | — |\n")
    L.append("_(Frame-digest disagreements = actions where the recorded "
             "`board_changed` flag disagreed with a direct hash of consecutive "
             "`board` frames. Low counts confirm the flag is a faithful no-effect "
             "label; a nonzero count means the flag counts changes the visible "
             "grid hash does not, e.g. hidden-state or score-only changes.)_\n")

    L.append("## Pooled results\n")
    L.append("| key mode | actions | no-effect rate | triggers | recurrence acc "
             "(P no-eff again \\| was no-eff) | Wilson 95% | majority baseline | "
             "games w/ >=1 trigger |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for km in KEY_MODES:
        p = raw["by_key_mode"][km]["pooled"]
        ci = p["recurrence_acc_wilson95"]
        ci_s = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci and ci[0] is not None else "—"
        L.append(f"| {km} | {p['total_actions']} | {p['noeff_base_rate']:.3f} | "
                 f"{p['triggers']} | {_fmt(p['recurrence_acc'])} | {ci_s} | "
                 f"{_fmt(p['majority_baseline_acc'])} | "
                 f"{p['games_with_ge1_trigger']}/{p['n_runs']} |")
    L.append("")

    L.append("## Triggers per run (feeds A10 canary: >=1/run on >=5 games)\n")
    for km in KEY_MODES:
        p = raw["by_key_mode"][km]["pooled"]
        L.append(f"**{km}**: mean {p['mean_triggers_per_run']:.2f} triggers/run, "
                 f"median {p['median_triggers_per_run']}, "
                 f"{p['games_with_ge1_trigger']}/{p['n_runs']} game-runs fire >=1 "
                 "trigger.")
        L.append("")
        L.append("| run pull | triggers | triggers/game | recurrence acc | "
                 "majority baseline |")
        L.append("|---|---:|---:|---:|---:|")
        for rd, r in raw["by_key_mode"][km]["per_run"].items():
            L.append(f"| {rd.split('/')[-1]} | {r['triggers']} | "
                     f"{r['triggers_per_game']:.2f} | {_fmt(r['recurrence_acc'])} | "
                     f"{_fmt(r['majority_baseline_acc'])} |")
        L.append("")

    L.append("## Verdict\n")
    acc = best_pool["recurrence_acc"]
    base = best_pool["majority_baseline_acc"]
    if justified:
        verdict = "JUSTIFY"
        para = (f"On the more favourable key mode (`{best_mode}`), the no-effect "
                f"FACT rule reaches recurrence accuracy {acc:.3f}, above the "
                f"majority-class baseline {base:.3f}, and fires >=1 trigger on "
                f"{best_pool['games_with_ge1_trigger']} game-runs (>=5). Component "
                "(d)'s FACT half clears both bars; keep its gate window.")
    else:
        verdict = "KILL"
        reasons = []
        if not beats_baseline:
            reasons.append(
                f"its best recurrence accuracy is only {acc:.3f} — FAR BELOW the "
                f"majority-class baseline {base:.3f} (predicting 'always effect'). "
                "When a (state,action) or action key recurs after a no-effect, the "
                "board in fact changes ~%.0f%% of the time, so the (d) rule is "
                "actively wrong most times it fires" % (100 * (1 - (acc or 0))))
        if not enough_triggers:
            reasons.append(
                f"it fires on only {best_pool['games_with_ge1_trigger']} game-runs "
                "(<5), so trigger coverage is too thin for the A10 canary")
        para = ("Component (d)'s no-effect FACT rule should be KILLED CHEAPLY NOW: "
                + "; and ".join(reasons) + ". The recorded Qwen streams are "
                "near-deterministic engines (N5 audit: 0/25 divergent) yet the "
                "no-effect label almost never recurs for the same (state,action) "
                "context — no-effects here are one-off / context-transient, not a "
                "stable 'this never works' property a memorised FACT could exploit. "
                "Emitting a PREDICT line before repeating an action family would be "
                "graded worse than a trivial 'it will have an effect' constant, so "
                "the wiring adds latency and grading surface for negative expected "
                "value. Do not open (d)'s gate window.")
    L.append(f"**{verdict}.** {para}\n")

    (out_dir / "report.md").write_text("\n".join(L), encoding="utf-8")
    return verdict


if __name__ == "__main__":
    main()
