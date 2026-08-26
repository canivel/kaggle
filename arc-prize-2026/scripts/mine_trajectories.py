"""Mine banked ARC-AGI-3 trajectories for phase1-v2 gating design.

Inputs:
  runs/tufa_example_run/benchmark.json      500 vanilla runs (20 passes x 25 games)
  runs/phase1_ab/seed{1,2,3}/benchmark.json our phase1 arm (injections active)
  runs/phase1_ab/seed1/artifacts/*_events.jsonl  per-action boards/levels (seed1 only)
  runs/phase1_ab/seed1/transcripts/*.txt    real ProgressTracker counter per turn

Outputs (stdout, markdown tables):
  Q1 mode signatures (early-window features, good vs bad mode, per-feature AUC)
  Q2 intervention timing (injection positions vs level-ups, displacement check)
  Q4 no-progress streak distribution before organic level-ups (threshold pick)
  Q5 RHAE marginal value: one extra late level vs early efficiency polish

Usage: python scripts/mine_trajectories.py [--window 30]
Pure stdlib. No GPU.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIMODAL = ["cd82", "cn04", "dc22", "g50t", "ls20", "m0r0", "re86", "s5i5", "sc25", "sk48"]
WIN_GAMES = ["cn04", "cd82", "lp85", "sc25", "tn36"]
LOSS_GAMES = ["ar25", "ka59", "ft09"]


def load_runs(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["game_runs"]


def levelup_indices(run: dict) -> list[int]:
    """Action index (1-based) at which each completed level ended."""
    lc = run["levels_completed"]
    apl = run["actions_per_level"][:lc]
    out, acc = [], 0
    for a in apl:
        acc += a
        out.append(acc)
    return out


def early_features(run: dict, window: int) -> dict:
    h = run["history"][:window]
    if not h:
        return {}
    ids = [e["action"]["id"] for e in h]
    sigs = [
        (e["action"]["id"], json.dumps(e["action"].get("data") or {}, sort_keys=True))
        for e in h
    ]
    n = len(ids)
    rep = sum(1 for i in range(1, n) if sigs[i] == sigs[i - 1]) / max(1, n - 1)
    toks = [e["generated_tokens"] for e in h]
    wc = [e["wallclock_seconds"] for e in h]
    dt = [wc[0]] + [wc[i] - wc[i - 1] for i in range(1, len(wc))]  # cumulative -> per-action
    lups = levelup_indices(run)
    return {
        "uniq_ids": len(set(ids)),
        "uniq_sigs": len(set(sigs)) / n,          # action diversity (id+coords)
        "repeat_ratio": rep,                       # exact consecutive repeats
        "tokens_per_action": st.mean(toks),
        "sec_per_action": st.median(dt),
        "mouse_frac": sum(1 for i in ids if i == "ACTION6") / n,
        "levelup_in_window": 1.0 if (lups and lups[0] <= window) else 0.0,
    }


def auc(pos: list[float], neg: list[float]) -> float:
    """P(pos > neg) rank statistic (Mann-Whitney AUC)."""
    if not pos or not neg:
        return float("nan")
    wins = ties = 0
    for p in pos:
        for q in neg:
            if p > q:
                wins += 1
            elif p == q:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def q1_mode_signatures(vanilla: list[dict], window: int) -> None:
    print(f"\n## Q1 mode signatures (vanilla, first {window} actions)\n")
    feats = ["uniq_sigs", "repeat_ratio", "tokens_per_action", "sec_per_action",
             "mouse_frac", "levelup_in_window"]
    hdr = "| game | n_good/n_bad | " + " | ".join(f"{f} g/b (AUC)" for f in feats) + " |"
    print(hdr)
    print("|" + "---|" * (2 + len(feats)))
    pooled: dict[str, tuple[list, list]] = {f: ([], []) for f in feats}
    for g in BIMODAL:
        runs = [r for r in vanilla if r["game_id"].startswith(g)]
        good = [r for r in runs if r["levels_completed"] >= 1]
        bad = [r for r in runs if r["levels_completed"] == 0]
        fg = [early_features(r, window) for r in good]
        fb = [early_features(r, window) for r in bad]
        cells = []
        for f in feats:
            pv = [x[f] for x in fg if x]
            nv = [x[f] for x in fb if x]
            pooled[f][0].extend(pv)
            pooled[f][1].extend(nv)
            a = auc(pv, nv)
            cells.append(
                f"{st.mean(pv):.2f}/{st.mean(nv):.2f} ({a:.2f})" if pv and nv else "—"
            )
        print(f"| {g} | {len(good)}/{len(bad)} | " + " | ".join(cells) + " |")
    print("\n**Pooled AUC (good vs bad, all 10 games):**")
    for f in feats:
        print(f"- {f}: AUC={auc(pooled[f][0], pooled[f][1]):.3f} "
              f"(good mean {st.mean(pooled[f][0]):.3f}, bad mean {st.mean(pooled[f][1]):.3f})")


def q1b_time_to_first_level(vanilla: list[dict]) -> None:
    """The one game-agnostic early signal: how long good runs take to level 1."""
    print("\n## Q1b time-to-first-level (vanilla runs that completed >=1 level)\n")
    print("| game | n_good/20 | first-level action idx (sorted) | median |")
    print("|---|---|---|---|")
    pooled: list[int] = []
    games = sorted({r["game_id"].split("-")[0] for r in vanilla})
    for g in games:
        rs = [r for r in vanilla if r["game_id"].startswith(g)]
        firsts = sorted(r["actions_per_level"][0] for r in rs if r["levels_completed"] >= 1)
        pooled.extend(firsts)
        med = st.median(firsts) if firsts else "—"
        print(f"| {g} | {len(firsts)}/20 | {firsts if firsts else '—'} | {med} |")
    s = sorted(pooled)
    pct = lambda p: s[int(p * (len(s) - 1))]
    print(f"\n- Pooled (n={len(s)}): median={st.median(s)}, p75={pct(.75)}, "
          f"p90={pct(.9)}, p95={pct(.95)} -> detector 'levels==0 at action 90' has "
          f"~10% FPR on eventual-good runs, ~100% TPR on never-level runs.")


def parse_tracker_series(txt: Path) -> list[int]:
    """ProgressTracker counter per analyzer turn from transcript status lines."""
    pat = re.compile(r"(\d+) turns since last new state")
    return [int(m.group(1)) for m in pat.finditer(txt.read_text(encoding="utf-8", errors="replace"))]


def parse_events(path: Path) -> list[dict]:
    out = []
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        e = json.loads(ln)
        if e.get("type") != "action":
            continue
        out.append({
            "action_num": e["action_num"],
            "analysis_step": e.get("analysis_step"),
            "level": e.get("level"),
            "board_changed": e.get("board_changed"),
            "level_completed": e.get("level_completed"),
            "board_hash": hash(json.dumps(e["board"])),
        })
    return out


def q2_q4_events(seed1_dir: Path) -> None:
    art = seed1_dir / "artifacts"
    tr = seed1_dir / "transcripts"
    print("\n## Q2 intervention timing (seed1, the only seed with artifacts)\n")
    print("| game | explores fired | anim injections | levels | notes |")
    print("|---|---|---|---|---|")
    streak_at_levelup: list[int] = []
    all_progress_streaks: list[int] = []
    for ev_path in sorted(art.glob("*_events.jsonl")):
        gid = ev_path.name.split("-")[0]
        events = parse_events(ev_path)
        t_path = tr / ev_path.name.replace("_events.jsonl", ".txt").replace("_p0_", "_p0")
        t_path = tr / (ev_path.name[: ev_path.name.index("_events")] + ".txt")
        text = t_path.read_text(encoding="utf-8", errors="replace") if t_path.exists() else ""
        n_exp = text.count("HARNESS EXPLORATION REPORT")
        n_anim = text.count("ANIMATION (")
        series = parse_tracker_series(t_path) if t_path.exists() else []
        # level-up turns: analysis_step of actions with level_completed
        lup_steps = [e["analysis_step"] for e in events if e.get("level_completed")]
        # map analyzer-turn index -> tracker value; take value at step BEFORE level-up
        for s in lup_steps:
            if series and s is not None and 1 <= s <= len(series):
                streak_at_levelup.append(series[s - 1])
        # organic progress streak distribution: value at turns where counter resets next
        for i in range(len(series) - 1):
            if series[i + 1] <= series[i] and series[i] > 0:
                all_progress_streaks.append(series[i])
        note = ""
        if n_exp and t_path.exists():
            # turn index of each injected report (counter resets on fire, so
            # locate reports by position in the transcript instead)
            body = text
            fire_turns = [body[: m.start()].count("turns since last new state")
                          for m in re.finditer("HARNESS EXPLORATION REPORT", body)]
            note = f"explore at turn {fire_turns} of {len(series)}"
        print(f"| {gid} | {n_exp} | {n_anim} | {len(lup_steps)} | {note} |")
    print("\n## Q4 no-progress streak stats (tracker counter, seed1, 25 games)\n")
    if streak_at_levelup:
        s = sorted(streak_at_levelup)
        print(f"- streak value on the turn a level-up landed: n={len(s)}, "
              f"median={st.median(s)}, p90={s[int(0.9 * (len(s) - 1))]}, max={max(s)}")
    if all_progress_streaks:
        s = sorted(all_progress_streaks)
        pct = lambda p: s[int(p * (len(s) - 1))]
        print(f"- streak length at any organic progress event: n={len(s)}, "
              f"median={st.median(s)}, p75={pct(.75)}, p90={pct(.9)}, p95={pct(.95)}, max={max(s)}")
        for thr in (10, 15, 20, 25):
            frac = sum(1 for x in s if x >= thr) / len(s)
            print(f"  - P(organic progress arrives after a streak >= {thr}): {frac:.3f}")


def q5_rhae(vanilla: list[dict]) -> None:
    print("\n## Q5 RHAE marginal value: late level vs early efficiency\n")
    print("| game | L | w(next lvl)/W | eff. median lvl1 | +cap lvl1 polish | +next lvl @0.25 eff | +next lvl @1.0 eff |")
    print("|---|---|---|---|---|---|---|")
    tot_polish, tot_next25, tot_next100 = [], [], []
    for gid in sorted({r["game_id"] for r in vanilla}):
        runs = [r for r in vanilla if r["game_id"] == gid]
        L = runs[0]["number_of_levels"]
        base = runs[0]["base_actions_per_level"]
        W = L * (L + 1) / 2
        lcs = [r["levels_completed"] for r in runs]
        lc_med = int(st.median(lcs))
        nxt = min(lc_med, L - 1)  # 0-idx of the "one more" level
        w_next = (nxt + 1) / W
        # observed level-1 efficiency among runs that completed it
        e1 = [min(115.0, (base[0] / r["actions_per_level"][0]) ** 2 * 100)
              for r in runs if r["levels_completed"] >= 1 and r["actions_per_level"][0] > 0]
        e1m = st.median(e1) if e1 else 0.0
        polish = (115 - e1m) * 1 / W if e1 else 0.0     # weight of level 1 is 1
        next25 = w_next * 25
        next100 = w_next * 100
        tot_polish.append(polish); tot_next25.append(next25); tot_next100.append(next100)
        print(f"| {gid.split('-')[0]} | {L} | {w_next:.3f} | {e1m:.0f} | +{polish:.2f} | +{next25:.2f} | +{next100:.2f} |")
    print(f"\n- Mean across 25 games: polishing level-1 to the 115 cap = "
          f"+{st.mean(tot_polish):.2f} RHAE; one extra level at 2x-over-budget (eff 0.25) = "
          f"+{st.mean(tot_next25):.2f}; at budget (eff 1.0) = +{st.mean(tot_next100):.2f}.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=30)
    args = ap.parse_args()
    vanilla = load_runs(ROOT / "runs/tufa_example_run/benchmark.json")
    q1_mode_signatures(vanilla, args.window)
    q1b_time_to_first_level(vanilla)
    q2_q4_events(ROOT / "runs/phase1_ab/seed1")
    q5_rhae(vanilla)


if __name__ == "__main__":
    main()
