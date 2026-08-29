"""UNTRIED-SET PROBE -- can a stagnation supervisor's redirect target ever FIRE?

WHY THIS EXISTS
    We have never built the STAGNATION SUPERVISOR, and our own banked numbers say
    it is the largest single gap in the harness:

        88%     of each game's clock elapses AFTER its last level clear
        45.2%   of actions are immediate repeats
        0       hard_noop_guard fires in 5,255 actions
        675/675 games died on the 7920s wall clock

    What the supervisor has never had is a REDIRECT TARGET -- something to DO
    instead of stalling. thtennant/arc3-duck-v28 proposes one (`untried`): press
    the declared controls the agent has never pressed. His offline sweep claims 91%
    of declared actions move the board, 69% on one press, and yet only 41% of his
    archived passes ever pressed the full declared set.

    CAMPAIGN RULE -- feedback_verify_treatment_can_fire:
        Prove an arm's TREATMENT CAN FIRE on OUR rail BEFORE building it.
    Banking was built, shipped, and could never fire: it needed a win, and there
    were 0 wins in 470 recorded game-runs. This probe is the gate that stops the
    same class of waste. It measures FIREABILITY ONLY. It does not measure effect.

WHAT IT READS  (artifacts only -- no GPU, no Kaggle slot, no model call)
    runs/kernel_pulls/<arm>/artifacts/<game>_events.jsonl

    Both halves of the measurement live in the per-turn transcript, in the USER
    PROMPT the harness itself writes:
        "Valid actions right now: UP, DOWN, LEFT, RIGHT, SPACE, MOUSE, ACTION7."
        "Executed actions: SPACE."
    The declared line is the game telling the agent, in advance and every single
    turn, which controls exist. The executed line is what it actually pressed.

VERDICTS
    FIRES            a non-empty untried set exists at stagnation, often enough to
                     act on. The supervisor has a target.
    DOES-NOT-FIRE    the agent already presses everything it is offered. The
                     redirect target is empty and the arm must be KILLED -- a clean
                     kill here is worth more than a build we cannot read.
    UNMEASURABLE     the declared/executed lines are absent for this arm. Reported
                     as such, NEVER silently dropped (feedback_audit_the_instrument).

USAGE
    scripts/untried_probe.py                       # every arm in runs/kernel_pulls
    scripts/untried_probe.py --pull runs/kernel_pulls/p1_v1
    scripts/untried_probe.py --json runs/untried_gate_0829/untried_gate.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PULLS = ROOT / "runs" / "kernel_pulls"

# These lines are written by the HARNESS into the user prompt. They must be read
# there and nowhere else: the assistant's own prose quotes them back ("it says
# 'Executed actions: DOWN'..."), and an unanchored search over the whole
# transcript silently harvests the model's speculation as if it were ground
# truth. Anchor to line-start, and only inside the [USER PROMPT] block.
RE_DECLARED = re.compile(r"^Valid actions right now:\s*([^\n]*)", re.M)
RE_EXECUTED = re.compile(r"^Executed actions:\s*([^\n]*)", re.M)
RE_STATE = re.compile(r"^Current state:\s*step\s+(\d+),\s*level\s+(\d+)", re.M)

# MOUSE is logged with its coordinates -- "MOUSE(row=23, col=60)" -- and that
# inner comma splits into two bogus tokens under a naive comma split. Normalise
# every parameterised press back to its control name.
RE_COORDS = re.compile(r"\s*\([^)]*\)")


def _user_prompt(t):
    """The harness-authored block only. Everything after the model's turn begins
    is the model talking about the prompt, not the prompt."""
    i = t.find("[USER PROMPT]")
    if i < 0:
        return ""
    j = t.find("[MODEL RESPONSE META]", i)
    if j < 0:
        j = t.find("[ASSISTANT]", i)
    return t[i:j] if j > 0 else t[i:]

# MOUSE is ACTION6: parameterised by (x, y). "Untried MOUSE" is not a one-press
# probe -- pressing it means choosing a coordinate out of 64x64. It is counted
# and reported SEPARATELY throughout; the headline gate is the discrete set.
PARAMETERISED = {"MOUSE"}


def _split(s):
    s = RE_COORDS.sub("", s)                      # MOUSE(row=23, col=60) -> MOUSE
    toks = [t.strip(" .") for t in s.split(",")]
    return [t for t in toks if t and t.replace("_", "").isalnum()]


def parse_pass(path):
    """One events.jsonl == one game pass. Returns a per-turn timeline."""
    turns, n_rows, n_transcript = [], 0, 0
    for line in path.open(encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        n_rows += 1
        t = row.get("transcript")
        if not t:
            continue
        n_transcript += 1
        up = _user_prompt(t)
        md, me, ms = RE_DECLARED.search(up), RE_EXECUTED.search(up), RE_STATE.search(up)
        turns.append({
            "action_num": row.get("action_num"),
            "level": row.get("level"),
            "score": row.get("score"),
            "declared": _split(md.group(1)) if md else None,
            # "Executed actions" describes the PREVIOUS sequence, so it lags the
            # prompt that reports it by one turn. Kept as-is: the union over a
            # pass is unaffected, and the gate reads strict prefixes.
            "executed": _split(me.group(1)) if me else [],
            "step": int(ms.group(1)) if ms else None,
            "t_level": int(ms.group(2)) if ms else None,
        })
    return {"path": str(path), "arm": path.parts[-3],
            "game": path.stem.replace("_events", ""),
            "n_rows": n_rows, "n_transcript": n_transcript, "turns": turns}


def stagnation_windows(turns, k):
    """Start indices of runs of >=k consecutive turns with no level gain and no
    score gain. The START is the moment a supervisor would fire."""
    starts, run_start = [], 0
    best_level = turns[0].get("level") or 0
    best_score = turns[0].get("score") or 0
    for i, t in enumerate(turns):
        lv, sc = t.get("level") or 0, t.get("score") or 0
        if lv > best_level or sc > best_score:
            best_level = max(lv, best_level)
            best_score = max(sc, best_score)
            run_start = i + 1
            continue
        if i - run_start + 1 == k:          # window matures exactly once
            starts.append(run_start)
    return starts


def analyse(passes, ks=(10, 25, 50)):
    out = {"n_passes": len(passes), "arms": {}, "pooled": {}}
    rec_hit = rec_tot = 0
    stab_ok = stab_tot = 0
    cov_full = cov_full_disc = cov_tot = 0
    gate = {k: {"windows": 0, "nonempty": 0, "nonempty_disc": 0, "sizes": []} for k in ks}
    press = collections.Counter()
    never_pressed = collections.Counter()
    unmeasured = []
    per_arm = collections.defaultdict(lambda: {
        "passes": 0, "turns": 0, "declared_turns": 0, "cov_passes": 0,
        "cov_full_disc": 0, "gate25_win": 0, "gate25_nonempty_disc": 0})

    for p in passes:
        turns = p["turns"]
        a = per_arm[p["arm"]]
        a["passes"] += 1
        a["turns"] += len(turns)
        if not turns:
            unmeasured.append(p["path"])
            continue

        # 1 -- recoverability
        d_turns = [t for t in turns if t["declared"]]
        rec_hit += len(d_turns)
        rec_tot += len(turns)
        a["declared_turns"] += len(d_turns)
        if not d_turns:
            unmeasured.append(p["path"])
            continue

        # 2 -- stability of the declared set within a level
        by_level = collections.defaultdict(set)
        for t in d_turns:
            by_level[t.get("level")].add(tuple(sorted(t["declared"])))
        for sets in by_level.values():
            stab_tot += 1
            stab_ok += (len(sets) == 1)

        # 3 -- coverage over the whole pass
        declared_union = set().union(*[set(t["declared"]) for t in d_turns])
        executed_union = set().union(*[set(t["executed"]) for t in turns])
        disc_declared = declared_union - PARAMETERISED
        cov_tot += 1
        a["cov_passes"] += 1
        cov_full += declared_union.issubset(executed_union)
        got_disc = disc_declared.issubset(executed_union)
        cov_full_disc += got_disc
        a["cov_full_disc"] += got_disc
        for act in disc_declared - executed_union:
            never_pressed[act] += 1
        for t in turns:
            for act in t["executed"]:
                press[act] += 1

        # 4 -- THE GATE: untried set at each stagnation-window start
        prefix, prefixes = set(), []
        for t in turns:
            prefixes.append(set(prefix))
            prefix |= set(t["executed"])
        for k in ks:
            for s in stagnation_windows(turns, k):
                dec = set(turns[s]["declared"] or declared_union)
                untried = dec - prefixes[s]
                untried_disc = untried - PARAMETERISED
                gate[k]["windows"] += 1
                gate[k]["nonempty"] += bool(untried)
                gate[k]["nonempty_disc"] += bool(untried_disc)
                gate[k]["sizes"].append(len(untried_disc))
                if k == 25:
                    a["gate25_win"] += 1
                    a["gate25_nonempty_disc"] += bool(untried_disc)

    out["pooled"] = {
        "declared_line_recoverability": [rec_hit, rec_tot],
        "declared_stable_within_level": [stab_ok, stab_tot],
        "passes_pressing_full_declared_set": [cov_full, cov_tot],
        "passes_pressing_full_DISCRETE_set": [cov_full_disc, cov_tot],
        "gate": {str(k): {
            "windows": v["windows"],
            "nonempty_any": v["nonempty"],
            "nonempty_discrete": v["nonempty_disc"],
            "mean_untried_size": (round(statistics.mean(v["sizes"]), 3) if v["sizes"] else None),
            "size_hist": dict(sorted(collections.Counter(v["sizes"]).items())),
        } for k, v in gate.items()},
        "press_counts": dict(press.most_common()),
        "never_pressed_in_pass": dict(never_pressed.most_common()),
        "unmeasured_passes": unmeasured,
    }
    out["arms"] = {k: dict(v) for k, v in sorted(per_arm.items())}
    return out


def pct(n, d):
    return "n/a" if not d else "%.1f%% (%d/%d)" % (100.0 * n / d, n, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", default=None, help="scope to one runs/kernel_pulls/<arm> dir")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    pattern = (str(Path(a.pull) / "artifacts" / "*_events.jsonl") if a.pull
               else str(PULLS / "*" / "artifacts" / "*_events.jsonl"))
    files = sorted(glob.glob(pattern))
    if not files:
        print("no event files matched %s" % pattern, file=sys.stderr)
        return 2
    passes = [parse_pass(Path(f)) for f in files]
    res = analyse(passes)
    p = res["pooled"]

    print("UNTRIED-SET PROBE -- %d game-passes across %d arms\n" % (len(files), len(res["arms"])))
    print("1. DECLARED-SET RECOVERABILITY")
    print("   turns carrying a 'Valid actions right now:' line   %s" % pct(*p["declared_line_recoverability"]))
    print("   passes with NO declared line (UNMEASURED)          %d" % len(p["unmeasured_passes"]))
    print("\n2. DECLARED-SET STABILITY")
    print("   (pass, level) blocks with ONE declared set         %s" % pct(*p["declared_stable_within_level"]))
    print("\n3. COVERAGE -- did the pass ever press every declared control?")
    print("   full declared set (incl. MOUSE)                    %s" % pct(*p["passes_pressing_full_declared_set"]))
    print("   discrete set only (MOUSE excluded)                 %s" % pct(*p["passes_pressing_full_DISCRETE_set"]))
    print("\n4. THE GATE -- untried set at the START of a stagnation window")
    print("   %4s %8s %24s %10s" % ("K", "windows", "non-empty (discrete)", "mean size"))
    for k, g in p["gate"].items():
        print("   %4s %8d %24s %10s" % (k, g["windows"],
                                        pct(g["nonempty_discrete"], g["windows"]),
                                        str(g["mean_untried_size"])))
    print("\n5. PRESS COUNTS (pooled)")
    for act, n in p["press_counts"].items():
        print("   %-10s pressed %7d   passes-never-pressing-it: %d"
              % (act, n, p["never_pressed_in_pass"].get(act, 0)))
    for act, n in p["never_pressed_in_pass"].items():
        if act not in p["press_counts"]:
            print("   %-10s pressed       0   passes-never-pressing-it: %d   <-- NEVER PRESSED ANYWHERE"
                  % (act, n))

    if a.json:
        out = Path(a.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
