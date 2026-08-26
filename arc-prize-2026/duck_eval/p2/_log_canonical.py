#!/usr/bin/env python3
"""Log the two canonical, bench-admissible rows for 2026-08-26.

`kaos bench push` refuses any record whose experiment metadata lacks
`mechanism` / `summary` / `lesson` -- and it still exits 0, so the refusal is
easy to miss (KAOS #37/#38/#39 already filed on exactly this). Metadata is
attached at log time and the journal is append-only, so the only way to publish
is a new row carrying all three keys at the TOP LEVEL.

Supersedes: exp 50 (execwm read), exp 51 + 52 (P2 build).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(r"F:\kaggle\arc-prize-2026")


def log(name, family, verdict, lock, results, meta):
    p = REPO / "runs" / ("bench_meta_%s.json" % name.replace("-", "_"))
    p.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    argv = ["uv", "run", "kaos", "experiment", "log",
            "--name", name, "--family", family, "--verdict", verdict,
            "--lock-sha256", lock, "--results-path", results,
            "--metadata-json", str(p)]
    r = subprocess.run(argv, cwd=r"F:\kaggle\kaos", capture_output=True, text=True,
                       env={**os.environ, "KAOS_DB": "f:/kaggle/arc-prize-2026/kaos.db"})
    print(r.stdout.strip() or r.stderr.strip())
    return r.returncode


# ---------------------------------------------------------------- exec-WM read
execwm_meta = {
    "mechanism": (
        "Executable world model: mine per-action object-delta rules from recorded "
        "history, verify them prequentially against held-out transitions, BFS-plan "
        "inside the verified program, and fall back PER LEVEL to the stock LLM agent. "
        "On the scored rail it cleared ls20 level 1 BY PLAN with llm_calls=0 and "
        "llm_tokens=0 (4 move rules at precision 1.0, n=33/21/23/19) -- the CPU proof "
        "transferred to Kaggle exactly. But the arm fell back on 31 of 32 in-scope "
        "levels: 14 levels produced ZERO candidate rules after the FULL 16-20 probe "
        "budget, and on 9 more the verifier refused every candidate (24 verified / 29 "
        "rejected overall). The binding constraint is therefore UPSTREAM of the "
        "planner and upstream of the verifier -- it is object identification and "
        "rule-class coverage. lc_total 25 vs a sealed NULL band of 24-34 and a "
        "comparator of 29.0 (pooled sd 2.80); board draw 1.05."),
    "summary": (
        "exec-WM v1 read NULL (lc 25) with the pre-stated decisive kill NOT triggered. "
        "It banked the campaign's first level cleared by deterministic search with no "
        "LLM in the loop, and it located its own defect precisely: the rule class is "
        "NARROW, not under-budgeted."),
    "lesson": (
        "TWO transferable lessons. (1) A DELIVERY GATE MUST COUNT ARM-REACHABILITY, "
        "NOT REPORT-PRESENCE, whenever the arm has a legitimate self-disable path. D1 "
        "failed 18/25 here purely because 7 click-only games were CORRECTLY refused by "
        "an arm that only models keyboard movement -- and emitted no report when "
        "disabled. The gate counted a correct refusal as a delivery failure. The gate "
        "is recorded FAILED as sealed rather than rewritten post-hoc; what changes is "
        "the next prereg. Same family as the 2026-08-20 arm-mismatch lesson, where two "
        "sealed scorers both returned INFRA DEATH on a healthy arm because one arm is "
        "defined by a marker being PRESENT and the other by its ABSENCE. "
        "(2) WHEN AN ARM FALLS BACK, READ *WHY* PER LEVEL BEFORE PROPOSING A V2. The "
        "probe histogram {4:2, 8:2, 16:11, 18:1, 20:16} shows the failing levels spent "
        "their FULL budget and still mined nothing, which arithmetically CLOSES "
        "'raise the probe budget' as a v2 -- the obvious tweak, and the wrong one."),
    "arm": "exec-WM v1", "exp_supersedes": 50, "kernel": "canivel/arc3-execwm-eval v1",
    "board_draw": 1.05, "lc_total": 25, "trim1": 2.330, "mean_score": 3.006,
    "band": "NULL (24-34)", "comparator": {"mean": 29.0, "pooled_sd": 2.80, "z": -1.43},
    "delivery": {"D1": "FAIL-as-written 18/25 (spec defect; 25/25 ARMED, 7 self-disabled)",
                 "D2": "PASS 5 games reached PHASE P (bar 3)",
                 "D3": "not triggered", "D4": "PASS"},
    "instrument_defects": [
        "D1 counts report-presence, not arm-reachability",
        "execwm_score.py reports disabled_games=0 because it counts disabled_reason "
        "only over PRESENT reports; the true count is 7 and is recoverable only from the log"],
    "prestated_v2_constraint": (
        "a v2 may NOT be a probe-budget tweak; it must widen the rule class / object "
        "model (click-addressable objects, non-constant deltas, multi-object dynamics)"),
    "open_bug": ("sb26 received 4 probes and bp35 8, against a 16-20 budget everywhere "
                 "else; sb26 carries 50.4% of the certified field floor's mean_score"),
}

# ------------------------------------------------------------------- P2 build
p2_meta = json.loads((REPO / "runs" / "p2_build_meta_2026-08-26.json").read_text(encoding="utf-8"))
p2_meta.update({
    "mechanism": (
        "P2 reset-anchored episodic retry. attempt(seq) runs a candidate action "
        "sequence from the CURRENT LEVEL START, reports what it reached, then issues "
        "its own RESET back to that same start -- so ONE LLM turn can evaluate K "
        "candidate plans instead of committing to one. It is composed entirely from "
        "the existing action() primitive inside the sandbox child process, so it needs "
        "no new host message type. A harness-side stuck trigger (H=4 consecutive "
        "ACTING turns on one uncleared level) arms it and advertises it as "
        "'retry_mode: on, episodes_available: K' on the python tool result. The design "
        "exploits the measured resource inversion on this rail: actions are nearly "
        "free (epsilon=0.17) while turns are the binding constraint (675/675 game-runs "
        "die on the 7,920 s clock at ~17 turns/game)."),
    "summary": (
        "P2 v1 BUILT, gated 57/0, and PUSHED as slot 1 after its blocking question was "
        "answered first: the H=4 trigger FIRES on 19/25 games on its own vehicle "
        "against a sealed bar of >=15/25, measured on retained artifacts before the "
        "build existed. No outcome claimed; the head rule was sealed pre-data."),
    "lesson": (
        "PROVE THE TRIGGER CAN FIRE ON RETAINED DATA BEFORE YOU SPEND A SLOT ON IT -- "
        "and the proof is cheap. This took ~20 minutes against benchmark.json histories "
        "already on disk, and it also produced the arm's honest upside cap: sb26 is one "
        "of the 6 games that correctly REFUSE to arm, and sb26 carries 50.4% of the "
        "field floor's entire mean_score, so P2 cannot lift our best game by "
        "construction. The same 20 minutes surfaced TWO silently-dead-arm bugs that "
        "would have shipped: the per-game counter keyed on a directory the shipped "
        "layout can SHARE across all 25 games (so the cleared-level count would have "
        "accumulated benchmark-wide and permanently disabled retry after the 4th clear "
        "anywhere), and a D2 instrument that lived only on stdout (unevaluable under "
        "the P1 0-byte-log class). Corollary, learned from P1 at 1.3% use against a 30% "
        "bar: instrument USE, not delivery -- count real attempt() CALLS by AST, split "
        "by whether the affordance was armed, and flush them to a JOB-DIR file so the "
        "read survives a truncated log."),
    "exp_supersedes": [51, 52],
})

rc = 0
rc |= log("execwm-v1-seed1-read-CANONICAL", "probe",
          "REJECT: NULL on primary (lc_total 25; sealed band NULL 24-34; -1.43 sigma vs "
          "comparator 29.0/2.80). The pre-stated decisive kill did NOT trigger, so the v1 "
          "exec-WM class is not dead by its own rule. D2 PASSED (5 games reached PHASE P, "
          "bar 3). D1 failed as written (18/25 reports) on a GATE-SPECIFICATION DEFECT: "
          "25/25 games ARMED and exactly 7 self-disabled 'no-keyboard-actions' because "
          "they are click-only and out of the v1 movement rule class by construction -- "
          "the gate counted a correct refusal as a delivery failure, and is recorded "
          "FAILED rather than rewritten post-hoc. BANKED: ls20 level 1 cleared BY PLAN on "
          "the scored rail with llm_calls=0 and llm_tokens=0, 4 move rules at precision "
          "1.0 -- the CPU proof transferred to Kaggle exactly, and it is this campaign's "
          "first level cleared by deterministic search with no LLM in the loop. DEFECT "
          "LOCATED UPSTREAM OF THE PLANNER: 14 of 32 in-scope levels produced ZERO "
          "candidate rules after the FULL probe budget and 9 more had every candidate "
          "refused by the verifier, so the rule class is NARROW rather than "
          "under-budgeted, and a probe-budget v2 is arithmetically closed. Supersedes "
          "exp 50, which was not bench-admissible for want of mechanism/summary/lesson.",
          "bb630a4a70fd41650b934b637106017db296db02dbeabf52c56d4e5ecadd0ad5",
          str(REPO / "learnings" / "war_room" / "execwm_seed1_read_2026-08-26.md"),
          execwm_meta)

rc |= log("p2-reset-retry-v1-build-CANONICAL", "probe",
          "ACCEPT (build + fireability gate discharged; NO OUTCOME CLAIMED). The ordered "
          "question was whether an H=4 stuck trigger can fire at all before a slot is "
          "spent on it -- this campaign's signature defect is a mechanism that ships "
          "armed and cannot fire (hard_noop_guard: armed, 0 blocks in 5,255 real "
          "actions). Measured on RETAINED REAL ARTIFACTS BEFORE THE BUILD EXISTED: the "
          "trigger fires on 19/25 games on the arm's OWN vehicle against a sealed D1 bar "
          "of >=15/25, and on >=15/25 across four independent corpora (field 19, "
          "budget_t3 23, p1_notes 19, execwm 19). The margin is not fragile -- 15 of 25 "
          "field-floor games have max_stuck_run >= 7. NEGATIVE CONTROL: 6/25 games "
          "correctly REFUSE and they are exactly the prompt clearers. The turn "
          "reconstruction reproduces an independent instrument EXACTLY (424/424 acting "
          "turns, 17.0/game). PRICED HONESTLY: sb26 is a refuser and carries 50.4% of the "
          "field floor's mean_score, so P2 cannot lift our best game by construction. TWO "
          "LATENT SILENTLY-DEAD-ARM BUGS FOUND AND FIXED PRE-PUSH: a per-game counter "
          "keyed on a directory the shipped layout can share across all 25 games, and a "
          "D2 instrument that existed only on stdout. GATES: local_gate --arm p2 --full "
          "PASS 57/0; episode smoke 18/18; trigger smoke 50/50; scorer selftest 33/33 "
          "including a healthy positive control and 6 real foreign artifacts all refused; "
          "p2_cell_smoke 20/20 executing the REAL notebook cell off-Kaggle with four "
          "loud-death negative controls. PULL-BACK: metadata EXACT, and the remote "
          "notebook minus the inserted patch cell is byte-identical to the certified "
          "floor's own remote copy. Supersedes exp 51 (shell-damaged) and exp 52 (not "
          "bench-admissible for want of mechanism/summary/lesson).",
          "346175882bf03eba3ebcf6eb1dcd22ff8a25470c9050519ddb56977ca57931ba",
          str(REPO / "learnings" / "war_room" / "p2_trigger_fireability_2026-08-26.md"),
          p2_meta)

sys.exit(rc)
