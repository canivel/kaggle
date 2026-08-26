#!/usr/bin/env python3
"""Log the corrected P2 build row via exact argv (no shell).

exp 51 was inserted through a shell that interpreted the backticks in the verdict
text as command substitution and silently deleted the word `cleared` in two
places. The KAOS journal is APPEND-ONLY by design, so the fix is a superseding
row that says what it supersedes -- not a quiet edit.
"""
import subprocess
import sys

VERDICT = (
    "ACCEPT (build + fireability gate discharged; NO OUTCOME CLAIMED). "
    "SUPERSEDES exp 51, which is byte-damaged: it was inserted through a shell that "
    "read the backticks in this text as command substitution and silently deleted the "
    "word 'cleared' in two places. Read this row, not 51. "
    "THE ORDERED QUESTION: can an H=4 stuck trigger fire at all, before a slot is spent "
    "on it? This campaign's signature defect is a mechanism that ships armed and cannot "
    "fire -- hard_noop_guard was armed and blocked 0 of 5,255 real actions. "
    "ANSWER, measured against RETAINED REAL ARTIFACTS BEFORE THE BUILD EXISTED: the "
    "trigger fires on 19/25 games on the arm's OWN vehicle (the certified field floor) "
    "against a sealed D1 bar of >=15/25, and on >=15/25 across four independent corpora "
    "(field 19, budget_t3 23, p1_notes 19, execwm 19). The margin is not fragile: 15 of "
    "25 field-floor games have max_stuck_run >= 7, so H would have to exceed 7 before "
    "delivery fell to the bar. NEGATIVE CONTROL: 6/25 games correctly REFUSE, and they "
    "are exactly the prompt clearers. The turn reconstruction was validated against an "
    "independent instrument and reproduces it EXACTLY (424/424 acting turns, 17.0/game). "
    "PRICED HONESTLY: sb26 is one of the refusers and carries 50.4% of the field floor's "
    "entire mean_score, so P2 cannot lift our best game by construction -- its upside is "
    "capped to the 19 stuck games. "
    "TWO LATENT BUGS FOUND AND FIXED PRE-PUSH, both of the silently-dead-arm class: "
    "(1) the counter would have keyed on state_path.parent, which the shipped layout can "
    "share across ALL games in a run, so the per-game cleared-level count would have "
    "accumulated benchmark-wide and PERMANENTLY disabled retry after the 4th level clear "
    "anywhere; (2) the D2 report existed only on stdout, which is unevaluable under the "
    "P1 0-byte-log class, and is now flushed per game to a job-dir file. "
    "D2 (USE, not delivery) is instrumented as AST-counted attempt() CALLS split "
    "armed/unarmed: P1 delivered at 96.3% and got 1.3% use, and its read was unevaluable "
    "precisely because nothing counted calls. "
    "GATES: local_gate --arm p2 --full PASS 57/0; episode smoke 18/18; trigger smoke "
    "50/50; scorer selftest 33/33 including a healthy positive control and 6 REAL foreign "
    "artifacts all refused; p2_cell_smoke 20/20, which EXECUTES THE REAL NOTEBOOK CELL "
    "off-Kaggle with four loud-death negative controls (late import, missing bundle, "
    "broken RESET invariant, tampered sha). "
    "PULL-BACK: metadata EXACT (model_sources survived), and the remote notebook MINUS "
    "the inserted patch cell is byte-identical to the certified floor's own remote copy. "
    "The head rule was SEALED PRE-DATA; the read happens tomorrow against lc mean 29.0 / "
    "pooled sd 2.80."
)

argv = [
    "uv", "run", "kaos", "experiment", "log",
    "--name", "p2-reset-retry-v1-build-CORRECTED",
    "--family", "probe",
    "--verdict", VERDICT,
    "--lock-sha256", "346175882bf03eba3ebcf6eb1dcd22ff8a25470c9050519ddb56977ca57931ba",
    "--results-path",
    "f:/kaggle/arc-prize-2026/learnings/war_room/p2_trigger_fireability_2026-08-26.md",
    "--metadata-json",
    "f:/kaggle/arc-prize-2026/runs/p2_build_meta_2026-08-26.json",
]

r = subprocess.run(argv, cwd=r"F:\kaggle\kaos", capture_output=True, text=True,
                   env={**__import__("os").environ,
                        "KAOS_DB": "f:/kaggle/arc-prize-2026/kaos.db"})
print(r.stdout.strip())
if r.returncode:
    print(r.stderr.strip(), file=sys.stderr)
sys.exit(r.returncode)
