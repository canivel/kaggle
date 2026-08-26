#!/usr/bin/env python3
"""Log the O1 finding (local screen vs board, above the floor) via exact argv."""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(r"F:\kaggle\arc-prize-2026")

meta = {
    "mechanism": (
        "Every promotion gate this campaign has sealed is defined on a LOCAL 25-game "
        "screen statistic (lc_total, trim1) -- exec-WM's SIGNAL >= 35, P2's kill at "
        "lc <= 21, the shared comparator mean 29.0 / pooled sd 2.80. Joining all 36 "
        "pulled artifacts that carry a benchmark.json to the board draw each actually "
        "produced shows only TWO matched pairs exist, and both sit at or below the "
        "certified floor: q38_field_v1 (lc 28, trim1 3.189) -> 1.16, and execwm_v1 "
        "(lc 25, trim1 2.330) -> 1.05. Two artifacts have screened ABOVE the floor -- "
        "budget_t3_v1 at lc 35 / trim1 5.021 and private_base_v1 at lc 30 / trim1 "
        "4.732 -- and NEITHER was ever submitted (the first is a 3x-budget config that "
        "is not a legal submission; the second is another lane's arm, held per its own "
        "order). Each omission is individually defensible; the consequence is not. The "
        "count of configs that have screened above the floor AND produced a board draw "
        "is ZERO, so the local->board mapping is entirely unsampled in the only region "
        "a promotion decision ever cares about."),
    "summary": (
        "O1 ANSWERED, and the answer is worse than 'unknown': we have never placed a "
        "single data point in the region our promotion gates are defined over. n=2 "
        "matched local/board pairs exist, both at or below the certified floor. The "
        "instrument (scripts/local_vs_board.py) deliberately REFUSES to fit a slope "
        "from n=2 rather than quote one it cannot support."),
    "lesson": (
        "A SCREENING STATISTIC IS ONLY CALIBRATED OVER THE RANGE YOU HAVE ACTUALLY "
        "SUBMITTED FROM. We have been sealing bands (>=35 SIGNAL, <=21 kill) across a "
        "region containing zero board observations, which means the bands encode a "
        "belief about the local->board curve rather than a measurement of it. The "
        "external corroboration is Kaggle discussion 736578: a competitor took the "
        "same duck harness from local 2.1% (board 1.4%) to their own harness at local "
        "5.0-5.4% and the board did not move off 1.4% -- a ~2.5x local gain buying "
        "nothing. That is exactly the failure mode our screen cannot currently detect. "
        "TWO RULES FOLLOW: (1) before sealing a band outside the sampled range, say so "
        "explicitly in the prereg and treat the band as a PRIOR, not a measurement; "
        "(2) spend a draw to place a calibration point in the empty region when a "
        "legal above-floor artifact already exists -- private_base_v1 screens lc 30 on "
        "a legal config and is already built and pull-back-verified, so it is the "
        "cheapest available point. SECOND-ORDER: I initially wrote that the R^2=0.990 "
        "trim1-to-LB fit was anchored entirely at the floor, could not find that fit's "
        "anchor list anywhere on disk, and WITHDREW the claim rather than let a "
        "convenient one stand -- the table above does not need it."),
    "instrument": "scripts/local_vs_board.py (read-only; writes nothing)",
    "matched_pairs": [
        {"pull": "q38_field_v1", "lc": 28, "trim1": 3.189, "mean": 6.173, "board": 1.16},
        {"pull": "execwm_v1", "lc": 25, "trim1": 2.330, "mean": 3.006, "board": 1.05},
    ],
    "above_floor_never_submitted": [
        {"pull": "budget_t3_v1", "lc": 35, "trim1": 5.021,
         "why": "3x budget -- not a legal submission config"},
        {"pull": "private_base_v1", "lc": 30, "trim1": 4.732,
         "why": "private arm, deliberately held per its own lane order"},
    ],
    "artifacts_with_benchmark": 36,
    "configs_above_floor_with_a_board_draw": 0,
    "external_corroboration": "Kaggle discussion 736578 (Nick Pellegrin, 2026-08-21)",
    "recommended_action": (
        "Sunday panel agenda: consider spending one nightly draw on private_base_v1 "
        "(lc 30, legal config, already built and verified) to buy the campaign's first "
        "calibration point above the floor. NOT this session's call -- the private arm "
        "is another lane's and its lock is explicit that pushes follow its own prereg."),
    "withdrawn_claim": (
        "An earlier draft asserted every anchor of the R^2=0.990 trim1 fit sits at the "
        "floor. The anchor list is not on disk; the claim is withdrawn as unverified."),
}

p = REPO / "runs" / "o1_local_vs_board_meta_2026-08-26.json"
p.write_text(json.dumps(meta, indent=1), encoding="utf-8")

VERDICT = (
    "ACCEPT (measurement, not a mechanism claim): THE LOCAL->BOARD MAPPING IS UNSAMPLED "
    "ABOVE THE CERTIFIED FLOOR. Across all 36 pulled artifacts carrying a benchmark.json, "
    "exactly TWO have a known board draw, and both sit at or below the floor: "
    "q38_field_v1 lc 28 -> 1.16 and execwm_v1 lc 25 -> 1.05. Two artifacts have screened "
    "ABOVE the floor -- budget_t3_v1 (lc 35) and private_base_v1 (lc 30) -- and NEITHER "
    "was ever submitted, for individually defensible reasons (illegal 3x-budget config; "
    "another lane's held arm). Net: the number of configs that screened above the floor "
    "AND produced a board draw is ZERO. This does NOT show the mapping is flat -- with "
    "n=2 pairs there is no regression to run and the instrument deliberately REFUSES to "
    "fit one -- it shows the mapping is UNTESTED in precisely the region where every "
    "sealed band lives (exec-WM SIGNAL >= 35, P2 kill at lc <= 21, comparator 29.0/2.80). "
    "We are calibrating promotion decisions on a curve we have never sampled. External "
    "corroboration: Kaggle discussion 736578 reports a ~2.5x local gain (2.1% -> 5.0-5.4% "
    "local) buying zero board movement (1.4% both times) on the same duck harness family. "
    "A claim in the first draft -- that the R^2=0.990 trim1 fit is anchored entirely at "
    "the floor -- was WITHDRAWN when its anchor list could not be found on disk; the "
    "finding does not rest on it."
)

argv = ["uv", "run", "kaos", "experiment", "log",
        "--name", "o1-local-screen-vs-board-calibration", "--family", "probe",
        "--verdict", VERDICT,
        "--results-path", str(REPO / "learnings" / "daily_brief_2026-08-26.md"),
        "--metadata-json", str(p)]

r = subprocess.run(argv, cwd=r"F:\kaggle\kaos", capture_output=True, text=True,
                   env={**os.environ, "KAOS_DB": "f:/kaggle/arc-prize-2026/kaos.db"})
print(r.stdout.strip() or r.stderr.strip())
sys.exit(r.returncode)
