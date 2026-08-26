# Pre-registration amendment — 2026-07-18 (panel R13 seals, filed BEFORE draw #5 observation)

Responds to panel round 13 (5× MAJOR-REVISION, 2 FATAL; `learnings/panel/round13/`).
Filed 2026-07-18 ~08:00 EDT. Draw #5 fires 20:07 EDT tonight and scores ~04:00
EDT Jul 19; nothing in this document may be revised after that observation.

## A8 — A5 variance-gate fail consequence (R13 methodology major, sealed pre-observation)

Panel arithmetic (verified): with ledger {0.91, 1.08, 0.88, 1.05}, A5
(χ²-CI-hi(σ) < 0.25 at df=4) passes only if draw #5 ∈ [0.955, 1.005]. The 0.25
threshold as written is therefore near-unpassable — a calibration error in A5,
conceded. Sealed consequences:

- **If A5 FAILS at n=5 (expected):** LB windows for the war-v1 arm remain
  **accumulation-only** (which they already are under A4) and additionally
  lose eligibility as an A/B *readout* arm at any future n. No mechanism
  line may cite war-arm LB deltas as evidence, positive or negative, until a
  re-registered gate with a recalibrated threshold (set from the frozen-fork
  control's own CI-hi at equal n, not an absolute constant) passes.
- **If A5 PASSES:** no new licenses are granted beyond what A4 already
  permits; a pass at a miscalibrated threshold is not evidence of low σ.
- The A5 threshold for future arms is re-based: CI-hi(arm) < 1.5 ×
  CI-hi(frozen control at same n), df-matched — relative, not absolute.

## A9 — Warpack reopening rule (R13 regime-transfer major, sealed pre-observation)

The A1 gate closed the warpack build-rail line on an offline bench that
suppresses budget-pressure firing conditions (R13: "refuted a composition,
not components" — conceded, and consistent with the banking trigger counter
never being observed live). Sealed now, before draw #5:

- The warpack line is reclassified from REFUTED to **UNTESTED-IN-REGIME,
  parked**. It does NOT reopen on any LB statistic at n=5. Reopening requires
  BOTH: (i) war-arm LB ledger at n≥8 with one-sided Welch p < 0.05 vs the
  frozen control ledger at equal-or-greater n, AND (ii) a budget-faithful
  build-rail bench (A10) demonstrating the banking/recovery trigger counters
  fire at ≥1 event/run on ≥5 games. Neither alone reopens. No other
  statistic, draw, or eyeballing reopens it.
- war-arm draws beyond tonight's #5 are NOT scheduled; the frozen fork
  resumes as default filler (its 1.33 demonstrates the order-stats floor;
  war accumulation past n=5 has no sealed purpose).

## A10 — Budget-faithful bench (R13 rl-planning/systems major)

Before any budget-regime mechanism (budget sentinel (a), per-game
re-allocation (g), banking soft-time (e)) enters a sealed gate: the build
rail must run a **compressed-budget bench** — per-game action/wall-clock
budgets scaled so the mechanism's trigger counter fires ≥1 time/run on ≥5
games (verified by canary before the gate seals, as the ledger canary did).
Mechanisms whose triggers cannot be made to fire on the rail get LB-ledger
accumulation status only, never a build-rail kill.

## A11 — Ledger conclusion relabeled (R13 methodology major, conceded)

"REFUTED" is withdrawn. Sealed label: **"trigger never fires as built
(mechanistic, certain: 0/1552); effect size unmeasured (n=1 screen,
p=0.86)."** The −0.128/−0.314 point estimates may not be cited in ranking
or retirement arguments. Ledger-as-built still does not enter scored
windows (no benefit channel); its firing-trigger upgrades compete in Q1 on
their own counting bounds.

## A12 — Unbundling (R13 unanimous major, conceded)

The (a)+(f) single-window lean is withdrawn. (f) game-over-continuation
ships FIRST as a standalone hygiene window with its own quick screen and a
pre-registered su15 exclusion from any later (a) evaluation. One flag per
window, no exceptions without full-panel sign-off.

## A13 — su15 wall verdict suspended (R13 prog-synthesis/llm-agents major, conceded)

"Accept-the-loss" is suspended pending one disambiguating experiment: after
the (f) fix lands in the local rig, re-probe su15 once with GPT-5.6-sol
(covered by the existing API credit; single game, 60-min/100-action caps,
$10 spend ceiling). Wall verdict re-affirmed or retracted on that evidence.
