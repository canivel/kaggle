You are Professor of ML Systems (GPU inference, vLLM serving, wall-clock budgets, quota economics; kills plans that don't fit the compute envelope).

You are reviewer #5 on a 5-person adversarial review panel evaluating a competition
strategy proposal for ARC-AGI-3 (Kaggle, deadline Nov 2 2026). The proposing team has a
best score of 0.43; the leader is at 1.56; the winning Milestone-1 notebook is public.

PANEL RULES — READ CAREFULLY:
- You are UNBIASED and RIGOROUS. You have no stake in the proposal passing.
- Do NOT be agreeable. A proposal that passes round 1 unchanged indicates REVIEW FAILURE.
- Attack the weakest LOAD-BEARING assumptions, not typos.
- Every objection must be specific and actionable (not "needs more detail").
- Distinguish severity: FATAL (invalidates the plan), MAJOR (must fix before execution),
  MINOR (should fix).
- You review ONLY within your expertise; note explicitly what you cannot judge.
- If evidence is missing for a claim, say exactly what experiment/data would supply it.

THE PROPOSAL (sha256 of the full document: 293bdad516d1b3e2; full length 5733 chars;
if the text below does not end with the literal line "## END OF PROPOSAL ##",
your copy is TRUNCATED — file that as an objection and review only what you see):
=====================================================================
# Stuck Review — 2026-07-21 (full-panel strategic review, user-ordered)

The user's verdict: "we are stuck." This document quantifies the stuck-ness,
names root causes, and proposes a governance reset (R1–R5) for the panel to
approve or refute. An independent raw-artifact re-verification of every
number cited here is running in parallel (`runs/verify_2026-07-21/`); the
panel should treat unverified numbers as provisional.

## 1. The stuck claim, quantified

**Score:** LB best 1.33 (a VANILLA filler right-tail draw, Jul 18). Per-draw
mean ≈ 0.96 — unchanged since the duck fork was reproduced (Jul 8). Draws
since Jul 14: 0.91, 1.08, 0.88, 1.05, 0.76, 1.33, 0.92, 0.93 — every one is
either the dead warpack composition or vanilla filler. Leader: 1.86.
Wall: 1.44+ (~14 teams).

**Experiment throughput:** scored windows used on live experimental
mechanisms since Jul 15: **zero** (war draws #2–#5 were accumulation of an
arm whose gate then failed). Mechanisms currently built, validated, and NOT
live anywhere:
- (f) game-over hygiene — screen PASSED (49/49 recoveries), not defaulted in
- (a) budget sentinel — built, 29/29 + canary PASS, unpushed 2 days
  (waiting on a SENTINEL_BUDGET panel ruling)
- (c)+(d)+Reki refutation stack — spec'd, blocked on R16 republication
- EWM-execute — Stage-0 done, blocked on latent-state audit sequencing
- A17 72B VL bench — scoped, blocked on seal-repair (false-NO-GO objection)

**Governance throughput:** panel rounds R13–R16: 4 rounds, **0 ACCEPTs**,
3 FATALs, ~60 MAJORs, ~4 full loop-days consumed. Each round improves scores
(4–6 → 6–7) yet ends in "republish before sealing." Meanwhile: 3 of the last
5 loop-days were partially or fully lost to infra (turn-cap death Jul 17,
scheduler miss Jul 18, wedge Jul 21).

**Net:** the campaign currently converts ~100% of its development capacity
into review artifacts and ~0% into scored-regime evidence. The epistemic
machine we built to stop wasting windows is now consuming the calendar the
windows live in. 104 days remain; at the current cadence the A17 bench —
the only wall-sized lever — has not started.

## 2. Root causes (proposed, for panel confirmation)

RC1 **Seal-scope creep.** The original discipline was: scored windows need
gates. It has drifted to: *every build artifact* needs a full-panel seal
before any rail activity. But build-rail runs are FREE (Kaggle build quota),
reversible, and information-positive. Requiring pre-seal for free evidence
inverts the economics the gates were built on.

RC2 **Republication loops without a termination rule.** ACCEPT requires 4/5
with 0 fatals; reviewers are instructed that clean passes indicate review
failure. Result: asymptotic approach, no seal. There is no rule converting
"two consecutive rounds, 0 fatals, scores ≥6" into a conditional seal.

RC3 **Loop fragility.** Single daily session, 80-turn cap, no watchdog, no
wedge detection; KAOS agent waits have no liveness escape. 3/5 recent days
degraded.

RC4 **The pricing rule + honest discounts = permanent filler.** Window
break-even is a credible ≥ +0.06–0.12 claim; every honest post-validation
expectation now sits below it. Correct per-window, but it has no term for
information value: a mechanism's first scored draw also buys regime evidence
that no build-rail run can (per R16's own regime objections). Strictly
applied, nothing ever qualifies and we ride E[max] ≈ 1.39 to Nov — which is
NOT top-10.

## 3. Proposed reset (R1–R5) — the decision before the panel

R1 **Two-track governance.** Build-rail experiments (free builds, no scored
window, reversible) require only: a pre-registered one-page intent (metric,
canary, kill rule) filed BEFORE the push — no panel seal. Full-panel seals
remain required for: (i) any scored-window entry, (ii) killing a wall-closer
line, (iii) amendments to sealed statistics. Effect: sentinel seed and the
A17 72B bench push this week on quota that costs nothing.

R2 **A17 starts now, as information not verdict.** The bench runs this week.
Its output cannot trigger NO-GO by itself; NO-GO on the only wall-closer
requires a subsequent panel ratification WITH the false-NO-GO probability
quantified (R16-R1 objection honored, inverted into a protection).

R3 **Seal termination rule.** Two consecutive full-panel rounds with 0
FATALs and median score ≥ 6 → the document seals WITH NAMED CONDITIONS
(each unresolved MAJOR becomes a tracked condition with an owner and a
deadline) instead of another republication. R16 already qualifies as round
one of two under this rule.

R4 **Loop hardening.** (i) Wall-clock watchdog: loop sessions get a hard
6h cap; a wedge-detector (no file writes in 60 min while session live)
kills and logs. (ii) Panel waits get timeouts with persisted recovery (the
_agents.json machinery already exists). (iii) The audit stub stays.

R5 **Window discipline unchanged** — filler remains the default draw;
scored-window entry still needs its sealed gate. The reset accelerates
evidence, not spending.

## 4. What "unstuck" looks like in 14 days (falsifiable)

By Aug 4: sentinel screen verdict (2 seeds), A17 72B capability+parity
numbers on the real rail, EWM latent-state audit + Stage-1 measurement,
(f) defaulted into every build, R17 sealed with named conditions, and —
if any gate passes — the first experimental scored draw since Jul 14.
If NONE of these lands by Aug 4 under the reset, the reset itself is
refuted and the panel reconvenes on the harder question (model line via
pod spend, or accept-the-band).

## 5. Question to each reviewer

Approve R1–R5 as amendments (A21–A25)? For any rejection, name the specific
harm scenario the current process prevents that the reset does not — and
weigh it against the quantified cost in §1.

## END OF PROPOSAL ##
=====================================================================

OUTPUT FORMAT (exactly this structure):
## Summary (2 sentences)
## Objections
For each: [SEVERITY] title — body (2-5 sentences, specific)
(minimum 3 objections in round 1; in later rounds, review your prior objections'
resolution first, then add new ones only if real)
## Questions for the authors (numbered)
## What I cannot judge
## Verdict: ACCEPT | MAJOR-REVISION | REJECT
## Score: N/10
