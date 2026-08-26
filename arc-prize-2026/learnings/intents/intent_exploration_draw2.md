# A22 intent — A21 exploration draw #2 (sentinel arm un-shelve rule)

**Status: DRAFT — NOT SEALED.** Build-rail intent only (A22: metric, canary, kill
rule). Nothing below is sealed; no sealed/amendment/prereg file is modified.
**Scheduling decision deferred** per the 07-27 restructure (weekday sealed gate or
Sunday panel 2026-08-02). Responds to **R21 directive #4 (3/5: methodology,
prog-synthesis, rl-planning)** and **NC-7** ("re-shelve only under the
pre-registered multi-draw rule, never on n=1"). Sources: `runs/lb_ground_truth.md`
(canonical, 07-27), `learnings/preregistration_amendment_2026-07-24_DRAFT.md`
(A21-E/P/R draft rules), ITERATION_LOG 07-23/07-24 (draw #1 record),
`learnings/panel/round21/_directives.md`.

## Why n=1 shelving is indefensible (panel arithmetic, independently verified)

Frozen ledger (API-verified 07-27): n = 13, mean 0.974, s ≈ 0.143.

- **The 0.71 draw:** panel quotes z ≈ −1.8. Verified: Gaussian
  z = (0.71 − 0.974)/0.143 = **−1.85** ✓. Under the declared t-predictive
  (ν = 12, scale 0.143·√(1+1/13) = 0.148): t = −1.78, one-sided **p ≈ 0.05** — not
  significant, and the frozen arm's own floor is 0.82 with a 0.76 in the closed war
  arm. (The 07-24 DRAFT's p ≈ 0.070 used the then-current n = 10 stats; both agree
  qualitatively.)
- **The 0.80 harm-pause:** panel quotes ~11% false-fire/draw. Verified: Gaussian
  P(draw < 0.80 | arm ≡ frozen) = Φ(−1.22) = **11.2%** ✓; t-predictive **13.2%**.
  (The 15.6% in the 07-24 DRAFT was the n = 10 figure; s shrank 0.156 → 0.143 at
  n = 13.) One in ~8 healthy arms trips the pause on any given draw — the pause is
  exposure control, not inference (C2 said so at seal time).

Standing record: the arm was harm-PAUSED 07-24 (mandatory, sealed) and then SHELVED
by the disposition memo on *eval-rail* evidence (RHAE negative on both seeds: s1
−0.315, s2 −0.166) — not by the 0.71 alone. This intent supplies the pre-registered
multi-draw path back, as directive #4 orders.

## Pre-registered trigger and date

- **Trigger:** draw #2 fires in the first daily scored slot **≥ 2 calendar days
  after the boristown A/B disposition memo is filed** (slot-priority order below),
  and in any case not before A17 v6 lands (panel's own suggested anchoring).
- **Nominal date:** **2026-08-03** if the A/B is gate-scheduled midweek and runs
  07-29 → 08-01; if A/B scheduling is deferred to the 08-02 Sunday panel, nominal
  date slides to **≈ 2026-08-08**. **Hard backstop: no later than 2026-08-10** —
  if the A/B has not completed by 08-08, draw #2 fires anyway (the exploration
  program may not be starved by upstream slippage; methodology: "structurally
  guaranteed to be killed by noise" is the failure mode being repaired).
- **Entry case (A21-E discipline):** filed before the window; must state the
  aggregated prior *including* the negative RHAE eval evidence and pre-register the
  explicit rail-transfer hypothesis the scored draw tests (A21-E(2) — the arm's
  aggregated prior is currently net-negative, so this clause is mandatory, not
  optional). 2-seed canary + non-harm screen already on record for this
  composition (07-23); re-run only if the build is re-composed.

## Disposition rule — no disposition at n < 4; target n = 5

**Metric:** public-LB scored draw per window, API-verified into the sentinel arm
ledger before any rule fires (NC-8). All draws accumulate the per-arm ledger (C2);
the 0.71 is draw 1/5 and is never excluded.

- **Sequential early re-shelve (panel's rule adopted verbatim):** re-shelve iff
  **2 consecutive draws < 0.80** OR **mean of first 3 draws (incl. 0.71) < 0.80**.
  Verified error rates under arm ≡ frozen: mean-of-3 clause needs draws 2–3 to
  average < 0.845 → Φ(−1.28) ≈ **10%**; 2-consecutive clause ≈ **3.5%** over draws
  2–5; union ≲ **13%** false early re-shelve.
- **Final disposition at n = 5 (no earlier, no later):**
  - **SHELVE** iff one-sided test vs frozen at α = 0.10:
    x̄₅ < 0.974 − 1.282 · 0.143 · √(1/5 + 1/13) = **x̄₅ < 0.878**.
  - **PROMOTE-track** iff the exceedance criterion (A21-P draft): ≥ 2 draws > 1.33
    OR ≥ 1 draw ≥ 1.44.
  - **Otherwise CLOSE-NEUTRAL:** arm closed as distribution-compatible; remaining
    windows revert to the pool under the A21 allocation policy.
  - Honest conditional arithmetic: with the sunk 0.71, SHELVE requires draws 2–5 to
    average < 0.92, which a healthy (≡ frozen) arm does with probability
    Φ(−0.76) ≈ **22%** — the sunk low draw biases toward shelving; that is the
    unavoidable price of C2 ledger accumulation and is stated here rather than
    hidden.
- **Harm-pause remains live** (sealed A21/C2): any single draw < 0.80 pauses the
  arm pending the next pre-registered step of *this* rule — under this intent a
  pause defers, it does not dispose (NC-7).
- **Supersession note:** the A21-R two-resume-draw rule (07-24 DRAFT §e) conflicts
  with the panel's n ≥ 4–5 order; neither is sealed. If this intent is ratified it
  supersedes A21-R for the sentinel arm.

## Slot budget

Up to 4 further scored draws (draws 2–5) from the **daily scored slot**; no extra
submissions, no GPU-h, $0 cloud. Opportunity cost ≈ 0.0024 E[max]-equiv total
(declared-model pricing, ≈ 0.0006/window).

## Slot-priority order (shared verbatim with intent_boristown_readiness_ab.md)

0. Sealed obligations pre-empt on their pre-registered dates (C4 lines; A17 v6
   scored bench).
1. **Boristown readiness-gate A/B** — n = 4 consecutive gated slots once scheduled.
2. **A21 exploration draw #2 (sentinel)** — first slot ≥ 2 days after the A/B
   disposition memo is filed.
3. Frozen-fork filler (default; affirmatively a strategy at P(touch 1.44) ≈ 0.33).

Justification: directive #1 carries 5/5 reviewers vs 3/5 for directive #4; the A/B
outcome (NC-6 regime switch) would change the default filler composition that draw
#2 is scored against, so sequencing A/B first avoids a mid-arm stratum split; the
resulting delay to draw #2 is ≤ ~5 windows ≈ 0.003 E[max]-equiv — immaterial. The
08-10 backstop above caps how long priority 1 can starve priority 2.

---
*Draft prepared 2026-07-27 (R21 directive #4 / NC-7). NOT SEALED. Do not queue,
push, or submit on the basis of this document alone.*
