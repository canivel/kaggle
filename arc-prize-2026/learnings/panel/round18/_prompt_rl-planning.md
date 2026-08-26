You are Professor of Reinforcement Learning and Planning (MCTS, model-based RL, exploration theory; 20 years; famously skeptical of under-specified search claims).

You are reviewer #1 on a 5-person adversarial review panel evaluating a competition
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

YOUR PRIOR-ROUND OBJECTIONS (verify each is resolved in the revision; state
RESOLVED / PARTIALLY-RESOLVED / UNRESOLVED for every one before new comments):
=====================================================================
## Objections

**[FATAL] R5 contradicts RC4 — the reset does not unstick the only thing that scores.** RC4's diagnosis is that the break-even rule (credible ≥ +0.06–0.12) has no information-value term, so "strictly applied, nothing ever qualifies." R5 then declares window discipline *unchanged*, and §4's promised "first experimental scored draw" is conditioned on "if any gate passes" — the exact gate RC4 just proved nothing passes. The reset therefore accelerates free build-rail evidence while leaving the scored-regime bottleneck mathematically intact; by the proposal's own arithmetic the terminal outcome (E[max]≈1.39, below the 1.44+ wall) is unaffected. Fix: an explicit amendment to the pricing rule — either a pre-registered exploration budget of k scored windows reserved for information draws, or a quantified VOI term added to break-even (see next objection for how to compute it). Without one of these, R1–R5 is a governance improvement attached to a losing policy.

**[MAJOR] The pricing rule appears denominated in the wrong currency for max-based scoring — quantify the true opportunity cost of an experimental draw.** The proposal's own numbers ("LB best 1.33," "ride E[max]≈1.39") imply the leaderboard takes the best draw, not the mean. Under max scoring, the cost of spending one window on an experiment is not the experiment's expected score deficit; it is P(a filler draw would have exceeded the current best) × E[exceedance] — and from the eight quoted draws (one at 1.33, mean 0.96), that tail probability looks like ~10-15% with small exceedance, i.e., an opportunity cost far below +0.06. A "credible ≥ +0.06–0.12" mean-improvement threshold is therefore miscalibrated by roughly an order of magnitude for information draws. Required evidence: fit a tail model to all filler draws to date, report P(draw > current best) and E[max over remaining filler windows], and re-derive break-even in max-scoring terms. This single calculation likely dissolves RC4 without any hand-waving about "information value."

**[MAJOR] §4's falsification criterion is a disjunction dominated by process artifacts — the reset can "succeed" while the score stays flat.** Of the six Aug-4 deliverables, at least three (R17 sealed, (f) defaulted in, EWM audit) are internal artifacts that move no leaderboard number, and the refutation trigger is "if NONE of these lands" — the weakest possible conjunction of failures. Under this rule, sealing one document validates the reset even at per-draw mean 0.96. Fix: make the 14-day success criterion require at least one *scored-regime* outcome (an experimental scored draw executed, or a pre-registered decision that none qualified with the max-scoring VOI calculation attached), and make refutation trigger on failure of that item alone.

**[MAJOR] Cited leaderboard state conflicts with the panel briefing — the quantified justification for the reset is unverified.** The panel briefing states team best 0.43 and leader 1.56; the proposal states LB best 1.33 and leader 1.86, and itself flags all numbers as "provisional" pending `runs/verify_2026-07-21/`. The entire urgency argument of §1 (and the wall-gap arithmetic in RC4) is load-bearing on these numbers. A governance reset should not be ratified against a §1 that the authors themselves have not verified and that disagrees with the panel's own record by 0.9 points on the headline number. Fix: attach the verification artifact (raw submission log hashes, draw-by-draw) before the ratification vote, and reconcile the 0.43-vs-1.33 discrepancy explicitly (fork? different account? stale briefing?).

**[MAJOR] R2's stopping rule is asymmetric in a way that ratchets toward GO.** NO-GO on A17 requires subsequent panel ratification *with quantified false-NO-GO probability*; no symmetric burden (quantified false-GO probability) is placed on proceeding. In sequential-decision terms this is a one-sided stopping rule: evidence can only be slowed in one direction, which biases the campaign toward committing spend to the wall-closer regardless of bench results. Fix: state both error probabilities, or pre-register the A17 decision thresholds (capability and parity numbers that constitute GO / NO-GO / CONTINUE) before the bench runs — this is a one-paragraph amendment and is standard pre-registration hygiene the proposal already demands of build-rail pushes in R1.

**[MINOR] R3's seal-termination rule is retroactive and gameable on severity labels.** Counting R16 as "round one of two" applies the rule to a round conducted before the rule existed, halving the check it purports to impose; and keying sealing on "0 FATALs, median ≥6" creates pressure to downgrade FATAL→MAJOR at the margin. Fix: R17 is round one; and require that any severity downgrade between rounds be logged with the downgrading reviewer named.

**[MINOR] "Build-rail runs are FREE" is asserted, not shown.** Kaggle build quota is finite (GPU-hours, weekly caps), and unsealed pushes to a shared rail are only "reversible" if artifacts like the sentinel don't alter accounting that later sealed scored draws depend on. Fix: one table — remaining weekly build quota, per-experiment quota cost for sentinel + A17 72B, and a statement of which rail state each push mutates.


=====================================================================

THE PROPOSAL (sha256 of the full document: a50f803e8c211e00; full length 5694 chars;
if the text below does not end with the literal line "## END OF PROPOSAL ##",
your copy is TRUNCATED — file that as an objection and review only what you see):
=====================================================================
# Stuck Review v2 — 2026-07-23 (revision for R18; R17 objections addressed)

Parent: `stuck_review_2026-07-21.md` (R17: 5× MAJOR-REVISION, 3 FATAL).
Independent verification since: `runs/verify_2026-07-21/report.md` — all gate
arithmetic reproduces exactly; 4 discrepancies found and adopted below.

## 0. Verified state update (post-verification corrections)

- Fork band corrected: **0.82–1.33** (n=12 now; new draws 1.14 and 0.82).
- Pricing re-based on the 15-draw pooled ledger (mean 0.962, σ̂ 0.144):
  **E[max@~102 remaining] ≈ 1.35, P(touch 1.44) ≈ 0.18** — down 40% from the
  stale 0.29. Filler-only is a losing policy with rising confidence.
- EWM clean carrier set shrinks to **{tn36, tu93, ls20, ft09-L1}** (tr87
  ALIASED-UNRESOLVED on binding holdout — R16 objection confirmed).
- Throughput (verified): 9 consecutive scored windows with zero new
  mechanism live; panels R10–R17 = 0/34 ACCEPTs, 169 majors; ≥7 mechanisms
  built-and-validated, 2 ever live (both killed); infra incidents 8/11 days.

## 1. Resolution of R17 FATAL 1 (R5-contradicts-RC4; pricing currency)

The methodology panel's max-currency insight is **adopted in full and
computed**. The LB keeps the max draw; the opportunity cost of giving one
window to an experiment is P(displaced filler draw > current best 1.33) ×
E[exceedance], not a mean-lift deficit. From the verified posterior
(mean 0.962, σ̂ 0.144): P(draw > 1.33) ≈ 0.5–1.5% per window (1.33 is
+2.6σ), E[exceedance | exceed] ≈ 0.05–0.08 → **opportunity cost ≈ 0.001–0.002
E[max]-equivalent per window** — two orders below the old +0.06–0.12 rule.

**A21 (new): Exploration window budget.** 12 scored windows (of ~102
remaining) are pre-registered as EXPLORATION draws for mechanism arms that
have passed their build-rail canary + screen (not full gate). Cost of the
full budget at the verified posterior: ΔE[max] ≈ −0.01 to −0.02 —
purchased against scored-regime evidence no rail can produce (R16's own
regime objections). Sequencing: no arm gets a 2nd exploration window before
its 1st is analyzed; a harm signal (arm draw < 0.80) pauses that arm.
The +0.06–0.12 mean-lift rule is retired for exploration draws; it remains
for PROMOTION (making an arm the default nightly draw).

## 2. Resolution of R17 FATAL 2 (quota is not free) — GPU-hour budget table

Weekly Kaggle GPU quota: 30 h. Rail SKU (verified from kernel logs, both
rails): RTX PRO 6000 Blackwell 96GB — NOT T4/P100/L4 (R17-systems' hardware
premise was wrong, but the finiteness point stands and is adopted).

| deliverable (by Aug 4) | runs × h/run | GPU-h |
|---|---|---|
| sentinel eval seeds 1–2 | 2 × 2.5 | 5.0 |
| A17 72B-VL capability+parity screen | 1 × 7.5 | 7.5 |
| (f)-default regression ride-along | 0 (shares above) | 0 |
| EWM Stage-1 measurement | CPU-only | 0 |
| contingency/re-run | — | 5.0 |
| **total (2 weeks available: 60 h)** | | **17.5** |

Fits with 3.4× headroom. A21 exploration draws use the daily submission
window, not GPU quota.

## 3. Resolution of R17 FATAL 3 (A17 envelope check) — one-pager

From `a17_72b_screen_scope.md` + kernel logs: Qwen2.5-VL-72B-Instruct-AWQ =
**43 GB weights on a 96 GB card** (53 GB headroom for KV at 32k ctx —
fits). Scored-rail SKU is IDENTICAL to bench SKU (verified: sm_120 both).
Throughput: 27B-FP8 baseline serves the full 8h window today; 72B-AWQ at
the scoped 2.5–3× decode penalty ⇒ ~⅓ the turns/window. The screen's
CAPABILITY prong measures exactly whether fewer-but-smarter turns net
positive levels; the ACTION-PARITY prong bounds the throughput loss. If
measured penalty exceeds 3.5× the screen self-reports envelope-infeasible —
that IS a valid NO-GO datum (distinct from capability NO-GO, no panel
ratification needed for physics). Serve-config risk (hermes parser, no
qwen3 thinking flags) is runtime-tested pre-push per scope doc.

## 4. Resolution of R17 MAJOR (watchdog kills the bench)

**A24 revised:** wedge signal = missing HEARTBEAT, not file-write silence.
Every registered long-run (bench pushes, panel rounds) must emit a
heartbeat line/file ≥ every 20 min (the harness already streams logs; the
loop's watchdog greps recency). Registered benches are exempt from the 6h
session cap; unregistered sessions are not. Kill only fires on 60 min of
missing heartbeat.

## 5. The reset, restated (A21–A25)

- **A21** exploration budget (§1) — 12 windows, canary+screen entry bar,
  harm-pause, promotion still gate-sealed.
- **A22** two-track governance — build-rail pushes need pre-registered
  one-page intent only (metric, canary, kill rule); full-panel seals only
  for scored-window PROMOTION, wall-closer kills, sealed-statistic edits.
  Quota governance per §2 table, republished weekly.
- **A23** A17 starts under §3 envelope; capability NO-GO requires panel
  ratification with false-NO-GO probability quantified; envelope NO-GO is
  self-certifying.
- **A24** loop hardening per §4 (heartbeat watchdog, bench exemptions,
  audit stub stays, panel recovery map stays).
- **A25** seal termination — two consecutive full-panel rounds, 0 fatals,
  median ≥6 → seal WITH NAMED CONDITIONS (tracked, owned, dated).

## 6. 14-day falsification (unchanged in spirit, dated from Jul 23)

By **Aug 6**: sentinel screen verdict; A17 capability/parity/envelope
numbers; EWM Stage-1 measurement on the clean 4-carrier set; (f) defaulted;
first exploration draw fired. If none lands, the reset is refuted and the
panel reconvenes on pod-spend vs accept-the-band.

## 7. Question to reviewers

Same as v1 §5, now with the R17 fatals resolved above: approve A21–A25, or
name the harm scenario the current process prevents that this reset does
not — weighed against the verified §0 throughput numbers.

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
