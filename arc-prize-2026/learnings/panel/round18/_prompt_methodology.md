You are Professor of Empirical ML Methodology and Statistics (experimental design, multiple-comparisons, noise-band inference; rejects any plan that draws conclusions from single noisy samples).

You are reviewer #4 on a 5-person adversarial review panel evaluating a competition
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

**[MAJOR] Provisional and internally conflicting numbers underpin the entire "stuck" claim.** The panel briefing I was issued states team best 0.43 and leader 1.56; the proposal states LB best 1.33 and leader 1.86, and itself flags all numbers as "provisional" pending `runs/verify_2026-07-21/`. A strategic reset justified by §1's quantification cannot be sealed while §1's numbers are unverified and contradict the panel's own briefing. Actionable fix: the verification run's output must be attached to the next round, with a reconciliation of the 0.43/1.33 and 1.56/1.86 discrepancies (stale briefing vs. wrong ledger — say which, with artifact paths).

**[MAJOR] RC4's core claim — a single scored draw "buys regime evidence" — is exactly the single-noisy-sample inference this panel exists to prevent.** The eight listed draws have mean ≈ 0.98 and sample SD ≈ 0.17. The break-even effect the pricing rule demands is +0.06–0.12, i.e., 0.35–0.7 SD. A one-sided test at α = 0.05 with 80% power to detect +0.09 against SD 0.17 needs on the order of 20+ draws per arm; one draw has power ≈ 12–18% and cannot distinguish a working mechanism from filler noise. If the argument is that the draw buys *regime* evidence (behavioral traces, failure modes) rather than a score estimate, then R1's intent template must pre-register *which non-score observables* constitute that evidence and what would count as a null result — otherwise the first draw will be interpreted post hoc in whichever direction is convenient.

**[MAJOR] R3's termination rule is fit to the data it is then retroactively applied to (double-dipping).** The rule "two consecutive rounds, 0 FATALs, median ≥ 6" is proposed *after* observing that R13–R16 scores climbed to 6–7, and the proposal immediately declares "R16 already qualifies as round one." A stopping rule chosen after seeing the sample, with the threshold set just below the observed values, is the textbook garden-of-forking-paths; it guarantees the rule fires on the round that motivated it. Fix: adopt the rule prospectively only — the two qualifying rounds must both occur *after* A23 is ratified — and pre-register the threshold before the next round's scores are seen. Also note reviewer scores are ordinal and reviewer-dependent; a median over 5 raters with no calibration is a fragile seal trigger — at minimum pair it with "0 FATALs and no *new* MAJORs," which is less gameable than a score threshold.

**[MAJOR] §4's falsification criterion is a disjunction over six heterogeneous items and is therefore nearly unfalsifiable.** "If NONE of these lands by Aug 4, the reset is refuted" means landing the single easiest item — defaulting (f) into builds, already 49/49 screened — confirms the reset, while the items that actually address the wall (A17 numbers, a scored experimental draw) can all fail without consequence. This is the multiple-endpoints problem: with six endpoints and success = any one, the reset "passes" under almost any world state. Fix: designate one or two *primary* endpoints (I propose: A17 capability+parity numbers on the real rail, and ≥1 gate reaching a sealed scored-entry decision — pass or fail) whose joint failure refutes the reset; the rest are secondary.

**[MAJOR] R1 removes the seal from build-rail experiments without adding any multiplicity control, creating a selection-then-promotion funnel.** Unlimited free experiments, each with a self-declared metric, will generate winners by chance; those winners will then be priced for scored windows using the same noisy build-rail estimates that selected them — a guaranteed winner's-curse overestimate at the gate. Fix: the one-page intent must pre-specify sample size (seeds/episodes) and the estimate presented at any subsequent scored-gate must either come from a held-out confirmation run or be shrunk for selection (report how many sibling experiments the candidate was selected from). Relatedly, "sentinel screen verdict (2 seeds)" in §4 is itself a 2-sample verdict — state what effect size 2 seeds can resolve, or run more seeds.

**[MAJOR] "E[max] ≈ 1.39 to Nov — NOT top-10" is asserted without a derivation and against a static-wall assumption.** No distributional model, no remaining-draw count, no CI on E[max], and — worse — the wall (1.44+, ~14 teams) is treated as fixed for 104 days while a winning Milestone-1 notebook is public, which historically drags the whole leaderboard band upward. The strategic conclusion (filler-only loses) is probably right, but the panel should not seal a magnitude claim with no shown work. Fix: attach the E[max] computation (n remaining draws, tail model, sensitivity to SD 0.15–0.20) and a wall-trajectory estimate from leaderboard history (even a linear fit over the last 4 weeks suffices).

**[MINOR] Pass/fail validation counts are reported without confidence bounds.** 29/29 and 49/49 sound decisive but by the rule of three give 95% upper bounds on failure rate of ~10% and ~6% respectively. State whether those failure-rate bounds are acceptable for a budget sentinel whose job is preventing budget death; if not, the free build-rail track R1 creates is exactly the place to cheaply extend n.


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
