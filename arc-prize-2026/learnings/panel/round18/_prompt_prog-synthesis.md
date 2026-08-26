You are Professor of Program Synthesis and Neurosymbolic AI (inductive program synthesis, world models as code, verification; insists on falsifiable synthesis-quality metrics).

You are reviewer #3 on a 5-person adversarial review panel evaluating a competition
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

**[MAJOR] §1's quantification contradicts the panel record and is self-declared provisional** — The proposal claims LB best 1.33, per-draw mean ≈ 0.96, leader 1.86; the panel briefing states team best 0.43 and leader 1.56. That is a ~3x discrepancy on the numbers that carry the entire "stuck" argument and the E[max] ≈ 1.39 projection in RC4, and the document itself says the re-verification (`runs/verify_2026-07-21/`) has not landed. A governance reset cannot be ratified on numbers its own authors flag as unverified and which conflict with the panel's record. Fix: attach the verification output and a reconciliation (metric definition, date, which board) before any of R1–R5 executes.

**[MAJOR] RC4 is diagnosed but not treated — the reset cannot produce scored-regime evidence** — RC4 correctly identifies that the pricing rule has no information-value term, so "nothing ever qualifies." But R5 declares window discipline unchanged and no amendment adds a VOI term, an exploration budget, or a first-draw exemption. §4's headline deliverable ("first experimental scored draw since Jul 14") is therefore gated on "if any gate passes" under the exact rule the proposal proves nothing passes. Fix: propose a concrete, pre-registered mechanism — e.g., a bounded exploration budget of N scored windows through Nov 2 with an explicit per-mechanism first-draw information credit and a stopping rule — or admit the reset only fixes build-rail throughput.

**[MAJOR] §4's refutation condition is disjunctive and gameable — this is not a falsifiable test** — Six heterogeneous deliverables are OR'd: landing the single easiest item ((f) defaulting in, which already passed 49/49) "confirms" the reset, while "If NONE of these lands" is the only refutation trigger. Several items have no pass/fail definition: what threshold makes A17 "capability+parity numbers" a pass, parity against which baseline (the public Milestone-1 notebook? the duck fork?), and what does "EWM Stage-1 measurement" measure? Fix: per-item binary criteria with numeric thresholds, and a conjunctive minimum (e.g., ≥4/6 with the A17 numbers mandatory, since the proposal itself calls A17 "the only wall-sized lever").

**[MAJOR] R2 protects only one tail — no false-GO guard on A17** — R2 correctly requires quantified false-NO-GO probability before killing the wall-closer, but is silent on false GO: promising bench numbers from an underpowered run will create momentum toward committing scored windows. Given the per-draw variance visible in §1 (0.76–1.33 across 8 draws), any capability/parity claim needs a pre-registered sample size and variance estimate to be interpretable at all. Fix: pre-register both GO and NO-GO thresholds with a power calculation *before* the bench runs, not after the numbers exist.

**[MAJOR] EWM line has no world-model fidelity metric — unfalsifiable by construction** — "Stage-0 done, blocked on latent-state audit sequencing" and "Stage-1 measurement" name no metric for what the world model is required to predict, against what held-out data, at what threshold. A learned/executable world model without a pre-registered fidelity gate (e.g., n-step transition prediction accuracy on held-out trajectories from the real rail, with a floor below which EWM-execute is killed) is exactly the kind of unmeasured synthesis line that later demands a scored window on vibes. Fix: define the Stage-1 fidelity metric and kill threshold in the R1 one-page intent before any push.

**[MAJOR] R3 grandfathers R16 retroactively and its "named conditions" have no fail-consequence** — Declaring "R16 already qualifies as round one of two" applies a new rule to reviewers who scored under different incentives; their 0-FATAL scores did not mean "consent to auto-seal" when given. And converting unresolved MAJORs into owner+deadline conditions is decorative unless a lapsed deadline has a defined consequence. Fix: start the two-round clock at R17, and specify that a missed condition deadline suspends the seal pending a single-reviewer spot check.

**[MINOR] "Build-rail runs are FREE" conflates quota cost with calendar and state cost** — §1 itself reports 3 of 5 recent loop-days lost to infra, including a wedge; build pushes consume the same fragile loop capacity and shared state, and the wedge shows the environment is not cleanly reversible in practice. Fix: bound R1 to N concurrent unsealed pushes, each with a checkpoint taken before push (the checkpoint machinery exists) so "reversible" is enforced rather than asserted.


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
