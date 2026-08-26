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

YOUR PRIOR-ROUND OBJECTIONS (verify each is resolved in the revision; state
RESOLVED / PARTIALLY-RESOLVED / UNRESOLVED for every one before new comments):
=====================================================================
## Objections

**[FATAL] "Build quota is FREE" is false under a capped weekly GPU budget, and R1's economics rest on it.** Kaggle GPU quota is a fixed weekly allocation (order ~30 GPU-hours/account/week) shared by *every* line in this campaign: the sentinel seed, EWM Stage-1, (f) regression runs, and the A17 72B bench all draw from the same pool. A 72B VL model on Kaggle-class accelerators (2×T4=32GB, P100=16GB, 4×L4=96GB) requires aggressive quantization (~40GB at 4-bit before KV cache) and will run at low single-digit tokens/s aggregate on L4s under vLLM tensor-parallel — a "capability+parity" bench of any statistical weight plausibly consumes the entire weekly quota by itself. RC1's claim that pre-seal review "inverts the economics" of free evidence is therefore wrong in sign: the evidence is *not* free, it is quota-priced, and R1 removes the only prioritization mechanism without replacing it. Fix: publish a GPU-hour budget table (mechanism × runs × hours/run × hardware) showing all §4 deliverables fit inside the quota available before Aug 4; without it R1 cannot be approved as written.

**[FATAL] No feasibility evidence that A17 (72B VL) fits the *scored* rail envelope — the "only wall-closer" may be dead on arrival.** The proposal calls A17 "the only wall-sized lever" but supplies no numbers for: (i) whether the model fits scored-notebook memory limits at any quantization, (ii) per-task inference latency versus the scored submission's wall-clock cap, (iii) whether "the real rail" bench hardware matches scored hardware (parity is asserted as a deliverable, not demonstrated as possible). If a 72B VL model cannot serve within the scored window's runtime limit, then R2's entire framing — protect the wall-closer from premature NO-GO — is protecting a line that quota physics has already killed, and the correct plan is the §4 fallback (smaller model line or pod spend) *now*, 14 days earlier. Supply: a one-page envelope check — weights+KV memory at target quantization, measured tokens/s on the bench GPU, tokens/task × tasks/window versus the scored wall-clock limit — *before* spending bench quota, not after.

**[MAJOR] R4's watchdog and wedge-detector will kill R2's benchmark.** A hard 6h session cap plus "no file writes in 60 min ⇒ kill" is incompatible with 72B inference on Kaggle hardware: model download + quantized load + graph capture alone can exceed 60 minutes with no artifact writes, and a full bench sweep at L4-class throughput will not finish in 6h. As specified, the reset's loop-hardening rule terminates the reset's flagship experiment and burns the quota already spent (non-reversible loss, contradicting R1's "reversible" premise). Fix: exempt registered long-run benches from the 6h cap, and make the wedge signal a process-level heartbeat (periodic progress file/log flush required of the bench harness) rather than generic file-write silence.

**[MAJOR] Headline numbers contradict the panel briefing and are self-declared provisional — the entire §1 quantification is unverified load-bearing evidence.** The panel briefing states team best 0.43 and leader 1.56; the proposal states LB best 1.33 and leader 1.86. One of these is stale or wrong, and the proposal itself says the raw-artifact re-verification (`runs/verify_2026-07-21/`) is still running. Since the whole case for the reset ("~0% scored-regime evidence," "E[max] ≈ 1.39 is NOT top-10") rests on §1, approval must be conditioned on the verification run confirming every §1 number; a governance reset ratified on numbers that later fail verification would itself be the harm scenario §5 asks us to name.

**[MAJOR] RC4 is diagnosed but not fixed — R5 preserves the exact pricing rule RC4 proves yields permanent filler.** §2 argues the break-even rule has no information-value term and that "strictly applied, nothing ever qualifies"; §3 then leaves window discipline "unchanged" (R5). So under the reset, all the newly accelerated build-rail evidence still faces the same gate that the proposal itself claims no honest mechanism can pass — meaning §4's "first experimental scored draw since Jul 14" is conditioned on a gate the authors have argued is unpassable. Fix: either propose the information-value term explicitly (e.g., a bounded per-mechanism first-draw allowance with a quantified expected-regret cap in windows), or retract RC4's "permanent filler" claim; you cannot hold both.

**[MINOR] E[max] ≈ 1.39 has no stated tail model.** With observed draws ranging 0.76–1.33 and a 1.33 already banked, the projection to Nov depends entirely on the assumed right-tail distribution and remaining draw count, neither of which is given; the gap to the 1.44+ wall is smaller than plausible tail-model uncertainty. State the distribution fit, draw budget, and CI, since "filler alone is NOT top-10" is used to justify risk-taking.


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
