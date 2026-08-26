# war-v1 LB draw #3 = 0.88 — analysis (2026-07-16)

## 1. Ledger at n=3 (prereg 2026-07-14 §3 + amendment A3)

war-v1 = {0.91, 1.08, 0.88}: **mean 0.957, σ̂ 0.108** (χ² 95% CI on σ, df 2: [0.056, 0.678]).

- **σ-recompute clause (A3): does NOT fire.** Trigger is point-estimate σ̂ > 0.15; 0.108 ≤ 0.15 → LB windows retain stopping-rule status for the R2 A/B. Caveat for the record: the n=3 CI cannot exclude σ > 0.15 (upper 0.678); recompute at n=5 is mandatory before any downgrade/clearance is treated as settled.
- **vs frozen control** {0.82, 0.89, 0.93, 1.02, 0.95} (mean 0.922, σ̂ 0.074): Δmean +0.035; descriptively inside the band except draw #2 (1.08 > control max 1.02, still inside Tufa band). Illustrative Welch t ≈ 0.49 — but prereg §3 forbids quoting standardized war-vs-frozen effects until n≥5. Verdict: **null-consistent, no lift, no harm.** Draw #3 alone: z ≈ −0.57 vs control (descriptive only).

## 2. Consistency with build rail

0.88 is exactly what the war-eval screen predicts. LB currency is RHAE; the screen showed **Δlc +0.272 (p=0.0074) but Δlog1p(RHAE) −0.036 (p=0.61)**. Warpack clears ~45% more levels **at full action cost** — the pooled-single-run first-clear tax converts extra L1s into ~zero RHAE. A flat LB draw therefore *confirms* the R11 Goodhart alarm (Δlc dissociates from scoring currency) and is fully consistent with the action-tax explanation. Nothing "went wrong" mechanically; the mechanism simply doesn't pay in RHAE yet. What went right: three clean draws, daemon fixed (00:07Z trigger fired), variance behaving near control scale.

## 3. Banking — still zero live evidence

`bank_fire_validation.json` (A2) showed banking fires only with ≥120s soft time AND non-randomized frames → plausibly **inert in scored runs** (budgets exhaust). With only a scalar LB score visible, in-scored-run confirmation requires one of:
1. **Score-keyed canary arm**: a build where replay is the *only* path to a clear on a designated game — score moves iff banking fired. Detectable but burns a window and confounds the ledger.
2. **Banking ON/OFF paired A/B**: MDE 0.12–0.17 at k=3–6 — unpowered for banking's plausible effect. Practically unconfirmable.
3. **Registered path (correct one)**: local engineered validation (DONE, fires + score-invariant on ar25/s5i5) + attempts/skips/aborts canary in build rail; scored-run inertness carried "on faith," ledgered as UNVERIFIED. Real fix = war-v3 soft-time threshold compatible with scored budgets so banking can fire at all.

## 4. Verdict for today's brief

**war-v1 n=3 COMPLETE: mean 0.957, σ̂ 0.108, null-consistent vs frozen control; no monitoring-only downgrade (σ̂ ≤ 0.15); no standardized effect quotable until n=5.** The draw corroborates, not contradicts, the levels-up/RHAE-flat picture.

**Yes — tomorrow's compound-gate look (Jul 17, war-eval seeds 1–3) remains THE decision point**, unchanged by this draw: LB draws are monitoring by design (A3: A/B unpowered). Gate = Δlc p<0.0125 AND mean Δlog1p(RHAE) ≥ 0. Seed 1 RHAE was −0.036, so criterion (ii) is the live risk; a (ii)-only fail → conversion-first mode (convert clears into clean clears), which draw #3's flatness already foreshadows. Precondition check: seed 3 must land; A2 replay-fires-locally condition for war-v2 windows is satisfied.
