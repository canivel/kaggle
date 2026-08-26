# Pre-registration amendment — 2026-07-16 (panel R12 responses)

Responds to panel round 12 (3× MAJOR-REVISION, 0 fatal; `learnings/panel/round12/`).
Parent documents: `preregistration_2026-07-14.md`, `preregistration_amendment_2026-07-15.md`.

## A4 — R2 A/B power disclosure and redesign (R12 N4 + methodology major)

Published power math for detecting Δ = +0.08 LB (the midpoint of R2's predicted
+0.05–0.10 reach) at 80% power, two-sided α = 0.05, two-sample:

| σ assumption | provenance | n per arm | nightly windows needed |
|---|---|---|---|
| 0.074 | frozen-fork ledger, n=5 | ~14 | 28 |
| 0.108 | war-v1 ledger, n=3 | ~29 | 58 |
| 0.213 | frozen-fork χ² CI-hi | ~111 | 222 (> calendar) |

At every variance candidate the A/B is unpowered within a feasible window
budget (~110 submission days remain; 28–58 of them on one contrast is not
EV-maximal). **Decision (R12 option (a), adopted): the R2 ledger mechanism is
gated on build-rail currency only.** Primary evidence = offline compound
screen (paired Δlc with exact sign-flip p, AND Δlog1p(RHAE) ≥ 0) on ≥1-seed
Kaggle build-rail ledger-ON vs ledger-OFF (the N6 screen). LB windows serve
ledger **accumulation, not inference**; the only LB read is a descriptive
non-inferiority harm check: at n=5 ledger-ON draws, flag HARM if
mean(ON) < mean(control ledger) − 0.15. A null LB difference is pre-declared
uninformative and will not be cited as evidence in either direction.

## A5 — A3 variance gate restated on the CI upper bound (R12 methodology major)

The 2026-07-15 A3 rule ("LB windows live iff σ̂ < 0.15") at df=2 passes
σ_true = 0.20 with p ≈ 0.43 — no discriminating power. Restated rule,
effective immediately:

- **LB windows live iff χ²-CI-hi(σ) < 0.25 with df ≥ 4** (i.e., n ≥ 5 draws
  in the active arm's ledger).
- With df < 4 the gate is **indeterminate** and windows default to
  accumulation-only status (which they already hold under A4) — no
  standardized-effect claims may cite the ledger σ̂ until df ≥ 4.
- Operating characteristics at n=5 (df=4): P(CI-hi < 0.25 | σ=0.10) ≈ 0.72,
  P(pass | σ=0.15) ≈ 0.20, P(pass | σ=0.20) ≈ 0.03 — the gate can now fail
  informatively.

## A6 — R3–R5 unconditional start (R12 rl-planning major)

The reach table caps every currently-budgeted line (order stats ≤ +0.15;
warpack ≤ +0.10 conditional) below the 0.42–0.84 gap to the wall. The sole
wall-closing line is grinder cracking (R3 perception / R4 execution-contract /
R5 undo-probe).

- **R3 scoping is UNCONDITIONAL on the R2 outcome and began 2026-07-16** —
  concretely: the GPT-5.6-sol local probe (user-provided OpenAI key,
  `duck_eval/gpt56_probe/`, hard spend cap) runs the duck harness with a
  frontier analyzer against the local engines on ft09 + sb26 + su15 + lp85 +
  control. Its transcripts decompose model-capability vs harness-bottleneck
  per grinder and are the distillation source for the R3 perception pack and
  war-v3. Legality: local development only; only game-agnostic harness/prompt
  changes we author ship to Kaggle.
- R3 build work (perception pack implementation) starts no later than
  **2026-07-20** regardless of R2/N6 outcomes.

## A7 — Seed-audit consequence (R12 methodology major, resolved)

The unseeded-randomness sweep (`runs/seed_audit_2026-07-16.md`) returned PASS
(no unseeded policy RNG in the submission path; residual nondeterminism =
LLM sampling, identical in all arms). **No fix was applied → no ledger reset;
all existing ledgers remain valid estimates of future-draw variance.** The
seed-only-diff certification for war-eval v1/v2/v3 is delegated to the seed-3
screen artifact (`runs/war_eval_v3/`), which must attach the code diff verdict
before the Jul 17 pooled gate look consumes the three seeds.
