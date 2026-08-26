# State of the War — 2026-07-18 deep-understanding synthesis

Inputs: `lb_process_model/report.md`, `winners_deepread_2026-07-18.md`,
`opine_world_deepread.md`, `grinder_cracking_design.md`, panel R13,
amendments A8–A13, GPT-5.6 probe + distill corpus. Panel target: R14/R15.

## What we now KNOW (high confidence, instrumented or replicated)

1. **The harness transfers; the analyzer is the gap.** GPT-5.6-sol through our
   unmodified scaffolding: ft09 5/6, sb26 5/8, lp85 4/8 vs Qwen's ~1 level.
   (Probe, 2026-07-16.)
2. **All 25 games are frame-deterministic** (N5 audit 0/25 divergent), and the
   published 20/25 OPINE-World result is built on exactly that property.
3. **The LB draw distribution is generatively explained by our own bench** —
   no hidden deep-play regime. A 1.33 night needs the measured *common-night
   correlation* (shared server/sampling luck across the 110 slots), and is
   44% ft09-level-2. σ̂=0.074 was a lucky-tight n=5 sample of a σ≈0.13–0.17
   process. (lb_process_model, 20k-night sims, exact scorer.)
4. **Honest window pricing:** E[max@107 remaining] ≈ 1.39 central (pooled-10
   posterior predictive); P(touch 1.44) ≈ 0.29; P(reach 1.86) ≈ 0.01.
   **Filler is a lottery ticket, not a plan.** Break-even for spending a
   window on an experiment: credible official-set lift ≥ +0.06–0.12. The
   existing +0.12 gate thresholds already price this correctly.
5. **Mechanical no-effect refutation + verify-before-act are THE convergent
   scaffolding primitives** — independently arrived at by Reki (dead-signature
   veto, his 0.64→0.86), the 3rd-place build, OPINE's counterexample loop,
   and our GPT-5.6 distillation; and their absence is exactly why our ledger
   idled (1552 digests / 0 escalations, prose triggers never fire).
6. **The 1.44+ band is not explained by any public artifact.** Winner-tier
   public code tops out at 0.86–1.21; Tufa credits "multimodality + better
   base models"; ~14 wall-breakers share nothing. No public competitor shows
   OPINE-style executable world models.
7. **Our fork is drift-free** vs the public duck notebook (retires the Jul-16
   re-fork question).
8. **The v3 micro-stack cannot close the wall**: real-scorer counting bounds
   give ceiling +0.31 rail / +0.17 LB, expectation +0.04–0.10 LB
   (grinder_cracking_design.md). Reclaimed actions on uncompleted levels
   score zero.

## What we BELIEVE (medium confidence, one good source or inference)

- **su15 is NOT an information-theoretic wall** — OPINE solved it 9/9. Our
  A13 re-probe is now expected-retraction (the wall verdict rested on a
  deadlock-confounded probe). ka59/sk48/lf52/bp35/s5i5 (OPINE's failures)
  are *search-budget* walls, not world-model walls — and our exec_wm sims
  are already saturated on lf52 (100%) and s5i5 (99.5%).
- The 1.86 leader is most plausibly an OPINE-family system or a
  frontier-tier analyzer; either way, per-draw mean — not draw volume — is
  what they have.
- Night-level correlation (shared vLLM/sampling temperature luck) is real
  and material to LB variance; deterministic build-rail RNG (3rd-place
  trick) buys gate power without sacrificing LB order-stats value.

## What we DON'T KNOW

- Whether the plan-execute-verify contract survives contact with the scored
  regime's budgets (A10 bench must fire its triggers first).
- Whether a 72B-tier AWQ analyzer nets positive after the 2.5–3× throughput
  penalty (war-v4's central risk; INT4-vs-FP8 reasoning quality on grinders
  unbenchmarked).
- What the sealed A5/A8 look will say tonight at n=5 (no peeking).

## The strategy stack this implies (for R14/R15 to ratify)

| priority | line | basis | ceiling/expectation | cost |
|---|---|---|---|---|
| 1 | **EWM-execute: OPINE plan-execute-verify contract on our 12 saturated exec_wm sims** (harness-side BFS, one action/step, hash-verify vs settled frame, fail-closed) | KNOW#2, BELIEVE#1, uncontested edge (KNOW#6) | ceiling +0.5/draw rail — exceeds entire v3 stack; expect +0.10–0.30 | 2–3 days, 0 LLM tokens, own gated window |
| 2 | **(c)+(d) flag with Reki dead-signature as the click component** | KNOW#5 (quadruple convergence) | ceiling ~+0.10 rail | 1–2 days, already spec'd |
| 3 | **war-v4 model swap scoping** (72B AWQ, free rail bench first) | KNOW#1, KNOW#6 | the only proven wall-sized lever | scoping Aug 1, gated |
| 3b | mixed-tier routing (27B grinds / 72B consulted) — bench row ONLY inside v4 scoping | Kimi-3 review 07-18 | contingent on throughput binding | costs incl. FRONT-LOADED penalty: two model loads before action 1 + split KV cache, not just per-decode; dual-serve in 9h unproven; simplicity-wins prior against |
| 4 | su15 GPT-5.6 re-probe post-(f) (A13; expected retraction) | BELIEVE#1 | epistemic repair | ~$10, capped |
| 5 | Filler in every window no experiment credibly beats (+0.06–0.12 rule) | KNOW#4 | lottery: ~29% touch 1.44 over 107 | free |

Retired/parked: warpack (A9 double-lock), ledger-as-built (A11), prompt-line
transfer (distill-proven inert), r11l as a variance story (contributes ~0).

## Window discipline (unchanged, now priced)

One flag per window (A12). A window goes to an experiment only when its
pre-registered expected lift ≥ +0.12 official-set equivalent; otherwise
filler. Tonight: war draw #5 (committed, completes n=5 → sealed look).
