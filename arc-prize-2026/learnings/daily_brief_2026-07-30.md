# Daily brief — 2026-07-30

## 1. Result deep-dive (validated interpretation, not raw numbers)

### A17 v7 (72B seed-2 confirmation) — branch B2a fires: 72B route DEAD
- **Measurement (VALID READ, `runs/a17_v7_gate_look_2026-07-30.json`):** ΣN₇₂B(seed-2)
  = **5** (vc33 4×ACTION6, sb26 1, ft09 0, lp85 0), 4/4 games, windows 7920.0–7922.8 s
  (max drift 0.036%), seed=2 served-confirmed in env + kernel log, vLLM cmd byte-identical
  to v6 except the seed.
- **Sealed arithmetic (threshold seal 4ecf49a):** 5 < 138 ⇒ **B2a** — concordant with v6
  (ΣN=5 both seeds). ρ_action frozen-form 96.0 vs kill line 3.5. **72B route DEAD: no
  third seed, no fix lane. NC-10 DISCHARGED** (k=2, no changes between seeds).
- **Mechanism concordance (`learnings/a17_v7_concordance_2026-07-30.md`):** v7 reproduces
  the v6 format-livelock signature exactly — engine healthy the whole window (1011 HTTP
  200s, stall_s=0), native hermes tool calls **0/1008**, all 5 actions via fenced-recovery
  in the first ~96 s, degenerate byte-identical re-emission afterwards, actions frozen
  from t=720 s to window end. Per-game distribution shifted (expected sampling divergence
  at temp 0.6 under a seed change); the **mechanism is seed-invariant**. Pre-registered
  expectation (concordance, per the 07-29 diagnosis calling the livelock deterministic)
  **met**.
- **What this closes:** the entire 72B capability question on Kaggle-local serving. The
  sealed screen's C4 (Aug 3) is discharged early by kill-line arithmetic. Build priority
  formally reverts to the boristown readiness-gate A/B (named in the seal itself).

### Overnight scored draw
- **0.85 frozen-fork filler (07-30 00:07Z)** — interior draw, z ≈ −0.91 vs the frozen
  n=15 control. No band change, no drift signal, no trigger. Record ledger now n=16
  (mean 0.9650, s 0.1334); **A/B control stays frozen at n=15 (0.9727/0.1343)** per
  prereg §3. `runs/lb_ground_truth.md` refreshed from live API.
- LB head unchanged: KOJIMA 1.86; 1.61/1.60/1.58 band; gold cutoff still ≈1.49
  (#13–14 at 1.49, #15 1.48). Our best 1.33 (#51 band).

## 2. Discussions sweep (`learnings/sweeps/discussions_2026-07-30.md`)
- 2 new posts since 07-29: beginner intro (**IGNORE**), neuro-symbolic MDL self-promo
  with no level-completion evidence, −4 votes (**IGNORE** — same low-ΣN mirage genre we
  just root-caused locally). 3 bumped housekeeping threads, no new method. **No plan
  change.**

## 3. Research sweep (`learnings/sweeps/research_2026-07-30.md`)
- **ADAPT (vocabulary only): ToolFailBench (arXiv:2607.04686)** — documents open-weight
  models emitting zero clean tool calls on tool-required tasks ("Tool-Skip"), the closest
  published analogue to our 72B livelock. Folded into `learnings/a17_error_model.md`
  (addendum 07-30). Diagnosis-only ⇒ independently supports 72B-DEAD, does not reopen it.
- IGNOREs: Deco-G (constrained decoding degrades reasoning — corroborates why the
  forced-tool_choice boot PASS masked the livelock, off-mechanism otherwise), aTTT
  (per-episode LoRA, incompatible with zero-budget 9 h window), AGI Maze (reinforces
  latent-state-audit concern, nothing adoptable). **No plan change.**

## 4. Today's development (build-rail, weekday — no panel, A22 intent-files)
Priority per B2a reversion: **boristown readiness-gate A/B** (prereg DRAFT
`learnings/war_room/boristown_ab_prereg_2026-07-29_DRAFT.md`; seal target Sat 08-01,
ratify Sun panel R23 08-02, gated draws 08-02→08-05 if entry gates land in time).
Two blockers being discharged today by background agents:
1. **Entry-gate #1 (BLOCKER 3):** 2-seed live-firing eval BUILD of `canivel/arc3-duck-gate`
   + non-harm screen vs `runs/null10` — staging by sentinel precedent (agent in flight;
   push decision after readiness note lands; 0/2 pushes used, A17 lane freed the slot).
2. **Preflight T3 (BLOCKER 2, option b):** `--max-diff-cells 1 --pin <boris_16 sha>`
   extension to `scripts/preflight.py` so the one-cell graft is certified mechanically
   instead of by waiver (agent in flight; regression-tested against strict mode).

## 5. Open questions
- Does the entry-gate eval build need both seeds pushed same-day (2/2 budget) or split
  across 07-30/07-31? (Answer with the readiness note; either fits the 08-02 calendar.)
- NC-12 GPU parity: attach the metadata-level parity note to the entry-gate build
  artifacts for Sunday panel.
- Sentinel draw #2 remains queued strictly behind the A/B (backstop 08-10).
