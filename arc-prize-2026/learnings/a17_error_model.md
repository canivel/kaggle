# A17 symmetric error model — C3 filing (2026-07-23, BEFORE bench push)

Discharges **C3** of `learnings/preregistration_amendment_2026-07-23.md`:
pre-register BOTH error probabilities of the A17 screen rule under the
verified draw distribution, before any 72B number is observed. Envelope
NO-GO (>3.5× penalty) self-certifies and is outside this model.

Machinery: `runs/a17_error_model/a17_error_model.py` (Monte Carlo, B=200,000
per cell, seed 20260723, + exact enumeration cross-check); inputs =
`runs/a17_repair/per_seed_table.json` (frozen 07-22 from raw benchmark.json,
consistent with `runs/verify_2026-07-21/report.md`). Artifact:
`runs/a17_error_model/a17_error_model.json`.

## 1. The rule being modeled (sealed, a17_72b_screen_scope.md)

GO iff **CAPABILITY** [Σ_g (72B per-game MAX lc over k seeds) ≥ Σ_g (27B
per-game MAX lc) + 2 = **8**] AND (**ACTION-PARITY** [ΣN₇₂B ≥ 0.90·ΣN₂₇B]
OR **THROTTLED** [Σ_g 72B MAX lc ≥ Σ null_adj(ρ) + 1]). Games
ft09/sb26/lp85/vc33; 27B side = per-game MAX over the certified seeds
(= {2,1,1,2}, Σ=6). At the ρ anchors 2.5/3.0, ACTION-PARITY is structurally
FALSE (it requires ρ ≤ 1.11), so within the modeled regime **GO ⇔ Σ_g 72B
MAX lc ≥ 8** (the throttled bar 5 (ρ2.5) / 4 (ρ3.0) is dominated by the
capability bar 8).

## 2. Pre-registered draw model

- 72B pseudo-seed = one certified row drawn uniformly from
  {war_eval_v1, war_eval_v2, war_eval_v3, w0_eval_s1} (row-wise, preserving
  within-seed correlation), mapped to the throttled regime by the frozen
  cumulative walk (throttled_lc at ρ). Throttled rows at ρ=2.5:
  v1/v2/v3 = {0,1,1,1} (Σ3), W0 = {0,1,1,2} (Σ4).
- **True lift +L** = L extra completed levels achieved in-window beyond the
  throughput-matched null (i.e. lift on the achieved/in-window scale — the
  scale the gate observes). Allocation: *uniform* (each level lands on a
  uniformly random game, capped at number_of_levels {6,8,8,7}) and
  *concentrated* (all L on one random game) — the latter is the most
  detection-favorable for a MAX statistic.
- k = 1 (the planned scored bench) and k = 2 (marginal-rule expansion); a
  "procedure" variant models the sealed marginal rule (extra seed drawn when
  the statistic lands within one level of either bar), approximated as a
  full-4-game extra seed — this can only raise P(GO), so its false-GO is an
  upper bound (conservative for the error the rule guards against).

## 3. Results (P from 200k MC; k=1 cells confirmed by exact enumeration)

### P(false NO-GO | true lift = +L)   [C3 required: L ∈ {+1,+2,+3}]

| ρ | scheme | k=1 | k=1+marginal | k=2 |
|---|---|---|---|---|
| 2.5 | uniform, L=+1 | **1.000** | 1.000 | 1.000 |
| 2.5 | uniform, L=+2 | **1.000** | 0.905 | 0.905 |
| 2.5 | uniform, L=+3 | **1.000** | 0.333 | 0.333 |
| 2.5 | concentrated, L=+3 | 1.000 | 0.251 | 0.251 |
| 3.0 | uniform, L=+1 | **1.000** | 1.000 | 1.000 |
| 3.0 | uniform, L=+2 | **1.000** | 1.000 | 1.000 |
| 3.0 | uniform, L=+3 | **1.000** | 0.926 | 0.564 |
| 3.0 | concentrated, L=+3 | 1.000 | 0.812 | 0.251 |

### P(false GO | true lift ≤ 0)   [C3 required]

**0.000 in every cell** (L = 0 and L = −1; both schemes, k ∈ {1,2},
procedure variant included; exact-enumeration 0 at k=1). Structural: the
null throttled distribution is capped at Σ=4 (ρ2.5) / 3 (ρ3.0), far below
the capability bar 8 — no draw noise can fake a GO.

### Detection frontier (smallest L with P(GO) ≥ 0.75)

ρ=2.5: k=1 → **L=+5** (L=+4 gives P(GO)≈0.25); k=2 uniform → **L=+4**
(P≈0.98). ρ=3.0: k=1 → **L=+5** (P≈0.75); k=2 uniform → **L=+4** (P≈0.91).

## 4. Pre-registered interpretation (filed before the bench)

1. **The screen is maximally asymmetric by construction.** False GO ≈ 0;
   false NO-GO = 1.0 (structural, not sampling) at k=1 for every true lift
   ≤ +3, because a single throttled seed can reach at most Σ = 4+3 = 7 < 8.
   The screen answers "does a ≥ +4-to-+5-level capability jump exist?" —
   the wall-closer question — NOT "is the 72B somewhat better?". A NO-GO
   with observed lift ≤ +3 is the designed outcome for a modest-lift world
   and must not be re-litigated post-hoc as a screen failure; conversely a
   GO is near-unimpeachable (P(false GO) ≈ 0).
2. **A +1..+3-level 72B advantage is invisible to this screen.** If the
   campaign later wants to detect modest lifts, that is a different,
   powered instrument (more seeds, mean-based statistic) and requires its
   own pre-registration — not a reinterpretation of this one.
3. **Marginal-rule dead zone (flagged, rule unchanged):** at ρ ∈ (2.6, 3.5]
   (bar 4) an observed Σ = 6 triggers neither the marginal rule (it is 2
   away from both bars) nor GO — the extra-seed escape is unavailable
   exactly one level below where it is at ρ ≤ 2.5. Filed for the panel; the
   sealed rule text stands.
4. Cells assume the expected throttled world (ρ ∈ [2.4, 3.1]); if measured
   ρ < 1.11 the parity branch re-opens (easier regime, errors strictly
   smaller than reported false-NO-GO); if ρ > 3.5 the envelope NO-GO
   self-certifies (C3).

## Addendum 2026-07-30 — post-mortem vocabulary (ToolFailBench)

The measured failure mode that killed the 72B route (v6/v7, both seeds:
zero native tool calls in ~1000+ free-form turns despite healthy engine,
actions frozen after the first ~90 s) matches the **"Tool-Skip"** failure
class of ToolFailBench (arXiv:2607.04686): the model produces tool-style
prose but never emits an executed call, documented there for open-weight
models on tool-required tasks. Adopted as vocabulary only (the paper offers
diagnosis, not mitigation) — it independently corroborates branch B2a
(72B route DEAD) rather than reopening it. Full local evidence:
`learnings/a17_v6_diagnosis_2026-07-29.md`,
`learnings/a17_v7_concordance_2026-07-30.md`,
`runs/a17_v7_gate_look_2026-07-30.json`.
