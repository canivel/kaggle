# Daily brief — 2026-07-23

## §1a Result deep-dive

### Filler draw

**Draw:** frozen-fork filler = **0.82** (00:07Z daemon fire, on schedule — third
consecutive clean gate pass for the audit-stub fix). Band-typical lower half.

**Pre-registered expectation:** plain draw from the frozen distribution. **Met.**
No mechanism claim; no kernel pull (filler runs are not evidence artifacts).

**Ledger update:**
- Frozen control n=10 {0.82, 0.89, 0.93, 1.02, 0.95, 1.33, 0.92, 0.93, 1.14,
  0.82}: mean **0.975**, σ̂ **0.156**.
- Pooled (frozen + closed war arm) n=15: mean **0.962**, σ̂ **0.144**.

**Interpretation:** distribution is stable and boring; nothing new. The real
result news today is (i) the sentinel W1 live run (below) and (ii) **rank
erosion**: we slid #44 → #45 at an unchanged 1.33 best (§1b). Filler holds the
line but loses rank as the field compresses upward — every credible experiment
window remains better-priced than filler.

### Sentinel W1 (seed 1) — live-run deep-dive (runs/sentinel_eval_analysis/report.md)

**Verdict: mechanism PASS / score NULL / behavior STRONG-NEGATIVE — "fires,
doesn't pay" (build-doc Open Risk #2 realized).**

- **Score prong:** pre-registered lift (+0.06–0.12 window pricing; build-doc
  honest +0.01–0.03/draw) **NOT met**: sentinel mean 0.855 vs war 3-seed
  baseline 1.454 (−0.60; −0.72 vs paired seed-1). But the gap is carried almost
  entirely by three high-variance NON-target games (ar25/ft09/sp80 ≈ the whole
  raw gap) and baselines span 1.16–1.73 → call is **NULL/underpowered at n=1**,
  not established regression. (Frozen LB band 0.99 is a different metric scale —
  not comparable.)
- **Mechanism prong: clean PASS.** 22 sidecars + 56 stdout `SENTINEL v=2` lines
  agree exactly; every threshold ≤once/game; cumulative game-envelope keying
  proven (identical fire-actions 75/113/135 across games; no re-arm across
  attempts). Open Risk #1 (inert-if-uncapped) cleared. The v1→v2 re-key is
  verified live on carriers ka59/re86 (fired 3/3 early where v1 was blind).
- **Missing sidecars (s5i5/tu93/vc33): EXPECTED** — sidecar is lazily created on
  first crossing; all three ended <75 actions. File-exists ⇔ ≥1 fire.
- **Behavioral effect: the warnings did nothing.** 1/22 fired games advanced
  after first warning; 21/22 kept grinding (wa30 560 actions stuck L1 after 3
  warnings); total actions UP +618 vs baseline. **tu93's 3.97/50-actions is NOT
  a sentinel win** (zero fires there — lucky efficient draw); flagging so it is
  never claimed as mechanism evidence.
- **Condition-4 envelope: PASSES on this pull** (tokens/game mean 64.3k, 23/25
  within 63k ±15%; B=150 needs no re-derivation).
- **Recommendation for R17:** seal the mechanism half; record the score prong as
  NULL with the fires-doesn't-pay label — (a) is **not a lift contributor** on
  this evidence. **W2 = $0 confirmatory-null free build** (pre-registered:
  expected mean inside 1.16–1.73, mechanism clean, behavior unchanged); no W3
  unless W2 surprises positive.

## §1b Discussions sweep (learnings/war_room/discussions_2026-07-23.md)

- **ADOPT (plan-relevant, loud): #728299 "Reading the score exactly"** (Busya
  PRIME) — dissects the shipped `arc_agi/scorecard.py`, verified to 1e-9.
  Resolves the host-silent 1.15x watch-item **from code**: 115 is a per-level
  cap; LB aggregate is **completion-weighted with a completion cap**. Hard
  consequences: **depth ≫ efficiency** (unreached level costs its weight twice;
  4/6 levels = 47.6 not 66.7) and **overshoot decays quadratically** (2× baseline
  actions = 25%, not 50%). Re-points EWM/A17 objectives at deeper levels, not
  action-trimming — and independently explains why the sentinel's stop-grinding
  signal cannot buy score unless the freed actions convert into level depth.
  Also ships a **no-API-key offline scoring atlas** of all 25 bundled games —
  adopt as free deterministic local scoring oracle.
- **Validation (no code):** #727505 Yakunin "Constraint Before Control"
  converges on our EWM+verifier topology; his weak results (0.17) died on
  runtime-env integration, not the idea. Borrowable: ternary
  required/forbidden/irrelevant hypothesis tagging.
- **LB: field compressed upward.** Old 1.44 wall is now the BOTTOM of a dense
  1.44–1.60 band; gold ~1.47; KOJIMA 1.86. **We slid #44→#45 at 1.33** — pure
  erosion. No new clones of our artifact above 1.39 (zoli800 still top public).
- **Watch:** #728220 asks when `arc_agi` bumps to 0.9.9 — silent scorer/env
  drift risk; preflight version assert stays mandatory. IGNORE: schema-harness
  self-report thread (closed-source, no semi-private number).

## §1c Research sweep (learnings/war_room/research_2026-07-23.md)

- **No new ADOPT; the window's substance is critique, all cutting in our favor.**
- **ADAPT (high) — arXiv 2607.12227** "Rethinking the Evaluation of Harness
  Evolution": held-out rebuttal of tune-on-public/report-on-public harness
  evolution (+0.6 avg on held-out Terminal-Bench). External charter for our
  gate: **beat-null10-on-held-out, never beat-baseline-on-tuning-games**; cite
  one-line in the R17 gate rationale.
- **ADAPT (medium) — arXiv 2606.24842** "World Models in Pieces": certification
  is **transition-local**, not model-global. Reframes the holdout collapse as a
  normal outcome (sb26 = the one transition-local certificate that generalized)
  and tightens EWM v1.1 wording: BFS-in-sim is sound only over transitions
  carrying a live local certificate.
- **Schema follow-ups:** no independent replication; HF traces unchanged;
  Kamradt critique (Jul 21) = warning label for A17 escalation economics —
  **serving-cost-only policy; per-game score feedback must never re-enter the
  agent's context** (document in A17′ amendment).
- **A17 serving:** still no independent 96GB-fit confirmation for 72B-AWQ/W8A8
  with vision tower + KV; settle empirically at the pre-Aug-1 screen.

## §2 Today's plan

**R17 panel (sealing round, full 5 reviewers, round16 priors).** Circulation =
`grinder_design_R17_sealing.md` (filed 07-22, all 9 checklist items discharged)
+ **addendum with post-filing evidence**: sentinel W1 live-run verdict
(mechanism-seal + score-NULL recommendation), scorer-dissection
(depth≫efficiency; completion-weighting), the two research ADAPTs, and the
W2-confirmatory-null proposal. After panel: implement directives; queue is
pre-filled with frozen-fork filler (submitted head unless panel redirects to a
validated alternative before 19:00 EDT).

## Open questions
1. Does R17 seal A14 with the sentinel score prong recorded as NULL, or does any
   reviewer treat fires-doesn't-pay as gate-relevant (the sentinel was a
   warn-only observable — lift was never its certified function)?
2. Depth≫efficiency re-pricing: does EWM v1.1's value proposition survive when
   the objective is deeper levels rather than fewer actions? (Schema's evidence
   says yes — their wins came from BFS-to-deeper-levels in-sim.)
3. Fork-wave/rank erosion at 1.33: when does a fresh-slug variance clone of the
   frozen artifact become better-priced than another filler draw? (Priced
   option, still not queued.)
4. W2 confirmatory null: push today (2/2 budget unused) or defer to after A17
   screen prep?
