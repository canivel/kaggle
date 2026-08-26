# A22 intent — boristown vLLM readiness-gate A/B on the frozen fork

**Status: DRAFT — NOT SEALED.** Build-rail intent only (A22: metric, canary, kill
rule). Nothing below is sealed; no sealed/amendment/prereg file is modified by this
document. **Scheduling decision deferred** per the 07-27 restructure: either a
weekday sealed arithmetic gate or the next Sunday panel (2026-08-02) must schedule
it before any slot is consumed. Responds to **R21 directive #1 (5/5 reviewers)** and
NC-6. Sources: `learnings/war_room/fork_diff_boristown_2026-07-24.md` (fork diff),
`runs/lb_ground_truth.md` (canonical ledger, refreshed 07-27),
`learnings/preregistration_amendment_2026-07-24_DRAFT.md` §(i) (fork policy),
`learnings/panel/round21/_directives.md`.

## Arms

- **A (control):** frozen duck fork, byte-identical filler (`canivel/arc3-duck-repro`
  v3 lineage). Control draws are the ongoing frozen-ledger filler draws (n=13 as of
  07-27; new fillers accrue during and around the test).
- **B (gated):** frozen fork + the **single audited diff** from the boristown 1.47
  artifact — nothing else. Per the fork-diff memo, 12/22 boristown cells are
  md5-identical to ours including every load-bearing cell; the only functional diff
  is the readiness gate below.

## Exact code delta (from runs/fork_diff_boristown, boris cell 16)

One standalone ~25-line cell, `wait_vllm_ready()`, inserted immediately **before**
the benchmark/run cell: polls `http://127.0.0.1:1234/v1/models` every 5 s, up to
180 s, and raises if vLLM never comes up. Closes a startup race the frozen fork has:
our fork waits for the **gateway** (`_wait_for_gateway`, 600 s) but never for the
**vLLM server**, which the solver setup commands launch async. Zero interaction
risk: runs before `bm.run(...)`, only polls localhost, no score-sensitive surface,
no new packages/datasets/keys. Anchor cells for the graft are byte-identical across
both artifacts (fork-diff §c).

## Hypothesis

- **H1 (tested):** the gate shifts the scored-draw distribution up (mechanistic
  story, 5/5 panel: without it, early-episode actions are burned while vLLM is still
  loading the 27B FP8 model — plausibly cold-start variance across the 0.82–1.33
  band). Anchor-implied effect if the gate explained boristown's 1.47:
  δ ≈ 1.47 − 0.974 ≈ **+0.50**.
- **Prior (on record, fork-diff §a):** honest EV is *floor-raise, not mean-shift* —
  1.47 is a right-tail draw of ≈ our distribution; "+0.14 is NOT budgeted as
  systematic."

## Metric, n, decision rule

- **Metric:** public-LB scored draw per window (completion-weighted, API-verified
  into the arm ledger before any rule fires — NC-8).
- **n = 4 gated draws** (panel band 3–5), consecutive scored slots once scheduled;
  control = frozen ledger n = 13 (mean 0.974, s 0.143; API-verified 07-27) plus any
  interleaved fillers. Stationarity of the control is on record (MK p = 0.47, CUSUM
  p = 0.72), so consecutive gated slots are acceptable.
- **Decision rule (one-sided, α = 0.05):** PROMOTE iff
  x̄_B(4) ≥ 0.974 + 1.645 · 0.143 · √(1/4 + 1/13) = **x̄_B(4) ≥ 1.108**.
- **σ̂ note:** rule frozen on s = 0.143 (frozen n = 13). Sensitivity: earlier ledger
  states give s = 0.156 (n = 10) / 0.144 (pooled n = 15) → threshold range
  1.108–1.120; the rule uses the canonical refreshed value.

## Error rates (stated pre-draw)

- **False positive (B ≡ frozen):** 5.0% Gaussian by construction; ≈ 6–7% under the
  declared t-predictive (ν = 12). 
- **Power:** at the anchor-implied δ = +0.50: > 0.999 (the rule cannot miss the
  effect the panel priced). At a modest floor-raise δ = +0.10: ≈ 34%; δ = +0.15:
  ≈ 58% — honest note: a MISS does not rule out a small floor effect, it rules out
  the anchor-sized one.

## Entry conditions (same bar as A21 entry)

1. **2-seed eval canary:** build COMPLETE on both seeds; banner echoes the gate
   ("GATE armed"); log shows the gate *observed firing* (poll count + vLLM ready
   latency ≤ 180 s). Smoke-test the graft cell pre-push
   (feedback_test_before_submit).
2. **Non-harm screen vs `runs/null10`:** mechanism fires AND Δ levels-completed not
   materially negative (same criterion the sentinel screen used).
3. Gated draws enter **their own arm ledger only** (pooling rule (b)(3) of the
   07-24 DRAFT: composition has a live mechanism diff — never frozen, never pooled,
   until closure + equivalence memo).

## Harm-pause and kill

- **Harm-pause:** any gated draw < 0.80 pauses the arm (A21/C2, sealed). Per-draw
  false-fire ≈ 11% Gaussian / 13% t-predictive under B ≡ frozen. Note the gate's
  whole mechanism is left-tail removal, so a pause is *evidence against* H1, not
  just exposure control.
- **PROMOTE ⇒** NC-6 fires as the panel wrote it: "the entire daily-filler regime
  switches to the gated variant"; the gate additionally becomes a default hygiene
  graft in all lineages (07-24 DRAFT §(i).1) and the changepoint monitor arms for
  the first 5 post-gate draws (§(i).3 stratum-split trigger).
- **MISS/kill ⇒** the mean-shift claim is dead: +0.14 confirmed non-systematic
  (variance hypothesis stands); no regime switch on this evidence. The *hygiene*
  question (free left-tail insurance at zero interaction risk) is explicitly NOT
  killed by a MISS — it returns to the Sunday panel on the fork-diff evidence
  alone. Fresh-slug byte-fork decision likewise returns to panel.

## Slot budget

4 scored draws, all from the **daily scored slot** (one/day, ARCDailySubmit) — no
extra submissions, no GPU-h beyond the normal window, $0 cloud. Opportunity cost
≈ 4 × 0.0006 ≈ 0.0024 E[max]-equiv (declared-model pricing).

## Slot-priority order (shared verbatim with intent_exploration_draw2.md)

0. Sealed obligations pre-empt on their pre-registered dates (C4 lines; A17 v6
   scored bench).
1. **Boristown readiness-gate A/B** — n = 4 consecutive gated slots once scheduled.
2. **A21 exploration draw #2 (sentinel)** — first slot ≥ 2 days after the A/B
   disposition memo is filed.
3. Frozen-fork filler (default; affirmatively a strategy at P(touch 1.44) ≈ 0.33).

Justification: directive #1 carries 5/5 reviewers vs 3/5 for directive #4; the A/B
outcome (NC-6 regime switch) would change the default filler composition that draw
#2 is scored against, so sequencing A/B first avoids a mid-arm stratum split; the
resulting delay to draw #2 is ≤ ~5 windows ≈ 0.003 E[max]-equiv — immaterial.

---
*Draft prepared 2026-07-27 (R21 directive #1). NOT SEALED. Do not queue, push, or
submit on the basis of this document alone.*
