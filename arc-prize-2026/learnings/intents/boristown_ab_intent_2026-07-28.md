# A22 intent — boristown vLLM readiness-gate A/B on the frozen fork (2026-07-28)

**Status: DRAFT — NOT SEALED.** Build-rail intent only (A22: metric, canary, kill
rule). Nothing below is sealed; no sealed/amendment/prereg file is modified by this
document. Responds to **R22 directive D2 (5/5 unanimous, carried from R21 #1):**
"Schedule the boristown readiness-gate A/B this week — replace filler draws, do not
arbitrate false scarcity against sentinel draw #2." Supersedes the undated draft
`learnings/intents/intent_boristown_readiness_ab.md` (same arms/rule; this version
is dated, ledger-refreshed to n=14, and points at the built canary artifacts).

Sources: `learnings/war_room/fork_diff_boristown_2026-07-24.md` (fork diff),
`runs/lb_ground_truth.md` (canonical ledger, refreshed 07-28),
`learnings/panel/round22/_directives.md` (D2 + Q2 disposition),
`learnings/preregistration_amendment_2026-07-24_DRAFT.md` §(i) (fork policy).

## What the readiness-gate diff actually is

Arm B = the **frozen duck fork** (`canivel/arc3-duck-repro` v3 lineage; the exact
bytes we score fillers on) **plus the single audited functional diff** from the
boristown 1.47 artifact — the `wait_vllm_ready()` cell (boristown cell 16), a
standalone ~25-line health check inserted immediately **before** the benchmark run
cell that polls `http://127.0.0.1:1234/v1/models` every 5 s (≤ 180 s) and raises if
vLLM never comes up. It closes a startup race the frozen fork has: our fork waits
for the **gateway** (`_wait_for_gateway`, 600 s) but never for the **vLLM server**,
which the solver setup commands launch async — so early-episode actions can be
burned while the 27B FP8 model is still loading. **Nothing else changes** (the fork
diff memo established 12/22 boristown cells are md5-identical to ours, including
every load-bearing cell; cell 16 is the sole functional delta).

### Diff ambiguity resolution (per directive: pick highest-anchor-value change)

The fork-diff memo is **unambiguous**: boristown cells 8–11/15 are deliberate
no-ops (their markdown describes an ACTION7 / animation-metadata monkey-patch that
the code *rolled back* — every `*_changed: False`), cell 21 is a display-only score
card gated off under `TRUE_SUBMISSION`, and cell 16 is "the only functional diff"
(memo table). So there is no multi-candidate ambiguity to arbitrate. Alternatives
considered and rejected as carriers: (a) the no-op patch cells — inert, would test
nothing; (b) the score-card cell — cannot affect a scored rerun. The
highest-anchor-value change **is** the readiness gate, and it is the only one.

## Arms

- **A (control):** frozen duck fork, byte-identical filler (`canivel/arc3-duck-repro`
  v3). Control draws = the ongoing frozen-ledger filler draws (n=14 as of 07-28;
  new fillers accrue during and around the test).
- **B (gated):** the built canary `canivel/arc3-duck-gate` (staged locally at
  `notebooks/duckgate/arc3-duck-gate.ipynb`, built by
  `duck_eval/a17/build_boristown_gate_canary.py`, smoke
  `duck_eval/a17/boristown_gate_smoke.py` = **47/47 PASS**). Fresh slug per
  feedback_fresh_kernel_slug; env metadata byte-identical to the frozen fork family.

## Exact code delta (built + smoke-verified)

One standalone cell inserted at code-cell position immediately before the
`await bm.run(...)` cell. Its `wait_vllm_ready()` body is **byte-identical to
boristown cell 16** (asserted at build time and in smoke S6); the only additions
are two additive `A17-GATE` banner prints — an "armed" line (endpoint/poll/timeout)
and an "observed-firing" line carrying the vLLM-ready latency — which satisfy entry
gate #1's "log shows the gate observed firing" without touching the polling logic.
Zero interaction risk: runs before `bm.run(...)`, only polls localhost, no
score-sensitive surface, no new packages (`requests` is already in the image),
datasets, or keys. The customization-hook cell (frozen fork cell 12) is left
byte-identical — **no solver graft**, unlike the sentinel arm; this is what keeps
the A/B a clean single-variable causal test.

## Hypothesis

- **H1 (tested):** the gate shifts the scored-draw distribution up (5/5 panel
  mechanistic story: without it, early-episode actions are burned while vLLM loads
  the 27B FP8 model — plausibly the cold-start component of the 0.82–1.33 band).
  Anchor-implied effect if the gate explained boristown's 1.47:
  δ ≈ 1.47 − 0.9686 ≈ **+0.50**.
- **Prior (on record, fork-diff §a):** honest EV is *floor-raise, not mean-shift* —
  1.47 is a plausible right-tail draw of ≈ our distribution; "+0.14 is NOT budgeted
  as systematic." The gate raises the **left tail** more than the mean.

## Metric, n, decision rule

- **Metric:** public-LB scored draw per window (completion-weighted, API-verified
  into arm B's own ledger before any rule fires — NC-8).
- **Control:** frozen ledger **n = 14, mean 0.9686, s ≈ 0.1384** (API-verified
  07-28) plus any interleaved fillers. Stationarity on record (MK p ≈ 0.47, CUSUM
  p ≈ 0.72), so consecutive gated slots are acceptable.
- **n = 4 gated draws** (panel band 3–5), consecutive scored slots once scheduled.
- **Final decision rule (one-sided, α = 0.05, Gaussian):** PROMOTE iff
  x̄_B(4) ≥ 0.9686 + 1.645 · 0.1384 · √(1/4 + 1/14)
  = 0.9686 + 1.645 · 0.1384 · 0.5976 = **x̄_B(4) ≥ 1.105**.
- **σ̂ note:** rule frozen on the canonical refreshed s = 0.1384 (n = 14). Earlier
  ledger states (s = 0.143 / 0.156) give thresholds 1.108–1.120; the difference is
  immaterial to any plausible outcome.

## Sequential stopping rule

Draws are read one per daily scored slot; the rule fires at these checkpoints:

1. **Harm-pause (per draw, sealed A21/C2):** any gated draw **< 0.80** pauses the
   arm pending panel review. Per-draw false-fire ≈ Φ((0.80−0.9686)/0.1384) =
   Φ(−1.22) ≈ **11%** under B ≡ frozen. Note: the gate's whole mechanism is
   left-tail removal, so a pause is *evidence against* H1, not just exposure
   control.
2. **Early PROMOTE (optional, conservative):** if any single gated draw ≥ **1.44**
   (the gold-cutoff-adjacent exceedance used across the exploration program), flag
   for panel — a single anchor-class draw is decision-relevant on its own, but the
   n = 4 mean rule remains the sealed test (no auto-promote on one draw).
3. **Final disposition at n = 4** via the PROMOTE rule above. If x̄_B(4) < 1.105 ⇒
   **MISS/kill:** the mean-shift claim is dead; +0.14 confirmed non-systematic
   (variance hypothesis stands); no regime switch on this evidence. **The hygiene
   question** (free left-tail insurance at zero interaction risk) is explicitly
   **NOT** killed by a MISS — it returns to the Sunday panel on the fork-diff
   evidence alone (07-24 DRAFT §(i).1).
4. **PROMOTE ⇒** NC-6 fires as written ("the entire daily-filler regime switches to
   the gated variant"); the gate additionally becomes default hygiene in all
   lineages and the changepoint monitor arms for the first 5 post-gate draws.

## Error rates (stated pre-draw)

- **False positive (B ≡ frozen):** 5.0% Gaussian by construction; ≈ 6–7% under the
  declared t-predictive (ν = 13).
- **Power:** at the anchor-implied δ = +0.50: > 0.999. At a modest floor-raise
  δ = +0.10: ≈ 34%; δ = +0.15: ≈ 58%. Honest note: a MISS rules out the
  anchor-sized effect, not a small floor effect.

## Entry gates (same bar as A21 exploration-draw entry)

Per `learnings/preregistration_amendment_2026-07-23.md` and the sentinel precedent,
entry to a scored draw = **2-seed canary PASS + non-harm screen**:

1. **2-seed eval canary:** build COMPLETE on both seeds; banner echoes the gate
   ("A17-GATE ... : GATE armed"); log shows the gate *observed firing*
   (`A17-GATE observed-firing vllm_ready_latency_s=... : GATE fired`, latency
   ≤ 180 s, plus boris's own "vLLM server ready" line). Graft smoke-tested pre-push
   (feedback_test_before_submit): **`boristown_gate_smoke.py` = 47/47 PASS** on the
   staged notebook.
2. **Non-harm screen vs `runs/null10`:** mechanism fires AND Δ levels-completed not
   materially negative (same criterion the sentinel screen used).
3. Gated draws enter **arm B's own ledger only** (pooling rule (b)(3) of the 07-24
   DRAFT: composition has a live mechanism diff — never frozen, never pooled, until
   closure + equivalence memo).

**Build-rail status against these gates (today):** entry gate #1 is *partially*
discharged — the canary is built, banners present, smoke 47/47, single-diff
invariant ALLOW (local preflight-equivalent T3). The **2-seed live-eval firing**
(the log-observed-firing half) and the **non-harm screen** are NOT yet run; those
require an eval-kernel BUILD (free, zero-spend), not a scored push. So the canary is
**staging-complete but not yet entry-gate-complete**.

## Slot plan — which filler draws it replaces

Per D2 ("readiness-gate A/B starts immediately at ~n=4 gated draws replacing the
next 4 fillers, zero incremental slots; sentinel draw #2 queued behind it; filler
cut to ~2/week"):

- **Replaces the next 4 daily-filler scored draws** (one/day, ARCDailySubmit) — no
  extra submissions, no GPU-h beyond the normal window, $0 cloud. Opportunity cost
  ≈ 4 × 0.0006 ≈ 0.0024 E[max]-equiv.
- **Nominal calendar (systems' ~Aug 2 anchor):** entry-gate canary BUILD 07-29;
  gated draws replace fillers on the first 4 scored slots on/after gate completion
  — nominally **2026-07-30 → 2026-08-02** if the eval canary lands 07-29, else
  slides one slot per day of canary slip. A/B disposition memo filed the day after
  draw #4.
- **Sentinel draw #2 queues behind** (its intent's backstop 08-10 caps starvation).

## Slot-priority order (shared verbatim with intent_exploration_draw2.md)

0. Sealed obligations pre-empt on their pre-registered dates (C4 lines; A17 v6
   scored bench).
1. **Boristown readiness-gate A/B** — n = 4 consecutive gated slots once scheduled.
2. **A21 exploration draw #2 (sentinel)** — first slot ≥ 2 days after the A/B
   disposition memo is filed.
3. Frozen-fork filler (default; cut to ~2/week per D2).

## Built artifacts (build-rail, staged locally — NO push)

- Build script: `duck_eval/a17/build_boristown_gate_canary.py` (fork-never-build:
  forks `notebooks/duckfork/tufa-labs-duck-harness-june-30-milestone-winner.ipynb`;
  anchor-exact; deterministic-from-base idempotence guard; asserts the pinned gate
  body still byte-matches `runs/fork_diff_boristown/cells/boris_16_code.txt`).
- Staged canary: `notebooks/duckgate/arc3-duck-gate.ipynb` (18 cells) +
  `notebooks/duckgate/kernel-metadata.json` (id `canivel/arc3-duck-gate`, env
  fields byte-identical to the frozen fork family).
- Smoke: `duck_eval/a17/boristown_gate_smoke.py` — **47/47 PASS**.
- Local preflight-equivalent (ported `preflight.py` trusted-fork T3, run on staged
  files since preflight has no local-file mode and the kernel is unpushed):
  single-diff invariant vs the frozen fork = **ALLOW** (the only differing frozen
  code cell is the cell-2 env-detect banner append; run cell + solver surface
  byte-identical).

---
*Draft prepared 2026-07-28 (R22 directive D2). NOT SEALED. Do not queue, push, or
submit on the basis of this document alone. The orchestrator holds the one remaining
push slot; this intent consumes zero pushes and does not touch submission_queue.json.*
