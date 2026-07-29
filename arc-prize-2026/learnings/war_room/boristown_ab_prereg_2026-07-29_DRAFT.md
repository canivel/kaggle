# Boristown vLLM readiness-gate A/B — pre-registration (DRAFT) — filed 2026-07-29

**STATUS: DRAFT — NOT SEALED. Does not fire.** This document fires only after all three
conditions in §7 are met (git-commit seal + gated-arm kernel exists & passes trusted-fork
preflight + governance ratification). No slot is consumed, no push is made, and
`submission_queue.json` is not touched on the basis of this file alone.

Responds to **panel R22 directive D2 (5/5 unanimous, carried from R21 directive #1)**:
"Schedule the boristown readiness-gate A/B this week — replace filler draws, do not arbitrate
false scarcity against sentinel draw #2 … it needs a calendar date (~Aug 2 per systems), not a
question mark." Discharges NC-6 (regime-switch trigger), NC-8 (API-verified ledger), and the
NC-11 discipline (a pre-registration may not fire without its error rates published).

Sources (all read 2026-07-29): `learnings/war_room/fork_diff_boristown_2026-07-24.md`;
`runs/lb_ground_truth.md` (refreshed 07-29, n=15); `learnings/panel/round22/_directives.md`
(D2 + Q2 disposition + NC-11); `learnings/intents/boristown_ab_intent_2026-07-28.md`
(build-rail intent this seals); `learnings/a17_threshold_commit_2026-07-28.md` (B2a reversion
target); `learnings/intents/intent_exploration_draw2.md` (sentinel un-shelve rule);
`scripts/preflight.py` (trusted-fork mode).

---

## ⚠ OPEN BLOCKERS (read first)

1. **The boristown kernel ref and its diff DO exist in-repo — NOT a blocker, resolved.** The
   fork is `boristown/agi-duck-harness-fast-eval` (144 upvotes, last run 2026-07-22, public LB
   **1.47**), fully audited in `fork_diff_boristown_2026-07-24.md`. **12 of its 22 cells are
   md5-identical** to our frozen fork including every load-bearing cell; the **single functional
   diff is boristown cell 16**, `wait_vllm_ready()` (pinned verbatim at
   `runs/fork_diff_boristown/cells/boris_16_code.txt`). Cells 8–11/15 are deliberate no-ops
   (rolled-back ACTION7 / animation-metadata patches, every `*_changed: False`); cell 21 is a
   display-only score card gated off under `TRUE_SUBMISSION`. There is no hidden diff carrier;
   the solver lives in the shared dataset `jeroencottaar/taaf-kaggle-source-share` (frozen
   2026-06-12, before our fork froze). **We are NOT inventing the diff.**

2. **BLOCKER (preflight-mode mismatch) — arm B is a *modified* fork, so strict trusted-fork
   T3 will FAIL, by design.** `scripts/preflight.py --mode trusted-fork` requires the fork's
   code cells to be **byte-identical to upstream** (check T3). Arm B is deliberately *not*
   identical to either the frozen fork or to boristown — it is the frozen fork **+ one added
   cell** (the gate) + two additive banner prints. Therefore:
   - Against upstream `= our frozen fork`: T3 reports **1 added code cell** (the gate) → FAIL.
   - Against upstream `= boristown`: T3 reports the two banner-print additions + the customization
     stub delta → FAIL.
   The stock trusted-fork gate cannot certify a one-cell graft. **Resolution (must be ratified
   before fire):** use the **single-diff-invariant preflight-equivalent** already run on the
   staged notebook (intent §"Built artifacts": ported preflight `T3`, verdict **ALLOW** — the
   only differing frozen code cell is the cell-2 env-detect banner append; run cell + solver
   surface byte-identical). This local T3 has **no upstream-pull / no COMPLETE-status leg**
   because the kernel is unpushed. Fire requires either (a) an explicit governance waiver
   recording that arm B is an *audited single-cell graft* and the local single-diff ALLOW
   substitutes for T3, OR (b) extending `preflight.py` with a `--max-diff-cells 1 --pin
   <boris_16 sha>` mode. **Until (a) or (b), the standard preflight cannot mechanically pass and
   the daily_submit daemon's auto trusted-fork tag does not apply to this slug.**

3. **BLOCKER (entry-gate #1, live-firing half) — NOT yet discharged.** The 2-seed eval canary
   is *built* (`notebooks/duckgate/arc3-duck-gate.ipynb`, smoke 47/47 PASS, local single-diff
   ALLOW) but the **live eval-kernel BUILD that logs the gate observed-firing** (`A17-GATE
   observed-firing vllm_ready_latency_s=…`, ≤180 s) and the **non-harm screen vs runs/null10**
   have **not been run**. These are free zero-spend kernel builds, not scored pushes, but they
   gate entry. Fire date slips one slot per day the eval canary slips (§2).

4. **NOTE (env parity, NC-12 analog) — LOW.** Arm B's metadata carries the **27B FP8** dataset
   (`vrfai-qwen3-6-27b-fp8-hf-snapshot`), i.e. the byte-identical frozen-fork family, **not** the
   A17 72B weights dataset. This is correct (the A/B tests the gate, not the model). The scored
   frozen draws already run on RTX PRO 6000; the canary metadata pins `machine_shape:
   NvidiaRtxPro6000`. GPU-parity confirmation (grep a scored frozen-fork log for the GPU string)
   is cheap and should be attached to the entry-gate build, but it is not expected to bind.

---

## 1. Arm definition

- **A (control):** the frozen duck fork, byte-identical filler `canivel/arc3-duck-repro` v3
  (exact fork of the Tufa/Cottaar duck-harness June-30 Milestone-1 winner
  `jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner`). Control = the banked
  frozen-ledger draws (**n=15** as of 07-29; new fillers may interleave — see §2). No new draws
  are commissioned *for* the control; it is already banked.

- **B (gated):** the built canary **`canivel/arc3-duck-gate`** (staged locally at
  `notebooks/duckgate/arc3-duck-gate.ipynb`, 18 cells; built by
  `duck_eval/a17/build_boristown_gate_canary.py`, fork-never-build from the Milestone-winner
  base; smoke `duck_eval/a17/boristown_gate_smoke.py` = **47/47 PASS**). Fresh slug per
  feedback_fresh_kernel_slug.

- **The single diff (byte-matched):** one standalone cell inserted immediately **before** the
  `await bm.run(...)` cell. Its `wait_vllm_ready()` body is **byte-identical to boristown cell 16**
  (asserted at build time against `runs/fork_diff_boristown/cells/boris_16_code.txt`): polls
  `http://127.0.0.1:1234/v1/models` every 5 s, up to 180 s, raises if vLLM never comes up. It
  closes a startup race the frozen fork has — our fork waits for the **gateway**
  (`_wait_for_gateway`, 600 s) but never for the **vLLM server**, launched async by the solver
  setup commands, so early-episode actions can be burned while the 27B FP8 model still loads. The
  only additions beyond the byte-matched body are two additive `A17-GATE` banner prints (an
  "armed" line and an "observed-firing" line carrying the ready latency) required by entry-gate
  #1. **Nothing else changes** — no solver graft (the customization-hook cell is left
  byte-identical), no budget/prompt/model/sampling/retry/reset change. This is what keeps the A/B
  a clean single-variable causal test.

- **Metadata (feedback_kaggle_env_match — byte-match discipline):** `kernel-metadata.json`
  (`notebooks/duckgate/kernel-metadata.json`) is field-for-field identical to the frozen-fork
  family except identity fields: `enable_gpu: true`, `enable_tpu: false`, `enable_internet:
  false`; the same 3 `dataset_sources` (`arc3-vllm-h100-wheelhouse-v3`, `taaf-kaggle-source-share`,
  `vrfai-qwen3-6-27b-fp8-hf-snapshot`, same order, no version pins); `docker_image` sha256
  `…be4cb13c`; `machine_shape: NvidiaRtxPro6000`; `competition_sources:
  [arc-prize-2026-arc-agi-3]`; empty `model_sources`/`kernel_sources`. Pull-back verification of
  the served metadata is mandatory at push (feedback_kaggle_model_attach / dataset_code_sync).

- **Preflight:** see BLOCKER 2. Nominal command (does not pass strict, kept for the record):
  `uv run python scripts/preflight.py --mode trusted-fork --kernel canivel/arc3-duck-gate
  --upstream jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner`. The **operative**
  gate is the local single-diff-invariant T3 = **ALLOW** plus a governance waiver (§7).

---

## 2. n, start date, interleaving

- **n = 4 gated draws** (panel band 3–5), each from the daily scored slot (one/day,
  ARCDailySubmit queue head) — **zero incremental slots**: they *replace* the next 4 frozen-filler
  draws, they do not add to the schedule (D2: "replace filler draws").

- **Start date: Aug 2 at the latest** (systems' ~Aug 2 anchor and the natural Sunday-panel
  ratification point). **Earlier if the build slot frees:** if the A17 lane closes on seed-2
  concordance (branch **B2a** of `a17_threshold_commit_2026-07-28.md` — 72B route DEAD, which the
  seal itself names as reverting build priority to *this* A/B), the entry-gate eval canary can run
  as soon as the freed kernel slot allows, and the first gated draw fires the next daily slot.
  Nominal calendar if the eval canary lands 07-30→08-01: gated draws replace fillers **08-02 →
  08-05**; A/B disposition memo filed the day after draw #4. Each day the entry-gate canary slips,
  the fire date slides one slot.

- **Interleaving: straight replacement (4 consecutive gated slots), NOT alternation.**
  One-line justification: the control is already banked (n=15) and stationary on record (MK
  p≈0.47, CUSUM p≈0.72), so alternation buys no de-confounding against drift and only stretches
  the arm across ~8 days, starving sentinel draw #2 longer; consecutive gated slots minimize
  time-to-disposition at no inferential cost.

---

## 3. Decision rule (one-sided vs the frozen ledger, n=15)

**Metric:** public-LB scored draw per window (completion-weighted), API-verified into arm B's own
ledger before any rule fires (NC-8). Gated draws enter **arm B's ledger only** — never frozen,
never pooled — until closure + an equivalence memo (pooling rule (b)(3) of the 07-24 DRAFT).

**Control parameters (frozen ledger, n=15, API-verified `runs/lb_ground_truth.md` 07-29):**
mean **x̄_C = 0.9727**, s **= 0.1343** (recomputed with today's 1.03 draw added; was
n=14 / 0.9686 / 0.1384). Verified by `uv run python` on the 15-value list
[0.82,0.89,0.93,1.02,0.95,1.33,0.92,0.93,1.14,0.82,1.05,0.84,1.02,0.90,1.03] → mean 0.972667,
s 0.134349.

**Hypotheses (one-sided):**
- **H0:** arm B ≡ frozen (the gate produces no upward distributional shift; μ_B = μ_C = 0.9727).
- **H1:** μ_B > μ_C (the gate shifts the scored-draw distribution up). Anchor-implied effect if
  the gate explained boristown's 1.47: δ = 1.47 − 0.9727 = **+0.4973**, i.e. Cohen's
  **d = δ/s = 0.4973/0.1343 = 3.70** (the panel's d≈3.6 verified against the n=15 ledger s — the
  small excess over 3.6 is because the mean rose and s tightened at n=15; either way the effect is
  enormous relative to the noise).

**α = 0.05, one-sided.** Common-s Welch-form comparison of x̄_B(4) against the banked control.
SE = s·√(1/n_B + 1/n_C) = 0.1343·√(1/4 + 1/15) = 0.1343·0.5627 = **0.0756**.

- **PROMOTE rule (Gaussian, sealed value):** PROMOTE iff
  **x̄_B(4) ≥ 0.9727 + 1.645·0.0756 = 1.0970.**
- **t-robust cross-check (Welch df ≈ 4.7, t_crit ≈ 2.040):** the same data would clear a t-based
  bar at **x̄_B(4) ≥ 1.1269.** The Gaussian value governs (sealed); the t bar is reported so a
  borderline PROMOTE (1.097 ≤ x̄_B < 1.127) is flagged for panel rather than auto-fired.

- **ABORT / harm-pause (per draw, sealed A21/C2):** any gated draw **< 0.80** pauses the arm
  pending panel review. Because the gate's entire mechanism is *left-tail removal*, a sub-0.80
  draw is **evidence against H1**, not merely exposure control.

- **MISS/kill (at n=4):** if x̄_B(4) < 1.0970 ⇒ the mean-shift claim is dead; boristown's +0.14
  over our mean is confirmed **non-systematic** (variance hypothesis stands); **no regime switch**
  on this evidence. The *hygiene* question (free left-tail insurance at zero interaction risk) is
  **NOT** killed by a MISS — it returns to the Sunday panel on the fork-diff evidence alone.

- **PROMOTE ⇒ NC-6 fires as written:** "the entire daily-filler regime switches to the gated
  variant"; the gate additionally becomes default hygiene in all lineages (frozen, duckwar,
  sentinel) and the changepoint monitor arms for the first 5 post-gate draws.

### Family error rates under the frozen null (NC-11 — published, not delegated)

All computed with `uv run python` (scipy) on the n=15 ledger; mimics the sentinel-rule
computation style (P(draw<x|frozen), predictive-t, family unions):

| quantity | Gaussian | t-predictive (ν = n−1 = 14) |
|---|---|---|
| **False-promote** P(x̄_B(4) ≥ 1.0970 \| B≡frozen) | **5.00%** (by construction) | **6.11%** |
| **False harm-pause / draw** P(draw < 0.80 \| B≡frozen) = Φ((0.80−0.9727)/0.1343) = Φ(−1.285) | **9.94%** | **11.69%** (t at −1.244, sd_pred = s·√(1+1/15) = 0.1388) |
| **Family false-abort** P(≥1 of 4 gated draws < 0.80 \| B≡frozen) | **34.2%** | **39.2%** |

Read: the promote rule false-fires ~5–6% of the time under the null (the number the rule is
built to hold). The per-draw harm-pause false-fires ~10–12% (one in ~9 healthy draws) — this is
*exposure control, not inference*; a pause defers to panel and does not by itself kill H1, but
because the gate is a left-tail-removal mechanism a pause is corroborating evidence *against* the
gate. The 34% family false-abort over 4 draws is the honest cost of a per-draw floor on a noisy
metric; it is accepted because a sub-0.80 draw is decision-relevant regardless.

---

## 4. Minimal detectable effect (80% power, one-sided α = 0.05)

MDE(80%) = (z_0.05 + z_0.20)·SE = 2.4865·s·√(1/n_B + 1/15). Power at the anchor δ and at
floor-raise δ's (Gaussian), all `uv run python`-verified:

| n_B | SE | PROMOTE thr | MDE(80%) | power @ δ=0.497 (anchor) | power @ δ=0.10 | power @ δ=0.15 | power @ δ=0.20 |
|---|---|---|---|---|---|---|---|
| 3 | 0.0850 | 1.1124 | **0.211** | ~1.0000 | 32.0% | 54.8% | 76.1% |
| 4 | 0.0756 | 1.0970 | **0.188** | ~1.0000 | 37.4% | 63.3% | 84.1% |
| 5 | 0.0694 | 1.0868 | **0.173** | ~1.0000 | 41.9% | 69.8% | 89.2% |

**Reading:** at the anchor-implied δ ≈ +0.50 the rule **cannot miss** at any n∈{3,4,5} (power
≈ 1). It is well-powered only down to δ ≈ 0.17–0.21. **A MISS therefore rules out the
anchor-sized (d≈3.7) effect the panel priced — it does NOT rule out a small floor-raise**
(δ ≈ 0.10 is detected only ~37% of the time at n=4). This is the honest limit and is why a MISS
returns the *hygiene* question to panel rather than closing it.

---

## 5. What this costs

- **Slots:** **zero incremental.** The 4 gated draws replace the next 4 daily frozen fillers
  (one/day scored slot). Opportunity cost ≈ 4 × 0.0006 ≈ **0.0024 E[max]-equiv** (declared-model
  pricing) — the value of 4 frozen fillers, which by P(single frozen draw ≥ gold cutoff 1.49) ≈
  2×10⁻⁴ cannot climb rank anyway.
- **GPU build hours:** **0 incremental for the scored A/B itself** — a byte-matched fork push is a
  *submission*, not a kernel build. The one non-zero build cost is the **entry-gate eval canary**
  (2-seed live-firing + non-harm screen), a **free Kaggle kernel BUILD** (~2 × 2.2 GPU-h against
  the weekly ~30 GPU-h quota, **$0 cloud** per feedback_arc_zero_budget). This is a prerequisite,
  not part of the scored A/B.
- **Kernel push budget:** max **2 kernel pushes/day**. Arm B consumes the fork push (1 push) on
  the first gated day; the frozen filler daemon is untouched (it re-uses v3, no push). No conflict
  with the A17 push budget on non-overlapping days.

---

## 6. Sentinel draw #2 — queued BEHIND this A/B (R22 Q2 disposition)

Per R22 open-question-2 disposition (option (b), adopted): the sentinel un-shelve rule is
**pre-registered but ranked strictly behind** this A/B on information value (5/5 directive,
single-variable causal test, d≈3.7 testable at n=3–4 vs the banked control; the sentinel's own
disposition memo says "no lift channel," n=1 at 0.71, p≈0.07). Already-computed un-shelve rule
(from `intent_exploration_draw2.md`, restated so it fires from one document):

- **Trigger:** draw #2 fires in the first daily scored slot **≥ 2 calendar days after this A/B's
  disposition memo is filed**, and not before A17 v6 lands. **Hard backstop 2026-08-10** (the
  exploration program may not be starved by upstream slippage).
- **n ≥ 4–5**, target n = 5 (the sunk 0.71 is draw 1/5, never excluded).
- **Sequential early re-shelve:** re-shelve iff **2 consecutive draws < 0.80** OR **mean of first
  3 draws (incl. 0.71) < 0.80**.
- **Family false-kill under frozen null (recomputed on n=15 ledger, `uv run python`):**
  per-draw P(<0.80) = **9.94%** (Gaussian); mean-of-3 clause (needs mean(d2,d3) < 0.845) =
  **8.95%**; 2-consecutive clause over draws 2–5 = **2.77%**; union ≈ **11.5%** (Gaussian). Under
  the tighter n=13 stats the panel/methodology quoted this as **≈6–7%**; the union rises to
  ~11% at the n=15 mean/s because the per-draw floor sits closer to the (higher) mean — **stated
  here rather than hidden** (NC-11). The doc that fires the sentinel draw must republish these
  under whichever ledger is canonical at fire time.
- **Final disposition at n=5** per the sentinel intent (SHELVE if x̄₅ < 0.878; PROMOTE-track if
  ≥2 draws > 1.33 or ≥1 draw ≥ 1.44; else CLOSE-NEUTRAL). Not restated in full here — this A/B
  prereg only records that draw #2 is queued behind and carries its error rates.

---

## 7. Status header — fire conditions

**DRAFT. This pre-registration fires only after ALL of:**

1. **Seal by git commit** — this file committed to the log with a recorded timestamp (house style:
   the A17 threshold seal `4ecf49a` is the precedent). The commit must predate the first gated
   push.
2. **The gated-arm kernel exists and passes the trusted-fork preflight** — arm B's slug
   `canivel/arc3-duck-gate` is pushed and its build status is COMPLETE, AND the preflight verdict
   is not BLOCK. **Per BLOCKER 2 the *strict* `--mode trusted-fork` T3 will not pass a one-cell
   graft**, so this condition is met either by (a) a recorded governance waiver accepting the
   local single-diff-invariant ALLOW + audited byte-match of the gate body as the T3 substitute,
   or (b) a `preflight.py` extension (`--max-diff-cells 1 --pin <boris_16 sha>`). Plus
   entry-gate #1 (2-seed live-firing log + non-harm screen vs `runs/null10`) discharged (BLOCKER
   3).
3. **Ratification per governance** — the **Sunday panel 2026-08-02** is the natural ratification
   point, and the panel itself demanded a date "~Aug 2." **Compliant path: seal Saturday
   (2026-08-01), ratify at the Sunday panel, first gated draw fires the Sunday-night queue (first
   scored slot on/after 08-02).** Earlier fire is permitted **only on an explicit principal order**
   (e.g. if B2a closes the A17 lane and frees the build slot before the weekend) — in which case
   the weekday sealed-arithmetic-gate path substitutes for the Sunday panel, and this file's seal
   commit + the entry-gate discharge remain mandatory.

---
*Draft prepared 2026-07-29 (R22 directive D2 / NC-6 / NC-11). NOT SEALED. Do not queue, push, or
submit on the basis of this document alone. All statistics `uv run python`-verified against the
n=15 API-verified frozen ledger (`runs/lb_ground_truth.md`, refreshed 07-29).*
