# R19 response memo — answers to every reviewer question (2026-07-24)

Companion to `learnings/preregistration_amendment_2026-07-24_DRAFT.md` (the
"amendment DRAFT" below; all statistics there are computed under the newly
declared t-predictive model, `runs/r19_hygiene/r19_hygiene_stats.{py,json}`,
seed 20260724). This memo is build-rail writing (A22): it binds nothing until
R20 ratifies.

**The headline first (the three-round FATAL):** the panel briefing's
"best 0.43 / leader 1.56" was a **stale hardcoded May-era template inside the
panel harness**, not a normalization and not a second account. Root-caused and
fixed 2026-07-24: `scripts/panel_round.py` now injects canonical LB state from
`runs/lb_ground_truth.md` at round build time and surfaces a missing/stale
artifact to reviewers instead of papering over it (see `scripts/panel_round.py`
lines 103–118, comment block "R19 FATAL fix … 3 rounds of stale hardcoded
0.43/1.56"). Canonical state, API-verified against account **canivel
(d.canivel@gmail.com)** with
`uvx --from kaggle==2.0.0 kaggle competitions submissions arc-prize-2026-arc-agi-3`:
our best **1.33** (frozen-fork filler, 2026-07-18, rank ~#49), leader KOJIMA
**1.86**, gold cutoff ≈ **1.49**. The draw-by-draw scored ledger (all 16 draws,
API-verified) is published in `runs/lb_ground_truth.md` lines 15–22.
Reconciliation of the stale number itself: 0.43 *was* our best in early May
(forge-era agents); the frozen duck fork moved the band to 0.82–1.33 from
07-05 on (`runs/lb_ground_truth.md` lines 10–13).

---

## rl-planning

1. **LB reconciliation (blocking; Q1).** Answered above: **1.33 / 1.86 is
   ground truth; 0.43 / 1.56 was a stale hardcoded template** in
   `scripts/panel_round.py`, fixed 2026-07-24. Draw-by-draw submission ledger
   with the account named: `runs/lb_ground_truth.md` (verification command on
   lines 4–5). The two-round survival was a process defect in the panel
   harness, and the fix makes the artifact load-bearing (missing artifact →
   surfaced as UNKNOWN, never defaulted).

2. **Was the W1 eval-rail evidence sealed before the sentinel window fired?
   (Q2.)** **Yes.** The eval-rail RHAE negatives were on record 2026-07-23,
   before the 00:07Z 07-24 fire: `ITERATION_LOG.md` (07-23 gate-chain section,
   line 444) records the screen with the caveat verbatim — "secondary
   Δlog1p(RHAE) negative on BOTH seeds (s1 −0.315 p=0.997; s2 −0.166 p=0.90) —
   **flag for the sealed 3-seed look, not gating the screen**." The sealed A21
   entry bar required only canary PASS + non-harm, so the gate fired as
   written; the *bar itself* was defective. **Conceded** (also conceded in
   `learnings/war_room/sentinel_disposition_2026-07-24.md`, Bookkeeping) and
   fixed prospectively in amendment DRAFT §(c): entry now requires aggregation
   of ALL prior evidence into one stated prior with dependence structure, plus
   positive right-tail evidence (eval point estimate > 0 on ≥ 1 seed, or a
   mechanism story with an identified depth channel).

3. **What generates "~0.001–0.002 E[max]-equiv"? (Q3.)** It was the Gaussian
   machinery of `learnings/stuck_review_v2_2026-07-23.md` §1 — your objection
   stands and the figure is superseded. Recomputed under the single declared
   model (amendment §(a), joint t-predictive, common-random-numbers MC):
   **ΔE[max] per displaced window ≈ 0.0006** (`r19_hygiene_stats.json`,
   `supplement_pricing`). Both load-bearing claims — "filler-only is losing"
   (E[max@101] ≈ 1.40 vs Nov-2 gold band 1.60–1.90) and "exploration is nearly
   free" (0.0006/window) — now derive from the same fitted model, as demanded.

4. **Promotion re-derivation date (Q4).** **Today.** Amendment DRAFT §(d),
   filed before exploration draw 2/12: promotion is re-denominated in
   exceedance currency — PROMOTE iff (≥2 draws > 1.33) OR (≥1 draw ≥ 1.44)
   within the arm's first ≤5 windows; false-promotion 4.9% under arm≡filler,
   power 13.8/22.5/35.5% at +0.10/+0.15/+0.20. The +0.06 mean-lift bar is
   retired for promotion upon ratification.

5. **Are the A17 GO thresholds timestamped before today's canary? (Q5.)**
   **Yes, three artifacts, all pre-canary:** (i) `runs/sealed/r17_thresholds.json`
   (`"sealed_at": "2026-07-22"`, `a17_gate` key: capability_bar 8,
   throughput_margin 1, ρ = ρ_action); (ii) the verbatim gate boolean sealed in
   `learnings/war_room/a17_72b_screen_scope_v2.md` §9.1; (iii) the full
   symmetric error model incl. false-GO ≈ 0 in `learnings/a17_error_model.md`
   (header: "C3 filing (2026-07-23, **BEFORE bench push**)"). External
   timestamp: all committed in git **a31f0fe, 2026-07-23 08:56:29 −0400** —
   before any 72B number existed.

6. **boristown re-baselining plan (Q6).** The fork-diff
   (`learnings/war_room/fork_diff_boristown_2026-07-24.md` — filed same-day, as
   you demanded) changes the premise of your question: boristown is
   **byte-equivalent to our frozen fork** (12/22 cells md5-identical including
   every load-bearing cell; identical solver-dataset bytes; zero metadata
   mismatches; only functional diff = a 25-line vLLM readiness gate; the
   claimed patches are deliberate no-ops). So adoption is a
   **filler-replacement with a hygiene graft, not a new filler distribution**:
   the control ledger continues (no n≥5 quarantine, harm-pause stays armed)
   WITH a noted left-tail caveat and a changepoint monitor armed for the first
   5 post-gate draws (MK + CUSUM after each; permutation p < 0.05 forces a
   pre/post stratum split before any rule fires). Amendment DRAFT §(i).

7. **Reallocation rule on A17 GO vs NO-GO (Q7).** Now written: amendment DRAFT
   §(f). GO → war-v4 is the next arm, entry case (named non-score observables +
   null criterion) filed before its first window; NO-GO → depth-lane/tr87
   becomes priority 1; unclaimed windows revert to filler; max 1 concurrent
   arm; spread not front-load (≤3 exploration windows per 7 days);
   paused-arm windows return to the pool immediately.

## methodology

1. **W1-sealed-before-entry timeline (Q1).** Same answer as rl-planning Q2
   with timestamps: eval negatives on record 07-23 (`ITERATION_LOG.md` line
   444, quoted above), scored window fired 07-24 00:07Z. The entry gate did
   not weigh it because the sealed bar text did not ask it to — the caveat was
   explicitly routed "for the sealed 3-seed look, not gating the screen."
   Defect conceded; fixed in amendment DRAFT §(c) (evidence aggregation with
   stated dependence structure is now entry-mandatory).

2. **Declared tail model (Q2).** **t-predictive with ν = n−1 and √(1+1/n)
   scale inflation — one model, applied everywhere** (joint hierarchical form
   for multi-draw quantities). Amendment DRAFT §(a) publishes the recomputed
   table: 0.71 → one-sided **p ≈ 0.070** (your ≈0.07 confirmed; Gaussian 0.044
   superseded); P(single filler ≥ 1.33/1.44/1.47/1.49) = 0.029/0.0096/0.0071/
   0.0058; E[max@101] = 1.403; the Gaussian "P(touch 1.44) ≈ 0.18" is formally
   RETIRED (its declared-model successor is 0.33 — the correction *helps* the
   filler lottery, i.e. it was not adopted because it flatters us).

3. **boristown re-baselining protocol (Q3).** Your "third category / baseline
   change" framing is answered by evidence rather than definition: the byte
   diff shows distribution-equivalence (see rl-planning Q6 answer), so the
   n=0-reset scenario your MAJOR priced does not arise. Protocol adopted
   instead: monitored continuation — ledger continues, left-tail caveat noted
   (the readiness gate raises the floor, so pre-gate control draws are
   conservatively left-skewed vs the new filler), changepoint monitor armed
   for 5 post-gate draws with a pre-registered split trigger. If the monitor
   trips, the stratum splits and the harm-pause threshold is re-derived on the
   post-gate stratum before A21 rules fire again — your k-draws re-baseline,
   but *conditional* on evidence of a shift rather than unconditional.

4. **Time-ordered changepoint check + prospective pooling rule (Q4).** Both
   now exist: amendment DRAFT §(b). Mann-Kendall frozen n=10 (chronological,
   order certain): z = 0.72, p = 0.47; CUSUM permutation p = 0.72. Pooled
   n=15 (documented interleave assumption, both plausible orders): MK p =
   0.55/0.49, CUSUM p = 0.93. **Verdict: no trend, no changepoint.** The
   prospective pooling rule for draws 16+ (strata, exclusion of open-arm
   compositions, no retroactive merging) generalizes the one-time 0.71
   exclusion.

5. **Which numbers are stale (Q5).** One sentence: **the briefing's 0.43/1.56
   is stale (May-era hardcoded template in `scripts/panel_round.py`, fixed
   2026-07-24); the canonical artifact is `runs/lb_ground_truth.md`** (1.33
   best / 1.86 leader / gold ≈ 1.49, API-verified, account canivel).

6. **war-v4 non-score observables + null criterion before its window (Q6).**
   Written into the GO branch of amendment DRAFT §(f): entry case filed BEFORE
   the first war-v4 window must name per-game levels_completed from the pull,
   realized per-game action counts N₇₂B(g), realized ρ_action (diagnostic),
   and the heartbeat/liveness observable; null criterion = arm distribution ≡
   frozen control with pause/KILL per amendment §(e); plus the scope-v2 §3
   obligation that a GO republishes the 25-game × 3-seed ledger before any
   promotion claim. Your "zero mechanism evidence from the scored rerun"
   critique is exactly why every named observable above is pull-derivable, not
   log-dependent.

## systems

1. **Reconciliation paragraph (Q1) — "what prevents this from being written
   today?"** Nothing; it is written today (headline above). The blocker was
   that nobody had audited the panel harness itself: the briefing text was
   generated from a hardcoded template frozen in May. Fix is structural, not
   editorial — `scripts/panel_round.py` lines 103–118 now read
   `runs/lb_ground_truth.md` at build time, and a missing artifact is
   surfaced, never defaulted.

2. **Numeric ρ_action GO/NO-GO threshold; matched concurrency; 27B control leg
   (Q2).** The C3 numbers (`learnings/a17_error_model.md` §1): ACTION-PARITY
   requires **ρ ≤ 1.11**; modeled anchors ρ ∈ {2.5, 3.0}; envelope NO-GO
   self-certifies at **ρ > 3.5**; in the expected throttled regime GO ⇔
   Σ 72B MAX lc ≥ 8. On concurrency: your #20 was accepted and sealed —
   scope v2 §9.2 (verbatim): "**null_adj is evaluated at the realized 72B
   per-game N from the pull; ρ_action is demoted to a pre-run planning
   diagnostic only**" — so no ρ-predicted N₇₂B enters the boolean, and §3
   records the residual bias: "the residual asymmetry (72B enjoys 4-game
   concurrency while the 27B comparator lc came from 25-game runs) is pro-72B
   on the capability prong. Accepted for a capability-existence screen…" with
   the sealed backstop that branch-1 false-GO still requires capability
   **Σ ≥ 8 against a 27B max-null of 6** — draw noise cannot fake it
   (P(false GO) = 0.000 in every modeled cell). Residual concern conceded on
   the PARITY prong (4-game 72B N vs 25-game 27B N flatters parity), so we
   propose this pre-registered correction as a draft line for R20:
   > **DRAFT (R20):** the ACTION-PARITY prong is DIAGNOSTIC-ONLY for this
   > canary; if the gate ever turns on branch 1 with parity within 10% of the
   > 0.90 bar, a 27B 4-game control leg is run at matched concurrency to
   > renormalize the numerator BEFORE GO is declared.

3. **Is 480 actions/7920 s an aggregate over 25 games? (Q3.)** **No — and the
   6× worry dissolves.** The frozen 27B numerator is the sum over the FOUR
   SCREEN GAMES ONLY: **ft09 39 + sb26 225 + lp85 147 + vc33 69 = 480**
   (scope v2 §3, lines 117–118: "147 (lp85) + 69 (vc33) = 480 actions /
   7920 s"), i.e. ~120 actions/game — not 480/25 ≈ 19/game. The counts come
   from the certified 25-game runs but only the 4 screen games' rows enter the
   numerator, matching the 4-game 72B denominator game-for-game.

4. **Does 7920 s include download/load/warmup? (Q4.)** No. Scope v2 §5
   (log-derived, not asserted): 7920 s is a **per-game soft deadline with all
   games running concurrently** (`max_runtime_s_per_game=7920.0,
   concurrency=28`; all 25 `started_at` within ~1 s; kernel wall 2h12–13m).
   Setup is outside the benchmark clock and budgeted separately: "One push =
   2.2 h window + 72B load/init (~0.2–0.3 h) ≈ 2.5 GPU-h." The 480/7920 s
   numerator was measured on the same accounting, so numerator and denominator
   are deployment-representative and consistent.

5. **W2 slot opportunity cost in E[max] currency (Q5).** Priced by
   rl-planning's own finding: **zero VOI** — the arm is harm-paused and cannot
   re-enter the LB without a new A21 entry case, so W2's outcome changes no
   live decision at any price. Disposition: sentinel SHELVED by memo
   (`learnings/war_room/sentinel_disposition_2026-07-24.md`) with evidence
   weights and dependence structure stated ("three independent signals"
   corrected to two dependence clusters); the calibrated W2 instrument is left
   armed and pre-registered at $0 should a future panel want the formal KILL.
   Slot 2 goes to tr87 (the only non-A17 depth-targeting line). The reason the
   question reached the panel: closing a sealed line by memo rather than by
   its pre-registered instrument needed methodology's sign-off, which the
   memo's precedent note now scopes narrowly (paused + decision-inert only).

6. **First-party device artifact (Q6) — "which is it?"** Attached
   retroactively from the certified run log, as you allowed:
   `runs/kernel_pulls/war_eval_v1/arc3-duck-war-eval.log` line 278 contains,
   verbatim:
   > `CUDA GPU check passed for rtx-pro-6000 x1: ['NVIDIA RTX PRO 6000 Blackwell Server Edition']`
   This is the scored/bench rail's own device query (the same check runs in
   every kernel of the lineage; the assertion machinery is visible at log
   lines 72–84). The 72B canary's own pull will carry the identical
   first-party line and will be attached to the canary report post-run.

---

## Disposition of the non-question objections (pointer table)

| Objection | Disposition |
|---|---|
| rl-planning FATAL (LB ground truth) | Fixed structurally; headline + `runs/lb_ground_truth.md` |
| rl-planning MAJOR (entry bar admitted known-negative arm) | Conceded; amendment §(c) |
| rl-planning MAJOR (boristown accounting) | Fork-diff filed same-day; amendment §(i) |
| rl-planning MINOR (W2 zero-VOI) | Adopted; disposition memo |
| rl-planning MINOR (stale target denomination) | Amendment §(g): Nov-2 gold band 1.60–1.90 |
| rl-planning MINOR (allocation policy) | Amendment §(f) |
| methodology MAJOR (tail-model shopping) | Amendment §(a): one declared model; Gaussian retired |
| methodology MAJOR ("three independent signals") | Corrected in disposition memo (two dependence clusters) |
| methodology MAJOR (fork invalidates ledger) | Amendment §(i): monitored continuation, evidence-based |
| methodology MINOR (rank-bleed trend claim) | Conceded; honest fit 2.1 ranks/day, amendment §(g) |
| methodology prior MINORs (§6 load-bearing, A25 prospectivity, rule-of-three) | Amendment §(h) 1/2/4 |
| systems MAJOR (ρ_action concurrency confound) | §9.2 realized-N seal + R20 draft correction line (Q2 answer) |
| systems MAJOR (hang mitigation diagnostic-only) | Not resolved here — liveness-gate/bench-smoke proposal deferred to the canary post-mortem; watch #684625. |
| systems MINOR (fork classification by evidence) | Evidence delivered (byte diff); amendment §(i) |
| seal-termination downgrade logging (carried MINOR) | Amendment §(h)3 |
