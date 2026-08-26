# Grinder-cracking design — R16 REPUBLICATION (post-(d)-kill)

Filed 2026-07-20. This is the republication R15 demanded (5/5 MAJOR-REVISION,
`learnings/panel/round15/_directives.md`): the A18 (d)-kill (recurrence accuracy
0.465, Wilson 95% [0.436, 0.494], vs majority baseline 0.903 —
`runs/predict_metric/report.md`) is here propagated into every number that
depended on it. **The recalibrated A14 gate (A14.1–A14.6,
`preregistration_amendment_2026-07-18b.md`) SEALS ON THIS CIRCULATION**, with the
amendments in §3, §6 and §8 below. Nothing in this document conditions on any
unobserved measurement; every threshold below is sealed before its measurement
runs (§13 seal-hygiene procedure).

Base document: `learnings/war_room/grinder_cracking_design.md` (unaltered on
disk; superseded section-by-section here). Evidence base unchanged, plus:
`runs/a5_a8_look_2026-07-19.json` (war arm CLOSED: n=5 {0.91, 1.08, 0.88, 1.05,
0.76}, mean 0.936, σ̂ 0.1309, χ²-CI-hi 0.376 ≥ 0.25 → FAIL; pooled n=11 σ̂ 0.154),
`runs/kernel_pulls/w0_eval_s1/screen_report.md` (W0: 49/49 game-over recoveries,
0 idle turns; 16 levels ∈ band {13, 22}), `learnings/war_room/
sentinel_build_2026-07-19.md` ((a) built; smoke 29/29; canary PASS; O5 predicate
49 budget deaths / 0 violations), `runs/ewm_dryrun/report.md` (Stage-0 dry-run).

---

## §1. What changed since the sealed base document

1. **(d) is DEAD** (A18, sealed threshold, observed after seal): a recurring
   "no-effect" (state, action) pair changes the board ~54% of the time. The FACT
   rule would be actively wrong most times it fired. The kill is propagated below
   into §2R, §4R, §5R, and Part D (§14). Root cause identified by R15 as
   **state-key aliasing** — one pathology behind the 0.465 accuracy, the EWM
   step-0 aborts, and the N5 prune bug (§10).
2. **The war LB arm is CLOSED** (A5/A8 FAIL at n=5, sealed consequence executed):
   no war-arm LB delta may be cited as evidence in any direction. Everything
   below lives on the build rail.
3. **W0 confirmed (f) as pure hygiene**: mechanism deterministic PASS (49/49),
   descriptive non-inferiority PASS (16 ∈ [13, 22]). Per R15: W0's 16 levels may
   NOT be cited downstream as a score claim; the citable claim is
   levels-in-band only. (f) is adopted as the default layer of every future
   build (R15 endorsement).
4. **(a) is built and canaried** (W1 owner per the A18 sealed consequence "W1
   becomes (a)'s window"): mechanism observable restated deterministic per R15
   O5 — "sentinel fired before every budget death", verified 49/0 on the three
   certified seeds. One unsealed design decision remains: SENTINEL_BUDGET (§12).

---

## §2R. Counting bounds and honest sum, republished with (d) REMOVED

Method unchanged (events × max value under the exact pooled-single-run scorer;
reclaimed actions on uncompleted levels are worth exactly zero). Per-component
bounds unchanged from the base doc except the killed flag. Two branches are
carried throughout, per R15: **B+** = banking window approved (full-panel
sign-off + A16 recompute + latent-state audit §10 all pass) and **B−** =
banking refused (the unconditional stack).

| component | ceiling (rail) | expectation (rail) | status |
|---|---|---|---|
| (f) continuation | 0.00 | 0.00 | shipped, default layer |
| ~~(d)+(c) mechanical refutation~~ | ~~+0.10~~ | ~~+0.02–0.05~~ | **(d) KILLED (A18); (c) KILLED (§7)** |
| (a) budget sentinel | +0.06 | +0.01–0.03 | built, W2 owner |
| (b) diff summarizer | +0.06 | +0.01–0.03 | W3, non-inferiority-guarded |
| banking-fixed (feasible) | +0.15 | +0.03–0.08 | conditional (B+ only; pre-A16-haircut) |

**The sums, shown:**

- **B+ raw sum** = 0.06 + 0.06 + 0.15 = **+0.27 rail**. Non-additivity haircut:
  the base doc deducted −0.06 for (a)/(b)/(d) reclaiming the same wasted
  actions (0.37 → 0.31). Recomputing without (d), the (a)/(b) pairwise overlap
  alone is bounded between −0.03 and −0.06 (both components' ceilings are the
  SAME "≈1 marginal clear per run panel-wide" event on the same lp85/tu93/re86
  waste). We seal the conservative end: **B+ ceiling = 0.27 − 0.06 = +0.21
  rail** — identical to the panel's own arithmetic (0.31 − 0.10).
- **B− raw sum** = 0.06 + 0.06 = **+0.12 rail**; (a)/(b) pairwise overlap −0.03
  → **B− ceiling = +0.09 rail** (the stricter joint-event bound gives +0.06;
  +0.09 is the generous end and is labeled as such).
- **Expectations:** B+ = 0.01–0.03 + 0.01–0.03 + 0.03–0.08 = **+0.05–0.14
  rail**. B− = 0.01–0.03 + 0.01–0.03 = **+0.02–0.06 rail** (the R15 quote
  "+0.02–0.08" included a registered (c); (c) is killed, so the lower figure
  stands; note the B− expectation's upper end touches its strict joint ceiling
  — that is the honest statement that (a)+(b) are the same money).

**LB conversion (0.56× ASSUMPTION, band 0.4–0.8 per A20):**

| branch | ceiling LB | expectation LB | fraction of 0.46 wall gap (ceiling / expectation) |
|---|---|---|---|
| B+ | +0.12 [0.08–0.17] | +0.03–0.08 | 26% / 7–17% |
| B− | +0.05 [0.04–0.07] | +0.01–0.03 | 11% / 2–7% |

**Conclusion, sharpened by the kill:** the base doc's verdict ("floor/mid
raiser; the wall needs more") is now stronger — the largest non-banking
component is gone, the unconditional stack closes ≤11% of the wall gap even at
ceiling, and **war-v4 (72B multimodal, A17 scope doc filed 07-19) remains the
only registered wall-closer.** Nothing here is smuggled toward the wall.

---

## §3R. α re-derivation (methodology N2)

The sealed α = 0.0125 was Bonferroni 0.05/4 for four per-window binding looks.
A14.3 demoted per-window looks to mechanism-prong + non-inferiority-guard only;
**exactly ONE binding score look remains** (the cumulative stack-vs-W0 look,
A14.2). The divisor's family no longer exists.

**Test family at the binding look, named:** {primary pooled per-game Δlc exact
sign-flip; secondary mean Δlog1p(RHAE) ≥ 0}. The decision rule is
**conjunctive** (PASS requires BOTH), so the combined size is ≤ the primary's
size — no Bonferroni is due for an AND-family (correction is for disjunctive
claim families). The secondary is a directional consistency check, not an
independent claim license.

**Sealed: α = 0.05 one-sided on the primary prong** (R15's lean, adopted).
First-order power consequence, shown: minimum uncontradicted wins to pass —

| nonzero pairs n | old critical (α=0.0125) | new critical (α=0.05) |
|---|---|---|
| 5 | impossible | 5/5 (p=0.031) |
| 6 | impossible | 6/6 (p=0.016) |
| 7 | 7/7 (p=0.0078) | 7/7 (p=0.0078) |
| 8 | 8/8 (p=0.0039) | 7/8 (p=0.035) |
| 9 | 9/9 | 8/9 (p=0.0195) |
| 10 | 9/10 (p=0.0107) | 9/10 (p=0.0107) |

The unpassable-gate defect A14 conceded (needing ≥7 clean wins against an
expectation of 1–4) is now materially repaired at the low-n end (5 clean wins
suffice).

---

## §4R. Per-game conversion targets, re-derived post-(d) (both branches)

The base §4 ft09 row was carried by (d) ("no-effect FACTs kill dead-target
re-probes; PREDICT gate kills retries") — that basis is retracted. su15 stays
excluded (A12; §13). Δclears = integer levels per run, honest.

| game | B− components | B− Δclears | B+ adds | B+ Δclears |
|---|---|---|---|---|
| ft09 (6) | none remaining ((d) basis retracted) | **0** | banking (top variance carrier, Δlc(max2) +0.44) | **0–1** |
| ka59 (7) | (a) — L2 base-109 grind never converts | **0** | banking (protect L1, free retries) | **0–1** |
| re86 (8) | (a) — v1 died L2 at 232 acts (base 42); v2 cleared L2+L3, capability exists | **0–1** | banking (variance harvest) | **0–1** |
| sc25 (6) | none | **0** | banking (floor protection) | **0–2** |
| tu93 (9) | (a) — v3 burned 301 acts on L1 (base 19) | **0–1** | banking | **0–1** |
| sb26 (8) | (c) killed; L2 semantics NOT-distillable | **0** | — | **0** |
| lp85 (8) | (a)/(b) survival ≠ solution | **0** | — | **0** |
| su15 (9) | excluded (A12) | — | — | — |

**Sums:** B− = **0–2 extra clears per run** (Δlc/draw ≈ +0.00–0.08); B+ =
**1–4 extra clears per run, banking-dominated variance harvest** (Δlc/draw ≈
+0.04–0.16). Expected nonzero positive game-level pairs at the binding look:
**B− ≈ 1–3; B+ ≈ 2–6** (down from the pre-kill 4–8). The two canonical grinders
still carry **zero** at Qwen tier — the model-gap finding is untouched by the
kill.

---

## §5R. P(pass | §4R expectations), republished with assumptions explicit

Binomial-sketch assumptions (all explicit, per R15): (i) the paired unit is the
game (A14.1), n = 24 games, exact zeros dropped; (ii) carrier games convert to
positive per-game means with the §4R ranges treated as uniform; (iii) noise
produces 0–2 spurious nonzero pairs of either sign (cross-seed lc variance
exists on ~8–12 games); (iv) independence across games (unmodeled common-night
correlation would push P(pass) DOWN, not up — stated adversely).

- **B+**: expected positives 2–6, negatives 0–2; pass requires ≥5 clean (or
  7/8, 8/9 per §3R). **P(pass) ≈ 0.10–0.30, point ≈ 0.2** (was 0.2–0.4
  pre-kill at the old α; the kill costs more than the α relaxation buys).
- **B−**: expected positives 1–3; P(≥5 positives) is the binding term.
  **P(pass) ≈ 0.02–0.10, point ≈ 0.05.** Stated plainly: the unconditional
  stack's cumulative look is a near-certain FAIL; its value claim rests on
  verified mechanisms + LB accumulation (A8), and the downside is bounded by
  the dismantle branch (§6.2). This is the honest price of shipping only
  hygiene-grade components after the largest flag died under its own sealed
  test.

---

## §6. The binding cumulative look — regime, dismantle branch, W1/W2 status

### §6.1 Budget regime (methodology N5)

The binding look runs at **FULL budget** — the scored-regime per-game envelope
(≈63k tokens/game; actions uncapped by the harness; SENTINEL_BUDGET exported
per §12). Compressed-bench (40%-cap) window passes are provisional and carry a
compressed-regime qualifier per A15. **The full-budget binding look discharges
A15's confirmation-replicate requirement by construction** (3 full-budget
certified seeds of the final stack ≥ the required 1). Scheduled: §12 quota
ledger.

### §6.2 Cumulative dismantle branch (rl-planning M4 — SEALED CONSEQUENCE)

**If pooled Δlc ≤ −0.10 at the binding cumulative look, the stack is
DISMANTLED to the (f)-only build** for all subsequent scored kernels. This
amends A14.6: the "ships with honest label" outcome now applies ONLY to the
middle band (score prongs not passed AND pooled Δlc > −0.10); a net-negative
stack does not ship under any label. Calibration of this threshold, published
(§8 arithmetic): under the null, with 3 ON seeds vs the n=4 control band
(§11), SE(Δ) ≈ 0.144 lc/game and P(trip | Δ=0) ≈ 0.24. Sealed at −0.10 as the
panel directed: the loss is asymmetric — a false dismantle forfeits an
unconfirmed +0.02–0.14 expectation; a false ship sends a real regression to
the scored LB. We accept the 24% and say so before observation.

### §6.3 Window status

- **W0 (f):** done; default layer; not a flag.
- **W1 = (a)** (per the A18 sealed consequence). Build complete
  (`sentinel_build_2026-07-19.md`): smoke 29/29, A10 canary PASS (23–25/25
  games fire), R15 O5 deterministic predicate PASS (49 budget-attributable
  GAME_OVERs, 0 violations — every death preceded by a strictly earlier
  firing in the same attempt; negative path validated). Mechanism prong at the
  window: the O5 predicate (code-checkable) + firing counter ≥1/run on ≥5
  games; pooled binomial (72/75 (game,seed) units fired) as fallback.
- **W2 = (b)** diff summarizer, non-inferiority-guarded (token cost),
  boundary per §8.
- **W3 = banking**, B+ branch only: full-panel sign-off + A16 recomputed
  ceiling + latent-state audit PASS (§10) all required before the window
  opens.
- **EWM Stage-1**: own window, own gate (§9), after its blocking prereqs.

---

## §7. (c)+Reki disposition: KILLED (decided, per R15 N8/Q3)

The bundle as circulated was an unregistered 3-way: (c) byte-identical
resubmit hard-block + Reki structural-signature suppression (a THIRD mechanism:
learned click suppression over signature families) + hard veto. Disposition
options were register-as-new-flag or kill. **Decision: KILL.** Reasoning:

1. **Standalone (c) forfeits under the MDE/2 rule** (R13's own rule, conceded
   in the base doc): direct ceiling +0.02/draw.
2. **The generalized (family-level) form is observable-state-keyed
   suppression — the exact premise A18 refuted.** At the closest measured
   keying, "this action does nothing here" is wrong ~54% of the time it
   recurs (0.465 vs 0.903). A signature-family veto firing beyond
   byte-identical matches would veto LIVE actions at a comparable rate until a
   non-aliasing key exists. R15's own root-cause finding (§10) says the key is
   the problem; building a veto on the broken key is building on the known
   fault.
3. **Windows are the scarce resource** (2 pushes/day; A17 must complete
   pre-Aug-1). A third mechanism needing its own counting bound, A19
   observable, keying re-run, and sign-off cannot out-compete the sentinel,
   the summarizer, or the 72B screen for a July window.

**Counting bound, published for the record (as R15 required if registered):**
dead-signature veto opportunities from the forensics corpus ≈ 30–70/run
(same-coord re-clicks 16–32/seed + SPACE 8–20/run + ACTION7 8–20/run);
reclaimable actions ≈ 70–110/run (sb26 50–70 + 20–40 on 1–2 other games);
conversion through the same clear-faster/clear-at-all channels → **ceiling
+0.02 (byte-identical) to +0.06 (family-level) rail — overlapping (a)/(b)'s
reclaim of the same actions**, expectation +0.00–0.02. Below single-window MDE
on every instrument; would only ever pay through the cumulative look.

**Sealed resurrection precondition (one path, no others):** (c)+Reki may be
re-proposed as a flag only after (i) the latent-state audit (§10) delivers a
per-game keying restoring ≥0.99 recurrence determinism on the target games,
AND (ii) `scripts/predict_metric.py` re-run under the exact Reki keying
(signature-family, level-scoped) clears **recurrence accuracy ≥ 0.90, sealed
here before that measurement is run**. Discharges R15's pre-window condition
by construction (the re-run is now an entry requirement, not a window step).

**KNOW#5 annotation (rl-planning M2 / prog-synthesis O2 item 1):**
`state_of_the_war_2026-07-18.md` KNOW#5 ("mechanical no-effect refutation +
verify-before-act are THE convergent primitives") is hereby conditioned:
**the primitives are convergent AT FRONTIER TIER AND ON NON-ALIASING STATE
KEYS; A18 killed observable-state-keyed no-effect FACTs, not refutation
machinery per se.** The EWM plan-execute-verify contract (which verifies
against the SETTLED FRAME, not a state-key lookup) is unaffected by this
annotation and remains registered.

---

## §8. Guard false-kill calibration (methodology N3)

**Inputs:** run-level lc/game means of the 3 certified ledger-OFF seeds:
22/25 = 0.88, 15/25 = 0.60, 13/25 = 0.52 → mean 0.667, **σ̂_run = 0.189
lc/game** (df=2; small-df caveat stated — this is the best available null and
is frozen here).

**Per-window guard as previously written (pooled Δlc ≤ −0.10, 3 ON vs 3
control seeds):** SE(Δ) = 0.189·√(2/3) = 0.154. **P(false kill | Δ=0) =
Φ(−0.10/0.154) = Φ(−0.65) ≈ 0.26 per window; familywise over 3 flagged
windows ≈ 1 − 0.74³ ≈ 0.59.** R15's suspicion confirmed: grossly uncalibrated
— the old guard was a coin-flip component shredder.

**Repaired per-window guard (SEALED):** flag OFF iff pooled Δlc ≤
**−z·SE(Δ)** with per-window z = 1.834 (one-sided α_kill = 0.0333 =
0.10/3), evaluated only at the window's sealed look after seed 3. At the
frozen σ̂ this is a boundary of **−0.28 lc/game**; familywise false-kill over
the ≤3 remaining flagged windows = 1 − 0.9667³ = **0.097 ≤ 0.10** (R15's
target). Honest MDE statement: a TRUE −0.20 regression trips this guard with
only ~30% probability — the per-window guard is a catastrophe tripwire, not a
fine instrument; the real net-negative protection is the cumulative dismantle
branch (§6.2), whose −0.10 threshold at SE 0.144 trips a true −0.10 regression
with 50% and a true −0.25 with ~85%.

**Mechanism-prong false-kill:** ≈ 0 under the null by construction — the
mechanism observables are deterministic counters ((a): fired-before-every-death
predicate, 49/49 on the certified seeds; (f): 0 idle turns, 49/49; (b):
recurrence counter), not statistics. A mechanism prong can only false-kill via
a code defect, which the canary-before-seal rule (A10) exists to catch.

---

## §9. EWM Stage-1 re-price (authors' job per R15) + the Stage-1 gate

### §9.1 Re-price on reliable carriers only

Dry-run facts (`runs/ewm_dryrun/report.md`): held-out saturation does NOT
transfer on-trajectory; cross-seed shadow step-accuracy of the reliable
carriers — tn36 0.53–1.00 (≥0.98 on 2 seeds), tr87 0.77–0.82, tu93 0.73–1.00,
ls20 0.64–0.92, ft09 0.56–1.00 (L1-scoped; the gpt56 depth probe measured
**0.07** on L2+ states — a direct depth-transfer measurement). vc33 (0.24–0.67)
and s5i5 (0.13–0.30) abort at step 0 on most plans and are STRUCK from the
target set.

**Q5 answered (what fraction was carried by vc33/s5i5):** the +0.5 ceiling
assumed 3 L1 conversions from {ls20, tn36, tr87, vc33, s5i5}. Candidate L1
point values: ls20 3.57, tn36 3.57, tr87 4.76, vc33 3.57, s5i5 2.78 (Σ =
18.25 pts). vc33+s5i5 = 6.35/18.25 = **35% of the candidate point mass**.
Additionally the "Qwen clears 0 levels" premise was WRONG for tn36 and ls20
(both clear L1 in ≥1–2 war_eval seeds), removing another 7.14/18.25 = 39%.
**≈74% of the original ceiling's basis is gone; the surviving new-clear
candidate is tr87 alone (+4.76 pts = +0.19/draw).**

**Depth-bounded arithmetic (fidelity^L; a deterministic-wrong sim is a wall,
not retryable noise — replans only help where alternate paths exist):**

| channel | value (pts) | acc range | assumed L | survival acc^L | expected pts |
|---|---|---|---|---|---|
| tr87 L1 new clear | +4.76 | 0.77–0.82 | 8–15 | 0.05–0.28 → w/ ≤3 replans ≈0.1–0.4 | +0.5–1.9 |
| tu93 L1 speed (base 19; current 0.96/2.99/0.01) | +0.9 mean marginal | 0.73–1.00 | 19 | 0.003–1.0 | +0.1–0.9 |
| ls20 L1 speed (current 0.46) | +3.1 potential | 0.64–0.92 | ~20 | <0.01–0.19 | +0.0–0.6 |
| ft09-L1 reliability (worst-seed repair; overlaps banking's +3.17) | +1.6 post-overlap | 0.56–1.00 | 43 | ~0–1.0 | +0.0–1.0 |
| tn36 (already at base, 3.57 achieved v1) | 0 | — | — | — | 0 |
| **sum** | | | | | **+0.6–4.4 pts** |

**Re-priced Stage-1: expectation ≈ +0.02–0.18 rail per draw, central ≈ +0.08
— down ~2.5× from the +0.10–0.30 the deep-read carried.** The undiscounted
ceiling (~+0.47) survives only as arithmetic; fidelity^L makes it unreachable
at measured accuracies. Stated honestly: EWM Stage-1 is still the largest
registered non-model line, but it is no longer plausibly wall-sized on its
own, and its central value is seed-fragile (tu93/ft09 accuracies swing
0.56–1.00 across seeds — the same aliasing pathology as §10).

### §9.2 Pre-registered Stage-1 gate (A14 form — EXISTS as of this filing)

**Blocking entry conditions (all before any window is consumed):**
1. Latent-state audit complete with per-game keying classification (§10).
2. **Cheap measurement (llm-agents):** BFS-plan step-accuracy on the 10 local
   engines matching the Kaggle build, on **sim-derived (not teacher-forced)
   states**. Sealed threshold: **≥0.70 at plan depth ≤10 on ≥3 of the 5
   reliable carriers**; FAIL → Stage-1 parked at zero window cost.
3. A10 canary: plan/abort/fallback triggers fire ≥1/run on ≥5 games on the
   compressed bench (already demonstrated in the dry-run replay).
4. Full-panel sign-off (new asset class in the kernel).

**Window prongs:** mechanism = plans executed ≥1/run on ≥5 games AND ≥1
`plan_done` on ≥2 games AND 0 post-abort deadlocks AND the per-game
live-fidelity breaker fires after k=3 step-0 aborts (emitting
`fallback reason=budget`); non-inferiority = §8 boundary; score = cumulative
look only, like every other component.

**Resync-before-abort = CONTRACT CHANGE v1.1 (not an ADAPT footnote), OFF by
default.** Sealed bounds if enabled: ≤1 resync per plan; a resync is a pure
recompute from the observed settled frame (0 live actions; wall-clock ≤2s
measured before adoption); 2 consecutive resync-mismatches → abort (loop
guard); own canary counter (`RESYNC` events ≥1/run on ≥3 games in dry-run
replay before the flag ships). Wasted-action bound: ≤ +0 live actions by
construction; the cost is wall-clock only, priced before adoption. OCM-style
pre-execution validation (arXiv:2607.02846) is adopted INSIDE the executor
spec — it runs before actions and therefore does not weaken fail-closed;
priced at sim speed (ms/step, negligible against 196–211 tok/s decode).
Double-run rejection stays (0 selfdiff / 11,747 lines — free).

---

## §10. Latent-state audit — REGISTERED (blocking prereq for EWM Stage-1 AND banking)

**One root cause, three symptoms (R15 rl-planning M2 + prog-synthesis O2):**
the observable state KEY aliases hidden state (timers / counters / phase) on
frame-deterministic engines. Symptoms: (i) predict-metric recurrence 0.465 on
engines that are 0/25-divergent; (ii) EWM step-0 abort dominance with small
median diff-cell counts (1–56 cells — timer rows, hidden-counter phase);
(iii) the N5 `prune_trace` bug (leading board_changed=False actions that
mutate hidden state).

**Registration:** offline, N5 deterministic replay traces, $0, 0 GPU-h,
computable this week. Scope: for each of the 25 games, (i) classify whether
the game carries hidden phase (recurrence non-determinism of
(state,action)→next-frame under the observable key); (ii) test whether
augmenting the key with action-count / level / GAME_OVER-count / candidate
phase counters restores determinism; (iii) output a per-game keying
classification with a qualifying bar of **≥0.99 recurrence determinism** on
that game's trace pairs (sealed here, before the audit runs). Protocol
document: `learnings/war_room/latent_state_audit_protocol.md` (drafted in
parallel today; the sealed bar above governs regardless of drafting details).

**Blocking consequences (sealed):** the EWM Stage-1 window (§9.2 condition 1)
and the banking window (§6.3 W3) may NOT open before the audit reports.
Banking's replay trigger depends on (state,action) replay fidelity — the same
key. The (c)+Reki resurrection path (§7) additionally requires the audit's
qualifying keying. No other component is blocked ((a)'s budget counter and
(f)'s continuation logic key on nothing aliased).

---

## §11. W0 control-arm seed count for the binding look (systems #12 / Q6)

**Question:** the cumulative look is 3 certified ON seeds vs "W0"; W0 has
n=1. A paired design without pairs, as R15 put it.

**Sealed answer: the control band is the 4-run set {war_eval_v1, war_eval_v2,
war_eval_v3, w0_eval_s1} — n=4 control runs; per-game control value = the
4-run per-game mean; paired unit remains the game (A14.1).** Legality of
admitting the three (f)-less ledger-OFF seeds: (f)'s counting bound is
**0.00, sealed before W0 ran** (ITERATION_LOG 2026-07-18), and W0's
descriptive screen confirmed it (16 levels ∈ [13, 22]; no game below the
3-seed floor; the 49 recovered game-overs were all recovered by the base
harness path in the war-eval seeds too). Under a sealed zero-effect bound the
ledger-OFF seeds ARE W0-equivalent on score, and using them quadruples the
control band at zero pushes. What may NOT be done with this band (R15): cite
W0's 16 levels as evidence of (f) benefit — the citable claim is
levels-in-band only. Author lean of 07-19 (1 seed suffices) stands for the
MECHANISM check (deterministic, 49/49); the CUMULATIVE look uses the n=4
band above.

**Pre-registered fallback (decision tree, sealed now):** if R16 rules the
(f)-less seeds inadmissible as controls, 2 additional W0 seeds run before the
look (2 pushes, 4.4 GPU-h — headroom exists in the §12 ledger) and the band
becomes {w0_s1..s3}, n=3. No other configuration is permitted.

---

## §12. SENTINEL_BUDGET proposal (UNSEALED — design decision for R16 to rule on) + quota ledger

**The problem:** the eval regime is UNCAPPED (`max_actions_per_game=None`),
and the sentinel is by design a silent no-op with no budget to warn against
(`sentinel_build_2026-07-19.md` open risk 1). Cell 2 of the W1 eval notebook
MUST export `SENTINEL_BUDGET=<value>` or the whole component is inert and the
window is void.

**Derivation from the certified logs (`runs/kernel_pulls/war_eval_v{1,2,3}/
summary.txt`):** the binding budget in BOTH regimes is the per-game token
envelope, not an action cap — per-game tokens sit at ≈56k–64k (median ≈63k)
across all 75 game-runs, an effectively uniform cap. Tokens per action:
1,559,428/3638 = 428.6 (v1), 1,680,057/4026 = 417.3 (v2), 1,604,469/3985 =
402.6 (v3). **Implied per-game action capacity = 63k / (403–429) ≈ 147–156
actions.** Realized medians: 117 / 144 / 165 actions/game; means 145.5 /
161.0 / 159.4. The eval runs complete 25 games in 2h12m at 196–211 tok/s; the
~8h scored rerun across 25+ games reflects more games and scheduling, not a
larger per-game envelope — the per-game token cap is the same harness config
(assumption; verifiable by the same tokens/game grep on any scored-run pull).

**Proposal: `SENTINEL_BUDGET=150`** (per level attempt, the patch's unit),
thresholds default 50/75/90% → warnings at actions 75 / 113 / 135 of an
attempt. Rationale: 150 = the token-implied scored-regime action capacity
(147–156, stable across all three certified seeds), which is exactly the
budget the sentinel is meant to model. Checks against the recorded deaths:
sb26 move-limit death at 140 — all three warnings precede; lp85 GAME_OVERs at
131–133 — the 50%/75% warnings precede (the in-game 60-click resource on lp85
is a game mechanic the sentinel cannot and should not model); tu93's
301-action L1 grind — the full ladder fires with 150 to spare. **Mandatory
pre-seal check (A10):** re-run `compressed_canary.py` at B=150 on the three
recorded seeds; the W2 gate does not seal unless ≥5 games fire per run
(predicted comfortably: 17/25 games exceed 88 recorded actions). Post-run
verification: grep the build log for `SENTINEL v=1 kind=budget_threshold`;
**zero events on a run containing any ≥75-action attempt ⇒ the budget was
unset ⇒ the window is VOID (not FAIL)** — the feedback_kaggle_dataset_code_sync
class of silent no-op is thereby excluded from ever counting as evidence.

**Quota ledger (systems #12 — the A14 look and A15 replicate, previously
unscheduled, now scheduled):** rail = free Kaggle GPU builds, 30 GPU-h/wk;
one 25-game eval ≈ 2.2 GPU-h.

| week | item | GPU-h |
|---|---|---|
| Jul 21–27 | W1 (a): canary@B=150 (CPU, 0) + 3 certified seeds | 6.6 |
| Jul 21–27 | W2 (b): 3 certified seeds | 6.6 |
| Jul 21–27 | A17 72B screen (4 games full budget + tokens/s bench on the named SKU) | ~10 |
| Jul 21–27 | latent-state audit (offline CPU) | 0 |
| | **week total** | **≈23.2 / 30** |
| Jul 28–Aug 3 | **A14 binding cumulative look**: 3 FULL-budget seeds of the final stack (discharges A15 by construction, §6.1) | 6.6 |
| Jul 28–Aug 3 | W3 banking IF B+ approved | 6.6 |
| Jul 28–Aug 3 | EWM Stage-1 IF §9.2 gate passes | 6.6 |
| Jul 28–Aug 3 | fallback W0 seeds 2–3 IF §11 fallback triggered | 4.4 |
| | **week total (max path)** | **≈24.2 / 30** |

Push budget 2/day is the binding constraint on the max path (≈11 pushes over
7 days); the conditional items are mutually orderable within it.

---

## §13. Seal hygiene, su15, A17 cross-reference

**Seal hygiene (methodology N6), adopted as standing procedure:** every
threshold sealed in this document (the §7 resurrection bar 0.90; the §9.2
cheap-measurement bar 0.70/L≤10/3-of-5; the §10 keying bar 0.99; the §8
z=1.834 boundary; the §6.2 dismantle −0.10; the §12 B=150 pending R16 ruling)
is extracted verbatim into its measurement's own hash-committed thresholds
file under `runs/sealed/` BEFORE the measurement script first runs; results go
to separate append-only artifacts. The circulation stamp on this document
hash-commits the master copies.

**su15 (rl-planning minor):** A12 exclusion HOLDS for the sealed cumulative
look. Registered now: after the A13 GPT-5.6 re-probe completes, an amendment
MAY re-admit su15 for war-v4 and EWM evaluations only, with full-panel
sign-off; it is never re-admitted retroactively into any look already sealed.

**A17 (4/5 reviewers):** the gate-boolean repair (GO iff [≥2 levels AND
actions ≥90% of 27B] **OR** [beats Σ null_adj with registered margin]),
comparator definition over the 3 certified 27B seeds, marginal-result rule,
and SKU naming/verification are discharged in
`learnings/war_room/a17_72b_screen_scope.md` (filed 2026-07-19, incorporating
the multimodal-harness finding: the swap target is Qwen2.5-VL-72B-AWQ). Not
re-litigated here; this document only schedules its quota (§12) and records
that the screen is pre-Aug-1 blocking.

---

## §14. Part D republished — strategy priorities with (d) and (c) struck

| priority | line | re-priced basis | status |
|---|---|---|---|
| 1 | EWM Stage-1 plan-execute-verify on reliable carriers {tn36, tr87, tu93, ls20, ft09-L1} | expectation **+0.02–0.18 rail, central +0.08** (§9.1); largest registered non-model line | BLOCKED by §10 audit + §9.2 cheap measurement + sign-off |
| 2 | (a) budget sentinel | ceiling +0.06, expectation +0.01–0.03 | **W1 owner**; built, canaried; SENTINEL_BUDGET ruling pending (§12) |
| 3 | war-v4 72B multimodal screen (A17) | **the only registered wall-closer** | scope doc filed; pre-Aug-1 |
| 4 | (b) diff summarizer | ceiling +0.06, expectation +0.01–0.03 | W2, guard per §8 |
| 5 | banking-fixed | ≤ +0.15 pre-A16-haircut, expectation +0.03–0.08 | conditional (B+): sign-off + A16 + §10 audit |
| 6 | su15 GPT-5.6 re-probe (A13) | epistemic repair, ~$10 | after (f) local rig |
| 7 | filler | E[max@~106] ≈ 1.39; ~29% touch 1.44 | every window nothing credibly beats (+0.06–0.12 rule) |
| — | ~~(c)+(d) / (c)+Reki refutation flag~~ | ~~+0.10~~ | **STRUCK: (d) A18-killed; (c) killed §7** |
| — | ~~(g) budget re-allocation~~ | — | dead (A20) |

---

## §15. Dream digest review (Dreams/2026-07-19-124559.md)

Reviewed as required. The 2026-07-19 KAOS digest is a recency-only dry-run
cycle (0.00s, window all-time): 3 episodes (2 completed, 1 in flight), $0
spend, and **skills_scored=0 — exactly what the sealed expectation predicts**
(the skill library is empty; `feedback_kaos_improvements` documents that GPU
benchmarks remain impractical locally, so no skills have been admitted). The
hot-memory table is retrieval-flat (all hits=0, scores 0.48–0.50 = pure
recency prior over the R15-cycle documents — it is correctly surfacing this
campaign's active corpus, which is a sanity signal, not information), the
Hebbian graph is empty, and no failure fingerprints or consolidation
proposals were emitted. The cold list surfaces pre-campaign v8–v14 documents
as natural archive candidates; no action required. **Nothing actionable;
nothing panel-worthy.**

---

## §16. Directive-discharge table (every R15 directive → where discharged)

| # | R15 directive | discharged at |
|---|---|---|
| 1 | §2 stack sum republished with (d) removed (post-kill ceiling ≈ +0.21) | §2R (sums shown: 0.27 − 0.06 = +0.21 B+; +0.09 B−) |
| 2 | §4 Δclears re-derived under BOTH banking branches | §4R |
| 3 | P(pass) republished, binomial-sketch assumptions explicit | §5R |
| 4 | (c) disposition DECIDED; Part D corrected before ratification | §7 (KILL + counting bound + sealed resurrection path); §14 |
| 5 | α re-derivation (N2): name family or reset 0.05 one-sided | §3R (family named; conjunctive; α = 0.05 sealed) |
| 6 | Cumulative dismantle branch (M4): Δlc ≤ −0.10 → (f)-only | §6.2 (sealed; null trip-rate 0.24 published) |
| 7 | Guard false-kill calibration (N3): publish P(false kill), repair if >0.10 | §8 (0.26/window, familywise 0.59 → SE boundary z=1.834, familywise 0.097) |
| 8 | Budget regime of binding look stated (N5) | §6.1 (FULL budget; discharges A15 by construction) |
| 9 | KNOW#5 conditioned on keying (O2-1) | §7 (annotation) |
| 10 | Latent-state audit registered, blocking EWM Stage-1 AND banking | §10 (offline, N5 traces, $0; 0.99 bar sealed) |
| 11 | Reki-keying predict_metric re-run w/ sealed threshold before any (c) window | §7 (converted to sealed resurrection precondition, bar 0.90) |
| 12 | EWM re-price: reliable carriers, Q5, fidelity^L, Stage-1 gate exists, resync = contract change w/ bounds, OCM priced, cheap measurement | §9.1–§9.2 |
| 13 | A17 gate boolean/comparator/SKU repairs | `a17_72b_screen_scope.md` (07-19), cross-referenced §13 |
| 14 | Quota ledger incl. A14 look + A15 replicate scheduled (systems #12) | §12 ledger |
| 15 | W0 control-arm seed count for the cumulative look stated | §11 (n=4 band {3 war-eval + w0_s1}; sealed fallback) |
| 16 | (a) mechanism observable deterministic (O5) | §6.3 / sentinel build report (49 deaths / 0 violations) |
| 17 | W0: default-layer adopt, no seed-2, 16 levels not citable, band claim only | §1.3, §11 |
| 18 | Seal hygiene: thresholds hash-committed pre-measurement (N6) | §13 |
| 19 | su15: keep A12 exclusion; register post-A13 re-admission path | §13 |

All 19 directives discharged. The recalibrated A14 gate — as amended by §3R
(α), §6.2 (dismantle branch), §8 (guard boundaries), and §11 (control band) —
**seals on this circulation.**

END OF R16 REPUBLICATION
