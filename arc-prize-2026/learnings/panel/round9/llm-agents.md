## Summary (2 sentences)

v4 is the first revision that arrives complete (sentinel present, sha256 stated), files the governance rules I demanded (H1–H4), executes the breach remediation for the 02:19Z draw, kills the sched lever by code forensics with live 4-decimal confirmation, and finally starts the assumption-free R0 leg — this is genuine progress, not disposition-table theater. However, three load-bearing problems remain: the team's own evidence base now shows a *vanilla resubmit at 1.21 versus their control mean of 0.922* while the entire proposed lever stack sums to less than that gap; the port-bundle attribution mechanism relies on the very local instrument v4 itself just demoted after a 3-seed sign flip; and R1b's survival is decided by a wall-discount parameter whose exploratory range (+0.22 → +0.037) straddles the kill threshold and whose derivation is not pre-registered.

## Objections

**Resolution of prior-round objections:**

**[LA-M1] Fork-delta audit — RESOLVED (execution).** R0 started 2026-07-13, pull #1 artifacts named (`runs/fork_audit/kevin_v9b/`), fork band mapped with three named forks, banking correctly killed via §Semantics(ii), and the LA-N6 gate is pre-registered. Prioritization complaint discharged; residual attribution problem re-filed as N13 below.

**[LA-M2] R2 shortlist / exec-WM metric — PARTIALLY-RESOLVED (carried, low urgency).** The checksummed v4 has now been filed, discharging the artifact-verification blocker, but the R2 decision table is still "due before first forensic transcript is read" — i.e., promised, not filed. Acceptable while R2 stays untouched and the $68 stays R2-gated; the table must exist before any transcript read, per v3's own language.

**[LA-M3] / [LA-M4] / [LA-M5] / [LA-m1] — RESOLVED** (dispositions stand).

**[N1] Vanilla gap — PARTIALLY-RESOLVED, and the new data makes it worse.** Execution started (credit), but the audit's own first finding — `caoyupeng` vanilla resubmit at 1.21 vs control mean 0.922 — is filed as "confirms the drift question" and then not acted on. See N12; the vanilla-gap experiment I originally demanded is now the highest-EV item in the entire plan.

**[N2] RESTART exchangeability pre-check — PARTIALLY-RESOLVED.** The live crush confirmation validates the *scoring formula*, not exchangeability. My specific fix — compare the empirical attempt-2 clear rate from the 18 actual restarts (4/18 ≈ 0.22) against the cross-seed proxy p used by the replay — is absent from the R1b pre-registration, which still says "same-game exchangeable draws" as an assumption. Fix: add a pre-registered exchangeability cell to R1b (exact binomial test of 4/18 vs proxy p per game class); if incompatible, the replay EV must be reweighted to the empirical rate before the kill/survive call.

**[N3] Tokens/action kill; 1.56 anchor — RESOLVED.** The v4-filing caveat is now closed by the complete filing.

**[N4] Incomplete document — RESOLVED.** This copy ends with the literal sentinel, sha256 and length are declared, and the pipeline fix is named (`scripts/panel_round.py`). The third-truncation clock stops.

**[N5] Wall-cost column — PARTIALLY-RESOLVED.** R1b names a "wall-cost cell from measured per-action wall in null10 histories" — that is the right source data. But the exploratory table applies three *arbitrary* discounts (1.0 / 0.6 / 0.365) with no derivation, and the answer flips across them. Re-filed with teeth as N14.

**[N6] Porting-gate track — RESOLVED (design).** The explicit choice is made: bundled Track A (+0.12 aggregate, 3 Kaggle-rail seeds, joint non-inferiority) with per-delta local ablations for attribution, Track B template fixed per RL-E. The design is internally inconsistent with §Instruments, however — see N13.

**[N7] Scoring-semantics method — RESOLVED.** Stands; residual wheel-vs-server risk is now honestly bounded (LA-N11 disposition), and the platform fact that LB reruns return no per-game artifacts is a legitimate closure of the (a) branch.

**[N8] Scored window on unapproved lever — RESOLVED.** Breach logged, draw #2 pulled and replaced with control σ-draw #6, ABORTED-BY-EXTERNAL-EVIDENCE ledgering with value-independent criterion, sched-v1's 0.90 excluded from the control pool, H1–H3 codified, and the audit-trail gate on `daily_submit.py` is live. This is exactly the fix I specified.

**[N9] Rail sign disagreement — RESOLVED (main), residual in N13.** The null-vs-null bootstrap (1-seed paired-Δ sd 0.52, P(Δ≥+0.19|null)=0.34) is the quantification I asked for; abolishing 1-seed screens and making the Kaggle build rail the binding instrument is the correct structural response. One provenance question below (Q3): this round's cross-rail exhibit (+0.169 pod / −0.73 Kaggle) does not match last round's reported pair (−0.54 local / +0.19 Kaggle) — I need the config mapping.

**[N10] Park-FP rate — RESOLVED.** Measured (10.1% of stuck-at-90 runs clear later; −0.068/draw), "parks cost nothing" retracted in writing, and the FP cell is fixed into the R1b pre-registration. This is the direct measurement I demanded.

**New objections:**

**[MAJOR] N12: The plan optimizes grafts onto a control that a public *vanilla resubmit* beats by ~0.29 — close that gap first or explain it.** Control pool mean is 0.922 (best draw 1.02); `caoyupeng`'s vanilla resubmit scores 1.21. The entire P-weighted lever stack (port bundle +0.08–0.20 × 0.4–0.7; restart +0.05–0.12 × 0.35) has lower expected value than simply matching an unmodified public baseline. This is either rail/config drift, model-version skew, or harness overhead in the frozen duck — all diagnosable for free. Fix: promote a "control-vs-vanilla diff" to R0 deliverable #1 — line-level diff of frozen-duck vs the 1.21 kernel's config (model id, temperature, prompt, action budget, tokens/action), plus one 2-seed Kaggle-rail run of the *literal* vanilla kernel under your harness; until this is explained, every graft EV estimate is anchored to a possibly-degraded base.

**[MAJOR] N13: Port-bundle attribution uses the instrument v4 just discredited.** §R0 gates the bundle on 3 Kaggle-rail seeds (good, per H4) but attributes per-delta effects via "local ablation evidence" — while §Instruments, in the same document, demotes pod/local to non-binding after a *3-seed* cross-rail sign flip (+0.169 vs −0.73). If the bundle passes at +0.12 but one graft is net-harmful (e.g., `shortcircuit` truncating exploration that pays off privately), local ablations cannot be trusted to finger it, and you will ship the harmful graft to the Nov-2 build. Fix: pre-register (a) local ablations at ≥2 seeds per H4, and (b) a conflict rule — if any local ablation's sign contradicts the bundle direction, the single most-suspect graft gets one leave-one-out 2-seed Kaggle-rail screen before the bundle is frozen into the deployment build.

**[MAJOR] N14: R1b's kill decision is controlled by an underived free parameter.** Exploratory Δ is +0.22 at discount 1.0, +0.105 at 0.6, +0.037 at 0.365, against a kill threshold of +0.10 — the lever lives or dies entirely on where the discount lands, and v4 nowhere states how 0.6 or 0.365 were obtained. "Measured per-action wall in null10" is a data source, not a model: fresh-context attempt-2 exploration inflates tokens/action (my N5 point, still uncomputed), so the effective discount is not the null-play per-action wall. Fix: before the confirmatory replay runs, file the discount derivation as a single number with its formula — (extra attempt-2 actions × measured attempt-2 tokens/action) against the per-game and session caps over the full 25-game draw — and state it in the pre-registration; a sensitivity grid chosen after seeing results is winner's-curse re-entry through the side door.

**[MINOR] N15: The "game-agnosticism check" in the R0 audit is named but undefined.** "No game-ID-keyed logic" is a hard constraint, but grafts can overfit the public set without ID keys — magic constants tuned to public grid sizes, color counts, or level-1 layouts (a plausible reading of `shortcircuit` and `recovery`). Fix: define the check as a concrete rubric in the audit deliverable — enumerate every literal constant in each graft diff and require a written game-independent justification or a parameterization; a graft failing the rubric routes to Track B or is dropped from the bundle.

## Questions for the authors (numbered)

1. (N12) What is the config delta between your frozen duck and the 1.21 vanilla resubmit — same model version, same prompt, same action budget? Will you run the literal public vanilla kernel on your rail as a 2-seed screen this week?
2. (N14) State the wall-discount number you will pre-register for R1b and show its derivation from the null10 per-action wall plus attempt-2 token inflation. If you cannot derive it before running, what stops the confirmatory replay from being exploratory-with-extra-steps?
3. (N9 residual) Map the two cross-rail exhibits: last round the panel was shown local −0.54 / Kaggle +0.19; this round shows pod +0.169 / Kaggle −0.73. Which configs (phase1-v1 vs v2), and where are the in-repo artifacts for each pair?
4. (N2) Will the R1b pre-registration include the 4/18 empirical attempt-2 clear rate vs cross-seed proxy comparison as a gating cell, yes or no?
5. (LA-M2) Hard date for the R2 decision table filing, given the Aug-3 one-pagers stand?

## What I cannot judge

The line-level correctness of the `arc_agi`/`scorecard.py`/`arcengine` citations in §Semantics (I accept the method and the 4-decimal live confirmation as strong corroboration, but I have not read the wheel myself — the code-forensics reviewer should countersign the WIN-gated `full_reset` path specifically). The statistical machinery of the 20k null-vs-null bootstrap and the exact-enumeration estimator's variance properties (methods/RL reviewer). Kaggle platform internals: whether LB reruns truly return no per-game artifacts, and kernel-push quota mechanics. RunPod cost realism for R2.

## Verdict: MAJOR-REVISION

## Score: 6/10

The governance layer, the semantics fact-check, the breach remediation, and the instrument recalibration are real and verifiable — this is the first round where the plan's epistemics are ahead of its execution rather than behind it. But N12 (your base underperforms a public vanilla resubmit by more than your whole lever stack), N13 (attribution via a discredited instrument), and N14 (a kill decision hanging on an underived parameter) are each load-bearing and each fixable within one round for zero dollars. Fix those three plus the N2 exchangeability cell and this passes.