## Summary (2 sentences)
v4 is the most responsive revision in this process: the scheduler is properly ledgered as killed-by-fact-check, the screen-power bootstrap I demanded was run and its result (sd 0.52) was allowed to abolish 1-seed screens, the park-FP claim was retracted with a measured number, and hard governance rules H1–H4 now exist. However, the plan's two remaining live legs (R0 port bundle, R1b restart lever) both route through a 3-seed Kaggle gate whose statistical power is uncharacterized — and possibly near zero given the team's own variance numbers — while R1b's exploratory EV is concentrated in three games and swings by 6× across plausible wall discounts, and a referenced normative section (§Instruments) does not exist in the shipped document.

## Objections

**Prior-round objection status (all thirteen):**

**[RESOLVED] RL-M2r** — Carried; nothing new due.

**[PARTIALLY-RESOLVED] RL-M3r (BFS dry-run)** — Unchanged; contingent on the Aug 3 measured tokens/action one-pagers, which v4 confirms "stand." Nothing due now.

**[PARTIALLY-RESOLVED] RL-M5r (compression thesis / fork audit)** — The audit finally started (pull #1 in `runs/fork_audit/kevin_v9b/`, fork band mapped, banking correctly killed by §Semantics ii). The deliverable — per-graft diff, game-agnosticism check, LB attribution — is not yet delivered; status upgrades to RESOLVED only when the bundle gate fires with that evidence attached.

**[RESOLVED] RL-A** — Carried.

**[RESOLVED] RL-B (track routing)** — The owed rule-text amendment is delivered as the LA-N6 porting gate (bundle → Track A with per-delta ablation attribution; countable-mechanism ports may route Track B under the fixed template). This is what I said I would verify in v4; verified.

**[RESOLVED] RL-C** — Carried.

**[RESOLVED] (D) Scoring/budget semantics** — Carried; residual verification burden tracked under (L) below.

**[PARTIALLY-RESOLVED] (E) λ₀ calibration** — The Track B template rule (generative null, event-count distribution from the replay corpus, 95th-percentile threshold, honest power, no per-event-rate powers) is adopted verbatim in the change-log, and deferral of the concrete derivation is legitimate since no Track B lever is currently live. But the rule purportedly lives in "§Instruments," which does not exist in this document — see new objection (M). E cannot close until the normative text exists somewhere certifiable.

**[PARTIALLY-RESOLVED] (H) Distribution truncation** — The sentinel + dynamic budgeting are implemented and this copy ends with the sentinel; genuine progress. But the printed "sha256" is a 16-hex-character prefix, not a full digest (see MINOR (S)), and the v3 sections incorporated by reference — R2 decision table ("still due"), §Windows, §Risks — remain uncertified by this panel, now joined by the missing §Instruments.

**[RESOLVED] (I) Window-spent-during-open-fact-check breach** — All three fixes implemented exactly as prescribed: draw #2 pulled and replaced with control σ-draw #6; ledgered ABORTED-BY-EXTERNAL-EVIDENCE citing §Semantics only (value-independent criterion); H1–H3 codified; the 0.90 permanently excluded from the control pool. This is what a resolved process objection looks like.

**[PARTIALLY-RESOLVED] (J) Q2 estimator conditioning / park FP** — Park FP measured (10.1%, −0.068/draw), "parks cost nothing" retracted, the conditioning defense (stuck seed → same-game exchangeable draws) is correct as stated, and the replay is properly downgraded to exploratory. The four-cell table is only fully populated by the pre-registered R1b run, which has not happened, and the confirmatory design has two new holes I file as (N) and (O).

**[RESOLVED] (K) 1-seed screen power** — The exact bootstrap I specified was run (sd 0.52; P(Δ≥+0.19|null)=0.34), the screen was abolished, H4 codified, and phase-1 closure upheld against the sign-negative Kaggle mean. Resolved — but the measurement's implications for 3-seed gates were not followed through; see new objection (P).

**[PARTIALLY-RESOLVED] (L) Semantics verification burden** — Live confirmation extended from n=1 to n=4 games at 4-decimal agreement, which validates the crush *formula*. The "one run per game_id" and "L2+ counts cleanly" claims on the *LB server* remain assumed, and the team's declaration that check (a) is "impossible" is contested — see new objection (Q).

---

**New objections:**

**[MAJOR] (M) §Instruments does not exist in the document —** It is cited twice as the normative home of two binding rule sets: the Track B statistical template (RL-E disposition) and the rail hierarchy (Kaggle-binding / pod-development, RL-K disposition). The sentinel confirms the document is complete, so the section was not truncated — it was never written. Rules that exist only as change-log summaries are not certifiable and will drift. Fix: v5 must contain §Instruments as full normative text (template, rail hierarchy, H1–H4 cross-references), or inline these rules and delete the dangling references.

**[MAJOR] (N) R1b's seed-split does not protect against game-level concentration, and the private set is 55 *unseen games* —** The exploratory gains concentrate in three games (ft09 +3.0, tn36 +0.88, tu93 +0.60); selecting (t,cap) on seeds 1–5 and estimating on 6–10 controls seed-overfitting but the *games are shared across both halves*, so the held-out Δ is still a bet on the prevalence of ft09-class deep-recovery games — a prevalence that is unestimated for the private set. Fix: the R1b pre-registration must add a leave-one-game-out jackknife on the held-out Δ; if dropping the single top game pushes Δ below +0.10, the lever is a 3-game bet and must be reported as such in the gate decision. Additionally, state which rail generated the null10 transcripts: if they are pod runs, the entire replay EV lives on the instrument v4 itself just declared non-binding after a demonstrated 3-seed sign flip.

**[MAJOR] (O) The R1b kill decision sits entirely inside the wall-discount uncertainty band, and the +0.10 threshold is unjustified —** The exploratory Δ is +0.22 / +0.105 / +0.037 across discounts 1.0 / 0.6 / 0.365 — a 6× swing straddling the kill threshold at every plausible discount. "Wall-cost cell from measured per-action wall in null10 histories" is not a specification: pre-register the exact formula mapping measured per-action wall time to the discount, the uncertainty on that measurement, and commit to evaluating the kill at the *conservative bound* of the discount, not its point estimate. Separately, derive +0.10 from something — e.g., the opportunity cost of the Track A window it would consume versus the port bundle's P-weighted EV — rather than asserting it.

**[MAJOR] (P) The team's own numbers imply the 3-seed gate — through which every future lever routes — may be underpowered by 3–5×, and two variance estimates are mutually inconsistent —** Bootstrap 1-seed paired-Δ sd = 0.52, yet control σ̂ = 0.074 implies a two-draw Δ sd of ≈ 0.105: a 5× discrepancy the document flags ("≈ 7× σ̂_control") but never reconciles. If 0.52 is the right replicate noise, a 3-seed gate has se ≈ 0.30 against target effects of +0.08–0.20 — the gate is then a coin flip and the entire Track A architecture (port bundle gate, R1b's 2-seed screen → 3-seed gate) is built on it. Fix (free, from existing transcripts): reconcile the two estimates (game-resampling vs run-replicate components of variance), then publish the implied se and power of the 3-seed Kaggle gate against +0.10 and +0.12; if power < ~0.5, H4's minimums must rise or the gate statistic must change (e.g., paired per-game Δ with variance-weighted games).

**[MAJOR] (Q) The wheel-vs-server identity check declared "impossible" has a feasible aggregate form —** LB submissions do return a *total* score. For a deterministic frozen build, reconstruct the expected LB total from local transcripts via the wheel formula and compare against the actual returned totals for all six historical submissions; six near-zero aggregate residuals would strongly support wheel-server identity (including the one-run-per-game_id claim, since a second run would perturb totals), at zero cost. The current mitigation ("aggregate gates only") is fine as a stopgap, but do not record (a) as impossible when its aggregate variant is sitting in the submission history.

**[MINOR] (R) Park-FP cell leaks held-out data into the confirmatory design —** The 10.1%/−0.068 figure was measured on the full corpus; fixing it as a constant in a protocol that estimates on seeds 6–10 imports information from the estimation set. Recompute the park-FP cell on seeds 1–5 only, or show the seed-split values agree.

**[MINOR] (S) The "sha256" is a 16-hex prefix, not a digest —** A full sha256 is 64 hex characters; a 64-bit prefix is fine as a document ID but should be labeled as a prefix, and the panel pipeline should verify against the full digest, otherwise the H-mechanism verifies less than it claims.

## Questions for the authors (numbered)
1. Which rail produced the null10 transcript corpus — pod or Kaggle kernel? If pod, how does R1b's replay EV survive the rule that the Kaggle rail is the binding instrument, given the documented 3-seed sign flip?
2. Reconcile the bootstrap 1-seed paired-Δ sd of 0.52 with control σ̂ = 0.074 (which implies ≈0.105): what fraction is game-resampling variance vs run-replicate noise, and what is the implied se and power of a 3-seed Kaggle gate against +0.10/+0.12?
3. Where is §Instruments? Will v5 ship it as certifiable normative text?
4. What is the derivation of the +0.10 R1b kill threshold, and at exactly which wall-discount value (point estimate or conservative bound?) is it evaluated?
5. What is the held-out Δ under leave-one-game-out — specifically, dropping ft09?
6. Can you run the aggregate LB-residual check (Q) on the six historical submissions this week, and will you record the result either way?
7. When a Track B lever next goes live, will the generative-null/λ₀ derivation be pre-registered as a standalone artifact before the statistic is used, and certified by whom?
8. Will the R2 decision table be brought before this panel before the first forensic transcript is read, per the standing commitment?

## What I cannot judge
Kaggle platform mechanics (kernel push limits, whether LB rerun artifacts are truly never returned — I contest the *aggregate* claim in (Q) but cannot verify the per-game claim); the licensing/ToS propriety of porting public fork code; dollar-cost accounting of the RunPod reserve; and the internal LLM-agent/prompt design of the duck builds. I defer those to the engineering and platform reviewers.

## Verdict: MAJOR-REVISION

## Score: 6/10