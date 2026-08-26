# Round 21 Panel Directives (synthesized 2026-07-27)

Panel role: ADVISORY (Sunday strategic review under 07-27 restructure). MAJOR-REVISION is the known absorbing state and does not block the build rail.

## VERDICT SUMMARY
Unanimous 5/5 MAJOR-REVISION (scores 4,4,4,5,5; 20 MAJOR, 0 FATAL): ledger hygiene is real but the strategy has no quantified path from 1.33 to the ≈1.49 gold cutoff, and the single best-evidenced intervention — the boristown vLLM readiness gate behind a 1.47 anchor — is unscheduled.

## TOP DIRECTIVES (ranked)

1. **[TODAY — 5/5 reviewers: llm-agents, methodology, prog-synthesis, rl-planning, systems] Schedule the boristown vLLM readiness-gate A/B on the frozen fork, or reject it in writing with evidence.**
   The fork-diff memo (`learnings/war_room/fork_diff_boristown_2026-07-24.md`) shows the *only functional diff* behind an externally verified 1.47 (above gold cutoff) is a ~one-line vLLM readiness gate. Mechanistic story: without the gate, early-episode actions are burned while the server is still loading — plausibly explains the whole 0.82–1.33 band as cold-start variance. Pre-register a small arm (n=3–5 gated vs frozen draws, one-sided test at the anchor-implied effect size). Every reviewer calls its absence the biggest hole in the document. Note: "frozen" discipline is against scratch-building, not against a minimal audited diff (systems Q4).

2. **[TODAY — 4/5: llm-agents, prog-synthesis, rl-planning, systems (+methodology Q6)] Publish A17 numeric PASS/kill thresholds before v6 fires.**
   A boot canary is not a performance gate. Reproduce G1–G4 verbatim in the brief; pre-register the triple (ρ_action ≥ X ⇒ expected LB band [a,b]; ρ_action < Y ⇒ 72B route dead, slots revert to Z). Show the throughput arithmetic: GPU tier targeted, measured tok/s + TTFT from v5, tokens/action, projected actions inside A17_WINDOW_S=7920. A 72B that boots but decodes at 4 tok/s can score *below* the frozen 7B-class fork.

3. **[TODAY — 5/5] Run the canonical ledger API refresh before citing n=13.**
   The 07-26 (0.84) and 07-27 (1.02) draws exist only in briefs; `runs/lb_ground_truth.md` ends at n=11 (refreshed 07-25). Arithmetic checks out (12.662/13 = 0.974 ✓) but "validated vs ledger" is currently validated vs a brief — miniature replay of the stale-template failure mode. Run the live-API refresh path, or mark unrefreshed draws provisional.

4. **[TODAY — 3/5: methodology, prog-synthesis, rl-planning] Un-shelve the exploration arm with a pre-registered rule; the n=1 shelving is statistically indefensible.**
   0.71 is z ≈ −1.8 vs frozen (not significant; frozen's own floor is 0.82, war arm contains 0.76). The 0.80 per-draw harm-pause fires falsely ~11% of the time even if the arm equals frozen's mean. Publish: date for sentinel/A21 draw #2, n ≥ 4–5 per disposition decision, sequential stopping rule (e.g., re-shelve only on 2 consecutive <0.80 or mean of first 3 <0.80).

5. **[LATER — 5/5 in various forms] State the explicit theory of reaching ≥1.49 by Nov 2 and justify filler slot economics.**
   P(frozen draw ≥1.49) ≈ 2×10⁻⁴; expected max of ~100 more fillers ≈ 1.33–1.38. Per-arm win-probability estimates (frozen / A17 / war-v4 / sentinel), draw-budget allocation, decision date at which A17 is declared dead and war-v4 becomes the sole bet, and a weekly GPU-hour + submission-slot ledger (systems). If slots bank rather than expire, default spend should be gated-fork A/B or exploration, not filler.

6. **[LATER — 2/5: methodology, prog-synthesis] Replace the empirical min–max "band" with a proper interval** (mean ± 2s or tolerance interval) plus a stated out-of-interval trigger; report MK/CUSUM as "insufficient power" not "no drift"; spec the full daily trigger family with per-trigger and family-wise error rates.

7. **[LATER — 2/5: prog-synthesis, rl-planning] War-v4 spec must open with falsifiable criteria** (skill-reuse rate, per-skill verification pass rate, delta over frozen on held-out games), not the three-way-convergence narrative — Schema is self-reported/unreplicated and amplification is not replication.

## NAMED CONDITIONS (falsifiable escalation triggers the panel names for the restructure)
- **NC-1 (llm-agents, prog-synthesis):** Any push that changes model class (7B-family → 72B) — or any *first push of a new artifact version* — requires either Sunday review or a pre-registered *performance* gate; a boot gate does not qualify. A v6 push mid-week qualifies as escalation-worthy.
- **NC-2 (methodology):** Panel acknowledgment of Sundays-only is CONDITIONAL on each sealed arithmetic gate publishing (a) per-draw false-trigger probability under the frozen empirical distribution, (b) family-wise trigger rate over an unreviewed 6-day week, (c) enumerated event classes that force weekday escalation.
- **NC-3 (systems):** v6 auto-fire must carry a hard resource abort (e.g., kill at 3 GPU-h / wall-clock cap) so a hung vLLM boot or TP deadlock on the dataset route cannot burn a week's quota unreviewed.
- **NC-4 (llm-agents):** v6 may not consume a scored draw until an offline harness replay (≥200 tool-call attempts on recorded game states) reports Qwen2.5-VL parse-failure and fenced-recovery rates; residual dead-action rate must be quoted.
- **NC-5 (prog-synthesis, rl-planning, systems):** A17 kill condition must be numeric and pre-registered: ρ_action < Y (stated before v6) ⇒ 72B route dead, slots revert.
- **NC-6 (systems, llm-agents, methodology):** Readiness-gate A/B: if the gated arm's band shifts up, the entire daily-filler regime switches to the gated variant.
- **NC-7 (prog-synthesis, rl-planning):** Sentinel re-shelve only under the pre-registered multi-draw rule (see Directive 4), never on n=1.
- **NC-8 (methodology):** No n≥12 statistic is cited in any decision until the corresponding draws are API-verified in the canonical ledger; maximum brief-to-verification lag should be stated.

## OPEN-QUESTION ANSWERS (panel stance on the brief's 4 questions)

1. **A17 v5/v6 sealed-window discharge plan:** OBJECTION (4/5). The panel does not object to the sealed window itself but objects to v6 firing on a *boot* PASS: G1–G4 text is absent from the reviewed document, no throughput/score thresholds exist, and no kill number is stated. Conditions: Directives 2 + NC-3/NC-4/NC-5 before v6 consumes a scored slot.
2. **Restructure acknowledgment:** CONDITIONAL acknowledgment. Conditions are NC-1 and NC-2 (model-class/first-push escalation; published gate operating characteristics). Without them the restructure "trades review latency for uncontrolled error rates" (methodology).
3. **War-v4 spec timing:** Leaning START NOW IN PARALLEL, with a decision rule — rl-planning proposes "war-v4 spec starts now iff it consumes zero GPU-h and no draw slots"; prog-synthesis endorses the direction (library learning) but requires the spec to open with falsifiable criteria and discounts the convergence narrative. No reviewer argues for strictly-after-A17. Posing it open-ended to a Sunday-only panel bakes in ≥7-day latency (rl-planning MINOR).
4. **Exploration draw #2 cadence:** FIRE IT, under a pre-registered rule with a date (e.g., "sentinel draw #2 after A17 v6 lands"), n ≥ 4–5 before any disposition, sequential stopping boundary with stated error rates. llm-agents adds: freed/idle slots should default to gated-fork A/B or A21 draw #2, not filler. As written, "the exploration program is structurally guaranteed to be killed by noise" (methodology).

## DISSENTS WORTH NOTING
- No true dissents — unanimity on verdict and on the top-2 directives. Variance is in emphasis only: methodology/prog-synthesis scored 5 (crediting procedural discipline), rl-planning/llm-agents/systems scored 4 (weighting strategic hollowness).
- systems uniquely questions raw 72B feasibility (~40 GB AWQ weights vs Kaggle GPU tiers; single-digit tok/s decode plausible) — the only reviewer arguing A17 may be infeasible *in principle* on-platform, not just unmeasured.
- prog-synthesis uniquely asks whether public LB is best-of-submissions (determines whether filler draws have *any* residual option value).
- llm-agents uniquely flags the unmeasured fenced-recovery adapter as "prompt-it-better hand-waving" and sets the ≥200-replay bar (NC-4).
