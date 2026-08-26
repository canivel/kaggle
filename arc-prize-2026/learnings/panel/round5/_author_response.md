# Author Response — Panel Round 5 on `path_forward_v1_2026-07-13.md`

**Author:** claude-fable-5 · **Date:** 2026-07-13 · **Revision delivered:** `learnings/path_forward_v2_2026-07-13.md`
**New evidence produced for this response:** a counterfactual replay on `runs/null10/merged_null_benchmark.json` (250 runs, per-action events), reported as §E of v2. Key outputs: 152 true level clears; 12/152 (7.9%) after cumulative action 120; 21/152 (13.8%) after action 90; 7/152 (4.6%) in the final 20% of run wall; first-clear median 25 / p90 114 / max 194 actions; truncating lc==0 runs at action 120 saves 1.20M tokens (7.9% of total, not 20.2%); restart-detector FP 14/126 (11.1%) at 90, 9/126 (7.1%) at 120; per-game lc vectors show 5 dead, 4 grinders, 16 flip games.

Legend: **ACCEPTED** (objection correct, plan changed), **PARTIALLY REBUTTED** (evidence narrows the objection; remainder accepted), **REBUTTED** (evidence contradicts the objection).

---

## rl-planning

**[FATAL] R1's EV contradicts Finding #3 (STUCK). Required: counterfactual replay.** — **ACCEPTED for L1, PARTIALLY REBUTTED for L2. The demanded replay was run.**
- L1: the replay confirms the objection quantitatively — only 7.9% of clears occur after action 120 and 4.6% in the final fifth of wall, so the marginal value of compute reallocated to progressing games is ≈0. The +0.10–0.30 EV is **retracted**; L1 survives only as a non-scoring throughput guard (parking) inside the merged attempt scheduler, with zero claimed EV.
- L2: the objection's second clause — "your 10/10-seed death data says these failures are deterministic per game" — holds only for the 5 dead games. The per-game lc vectors show **16 games flip across seeds** (ft09 [0,0,0,1,2,2,2,2,2,3]; tn36 [0×6,1,1,1,2]; cn04 [0×5,1×5]; …). Failure on flip games is within-game attempt variance, which is exactly what a restart re-rolls. The +0.13 EV was derived (headroom §3, lever 2) from the 8 material flip games only, never from the dead 5 — re-derivation in v2 §E gives +0.126, minus a measured FP cost (11.1% of good runs) → net **+0.10 ± 0.05**, which is now the only R1 EV claimed. (Also answers your Q1 and Q5.)

**[MAJOR] L1/L2 threshold interaction incoherent.** — **ACCEPTED.** v2 merges them into one attempt scheduler with explicit semantics: restart trigger uses a per-attempt counter (resets on restart); the cap (2 restarts/game) uses a cumulative counter that never resets; after the cap the game is parked (resumes only if everything else is finished); park dominates restart. Dead games are bounded at ≤270 actions then parked — strictly less than today's grind-to-wall; no thrash loop is possible. Simulated dead-game behavior on null10 transcripts ships with the build. (Answers Q2.)

**[MAJOR] R2 gate treats 9 clustered trials as independent.** — **ACCEPTED.** Primary endpoint changed to level-2 clears on **≥2 distinct games** with ≥2/3 seeds each; null computed at the game level (0/30 in null10, per-game rule-of-three p<0.1, two independent game events → p<0.01 under full seed clustering). (Answers Q4: yes.)

**[MAJOR] "Duck+BFS hybrid" under-specified.** — **ACCEPTED.** Your dichotomy is correct and v2 commits to the honest branch: there is no forward model, so BFS is **executed** systematic exploration — node = the duck's segmentation-graph state abstraction (color/shape-hash/containment/adjacency, already built into the harness), edges = real env actions. It is stall-scoped (fires only after 90 stalled actions on a level), **replaces** analyzer turns instead of adding context, hard-capped at 40 actions/burst, LLM only for frontier scoring — and its one-page design (due Aug 3, pre-registered) must state the predicted tokens/action delta against the >10% kill criterion before any GPU spend. On the RHAE concern: the quadratic penalty you cite is on actions per *completed* level; a stall-scoped burst on a level that is otherwise never completed risks the efficiency term only if it clears — the design doc must include that arithmetic. (Answers Q3.)

**[MAJOR] Window gate uninterpretable at σ̂ upper endpoint.** — **ACCEPTED.** v2: (a) σ̂ df grows ~1/week because sentinels and best-build redraws are control-class draws (df≈8 at first candidate gate, ≈15+ by Sep); (b) a pre-registered **sign-flip rule**: decision differing across CI endpoints → one extra window; still flipped → decide at point estimate, mark provisional, and provisional promotes must re-confirm inside the stack gate. Error rates are printed as numbers, not decoration: false-promote at Δ=0 = 2.4%/24.5% (σ = 0.074/0.213); false-kill of +0.10 = 4.9%/28.3%.

**[MAJOR] Nov-2 compression thesis asserted with zero evidence.** — **ACCEPTED (downgraded to a tested hypothesis).** Correction of premise: the Milestone-1 winner is not unexamined — it is our substrate (Tufa duck harness; see `panel_research_winners.md`, built from the winner's open-sourced notebook and source bundle). The missing audit is of the **1.28–1.56 fork band** above it. v2 R0 adds that audit with a pre-registered test: count game-ID-keyed logic/public-set tuning per top fork. Supporting (not sufficient) evidence already on file: `panel_research_lb.md` — private ≈55 vs 25 public games; 82% of the ≥1.0 cohort appeared within 2 days of the June-30 open-source; no dominant exact-score cluster above 1.0 (many small forks). Pre-registered failure consequence: if top forks are predominantly game-agnostic, Risk #1 is rewritten and contingency windows convert to wholesale porting. (Answers Q6.)

**[MINOR] Exploration closed as a family when only one implementation was falsified.** — **ACCEPTED.** The state-coverage metric (distinct segmentation-graph states per 100 actions, null vs stuck) is added to R2 forensics; the closure is re-scoped to *always-on context injection*, and stall-scoped exploration remains on the R2 shortlist.

**Q7 (difficulty ratio refit):** yes — the R1 gate reports per-game official deltas and the 0.55 ratio is refit before any Sep-1 re-baseline arithmetic uses it.

## llm-agents

**[MAJOR] Public Milestone-1 winning notebook absent.** — **PARTIALLY REBUTTED, then ACCEPTED in stronger form.** The duck *is* the Milestone-1 winner (Tufa, 1.21; Qwen3.6-27B-FP8 on vLLM, python-as-only-tool coding agent, segmentation-first object graph — full teardown in `panel_research_winners.md`, including source-bundle inspection). Our 1.02 vs their 1.21: Cottaar himself reports the cleaned notebook "hasn't had the same lucky result." So "run the winner on null10" is our baseline, already done 10×. What the objection correctly exposes: no audit of the **forks above us** (leader 1.56 = Mathurin Ache duck fork; the 1.28–1.56 band is small public deltas — model swaps, prev-frame tweaks). v2 R0 adds the mandatory fork-delta audit: diff table, game-agnostic vs game-keyed classification, top-2 portable deltas each gated on 2 windows. (Answers Q1.)

**[MAJOR] R2 shortlist is hand-waving; no forensics→intervention mapping; interventions are themselves injections.** — **ACCEPTED.** v2 pre-registers the decision table (mechanic-never-stated → search hybrid, not prompts; stated-then-lost → context fix; stated-but-misexecuted → grounding fix; low coverage → BFS), one-page designs with token-overhead budgets due Aug 3, and component fidelity gates before GPU. On the injection tension: the shipped interventions are **stall-scoped** (fire only in a state the data shows is already lost ~86–92% of tokens deep) and replace turns rather than adding always-on context; each must independently pass the >10% tokens/action rule. On Q2 (why defer the verbalization check to Aug 3): accepted — it is the first forensics task, started immediately at R2 open (Jul 21); Aug 3 is the deadline, not the start.

**[MAJOR] Reallocator EV contradicts Evidence #3.** — **ACCEPTED; the demanded replay was run** (see rl-planning FATAL disposition and v2 §E). L1's EV retracted; the freed-compute recipient is re-specified: not "progressing games" (measured worthless) but **fresh attempts on flip games** via restarts, whose EV survives at +0.10 ± 0.05. (Answers Q4.)

**[MAJOR] L1/L2 interaction can thrash on dead games.** — **ACCEPTED.** See attempt-scheduler semantics above: cumulative cap → no loop; dead games bounded at 270 actions; precedence specified; null10 replay of the combined policy ships before the first window. (Answers Q3.)

**[MAJOR] R2 gate doesn't test generalization.** — **ACCEPTED.** ≥2 distinct games primary + **r11l held out of intervention development entirely** with a pre-registered directional prediction that can block confirmation. (Answers Q5: yes.)

**[MINOR] Baseline staleness under drift.** — **ACCEPTED.** Rolling 6-draw control; pre-registered refresh trigger (per-game >2 game-level sd or aggregate >2σ̂ → freeze in-flight gates, confirm with one sentinel, re-center on post-drift draws). (Answers Q6.)

**Q7 (rival volatility):** not yet measured; the R0 fork audit is the pre-registered instrument.

## methodology

**[MAJOR] "Redraws are dead" fails the both-endpoints rule.** — **ACCEPTED with recompute.** E[best-of-80] = 1.03 / 1.10 / 1.45 and P(≥1.17 in 80) = 0.000 / 0.032 / 1.000 at σ = 0.044 / 0.074 / 0.213. Bootstrap on the empirical draws is degenerate for the max (5 points, observed max 1.02) — noted; the 5-draw range (0.20) gives a range-based σ̂ ≈ 0.086, close to the point estimate, but the decision no longer depends on that: the **ban is replaced by a priority rule** (gates preempt; every unused window defaults to a best-build redraw), so the upper-endpoint lottery is retained at zero opportunity cost while ~45 windows still fund gates. (Answers Q2.)

**[MAJOR] R1 EV self-refuted.** — **ACCEPTED; replay run.** See rl-planning FATAL disposition: marginal-compute channel measured ≈0 and retracted; the surviving mechanism is attempt variance on flip games (+0.10 ± 0.05, derivation in v2 §E). The RHAE wall-clock channel you hypothesized is claimed at zero. (Answers Q3.)

**[MAJOR] Window gate assumes a stationary baseline.** — **ACCEPTED.** Rolling control (6 most recent control-class draws), candidate-vs-recent-control differences, pre-registered drift alarm that freezes in-flight gates and re-centers on post-drift draws. (Answers Q5: it was frozen; it is now rolling.)

**[MAJOR] R2 gate independence false.** — **ACCEPTED.** Game-level endpoint (≥2 distinct games, ≥2/3 seeds each); game-cluster null p<0.01. Recomputed false-pass against a single-game fluke: a one-game artifact now fails by construction. (Answers Q6.)

**[MAJOR] v1 rehabilitation is unregistered subgroup analysis.** — **ACCEPTED.** v1 = +0.13/+0.42 reclassified as a hypothesis; the version-stability criterion (game version-suffix match, an objective property of the run logs) is now pre-registered for all future paired scoring, but its original application was post-hoc and is treated as such; explore-min enters R3 with **zero prior credence**; the p=0.22 decomposition is no longer used as evidence. (Answers Q4.)

**[MINOR] Multiplicity and additive stacking.** — **ACCEPTED.** Expected false promotions across ~6 families: 0.14 (σ̂ point) to 1.5 (upper). Mitigations: sequential gating against the updated rolling control (promoted builds enter the control, so later gates measure marginal effect, not assumed additivity); final stack must beat the **vanilla-duck fork** on its own pre-registered 2-window test. (Answers Q8.)

**[MINOR] Secondary no-collateral gate powerless at n=3.** — **ACCEPTED.** Replaced by a per-game paired sign statistic over the ~20 version-matched non-wall games (game as exchangeable unit), in actions/token units.

**Q1 (six baseline draws / σ̂ provenance):** draws {0.82, 0.89, 0.93, 0.95, 1.02} (draw #6 pending at R0), same frozen fork, Jun–Jul windows; σ̂ = 0.074 is the sample sd and therefore *includes* any within-period drift variance; exchangeability is exactly what the rolling-control + per-game drift statistic now monitors instead of assuming. **Q7 (0.55 ratio):** fit from one build family (substrate arms vs null10 against official draws); CI not meaningfully estimable at that n — which is why v2 makes the window gate the sole promotion authority and refits the ratio from R1's per-game official deltas.

## prog-synthesis

**[MAJOR] R2 has no engineering substance / no component gates.** — **ACCEPTED.** Free CPU component gates pre-registered before any GPU dollar: segmentation-fidelity ≥90% object-identity/transition consistency on 20 hand-labeled grinder frames (the duck's 4-connected single-color fragmentation is the known failure to quantify); exec-WM ≥70% next-state prediction on held-out logged transitions; fail → struck from shortlist. One-page designs with per-node token costs due Aug 3. (Answers Q2: the object graph is the duck's built-in segmentation — id, color, shape-hash, containment, adjacency — and no, its fidelity has not been measured; that is now a blocking gate. When extraction is wrong, BFS dedup fails conservatively — revisits, bounded by the 40-action burst cap.)

**[MAJOR] R2 primary gate incoherent with its own confirmation gate.** — **ACCEPTED.** Game-level endpoint (≥2 distinct games, ≥2/3 seeds each). Your one-game arithmetic is the decisive point: ~+0.17 local → ~+0.09 official < +0.12 threshold, so v1's local gate could pass what its confirmation gate must kill. The 2-game endpoint (~+0.35 local → ~+0.19 official) restores consistency. Pre-registered decision for a one-game crack: near-miss → at most one reserve retry, never a window. (Answers Q3.)

**[MAJOR] L1 EV contradicts STUCK; run the replay.** — **ACCEPTED; replay run** (v2 §E): late clears 7.9%/4.6%; truncation saves 7.9% of tokens (correcting the 20.2% framing); EV retracted; R1 re-scoped to the restart lever with per-game attribution in the derivation. (Answers Q5.)

**[MAJOR] Forensics overfit loop on the same 3 games.** — **ACCEPTED.** r11l pre-registered as directional holdout, excluded from all intervention development; a pre-GPU falsifiable r11l prediction is required and a contradicting result blocks confirmation. On Q4 (mechanic ground truth): labels are human-derived from playing the three public games; that leakage is real and is contained by (a) the holdout, (b) the no-game-ID rule, (c) the intervention classes being mechanism-shaped (search, pinning, verification), not mechanic-shaped.

**[MAJOR] null10 treated as frozen in a drifting environment.** — **ACCEPTED.** null10 is version-pinned per game (game IDs carry version-hash suffixes; e.g. `wa30-ee6fef47`); paired scoring restricted to version-matched games as a pre-registered criterion; refresh triggers defined (any gate-relevant game bump invalidates its rows; >5 bumps void in-flight local gates pending partial re-collection). (Answers Q6.) On Q7 (offline scorer exactness): reproduced on the v2 gate report (`gate_report_FINAL`, Δ = −0.542 exactly) — one build family; exactness under a version bump is precisely what the version-pinning rule no longer assumes.

**[MAJOR] Window gate unstable across σ̂ CI; family-wise inflation.** — **ACCEPTED.** Sign-flip rule + provisional-promote re-confirmation inside the stack gate + df growth via control-class draws + printed false-promote numbers (2.4%/24.5%). We did not adopt "promote only if it holds at the upper endpoint" for every gate: at σ = 0.213 that is a +0.22 threshold, which false-kills essentially every realistic candidate (+0.10–0.19 expected) — the freeze criterion binds at both endpoints, intermediate gates use the flip rule. This is a deliberate power/error trade and is now stated as such.

**Q1 (ar25 p-values):** unit is the game; p = P(3 random null seeds from the 10-seed null10 empirical distribution score this game this low) — a game-level bootstrap against a 10-seed null, not an n=3 t-test; replication = the same tail in both independent arms (0.009 and 0.008). This is consistent with the ban on n=3 *ungated A/B inference*, which concerns arm-level conclusions, not per-game diagnostics.

## systems

**[MAJOR] $15–25 has no provenance; token inflation ignored.** — **PARTIALLY REBUTTED, provenance ACCEPTED.** Rebuttal on inflation: runs are wall-clock-capped (7,920 s/game; ~25 games compressed to ~12 GPU-h/seed by 28-way concurrency on one vLLM server — `panel_research_winners.md` compute profile), so a treated arm that inflates tokens/action does the same wall time for the same dollars; it loses *actions*, not money. Provenance now stated: 3 seeds ≈ 36 GPU-h on the null10 A40-class SKU at $0.39–0.79/h = $14–28. Accepted: a mandatory 1-seed calibration before the full spend, with a pre-registered de-scoping table (>$35 → 2 seeds × 3 games at 2/2-seed gate; >$50 → 1-seed screen, windows only). (Answers Q1, Q6: the fallback is de-scoping, never cannibalizing — and confirmation was never GPU, see next.)

**[MAJOR] No Kaggle quota ledger.** — **ACCEPTED.** Ledger in v2: scoring is organizer-side; submitting an already-committed kernel version costs 0 participant GPU-h (the daily daemon has operated on this basis since May); a *new build* costs one ~12 h commit → cap 2 new builds/wk; weekly total ≈ 27 h ≤ 30 h; measured commit-hours verified at R0 exit with a pre-registered fallback (cap 1/wk, cut contingency candidates first). (Answers Q2.)

**[MAJOR] L1 EV contradicts STUCK.** — **ACCEPTED; replay run.** Direct answers to Q3 from the replay: clears after action 120 = 7.9%; clears in the final 20% of run wall = 4.6%; the 9 still-progressing runs are 7% of good runs — the marginal-budget curve is flat and L1's EV is retracted (v2 §E).

**[MAJOR] 80-window budget contradictory; 45+ unaccounted.** — **ACCEPTED.** Reconciled ledger in v2: 45 enumerated (R0 1, fork ports 6, R1 5, R2 3, explore-min 3, stack 3, second-generation contingency 8 — named class, same pre-registration rules, filled only by candidates that clear the same bar; sentinels 11, corrected from 8; selection 5) + ~34 default best-build redraws. The redundancy-vs-ban contradiction is dissolved: the ban is now a priority rule, and redraws of anything but the current best build remain banned. (Answers Q4.)

**[MAJOR] Gate underpowered at upper endpoint; R2 confirmation authority ambiguous.** — **ACCEPTED.** False-promote quantified (2.4%/24.5% per family; expected 0.14–1.5 over ~6 families) with the stack-gate/vanilla-floor backstop; ambiguity resolved: **the free 2-window gate is the sole promotion authority**, the GPU run is a local screen, and "confirmation sweep" is deleted from the reserve's uses. (Answers Q5, Q7.)

**[MINOR] Local wall-clock doesn't transfer for RHAE.** — **ACCEPTED.** All local gates in actions-per-completed-level and tokens/action; wall-clock banned as a local gate metric.

**[MINOR] Sentinel blind at n=1/week.** — **ACCEPTED.** Per-game drift statistic from the sentinel event log (>2 game-level sd flags), which is the trigger for the rolling-control refresh.

---

## Summary of dispositions

- **Accepted outright:** 21 objections (all interaction-semantics, gate-statistics, baseline/drift, ledger, component-gate, holdout, subgroup-analysis, and power items).
- **Accepted after running the demanded analysis:** the L1-EV family (RL-F1, LA-M3, ME-M2, PS-M3, SY-M3) — the counterfactual replay confirmed the objection for L1 (EV retracted) and preserved a smaller, better-derived restart EV (+0.10 ± 0.05) from the flip-game data.
- **Partially rebutted with evidence:** LA-M1 (the "absent" winner notebook is our substrate; the real gap — the fork band — is now audited), SY-M1 (wall-capped runs make token inflation cost-neutral; provenance gap conceded and closed), RL-F1's determinism clause (16/25 games flip across seeds).
- **No objection was left unaddressed.**
