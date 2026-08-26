# Daily Brief — 2026-08-04 (Tuesday; weekday cadence, no panel)

## 1. Results deep-dive

### 1a. Overnight scored draw + state reconciliation
- **0.97** (frozen-fork filler, 08-04 00:07Z, API COMPLETE). Interior (z ≈ +0.17 vs n=20
  stats) → record ledger **n=21, mean 0.9443, s 0.1514**. Third consecutive interior draw
  after the 0.65/0.68 dip — the NC-15 stationarity verdict keeps holding.
- LB 08-04: head frozen (KOJIMA 1.86, Andy liu 1.69, GeniusYY 1.64); **gold cutoff
  (top-13) 1.56** (08-03: 1.54; 07-28: 1.49), top-10 cutoff 1.58; new name FOYSAL 1.61
  at #5. ~0.02/day drift at the gold line. Our 1.33 slides on pure drift; only a
  mechanism win moves us.
- Reconciliation: 08-03 session analyzed the A22 v1 seed-1 screen (M1/M2/M3 ALL FAIL,
  K3 fired, arm paused — full deep-dive in yesterday's brief §1b, still the operative
  result) and queued the filler, but wrote no log entry and did not build v2. Backfilled
  in ITERATION_LOG 08-04; **v2 build is today's single development target** (in flight,
  results appended in §4 below).

### 1b. No new scored evidence today
The v1→v2 pivot decision was made 08-03 on the seed-1 screen; nothing new scored since.
Tonight fires the frozen filler (queued, trusted-fork ALLOW re-verified); v2 is a free
Kaggle build-time eval, measurement-only — nothing can pass promotion gates today.

## 2. Discussions sweep (`learnings/sweeps/discussions_sweep_2026-08-04.md`)
Third consecutive quiet window. 1 new thread (teammate-wanted ad, hand-played "100%") →
**IGNORE** (hand-authored per-game DSL = the overfitting `feedback_arc_generalization_first`
forbids). 1 host comment (Greg Kamradt): upstream **PR arcprize/ARC-AGI-3-Agents#74 merged
— bare `arcprize.org` now canonical over `three.arcprize.org`** → **ADAPT, EXECUTED
same-day**: H2 host-gate substring test widened to any-subdomain regex + 2 new tests
(23/23 PASS); production frozen-fork preflight re-verified **ALLOW unchanged** (addendum in
`duck_eval/a17/preflight_host_gates_2026-08-02.md`). No deadline/scoring/infra change.
Absence-signal: no public disclosure above the 1.17 duck since 07-31 while gold drifts up —
4-day pattern; consider dropping the sweep to every-other-day if it holds through 08-06.

## 3. Research sweep (`learnings/sweeps/research_sweep_2026-08-04.md`)
Methodology correction: arXiv 2608.* IS indexed via the export API (the monthly listing
page lies); yesterday's "not yet indexed" was wrong — future sweeps use the API. Recovered
a dense 08-01..08-03 slice: 18 relevant items, **3 ADOPT / 7 ADAPT**. Headlines:
- **PRO-LONG (2607.20064): +18.0 pts on the full ARC-AGI-3 public set at 4.2-5.8×
  fewer tokens via a lossless interaction log — nothing summarized.** First hard
  context-management number on our benchmark. Its regex/log-search method is
  private-track-illegal for us; we adapt the *discipline*: evict from presentation,
  never from the record.
- **Four independent lines now say v1's digest failure was the predicted outcome, not an
  execution error**: PRO-LONG, LightMem repro, 2605.12978 (GPT-5.4 loses 54% of
  previously-solved ARC tasks even consolidating from ground truth), 2608.00902 (eviction
  beats summarization under imperfect proxies). v2's demote-digest/promote-eviction
  direction is corroborated pre-outcome.
- **Toxic-digest mechanism named**: Authority Collapse (2608.01679) + Governance
  Decay/Constraint Pinning (2606.22528 — violations 0% when a constraint survives
  compaction, 38% when elided; training-free pinning restores 0%). Implication: hygiene
  *filters* are blind to omission by construction → **pin invariants (refuted list) rather
  than gate-protect them**, and status-tag claims (CONFIRMED/HYPOTHESIS/REFUTED).
  → **v3 candidate / R24 agenda — NOT injected into v2 mid-build** (sealed-spec
  discipline; v2 already demotes the digest to empty-allowed).
- **Blind-batching fix**: 2608.02464 — LLM-free deterministic post-batch verifier
  (60-96% catch, 0/63 FP, ~200µs/step): abort batch → single-action on mismatch.
  Cheapest concrete fix if batching harm survives RETAIN-off. → v3 candidate / R24.
- Absences: ARC blog quiet (3rd sweep); no new ARC-AGI-3 arXiv since Tycho (07-30);
  banking/replay quiet (2nd sweep). Tooling gap: arcprize.org/leaderboard is
  client-rendered — needs a browser-driven check for drift.

## 4. Today's development target: A22 compaction v2 (build in flight)
Spec sealed 08-03 (brief §4): region-aware eviction w/ pinning (system prompt +
scientist-note + latest reasoning never evicted; stale action-episodes first),
hygiene-gated demoted digest (empty-allowed), RETAIN decoupled OFF-by-default,
suppress-cut-while-stuck; M1/M2/M3 + K1-K5 inherited; M2 gains budget-relief attribution.
Build agent deliverables: sealed intent file `learnings/war_room/
a22_compaction_v2_prereg_2026-08-04.md` pre-build, patch v2 + extended smoke, builder
regression, NO pushes (orchestrator verifies then pushes; 2/2 slots free).
**[RESULTS APPENDED BELOW WHEN BUILD REPORTS]**

## 5. Open questions
- (carried) Does region-aware eviction alone (RETAIN off, digest empty-allowed) recover
  M1 ≥ −0.128 / worst ≥ −1.0 vs war-eval seed-1? The ONLY question v2 seed-1 answers.
- For R24 (Sunday): v3 riders ranked — (a) pin refuted-list as surviving invariant
  (constraint-pinning result), (b) status-tag claims, (c) post-batch verifier; plus
  PRO-LONG lossless-record framing as the lane's theoretical anchor.
- Sweep cadence: drop discussions to every-other-day if quiet through 08-06?
- Boristown A/B: DORMANT (NC-14), unchanged. A17: closed, pin dissolved 08-03.
